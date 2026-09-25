"""入库流水线编排。

一次入库、多次查询。顺序：
    探测 → 播放代理 → 抽音频 → 静音切段 → ASR → 课件抽帧 OCR
         → 语义分段 → 分层摘要 → 向量索引

设计要点：
- 每个阶段都写状态，中断后可以看清卡在哪一步。
- 幻灯片抽帧用原视频（不是代理），时间轴与 ASR 一致（都基于原始时间）。
- 单课的 chunk 数只有 10²–10³，向量用 numpy 暴力检索即可，不需要 ANN。
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .embedding import EmbedderLike
from .glossary import Glossary, apply_to_segments
from .ingest.asr_volc import VolcASRClient
from .ingest.media import (
    build_proxy,
    detect_silences,
    extract_audio,
    pick_split_points,
    probe,
    slice_audio,
)
from .ingest.segment import attach_parents, build_chapters, build_chunks
from .ingest.slides import detect_slides
from .llm.client import LLMClient
from .schema import Segment, Video, VideoStatus
from .store import Store
from .summarize import build_summary_tree

log = logging.getLogger(__name__)


@dataclass
class Progress:
    stage: str
    done: int = 0
    total: int = 0
    message: str = ""

    @property
    def percent(self) -> float:
        return (self.done / self.total * 100.0) if self.total else 0.0


def video_id_for(path: Path) -> str:
    """用「路径 + 大小 + 修改时间」生成稳定 ID。

    刻意不做全文件内容哈希：2 小时课程视频动辄 1–2GB，每次入库全量读一遍
    代价太高。路径+大小+mtime 已足以判定「同一个文件」。
    """
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def reindex_videos(
    store: Store,
    embedder: EmbedderLike,
    video_ids: list[str] | None = None,
    *,
    progress=None,
) -> dict[str, int]:
    """只为已入库课程重建向量索引，**不重新转写**。

    存在的理由：本地嵌入模型是可选的，常见顺序是先入库一批课程、后下载模型。
    那时库里没有向量，向量召回静默失效，而重跑 ingest 会再付一次 ASR 的钱。
    这里只读已有的文本块、在本地编码，零 API 成本。

    返回 {video_id: 文本块数}；跳过的课程不在返回值里。
    """
    if not embedder.available:
        raise RuntimeError("嵌入模型不可用，无法重建索引")

    targets = video_ids if video_ids is not None else [v["video_id"] for v in store.list_videos()]
    counts: dict[str, int] = {}
    for i, video_id in enumerate(targets):
        chunks = store.get_chunks(video_id)
        if not chunks:
            continue
        vectors = embedder.encode([c.combined_text for c in chunks])
        store.upsert_embeddings(
            video_id, zip([c.chunk_id for c in chunks], vectors, strict=False)
        )
        counts[video_id] = len(chunks)
        if progress:
            progress(i + 1, len(targets), f"已重建 {video_id}（{len(chunks)} 块）")
    return counts


def repair_videos(
    cfg: Config,
    store: Store,
    video_ids: list[str] | None = None,
    *,
    embedder: EmbedderLike | None = None,
    dry_run: bool = False,
    progress=None,
) -> dict[str, dict[str, int]]:
    """按术语表纠正**已入库**课程的文本，不重新转写。

    存在的理由：术语表是随课程不断补充的。补完之后，已入库课程里仍留着错字——
    转写、块文本、块摘要、章节摘要、课件 OCR、全课摘要，六处都有。
    重跑 ingest 会再付一次 ASR 的钱，而转写根本没变，错的只是那几个名词。

    只做确定性替换，所以对同一份数据反复执行是幂等的（第二次全部计 0）。
    返回 {video_id: {字段: 替换次数}}。
    """
    glossary = Glossary.load(cfg.glossary_path)
    targets = video_ids if video_ids is not None else [v["video_id"] for v in store.list_videos()]
    report: dict[str, dict[str, int]] = {}

    for i, video_id in enumerate(targets):
        stats: dict[str, int] = {}

        def tally(field_name: str, fixes) -> int:
            n = sum(f.count for f in fixes)
            if n:
                stats[field_name] = stats.get(field_name, 0) + n
            return n

        # --- 转写
        segments = store.get_segments(video_id)
        if segments:
            _, fixes = apply_to_segments(segments, glossary)
            if tally("segments", fixes) and not dry_run:
                store.replace_segments(video_id, segments)

        # --- 块（正文 / 课件文字 / 标题 / 摘要）
        chunks = store.get_chunks(video_id)
        chunk_hits = 0
        if chunks:
            for c in chunks:
                for attr in ("text", "ocr_text", "title", "summary"):
                    val = getattr(c, attr) or ""
                    if not val:
                        continue
                    new, fixes = glossary.correct(val)
                    if fixes:
                        chunk_hits += sum(f.count for f in fixes)
                        setattr(c, attr, new)
            if chunk_hits:
                stats["chunks"] = chunk_hits
                if not dry_run:
                    store.replace_chunks(video_id, chunks)

        # --- 章节
        chapters = store.get_chapters(video_id)
        chapter_hits = 0
        if chapters:
            for ch in chapters:
                for attr in ("title", "summary"):
                    val = getattr(ch, attr) or ""
                    if not val:
                        continue
                    new, fixes = glossary.correct(val)
                    if fixes:
                        chapter_hits += sum(f.count for f in fixes)
                        setattr(ch, attr, new)
            if chapter_hits:
                stats["chapters"] = chapter_hits
                if not dry_run:
                    store.replace_chapters(video_id, chapters)

        # --- 课件 OCR
        slides = store.get_slides(video_id)
        slide_hits = 0
        if slides:
            for s in slides:
                val = s.ocr_text or ""
                if not val:
                    continue
                new, fixes = glossary.correct(val)
                if fixes:
                    slide_hits += sum(f.count for f in fixes)
                    s.ocr_text = new
            if slide_hits:
                stats["slides"] = slide_hits
                if not dry_run:
                    store.replace_slides(video_id, slides)

        # --- 全课摘要
        summary = store.get_video_summary(video_id)
        if summary:
            new, fixes = glossary.correct(summary)
            if tally("video_summary", fixes) and not dry_run:
                store.set_video_summary(video_id, new)

        # --- 文本变了，向量必须重算（否则检索还在按旧文本召回）
        if not dry_run and embedder is not None and embedder.available:
            fresh = store.get_chunks(video_id)
            if fresh:
                vectors = embedder.encode([c.combined_text for c in fresh])
                store.upsert_embeddings(
                    video_id, zip([c.chunk_id for c in fresh], vectors, strict=False)
                )
                stats["reembedded"] = len(fresh)

        report[video_id] = stats
        if progress:
            total = sum(v for k, v in stats.items() if k != "reembedded")
            progress(i + 1, len(targets), f"{video_id} 纠正 {total} 处")

    return report


class IngestPipeline:
    def __init__(
        self,
        cfg: Config,
        store: Store,
        *,
        llm: LLMClient | None = None,
        embedder: EmbedderLike | None = None,
        asr: VolcASRClient | None = None,
    ):
        self.cfg = cfg
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.asr = asr
        # 术语表在构造时载入：纠错要在转写落库前生效，晚加载就晚了
        self.glossary = Glossary.load(cfg.glossary_path)

    def run(
        self,
        video_path: Path,
        *,
        reuse_proxy: bool = True,
        reuse_slides: bool = True,
        skip_summary: bool = False,
        progress=None,
    ) -> Video:
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在：{video_path}")

        def report(stage: VideoStatus, done=0, total=0, message=""):
            self.store.set_status(video_id, stage)
            if progress:
                progress(Progress(stage.value, done, total, message or stage.label))

        video_id = video_id_for(video_path)
        work_dir = self.cfg.library_dir / video_id
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # ---------------------------------------------------- 1. 探测
            report(VideoStatus.PROBING)
            info = probe(self.cfg.media.ffprobe, video_path)
            video = Video(
                video_id=video_id,
                path=str(video_path),
                title=video_path.stem,
                duration_ms=info.duration_ms,
                status=VideoStatus.PROBING,
                size_bytes=video_path.stat().st_size,
            )
            self.store.upsert_video(video)

            # ---------------------------------------------------- 2. 播放代理
            # Chromium 不支持 AC3/EAC3、MKV/FLV 不可靠，不生成代理第一周就会打不开文件
            report(VideoStatus.PROXY)
            proxy_path = work_dir / "proxy.mp4"
            if reuse_proxy and proxy_path.exists():
                log.info("复用已有播放代理 %s", proxy_path)
            else:
                _reencoded, detail = build_proxy(
                    self.cfg.media.ffmpeg,
                    video_path,
                    proxy_path,
                    video_codec=self.cfg.media.proxy_video_codec,
                    audio_codec=self.cfg.media.proxy_audio_codec,
                    crf=self.cfg.media.proxy_crf,
                    preset=self.cfg.media.proxy_preset,
                )
                log.info("播放代理：%s", detail)
            video.proxy_path = str(proxy_path)

            # 代理建好后立刻落库，这样界面即使中途失败也能播放已转好的文件
            self.store.upsert_video(video)

            # ---------------------------------------------------- 3. 抽音频
            report(VideoStatus.AUDIO)
            audio_path = work_dir / "audio.mp3"
            if not (reuse_proxy and audio_path.exists()):
                extract_audio(
                    self.cfg.media.ffmpeg,
                    video_path,
                    audio_path,
                    sample_rate=self.cfg.media.audio_sample_rate,
                    bitrate=self.cfg.media.audio_bitrate,
                )

            # ---------------------------------------------------- 4. ASR
            # 转写是整条链路里唯一按小时计费的环节，也是唯一不可复现的环节。
            # 只要已有转写结果且音频没被重新生成，就直接复用——否则「摘要失败后
            # 重跑」会再付一次转写费，而转写结果本来是一模一样的。
            report(VideoStatus.ASR, 0, 1, "语音转写")
            existing = self.store.get_segments(video_id) if reuse_proxy else []
            if existing:
                segments = existing
                log.info("复用已有转写：%d 句（跳过 ASR，不产生费用）", len(segments))
            else:
                if self.asr is None:
                    raise RuntimeError("未配置火山 ASR 凭证，无法转写")
                segments = self._transcribe(audio_path, info.duration_ms, work_dir, report)
                log.info("转写完成：%d 句", len(segments))

            # 转写纠错：ASR 会把专有名词听错（讲师说 uiautomator，转写成
            # "URL to meta"）。错字会一路穿过切块、摘要、章节笔记，最后变成
            # 「通顺但错误」的结论——比明显乱码危险得多。
            # 必须在落库前纠正，否则下游全是脏的；且对复用路径也生效，
            # 这样改了术语表不用重新转写（转写是唯一按小时计费的环节）。
            segments, fixes = apply_to_segments(segments, self.glossary)
            if fixes:
                log.info(
                    "转写纠错 %d 处：%s",
                    sum(f.count for f in fixes),
                    "；".join(f"{f.wrong}→{f.right}×{f.count}" for f in fixes),
                )
            self.store.replace_segments(video_id, segments)

            # ---------------------------------------------------- 5. 课件
            slides = []
            if self.cfg.slides.enabled:
                report(VideoStatus.SLIDES, 0, 1, "课件抽帧与 OCR")
                cached = self.store.get_slides(video_id) if reuse_slides else []
                # 复用条件：幻灯片仍在库里，图片文件仍存在，且 OCR 要么关闭，
                # 要么已经产出过文字。最后一条同时覆盖了升级路径：
                # 先入库（无 OCR）后装 OCR，此时 cache 里全是空文字，会重跑。
                # OCR 是本地最贵的一步（单张 3–12 秒），整门课要几十分钟，
                # 重复做这一遍纯属浪费——但改过抽帧参数时需用 --refresh-slides。
                reusable = bool(cached) and all(Path(s.image_path).exists() for s in cached)
                if reusable and self.cfg.slides.ocr_enabled and not any(s.ocr_text for s in cached):
                    reusable = False
                    log.info("已有课件但没有文字，将重新抽帧并 OCR")
                if reusable:
                    slides = cached
                    log.info(
                        "复用已有课件：%d 张（其中 %d 张有文字）",
                        len(slides),
                        sum(1 for s in slides if s.ocr_text),
                    )
                else:
                    slides = detect_slides(
                        self.cfg.media.ffmpeg,
                        video_path,
                        info.duration_ms,
                        work_dir / "slides",
                        self.cfg.slides,
                        progress=lambda d, t, m: report(VideoStatus.SLIDES, d, t, m),
                    )
                    self.store.replace_slides(video_id, slides)
                log.info("课件：%d 张（其中 %d 张有文字）", len(slides), sum(1 for s in slides if s.ocr_text))

            # ---------------------------------------------------- 6. 语义分段
            report(VideoStatus.SEGMENT)
            chunks = build_chunks(video_id, segments, slides)
            chapters = build_chapters(video_id, chunks)
            attach_parents(chunks, chapters)
            self.store.replace_chunks(video_id, chunks)
            self.store.replace_chapters(video_id, chapters)
            log.info("分段：%d 块，%d 章", len(chunks), len(chapters))

            # ---------------------------------------------------- 7. 分层摘要
            summary_note = ""
            if self.llm is not None and not skip_summary:
                report(VideoStatus.SUMMARY, 0, len(chunks), "生成分层摘要")
                result = build_summary_tree(
                    self.llm,
                    self.store,
                    video,
                    chunks,
                    chapters,
                    progress=lambda d, t, m: report(VideoStatus.SUMMARY, d, t, m),
                )
                if result.video_title:
                    # 用模型给的标题，比文件名可读
                    video.title = result.video_title[:80]
                self._write_concepts(video_id, result)
                if result.degraded:
                    summary_note = result.describe()
                    log.warning(summary_note)

            # ---------------------------------------------------- 8. 向量索引
            if self.embedder is not None and self.embedder.available:
                report(VideoStatus.EMBED, 0, len(chunks), "建立向量索引")
                texts = [c.combined_text for c in chunks]
                vecs = self.embedder.encode(texts)
                self.store.upsert_embeddings(
                    video_id, zip([c.chunk_id for c in chunks], vecs, strict=False)
                )

            # 摘要降级时把说明留在 video.error 里（该字段同时充当「状态说明」），
            # 这样 UI / info 能看到，而不是只在日志里一闪而过。
            video.error = summary_note or None

            video.status = VideoStatus.READY
            self.store.upsert_video(video)
            if progress:
                message = "入库完成" if not summary_note else "入库完成（摘要部分降级）"
                progress(Progress(VideoStatus.READY.value, 1, 1, message))
            return video

        except Exception as exc:  # noqa: BLE001
            log.exception("入库失败")
            self.store.set_status(video_id, VideoStatus.FAILED, error=str(exc))
            if progress:
                progress(Progress(VideoStatus.FAILED.value, 0, 1, f"入库失败：{exc}"))
            raise

    # ------------------------------------------------------------------ 内部

    def _transcribe(self, audio_path: Path, duration_ms: int, work_dir: Path, report):
        """决定「一次性提交」还是「切段提交再按偏移合并」。

        默认走极速版 + base64：本地文件开箱即用，不需要对象存储。
        """
        size = audio_path.stat().st_size
        single_shot = size <= self.cfg.asr.max_upload_bytes

        if single_shot:
            result = self.asr.transcribe_file(audio_path)
            segments = [
                Segment(
                    idx=i,
                    start_ms=u.start_ms,
                    end_ms=u.end_ms,
                    text=u.text,
                    speaker=u.speaker,
                    words=[
                        {
                            "start_ms": int(w.get("start_time", 0)),
                            "end_ms": int(w.get("end_time", 0)),
                            "text": w.get("text", ""),
                            "confidence": w.get("confidence"),
                        }
                        for w in u.words
                    ],
                )
                for i, u in enumerate(result.utterances)
            ]
        else:
            # 超长：先找静音切点，切在静音中点，避免切断句子
            log.info("音频 %.1fMB 超过单次上限，将切段转写", size / 1e6)
            silences = detect_silences(self.cfg.media.ffmpeg, audio_path)
            spans = pick_split_points(
                duration_ms,
                silences,
                chunk_ms=self.cfg.asr.chunk_ms,
                overlap_ms=self.cfg.asr.overlap_ms,
            )
            report(VideoStatus.ASR, 0, len(spans), f"转写中（分 {len(spans)} 段）")
            parts_dir = work_dir / "asr_parts"
            parts_dir.mkdir(parents=True, exist_ok=True)

            def slice_fn(start: int, end: int, index: int) -> Path:
                dst = parts_dir / f"part_{index:03d}.mp3"
                slice_audio(self.cfg.media.ffmpeg, audio_path, dst, start, end)
                report(VideoStatus.ASR, index, len(spans), f"转写第 {index + 1}/{len(spans)} 段")
                return dst

            segments = self.asr.transcribe_spans(spans, slice_fn=slice_fn)

        if not segments:
            # 全片无语音时给出可照做的指引，而不是留下一个空课程让人困惑
            raise RuntimeError(
                f"整段音频没有识别到任何语音（时长 {duration_ms / 1000:.0f} 秒）。"
                "请确认：1) 该视频确实含人声讲解（不是纯音乐/纯演示）；"
                "2) 音轨没有被抽成静音（可用播放器打开 data/library 下的 audio.mp3 试听）。"
            )
        return segments

    def _write_concepts(self, video_id: str, result) -> None:
        if not result.concepts:
            return
        lines = ["# 术语表", ""]
        for concept in result.concepts:
            from .schema import ms_to_hms

            lines.append(f"- **{concept.term}**（{ms_to_hms(concept.start_ms)}）：{concept.definition}")
        path = self.cfg.library_dir / video_id / "concepts.md"
        path.write_text("\n".join(lines), encoding="utf-8")

    def purge(self, video_id: str) -> None:
        """删除某课程的入库产物（转写、幻灯片、代理、向量）。"""
        work_dir = self.cfg.library_dir / video_id
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        for table in ("segments", "chunks", "chapters", "slides", "embeddings"):
            self.store.conn.execute(f"DELETE FROM {table} WHERE video_id=?", (video_id,))  # noqa: S608
        self.store.conn.execute("DELETE FROM chunks_fts WHERE video_id=?", (video_id,))
        self.store.conn.execute("DELETE FROM segments_fts WHERE video_id=?", (video_id,))
        self.store.conn.execute("DELETE FROM videos WHERE video_id=?", (video_id,))
        self.store.conn.commit()
