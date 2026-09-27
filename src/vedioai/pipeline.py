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
from . import ledger
from .ingest.asr_volc import VolcASRClient
from .ingest.media import (
    build_proxy,
    detect_silences,
    extract_audio,
    pick_split_points,
    probe,
    slice_audio,
)
from .ingest.document import is_document_path, load_document_text, parse_document
from .ingest.segment import attach_parents, build_chapters, build_chunks
from .ingest.slides import detect_slides, reocr_slides
from .llm.client import LLMClient
from .schema import Chapter, Chunk, ContentKind, Segment, Video, VideoStatus
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
    targets = video_ids if video_ids is not None else [v["video_id"] for v in store.list_videos()]
    report: dict[str, dict[str, int]] = {}

    for i, video_id in enumerate(targets):
        # 每课单独加载：courses.<video_id> 下的界面纠错只对本课生效
        glossary = Glossary.load(cfg.glossary_path, video_id=video_id)
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


def reocr_video(
    cfg: Config,
    store: Store,
    video_id: str,
    *,
    embedder: EmbedderLike | None = None,
    progress=None,
) -> dict[str, int]:
    """用原始分辨率重跑**已入库**课程的课件 OCR，不重新转写。

    为什么单独有这个函数，而不是重跑一遍 ingest：
    - ASR 是整条链路里唯一按小时计费的环节，而转写与课件 OCR 毫无关系；
    - 每张课件的时间轴（变化检测的结果）已经在库里，不必重算。

    背景：早期版本把代表帧压到 960 宽再 OCR，导致
    "uiautomatorviewer.bat" 被读成 "uiautomatoniewer.bet" 之类的错字。
    这些错字已经进了库、进了摘要和笔记，所以必须回补一遍。
    """
    video = store.get_video(video_id)
    if video is None:
        raise ValueError(f"未找到课程：{video_id}")
    src = Path(video.path)
    if not src.exists():
        raise FileNotFoundError(f"源视频已不在原位置，无法重抽帧：{src}")

    slides = store.get_slides(video_id)
    if not slides:
        return {}

    out_dir = cfg.library_dir / video_id / "slides"
    slides = reocr_slides(
        cfg.media.ffmpeg,
        src,
        slides,
        out_dir,
        width=cfg.slides.ocr_width or None,
        progress=progress,
    )
    store.replace_slides(video_id, slides)

    # 课件文字换了，块里的「（课件）…」段也要跟着换。
    # 块的 slide_idxs 入库时已算好，直接复用——重建分段会连带清掉块摘要，
    # 那是 LLM 花钱生成的，不能白扔。
    by_idx = {s.idx: s for s in slides}
    touched = 0
    for chunk in store.get_chunks(video_id):
        new_ocr = "\n".join(
            by_idx[i].ocr_text
            for i in chunk.slide_idxs
            if i in by_idx and by_idx[i].ocr_text
        ).strip()
        if new_ocr != (chunk.ocr_text or ""):
            store.update_chunk_texts(video_id, chunk.chunk_id, chunk.text, new_ocr)
            touched += 1

    stats: dict[str, int] = {"slides": len(slides), "chunks": touched}
    # 新 OCR 文字还没过术语表；顺带纠错并重算向量（repair 本身幂等）
    stats.update(repair_videos(cfg, store, [video_id], embedder=embedder).get(video_id, {}))
    return stats


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

    def _carry_over_summaries(
        self,
        video_id: str,
        chunks: list[Chunk],
        chapters: list[Chapter],
    ) -> None:
        """把库里已有的标题/摘要搬到新构建的块与章节上。

        存在的理由：分段重建是「先删后建」，而摘要生成紧随其后。若摘要这一步
        失败（LLM 返回格式异常、网络中断、额度用尽），库里留下的就是
        「摘要全空」的残局——一次本可降级的失败被升级成了数据损失。

        为什么按 id 搬运是安全的：``chunk_id`` 由序号决定
        （``{video_id}-c{idx:04d}``），而块边界只取决于转写文本。
        只要转写没变（复用路径下必然如此），新旧块就一一对应；
        章节同理（``{video_id}-h{idx:04d}``）。搬运后若摘要重新生成成功会覆盖，
        失败则原地保留旧值——两种结局都不会比原来更差。
        """
        old_chunks = {c.chunk_id: c for c in self.store.get_chunks(video_id)}
        for chunk in chunks:
            old = old_chunks.get(chunk.chunk_id)
            if old is None:
                continue
            # 只在旧块正文没变时搬运：正文变了说明分段口径变了，旧摘要已过期
            if old.text == chunk.text:
                chunk.title = chunk.title or old.title
                chunk.summary = chunk.summary or old.summary

        old_chapters = {c.chapter_id: c for c in self.store.get_chapters(video_id)}
        for chapter in chapters:
            old = old_chapters.get(chapter.chapter_id)
            if old is None:
                continue
            chapter.title = chapter.title or old.title
            chapter.summary = chapter.summary or old.summary

    def run(
        self,
        video_path: Path,
        *,
        reuse_proxy: bool = True,
        reuse_slides: bool = True,
        skip_summary: bool = False,
        progress=None,
        title: str | None = None,
        video_id: str | None = None,
    ) -> Video:
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在：{video_path}")

        def report(stage: VideoStatus, done=0, total=0, message=""):
            self.store.set_status(video_id, stage)
            if progress:
                progress(Progress(stage.value, done, total, message or stage.label))

        video_id = (video_id or "").strip() or video_id_for(video_path)
        work_dir = self.cfg.library_dir / video_id
        work_dir.mkdir(parents=True, exist_ok=True)
        display_title = (title or "").strip() or video_path.stem

        try:
            # ---------------------------------------------------- 1. 探测
            report(VideoStatus.PROBING)
            info = probe(self.cfg.media.ffprobe, video_path)
            video = Video(
                video_id=video_id,
                path=str(video_path),
                title=display_title,
                duration_ms=info.duration_ms,
                status=VideoStatus.PROBING,
                size_bytes=video_path.stat().st_size,
                kind=ContentKind.VIDEO,
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
                # 转写是唯一按音频时长计费的环节，记一笔账。
                # 复用已有转写时**不记**（没有产生新费用），否则重跑会因为
                # 记账而看起来越来越贵。
                self._record_asr_usage(video_id, info.duration_ms)

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
            # 复用旧摘要：chunk_id 由「序号」决定（{video_id}-cNNNN），而块的边界
            # 只取决于转写，所以只要转写没变，新旧块就是一一对应的。
            #
            # 为什么必须复用：接下来这一步是**先删后建**——如果没有摘要就落库，
            # 而摘要生成中途失败（LLM 返回格式异常、网络中断、额度用尽），
            # 库里就会留下「摘要全空」的残局。旧行为把一次可降级的失败
            # 升级成了数据损失。带过来之后，生成成功会覆盖，失败则原地保留。
            self._carry_over_summaries(video_id, chunks, chapters)
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

    def run_document(
        self,
        doc_path: Path | str,
        *,
        skip_summary: bool = False,
        progress=None,
        title: str | None = None,
        video_id: str | None = None,
    ) -> Video:
        """文档课程入库：解析切分 → 轻量摘要 → 向量。不跑 ASR / 代理 / 笔记。"""
        doc_path = Path(doc_path).resolve()
        if not doc_path.exists():
            raise FileNotFoundError(f"文档不存在：{doc_path}")
        if not is_document_path(doc_path):
            raise ValueError(f"不支持的文档格式：{doc_path.suffix}（支持 .md / .txt / .markdown）")

        def report(stage: VideoStatus, done=0, total=0, message=""):
            self.store.set_status(video_id, stage)
            if progress:
                progress(Progress(stage.value, done, total, message or stage.label))

        video_id = (video_id or "").strip() or video_id_for(doc_path)
        work_dir = self.cfg.library_dir / video_id
        work_dir.mkdir(parents=True, exist_ok=True)
        display_title = (title or "").strip() or doc_path.stem

        video = Video(
            video_id=video_id,
            path=str(doc_path),
            title=display_title,
            duration_ms=0,
            status=VideoStatus.PENDING,
            size_bytes=doc_path.stat().st_size if doc_path.exists() else 0,
            proxy_path=None,
            kind=ContentKind.DOCUMENT,
        )
        self.store.upsert_video(video)

        try:
            report(VideoStatus.SEGMENT, 0, 1, "解析文档")
            text = load_document_text(doc_path)
            parsed = parse_document(text, video_id, title=display_title)

            # 原文落盘，供阅读区渲染（不生成视频风格 notes.md）
            source_path = work_dir / "source.md"
            source_path.write_text(parsed.text, encoding="utf-8", newline="\n")

            video.title = parsed.title
            video.duration_ms = parsed.duration_ms
            video.status = VideoStatus.SEGMENT
            video.size_bytes = doc_path.stat().st_size
            self.store.upsert_video(video)

            # 文档课无转写 / 课件
            self.store.replace_segments(video_id, [])
            self.store.replace_slides(video_id, [])

            chunks = parsed.chunks
            chapters = parsed.chapters
            self._carry_over_summaries(video_id, chunks, chapters)
            self.store.replace_chunks(video_id, chunks)
            self.store.replace_chapters(video_id, chapters)
            log.info("文档分段：%d 块，%d 章", len(chunks), len(chapters))

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
                    video.title = result.video_title[:80]
                self._write_concepts(video_id, result)
                if result.degraded:
                    summary_note = result.describe()
                    log.warning(summary_note)

            if self.embedder is not None and self.embedder.available:
                report(VideoStatus.EMBED, 0, len(chunks), "建立向量索引")
                texts = [c.combined_text for c in chunks]
                vecs = self.embedder.encode(texts)
                self.store.upsert_embeddings(
                    video_id, zip([c.chunk_id for c in chunks], vecs, strict=False)
                )

            video.error = summary_note or None
            video.status = VideoStatus.READY
            self.store.upsert_video(video)
            if progress:
                message = "入库完成" if not summary_note else "入库完成（摘要部分降级）"
                progress(Progress(VideoStatus.READY.value, 1, 1, message))
            return video

        except Exception as exc:  # noqa: BLE001
            log.exception("文档入库失败")
            self.store.set_status(video_id, VideoStatus.FAILED, error=str(exc))
            if progress:
                progress(Progress(VideoStatus.FAILED.value, 0, 1, f"入库失败：{exc}"))
            raise

    # ------------------------------------------------------------------ 内部

    def _record_asr_usage(self, video_id: str, duration_ms: int) -> None:
        """记一笔转写用量。

        注意这与「花费」是两件事：记录的是**音频时长**这个事实，
        cost 只是按标价估的值。官方给录音文件识别 20 小时免费额度，
        额度内实际不扣钱——所以界面上这项要标明是估算。
        """
        self.store.record_usage(
            kind="asr",
            model="volc.bigasr.auc_turbo",
            video_id=video_id,
            audio_ms=duration_ms,
            cost_yuan=ledger.estimate_asr_cost(duration_ms),
        )

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
        """删除某课程的入库产物（转写、幻灯片、代理、向量、会话）。

        用量账本保留（钱已经花了），但会先把课名写入 course_labels，
        避免界面「按课程」里只剩一串 video_id。
        """
        video = self.store.get_video(video_id)
        title = (video.title if video else "") or ""
        self.store.remember_course_label(video_id, title, deleted=True)

        work_dir = self.cfg.library_dir / video_id
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        for table in ("segments", "chunks", "chapters", "slides", "embeddings"):
            self.store.conn.execute(f"DELETE FROM {table} WHERE video_id=?", (video_id,))  # noqa: S608
        self.store.conn.execute("DELETE FROM chunks_fts WHERE video_id=?", (video_id,))
        self.store.conn.execute("DELETE FROM segments_fts WHERE video_id=?", (video_id,))
        # 会话消息靠 FK CASCADE；先删 sessions
        try:
            self.store.conn.execute("DELETE FROM chat_sessions WHERE video_id=?", (video_id,))
        except Exception:  # noqa: BLE001 — 旧库可能还没有会话表
            pass
        self.store.conn.execute("DELETE FROM videos WHERE video_id=?", (video_id,))
        self.store.conn.commit()
