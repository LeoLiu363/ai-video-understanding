"""端到端集成测试：合成视频 → 完整入库 → 问答 → HTTP 服务。

用桩替换 ASR 与 LLM，因此**不需要任何网络与密钥**，可以随时跑。
这是「改一处怕碰坏另一处」的保险：它验证的是各层之间的对接，而不是单层逻辑。

运行：
    .venv\\Scripts\\python.exe -m pytest tests/test_integration_pipeline.py -q
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from vedioai.ask import AskService  # noqa: E402
from vedioai.config import Config, MediaConfig, SlideConfig  # noqa: E402
from vedioai.ingest.asr_volc import TranscribeResult, Utterance  # noqa: E402
from vedioai.llm.client import Reply, Usage  # noqa: E402
from vedioai.pipeline import IngestPipeline  # noqa: E402
from vedioai.schema import VideoStatus, ms_to_hms  # noqa: E402
from vedioai.store import Store  # noqa: E402

FFMPEG_CANDIDATES = [
    r"D:\videoplayer\MediaPlayer\ffmpeg-4.2.2\bin\ffmpeg.exe",
    "ffmpeg",
]


def _find_ffmpeg() -> str | None:
    for cand in FFMPEG_CANDIDATES:
        exe = shutil.which(cand) or (cand if Path(cand).exists() else None)
        if exe:
            return exe
    return None


FFMPEG = _find_ffmpeg()
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="需要 FFmpeg 才能跑集成测试")


# --------------------------------------------------------------------- 桩


class StubASR:
    """假 ASR：按固定节奏返回中文句子，带字级时间戳。"""

    def __init__(self, total_ms: int = 40_000, step_ms: int = 4_000):
        self.total_ms = total_ms
        self.step_ms = step_ms

    def _build(self, offset: int = 0) -> list[Utterance]:
        lines = [
            "这一节我们讲快速排序",
            "快速排序的核心是选取基准元素并划分",
            "最坏情况的时间复杂度是平方级",
            "解决办法是随机化选取基准元素",
            "接下来讲归并排序",
            "归并排序需要额外的线性空间",
            "但它能保证稳定的对数线性复杂度",
            "最后比较一下这两种排序的稳定性",
            "归并排序是稳定的排序算法",
            "快速排序则是不稳定的排序算法",
        ]
        out = []
        for i, text in enumerate(lines):
            start = offset + i * self.step_ms
            words = [
                {
                    "start_time": start + j * 300,
                    "end_time": start + j * 300 + 260,
                    "text": ch,
                    "confidence": 0.95,
                }
                for j, ch in enumerate(text)
            ]
            out.append(
                Utterance(
                    start_ms=start,
                    end_ms=start + len(text) * 300,
                    text=text,
                    words=words,
                )
            )
        return out

    def transcribe_file(self, audio_path, audio_url=None):
        return TranscribeResult(utterances=self._build(), duration_ms=self.total_ms)

    def transcribe_spans(self, spans, slice_fn=None):
        # 走分段路径时也要能还原绝对时间
        from vedioai.schema import Segment

        segments = []
        idx = 0
        for i, (start, end) in enumerate(spans):
            for utt in self._build():
                if start + utt.start_ms >= end:
                    continue
                if segments and start + utt.end_ms <= segments[-1].end_ms:
                    continue
                segments.append(
                    Segment(
                        idx=idx,
                        start_ms=start + utt.start_ms,
                        end_ms=start + utt.end_ms,
                        text=utt.text,
                    )
                )
                idx += 1
        return segments

    def close(self):
        pass


class SilentASR:
    """模拟「音频里没有语音」：服务返回 20000003，不是错误。"""

    def transcribe_file(self, audio_path, audio_url=None):
        return TranscribeResult(utterances=[], no_speech=True, duration_ms=2000)

    def transcribe_spans(self, spans, slice_fn=None):
        return []

    def close(self):
        pass


class StubLLM:
    """假 LLM：问答时把前缀里的时间戳回抄进答案，用于验证引用链路。"""

    def __init__(self, text: str | None = None):
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def ask(self, prefix: str, question: str) -> Reply:
        self.calls.append((prefix, question))
        text = self.text
        if text is None:
            # 从前缀里取一个时间戳，模拟「带引用的回答」
            import re

            stamps = re.findall(r"【(\d{2}:\d{2}(?::\d{2})?)】", prefix)
            stamp = stamps[0] if stamps else "00:00"
            text = f"根据课程内容，快速排序最坏是平方复杂度，解决办法是随机化基准元素。[{stamp}]"
        return Reply(
            text=text,
            usage=Usage(prompt_tokens=12_000, completion_tokens=120, cached_tokens=11_500),
            model="stub",
        )

    def chat(self, messages, **kwargs) -> Reply:
        return Reply(text="{}", usage=Usage(), model="stub")

    def chat_json(self, messages, **kwargs):
        return {}, Reply(text="{}", usage=Usage(), model="stub")

    def ask_with_images(self, prefix, question, image_paths, **kwargs) -> Reply:
        self.calls.append((prefix, f"{question} [imgs={len(image_paths)}]"))
        return Reply(text=f"看图后回答。[00:04]", usage=Usage(prompt_tokens=100), model="stub")

    def close(self):
        pass


# ------------------------------------------------------------------- 夹具


@pytest.fixture(scope="module")
def video_file(tmp_path_factory) -> Path:
    """合成一段 40 秒的 AC3/MKV —— 刻意选 Chromium 不支持的音频编码，用来验证代理转码。"""
    tmp = tmp_path_factory.mktemp("media")
    out = tmp / "lesson.mkv"
    subprocess.run(
        [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:duration=40",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "ac3", "-shortest", str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    c = Config()
    c.data_dir = tmp_path / "data"
    c.data_dir.mkdir(parents=True, exist_ok=True)
    c.media = MediaConfig(ffmpeg=FFMPEG, ffprobe=str(Path(FFMPEG).with_name("ffprobe.exe")))
    c.slides = SlideConfig(enabled=True, sample_interval_ms=5000, ocr_enabled=False)
    return c


# ------------------------------------------------------------------- 测试


def test_carry_over_summaries_survives_rebuild(cfg: Config, video_file: Path, monkeypatch):
    """重建分段时必须把已有摘要搬过去，否则摘要生成一失败就是数据损失。

    这是真实事故的回归护栏：流水线先 replace_chunks（新块摘要为空）再生成摘要，
    一旦摘要那步抛错，库里就留下「摘要全空」的残局——旧摘要明明还在库里，
    只是被「先删后建」带走了，而且是在转写已付过费之后。
    """
    from vedioai import pipeline as pipeline_mod

    def fake_detect(ffmpeg, src, duration_ms, out_dir, slide_cfg, *, progress=None):
        return []

    monkeypatch.setattr(pipeline_mod, "detect_slides", fake_detect)

    # 第一次入库：写入摘要
    store = Store(cfg.db_path)
    pipe = IngestPipeline(cfg, store, llm=None, asr=StubASR())
    video = pipe.run(video_file, skip_summary=True)
    vid = video.video_id

    for chunk in store.get_chunks(vid):
        store.update_chunk_summary(vid, chunk.chunk_id, "旧标题", "旧摘要内容")

    before = {c.chunk_id: c.summary for c in store.get_chunks(vid)}
    assert any(before.values()), "前置条件：应该已经有摘要了"
    store.close()

    # 第二次入库（复用转写与课件），摘要生成故意全部失败。
    # 这里必须给一个非 None 的 llm——流水线只在 llm 非空时才走摘要分支，
    # 给 None 就绕过了要测的那条路径（踩过这个坑：断言 DID NOT RAISE）。
    store = Store(cfg.db_path)
    pipe = IngestPipeline(cfg, store, llm=object(), asr=StubASR())

    def boom(*a, **kw):
        raise RuntimeError("模拟摘要生成失败")

    monkeypatch.setattr(pipeline_mod, "build_summary_tree", boom)
    with pytest.raises(RuntimeError):
        pipe.run(video_file, reuse_slides=True)

    after = {c.chunk_id: c.summary for c in store.get_chunks(vid)}
    assert after == before, "摘要生成失败不该把已有摘要清空"
    store.close()


def test_full_ingest_produces_structured_ir(cfg: Config, video_file: Path):
    store = Store(cfg.db_path)
    pipeline = IngestPipeline(cfg, store, llm=None, asr=StubASR())

    stages: list[str] = []
    video = pipeline.run(
        video_file, skip_summary=True,
        progress=lambda p: stages.append(p.stage),
    )

    # 状态机走完所有阶段
    assert video.status is VideoStatus.READY
    for expected in ("probing", "proxy", "audio", "asr", "slides", "segment", "ready"):
        assert expected in stages, f"缺少阶段 {expected}：{stages}"

    # 播放代理确实是 Chromium 能播的 H.264/AAC
    from vedioai.ingest.media import plan_proxy, probe

    proxy = Path(video.proxy_path)
    assert proxy.exists()
    need, reason = plan_proxy(probe(cfg.media.ffprobe, proxy))
    assert not need, f"代理仍需转码：{reason}"

    # 句级转写：带原生时间戳与字级 words
    segments = store.get_segments(video.video_id)
    assert len(segments) >= 8
    assert segments[0].start_ms == 0
    assert all(s.end_ms > s.start_ms for s in segments)
    assert segments[0].words, "字级时间戳丢失，引用会漂移"

    # 语义块与章节
    chunks = store.get_chunks(video.video_id)
    chapters = store.get_chapters(video.video_id)
    assert chunks and chapters
    assert all(c.parent_id for c in chunks), "有块没挂到章节上"
    assert sum(len(ch.chunk_ids) for ch in chapters) == len(chunks)

    # 块必须带可点击的时间区间
    assert all(c.end_ms > c.start_ms for c in chunks)

    # 课件抽帧
    slides = store.get_slides(video.video_id)
    assert slides, "没有抽出任何课件帧"
    assert all(Path(s.image_path).exists() for s in slides)

    # 关键词检索在真实入库数据上可用
    hits = store.search_keyword(video.video_id, "归并排序", 5)
    assert hits, "在真实入库的转写上 FTS5 检索没有命中"

    store.close()


def test_ask_returns_clickable_citations(cfg: Config, video_file: Path):
    store = Store(cfg.db_path)
    IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(video_file, skip_summary=True)
    video_id = store.list_videos()[0]["video_id"]

    llm = StubLLM()
    service = AskService(cfg, store, llm)

    answer = service.ask(video_id, "快速排序最坏复杂度是多少？")
    assert answer.citations, "回答没有带引用，前端就无法做点击跳转"
    for cite in answer.citations:
        assert cite.end_ms > cite.start_ms
        assert ms_to_hms(cite.start_ms) == cite.label
    assert answer.intent == "local"

    # 前缀必须带完整逐字稿，且稳定（命中缓存的前提）
    prefix, _question = llm.calls[-1]
    assert "逐字稿" in prefix
    assert "归并排序" in prefix

    # 全局型问题走另一套提示词
    service.ask(video_id, "这门课一共讲了几种排序算法？")
    assert service.ask(video_id, "这门课一共讲了几种排序？").intent == "global"

    # 视觉型问题应当把课件图喂给视觉模型
    vision = StubLLM()
    service2 = AskService(cfg, store, llm, vision)
    ans = service2.ask(video_id, "屏幕上那张表写了什么？")
    assert ans.intent == "visual"
    assert ans.images, "视觉型问题没有附带任何课件图"

    store.close()


def test_prefix_is_stable_across_questions(cfg: Config, video_file: Path):
    """同一视频的多次提问，前缀必须逐字节一致，否则缓存永不命中。"""
    store = Store(cfg.db_path)
    IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(video_file, skip_summary=True)
    video_id = store.list_videos()[0]["video_id"]

    llm = StubLLM()
    service = AskService(cfg, store, llm)
    service.ask(video_id, "问题一")
    prefix1, _ = llm.calls[-1]
    service.ask(video_id, "问题二", current_ms=12_000)
    prefix2, _ = llm.calls[-1]

    assert prefix1 == prefix2, "前缀不稳定：把变化的播放进度混进前缀会让缓存全废"
    # 变化的播放进度只应出现在问题那一侧
    _p, question2 = llm.calls[-1]
    assert "00:12" in question2

    store.close()


def test_http_server_endpoints(cfg: Config, video_file: Path):
    from fastapi.testclient import TestClient

    from vedioai.server import create_app

    store = Store(cfg.db_path)
    IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(video_file, skip_summary=True)
    store.close()

    app = create_app(cfg)
    with TestClient(app) as client:
        # 首页
        assert client.get("/").status_code == 200
        assert "vedioAI" in client.get("/").text

        health = client.get("/api/health").json()
        assert health["ok"] is True

        lib = client.get("/api/library").json()
        assert len(lib["items"]) == 1
        video_id = lib["items"][0]["video_id"]

        detail = client.get(f"/api/library/{video_id}").json()
        assert detail["chapters"]
        assert detail["stats"]["chunks"] > 0

        # 播放代理：必须支持 Range，否则进度条拖不动
        full = client.get(f"/media/{video_id}")
        assert full.status_code == 200

        ranged = client.get(f"/media/{video_id}", headers={"Range": "bytes=0-1023"})
        assert ranged.status_code == 206
        assert ranged.headers["content-range"].startswith("bytes 0-1023/")
        assert len(ranged.content) == 1024

        # 课件图可访问
        slide = detail["slides"][0]
        assert client.get(f"/api/slides/{video_id}/{slide['idx']}").status_code == 200

        # 入库任务：路径不存在要给出人话报错
        bad = client.post("/api/ingest", json={"path": "D:/nope/none.mp4"})
        assert bad.status_code == 400


def test_http_reads_notes_document_from_disk(cfg: Config, video_file: Path):
    """界面必须读得到磁盘上的 notes.md。

    以前只有 POST /api/notes（生成）而没有 GET（读取），于是界面只能显示
    「本次会话刚生成、还留在内存里」的那一份——服务一重启，已经生成好的
    长篇学习文档就从界面上消失了。文档是产物，不该依赖进程生命周期。
    """
    from fastapi.testclient import TestClient

    from vedioai.server import create_app

    store = Store(cfg.db_path)
    vid = IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(
        video_file, skip_summary=True
    ).video_id
    store.close()

    app = create_app(cfg)
    with TestClient(app) as client:
        # 还没生成过：返回 exists=False 而不是 404，界面才能区分
        # 「这门课没有文档」与「课程不存在」两种状态。
        r = client.get(f"/api/notes/{vid}")
        assert r.status_code == 200
        assert r.json()["exists"] is False

        out_dir = Path(cfg.library_dir) / vid
        out_dir.mkdir(parents=True, exist_ok=True)
        doc = "# 标题\n\n| 章 | 时间 |\n|---|---|\n| 甲 | 00:05 |\n"
        (out_dir / "notes.md").write_text(doc, encoding="utf-8")
        (out_dir / "concepts.md").write_text("术语表内容", encoding="utf-8")

        d = client.get(f"/api/notes/{vid}").json()
        assert d["exists"] is True
        assert "00:05" in d["markdown"]
        assert d["concepts"] == "术语表内容"
        # chars 是界面显示「多少字」的依据，必须与实际内容一致
        assert d["meta"]["chars"] == len(doc)
        assert d["meta"]["mtime"]

        assert client.get("/api/notes/不存在的课程").status_code == 404


def test_library_payload_drops_fields_the_ui_never_reads(
    cfg: Config, video_file: Path, monkeypatch
):
    """列表与详情别再白传界面用不到的重字段。

    实测这门课：详情响应 222KB 里 162KB（89.5%）是 slides[].ocr_text，而界面
    点开课程时从不读 slides；列表则每门课都捎带一份 4500 字的 video_summary。
    课程一多，这些就是纯浪费的带宽与解析开销。
    """
    from fastapi.testclient import TestClient

    from vedioai.server import create_app

    cfg.slides = SlideConfig(enabled=True, sample_interval_ms=5000, ocr_enabled=True)
    marker = "NATIVE BASE64 AES 课件文字"

    def make_slides(Slide, src):
        return [
            Slide(
                idx=0, start_ms=0, end_ms=5000,
                image_path=str(src), ocr_text=marker, phash="0",
            )
        ]

    _counting_detect(monkeypatch, make_slides)
    store = Store(cfg.db_path)
    IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(video_file, skip_summary=True)
    vid = video_id_of(store)
    store.set_video_summary(vid, "摘要正文\n\n大纲：\n- 甲章：讲甲\n")
    store.close()

    app = create_app(cfg)
    with TestClient(app) as client:
        item = client.get("/api/library").json()["items"][0]
        assert "video_summary" not in item, "列表不需要整篇摘要，界面只用标题/时长/章数"
        assert item["duration_label"]

        plain = client.get(f"/api/library/{vid}").json()
        assert plain["slides"], "课件帧的元信息（idx/时间）仍要保留，界面用它定位"
        assert all(s["ocr_text"] == "" for s in plain["slides"]), (
            "默认不该发课件 OCR 全文"
        )

        # 需要时显式索取，能力不能丢
        full = client.get(f"/api/library/{vid}", params={"with_ocr": True}).json()
        assert any(marker in s["ocr_text"] for s in full["slides"])


def test_http_usage_ledger_endpoint(cfg: Config, video_file: Path):
    """用量账本接口：全库汇总 + 单课汇总，都要能读出来。

    这个接口存在的理由：以前没有任何地方能回答「这门课花了多少钱」——
    summarize 把 usage 丢掉了、notes 算了却不落库。现在它是可查的。
    """
    from fastapi.testclient import TestClient

    from vedioai.server import create_app

    store = Store(cfg.db_path)
    vid = IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(
        video_file, skip_summary=True
    ).video_id
    store.close()

    app = create_app(cfg)
    with TestClient(app) as client:
        # 入库时跑过转写，所以账本里已经有 ASR 那一笔——这本身就是要验证的：
        # 转写是唯一按音频时长计费的环节，必须自动进账。
        base = client.get("/api/usage").json()
        assert base["calls"] == 1
        assert base["by_kind"][0]["kind"] == "asr"
        assert base["audio_ms"] > 0
        assert base["cost_yuan"] > 0
        # 单价表要随接口返回，界面才能解释「这个数怎么算出来的」
        assert "deepseek-flash" in base["prices"]
        assert isinstance(base["peak_now"], bool)

        store2 = Store(cfg.db_path)
        store2.record_usage(
            kind="ask", model="deepseek-flash", video_id=vid,
            prompt_tokens=1_000_000, cached_tokens=1_000_000, cost_yuan=0.02,
        )
        store2.record_usage(
            kind="summarize", model="deepseek-flash", video_id=vid,
            prompt_tokens=2_000_000, cached_tokens=0, completion_tokens=5_000,
            cost_yuan=2.02, peak=True,
        )
        # 没有价目的调用（如视觉模型）也要能进账，并被标为未计价
        store2.record_usage(kind="ask", model="doubao-seed", video_id=vid, cost_yuan=None)
        store2.close()

        all_usage = client.get("/api/usage").json()
        assert all_usage["calls"] == 4
        assert all_usage["unpriced_calls"] == 1
        assert all_usage["cost_yuan"] == pytest.approx(base["cost_yuan"] + 2.04)
        assert all_usage["by_video"][0]["video_id"] == vid

        one = client.get("/api/usage", params={"video_id": vid}).json()
        assert one["calls"] == 4
        assert one["video_id"] == vid
        assert {r["kind"] for r in one["by_kind"]} == {"ask", "summarize", "asr"}


def test_scaffold_generates_40_slots(cfg: Config, video_file: Path, tmp_path: Path):
    from evals.run_eval import scaffold

    store = Store(cfg.db_path)
    IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(video_file, skip_summary=True)
    video_id = store.list_videos()[0]["video_id"]
    store.close()

    out = tmp_path / "questions.yaml"
    assert scaffold(video_id, cfg, out) == 0

    import yaml

    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert len(data["questions"]) == 40
    kinds = [q["type"] for q in data["questions"]]
    for kind in ("factual", "cross", "visual", "global"):
        assert kinds.count(kind) == 10


class CountingASR(StubASR):
    """记录转写调用次数的假 ASR，用于验证「不要重复付费」。"""

    def __init__(self):
        super().__init__()
        self.file_calls = 0
        self.span_calls = 0

    def transcribe_file(self, audio_path, audio_url=None):
        self.file_calls += 1
        return super().transcribe_file(audio_path, audio_url)

    def transcribe_spans(self, spans, slice_fn=None):
        self.span_calls += 1
        return super().transcribe_spans(spans, slice_fn)


class BrokenLLM:
    """所有摘要调用都失败的假 LLM。"""

    def __init__(self, message: str = "输出被 max_tokens 截断（budget=4096，正文 0 字符）"):
        self.message = message

    def chat_json(self, messages, **kwargs):
        from vedioai.llm.client import LLMError

        raise LLMError(self.message)

    def chat(self, messages, **kwargs):
        from vedioai.llm.client import LLMError

        raise LLMError(self.message)

    def ask(self, prefix, question, **kwargs):
        from vedioai.llm.client import LLMError

        raise LLMError(self.message)

    def ask_with_images(self, prefix, question, image_paths, **kwargs):
        from vedioai.llm.client import LLMError

        raise LLMError(self.message)


def test_summary_failure_does_not_destroy_ingest(cfg: Config, video_file: Path):
    """回归：摘要失败不能让整次入库作废。

    真实事故：视频摘要那一步返回被截断，异常一路抛出，前面已经付过费的
    转写成果全部被丢弃，课程变成 FAILED。摘要是锦上添花，转写才是钱。
    """
    store = Store(cfg.db_path)
    asr = CountingASR()
    pipeline = IngestPipeline(cfg, store, llm=BrokenLLM(), asr=asr)

    video = pipeline.run(video_file)

    # 入库仍然成功，且可正常使用
    assert video.status is VideoStatus.READY
    assert video.error, "降级必须留在可见的状态说明里，不能静默"
    assert "摘要" in video.error

    # 转写、分段、课件全部完好
    assert store.get_segments(video.video_id)
    assert store.get_chunks(video.video_id)
    assert store.get_chapters(video.video_id)
    assert store.get_slides(video.video_id)

    # 并且真的花了转写钱（证明这次确实走到了 ASR）
    assert asr.file_calls == 1


def test_reingest_reuses_transcript_and_does_not_pay_asr_again(
    cfg: Config, video_file: Path
):
    """回归：重跑入库必须复用已有转写。

    真实事故：摘要失败后重跑 ingest，代理和音频都复用了，唯独 ASR 每次都重跑。
    转写是整条链路唯一按小时计费、且唯一不可复现的环节，重复付费没有任何收益。
    """
    store = Store(cfg.db_path)

    first_asr = CountingASR()
    pipeline = IngestPipeline(cfg, store, llm=None, asr=first_asr)
    video = pipeline.run(video_file, skip_summary=True)
    assert first_asr.file_calls == 1, "首次入库应当真的调用 ASR"
    segments_before = store.get_segments(video.video_id)

    # 第二次入库：模拟「摘要失败后重跑」
    second_asr = CountingASR()
    pipeline2 = IngestPipeline(cfg, store, llm=None, asr=second_asr)
    video2 = pipeline2.run(video_file, skip_summary=True)

    assert video2.video_id == video.video_id
    assert second_asr.file_calls == 0, "已有转写时必须跳过 ASR，避免重复付费"
    assert second_asr.span_calls == 0

    # 转写内容保持原样（不是被重新生成了一遍）
    segments_after = store.get_segments(video.video_id)
    assert len(segments_after) == len(segments_before)
    assert [s.text for s in segments_after] == [s.text for s in segments_before]


def test_reingest_without_existing_transcript_still_calls_asr(cfg: Config, video_file: Path):
    """复用逻辑不能把「首次入库」也跳过。"""
    store = Store(cfg.db_path)
    asr = CountingASR()
    IngestPipeline(cfg, store, llm=None, asr=asr).run(video_file, skip_summary=True)

    assert asr.file_calls == 1


def video_id_of(store: Store) -> str:
    """库里那门课的 video_id（测试里只入库一门课）。"""
    return store.list_videos()[0]["video_id"]


def _counting_detect(monkeypatch, make_slides):
    """把 detect_slides 换成计数器，用于观察课件是否被重新抽取。

    OCR 单张实测 3–22 秒（随文字量增长），整门课几十分钟。测试里不能真跑 OCR，
    所以这里只关心**是否调用了**抽帧这一步。
    """
    import vedioai.pipeline as pipeline_mod
    from vedioai.schema import Slide

    calls: list[int] = []

    def fake_detect(ffmpeg, src, duration_ms, out_dir, slide_cfg, *, progress=None):
        calls.append(1)
        return make_slides(Slide, src)

    monkeypatch.setattr(pipeline_mod, "detect_slides", fake_detect)
    return calls


def test_slides_reused_when_ocr_text_already_present(
    cfg: Config, video_file: Path, monkeypatch
):
    """已有 OCR 文字时应复用课件，不再重新抽帧。

    OCR 是本地最贵的一步。用户有整季 17 讲，若每次重跑入库都重做 OCR，
    光这一项就是几小时的无谓开销。
    """
    cfg.slides = SlideConfig(enabled=True, sample_interval_ms=5000, ocr_enabled=True)

    def make_slides(Slide, src):
        # 模拟「OCR 已经产出文字」
        return [
            Slide(
                idx=0, start_ms=0, end_ms=5000,
                image_path=str(src), ocr_text="NATIVE BASE+MD5 AES", phash="0",
            )
        ]

    calls = _counting_detect(monkeypatch, make_slides)
    store = Store(cfg.db_path)

    IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(video_file, skip_summary=True)
    assert len(calls) == 1, "首次入库应当抽帧"

    IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(video_file, skip_summary=True)
    assert len(calls) == 1, "已有文字时必须复用，不能重做 OCR"

    # 文字确实还在库里
    assert any(s.ocr_text for s in store.get_slides(video_id_of(store)))


def test_slides_reextracted_when_ocr_was_previously_skipped(
    cfg: Config, video_file: Path, monkeypatch
):
    """升级路径：先入库（没 OCR）后装 OCR，重跑必须真的去跑 OCR。

    这是最容易漏的分支——如果只判断「课件已存在」，装好 OCR 后重跑会
    静默复用一堆空文字，用户以为装好了，实际什么都没变。
    """
    cfg.slides = SlideConfig(enabled=True, sample_interval_ms=5000, ocr_enabled=True)

    def make_slides(Slide, src):
        return [
            Slide(
                idx=0, start_ms=0, end_ms=5000,
                image_path=str(src), ocr_text="", phash="0",  # 没有文字
            )
        ]

    calls = _counting_detect(monkeypatch, make_slides)
    store = Store(cfg.db_path)

    IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(video_file, skip_summary=True)
    IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(video_file, skip_summary=True)

    assert len(calls) == 2, "没有文字时必须重新抽帧跑 OCR，否则等于静默失败"


def test_refresh_slides_forces_reextraction(cfg: Config, video_file: Path, monkeypatch):
    """改了抽帧参数时用 reuse_slides=False 强制重做。"""
    cfg.slides = SlideConfig(enabled=True, sample_interval_ms=5000, ocr_enabled=True)

    def make_slides(Slide, src):
        return [
            Slide(
                idx=0, start_ms=0, end_ms=5000,
                image_path=str(src), ocr_text="已有文字", phash="0",
            )
        ]

    calls = _counting_detect(monkeypatch, make_slides)
    store = Store(cfg.db_path)

    IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(video_file, skip_summary=True)
    IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(
        video_file, skip_summary=True, reuse_slides=False
    )

    assert len(calls) == 2, "--refresh-slides 必须绕过复用"


def test_slides_with_ocr_text_flow_into_chunks(cfg: Config, video_file: Path, monkeypatch):
    """课件文字必须真的进入分块的 combined_text。

    只把文字存进 slides 表是没用的——检索和摘要读的是 chunk.combined_text。
    这一步断了，OCR 等于白做，而且不报错。
    """
    cfg.slides = SlideConfig(enabled=True, sample_interval_ms=5000, ocr_enabled=True)
    marker = "NATIVE_BASE_MD5_MARKER"

    def make_slides(Slide, src):
        # 覆盖整段时间，确保必然落在某个块里
        return [
            Slide(
                idx=0, start_ms=0, end_ms=10_000_000,
                image_path=str(src), ocr_text=marker, phash="0",
            )
        ]

    _counting_detect(monkeypatch, make_slides)
    store = Store(cfg.db_path)

    IngestPipeline(cfg, store, llm=None, asr=StubASR()).run(video_file, skip_summary=True)

    chunks = store.get_chunks(video_id_of(store))
    assert chunks
    assert any(marker in c.combined_text for c in chunks), (
        "课件文字没有进入 combined_text，OCR 成果对检索不可见"
    )
    assert any(marker in c.ocr_text for c in chunks)


