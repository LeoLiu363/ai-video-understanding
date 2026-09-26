"""不依赖网络与 FFmpeg 的单元测试：存储、分段、检索、上下文、评分。

这些是「改一处怕碰坏另一处」最容易出问题的部分，所以先钉住。

运行：
    .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.run_eval import (  # noqa: E402
    normalize,
    score_citation,
    score_keypoints,
    strip_extension_blocks,
)
from vedioai.context import build_full_prefix, build_outline_prefix, estimate_tokens  # noqa: E402
from vedioai.ingest.media import MediaInfo, StreamInfo, pick_split_points, plan_proxy  # noqa: E402
from vedioai.ingest.segment import attach_parents, build_chapters, build_chunks  # noqa: E402
from vedioai.config import Config, RetrieveConfig, SlideConfig  # noqa: E402
from vedioai.embedding import explain_ollama_failure  # noqa: E402
from vedioai.glossary import Glossary, render_findings  # noqa: E402
from vedioai import ledger  # noqa: E402
from vedioai.selfcheck import check_embedding  # noqa: E402
from vedioai.retrieve import Retriever  # noqa: E402
from vedioai.schema import Chapter, Segment, Slide, Video, VideoStatus, hms_to_ms, ms_to_hms  # noqa: E402
from vedioai.store import Store, query_terms, tokenize_zh  # noqa: E402


# --------------------------------------------------------------------- 夹具


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


def make_chunk(video_id: str, idx: int, start: int, end: int, text: str, ocr: str = "", slide_idxs=None):
    from vedioai.schema import Chunk

    return Chunk(
        chunk_id=f"{video_id}-c{idx:04d}",
        idx=idx,
        start_ms=start,
        end_ms=end,
        text=text,
        ocr_text=ocr,
        slide_idxs=slide_idxs or [],
    )


def make_segments(texts: list[tuple[int, int, str]]) -> list[Segment]:
    return [
        Segment(idx=i, start_ms=s, end_ms=e, text=t) for i, (s, e, t) in enumerate(texts)
    ]


@pytest.fixture()
def sample_video(store: Store) -> Video:
    video = Video(
        video_id="v1",
        path="D:/courses/a.mp4",
        title="排序算法",
        duration_ms=600_000,
        status=VideoStatus.READY,
    )
    store.upsert_video(video)
    return video


# ------------------------------------------------------------------- 基础


def test_time_format_roundtrip():
    assert ms_to_hms(0) == "00:00"
    assert ms_to_hms(75_000) == "01:15"
    assert ms_to_hms(3_725_000) == "01:02:05"
    assert hms_to_ms("01:15") == 75_000
    assert hms_to_ms("1:02:05") == 3_725_000


def test_tokenize_and_query_terms():
    tok = tokenize_zh("快速排序的最坏时间复杂度")
    assert " " in tok
    assert "排序" in tok

    terms = query_terms("快速排序的最坏时间复杂度是多少")
    assert any("排序" in t for t in terms)
    # 单字被丢弃，避免「的」「是」之类噪声把 BM25 结果拉歪
    assert all(len(t) >= 2 for t in terms)


# ------------------------------------------------------------------- 存储


def test_store_roundtrip(store: Store, sample_video: Video):
    segs = make_segments(
        [(0, 4000, "今天讲快速排序"), (4000, 9000, "最坏情况是平方复杂度")]
    )
    store.replace_segments("v1", segs)
    assert len(store.get_segments("v1")) == 2

    slides = [
        Slide(idx=0, start_ms=0, end_ms=5000, image_path="a.jpg", ocr_text="复杂度对比表"),
        Slide(idx=1, start_ms=5000, end_ms=10000, image_path="b.jpg", ocr_text="递归树"),
    ]
    store.replace_slides("v1", slides)
    assert len(store.get_slides("v1")) == 2
    assert [s.idx for s in store.slide_at("v1", 6000)] == [1]

    chunks = [
        make_chunk("v1", 0, 0, 9000, "今天讲快速排序\n最坏情况是平方复杂度", "复杂度对比表", [0])
    ]
    store.replace_chunks("v1", chunks)
    assert store.get_chunks("v1")[0].ocr_text == "复杂度对比表"
    assert store.get_chunks("v1")[0].slide_idxs == [0]


def test_keyword_search_hits_chinese(store: Store, sample_video: Video):
    chunks = [
        make_chunk("v1", 0, 0, 60_000, "这一节我们讲快速排序，最坏情况是平方复杂度"),
        make_chunk("v1", 1, 60_000, 120_000, "接下来讲归并排序，它的复杂度稳定在 n log n"),
    ]
    store.replace_chunks("v1", chunks)

    hits = store.search_keyword("v1", "归并排序的复杂度", 5)
    assert hits, "FTS5 中文检索没有命中，说明 jieba 预分词没生效"
    assert hits[0][0] == "v1-c0001"


def test_vector_search(store: Store, sample_video: Video):
    chunks = [make_chunk("v1", i, i * 1000, i * 1000 + 900, f"片段{i}") for i in range(3)]
    store.replace_chunks("v1", chunks)
    store.upsert_embeddings(
        "v1",
        [
            ("v1-c0000", np.array([1.0, 0.0], dtype=np.float32)),
            ("v1-c0001", np.array([0.0, 1.0], dtype=np.float32)),
            ("v1-c0002", np.array([0.9, 0.1], dtype=np.float32)),
        ],
    )
    assert store.has_embeddings("v1")
    hits = store.search_vector("v1", np.array([1.0, 0.0], dtype=np.float32), 2)
    assert hits[0][0] == "v1-c0000"
    assert hits[0][1] > hits[1][1]


def test_update_chunk_texts_refreshes_index(store: Store, sample_video: Video):
    chunks = [make_chunk("v1", 0, 0, 5000, "原始内容是冒泡排序")]
    store.replace_chunks("v1", chunks)
    # 用「冒泡」而不是「冒泡排序」：后者会被 jieba 切出「排序」这个共同词，
    # 换词后仍会命中，测不出索引有没有真的重建
    assert store.search_keyword("v1", "冒泡", 5)

    # 用户纠错后局部重嵌入，索引必须同步更新
    store.update_chunk_texts("v1", "v1-c0000", "纠正为插入排序", "")
    assert store.search_keyword("v1", "插入排序", 5)
    assert not store.search_keyword("v1", "冒泡", 5)


def test_set_status_and_summary(store: Store, sample_video: Video):
    store.set_status("v1", VideoStatus.FAILED, error="ASR 超时")
    video = store.get_video("v1")
    assert video.status is VideoStatus.FAILED
    assert video.error == "ASR 超时"
    store.set_video_summary("v1", "本课讲排序")
    assert store.get_video_summary("v1") == "本课讲排序"


# ----------------------------------------------------------------- 分段


def test_build_chunks_respects_gaps_and_bounds():
    segs = make_segments(
        [
            (0, 20_000, "第一段很长很长" * 10),
            (21_000, 40_000, "紧接上一段"),
            # 大停顿 → 应该切开
            (60_000, 80_000, "新话题开始"),
        ]
    )
    chunks = build_chunks("v1", segs)
    assert len(chunks) == 2, [c.text[:10] for c in chunks]
    assert chunks[0].start_ms == 0
    assert chunks[1].start_ms == 60_000
    # chunk_id 必须稳定，供引用与增量重建
    assert chunks[0].chunk_id == "v1-c0000"


def test_build_chunks_empty():
    assert build_chunks("v1", []) == []


def test_chunk_max_duration_forces_split():
    # 连续无停顿的长讲稿也要被切开，否则单块过大
    segs = make_segments([(i * 10_000, i * 10_000 + 9_500, "连续讲" * 20) for i in range(12)])
    chunks = build_chunks("v1", segs)
    assert len(chunks) > 1
    assert all(c.end_ms - c.start_ms <= 120_000 for c in chunks)


def test_attach_slide_ocr_to_chunk():
    # 两段之间留出停顿，确保被切成两个语义块
    segs = make_segments([(0, 30_000, "看这张表"), (45_000, 75_000, "再看下一张")])
    slides = [
        Slide(idx=0, start_ms=0, end_ms=31_000, image_path="a.jpg", ocr_text="表一内容"),
        Slide(idx=1, start_ms=31_000, end_ms=60_000, image_path="b.jpg", ocr_text="表二内容"),
    ]
    chunks = build_chunks("v1", segs, slides)
    assert len(chunks) == 2
    assert "表一内容" in chunks[0].ocr_text
    assert chunks[0].slide_idxs == [0]
    assert chunks[1].slide_idxs == [1]


def test_adjacent_segments_merge_into_one_chunk():
    """无停顿的连续讲解应当合成一块，不要被硬切成碎片。"""
    segs = make_segments([(0, 30_000, "第一句"), (30_000, 60_000, "紧接着第二句")])
    chunks = build_chunks("v1", segs)
    assert len(chunks) == 1


def test_chapters_and_parents():
    segs = make_segments([(i * 600_000, i * 600_000 + 400_000, "内容" * 50) for i in range(3)])
    chunks = build_chunks("v1", segs)
    chapters = build_chapters("v1", chunks)
    assert len(chapters) >= 2, "大间隔没有被识别为章节边界"
    attach_parents(chunks, chapters)
    assert all(c.parent_id for c in chunks)
    total = sum(len(ch.chunk_ids) for ch in chapters)
    assert total == len(chunks), "有块没被归入任何章节"


# ----------------------------------------------------------------- 检索


def test_retriever_hybrid_and_fallback(store: Store, sample_video: Video):
    chunks = [
        make_chunk("v1", 0, 0, 60_000, "快速排序的最坏复杂度是平方级"),
        make_chunk("v1", 1, 60_000, 120_000, "归并排序需要额外 O(n) 空间"),
    ]
    store.replace_chunks("v1", chunks)
    retriever = Retriever(store)

    hits = retriever.search("v1", "归并排序的空间开销")
    assert hits and hits[0].chunk.chunk_id == "v1-c0001"
    assert "keyword" in hits[0].sources

    # 完全检索不到时必须退化为「开头若干块」，而不是返回空
    empty = retriever.search("v1", "量子纠缠与咖啡因")
    assert empty, "检索无果时不应该返回空列表"


class FakeEmbedder:
    """假嵌入器：按关键词给向量，用来验证向量召回真的接进了问答链路。

    真实模型（bge-m3）不在本地也能测——这正是要钉住的点：
    接线错误时向量那一路会静默失效，只有断言 sources 才看得出来。
    """

    available = True

    def __init__(self, table: dict[str, list[float]]):
        self.table = table
        self.encode_calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.encode_calls += 1
        rows = []
        for text in texts:
            for key, vec in self.table.items():
                if key in text:
                    rows.append(vec)
                    break
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


class FakeReranker:
    available = True

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [float(len(d)) for d in docs]


def test_ask_service_passes_embedder_into_retriever(store: Store, sample_video: Video):
    """回归：AskService 必须把嵌入模型交给内部 Retriever。

    曾经的 bug 是 cli/server/eval 三处构造 AskService 时都漏传 embedder，
    导致入库算好的向量在查询时被完全闲置——向量召回静默失效、重排永不生效。
    直接断言 AskService 内部的 retriever，才能钉住这个位置。
    """
    from vedioai.ask import AskService
    from vedioai.config import Config

    embedder = FakeEmbedder({"归并": [1.0, 0.0, 0.0]})
    reranker = FakeReranker()
    service = AskService(
        Config(), store, client=None, vision=None, embedder=embedder, reranker=reranker
    )

    assert service.retriever.embedder is embedder, "embedder 没有传进 Retriever"
    assert service.retriever.reranker is reranker, "reranker 没有传进 Retriever"


def test_ask_service_vector_recall_reaches_search(store: Store, sample_video: Video):
    """端到端：走完 AskService → Retriever，向量召回确实参与融合。"""
    from vedioai.ask import AskService
    from vedioai.config import Config

    store.replace_chunks(
        "v1",
        [
            make_chunk("v1", 0, 0, 60_000, "这节课讲归并排序的合并过程"),
            make_chunk("v1", 1, 60_000, 120_000, "这节课讲快速排序的划分过程"),
        ],
    )
    embedder = FakeEmbedder(
        {"归并": [1.0, 0.0, 0.0], "快速": [0.0, 1.0, 0.0], "归并怎么合并": [1.0, 0.0, 0.0]}
    )
    chunks = store.get_chunks("v1")
    vectors = embedder.encode([c.combined_text for c in chunks])
    store.upsert_embeddings("v1", zip([c.chunk_id for c in chunks], vectors, strict=False))

    service = AskService(Config(), store, client=None, embedder=embedder)
    hits = service.retriever.search("v1", "归并怎么合并")
    assert any("vector" in h.sources for h in hits)


def test_retriever_degrades_to_keyword_without_embedder(store: Store, sample_video: Video):
    """模型缺失时必须优雅退化为关键词检索，而不是报错。"""
    store.replace_chunks(
        "v1", [make_chunk("v1", 0, 0, 60_000, "归并排序需要额外 O(n) 空间")]
    )
    hits = Retriever(store, embedder=None, reranker=None).search("v1", "归并排序")
    assert hits and "keyword" in hits[0].sources


def test_reindex_builds_vectors_without_touching_asr(store: Store, sample_video: Video):
    """回归：后装模型的人必须能只重建向量，而不必重跑（付费的）转写。"""
    from vedioai.pipeline import reindex_videos

    store.replace_chunks(
        "v1",
        [
            make_chunk("v1", 0, 0, 60_000, "归并排序的合并过程"),
            make_chunk("v1", 1, 60_000, 120_000, "快速排序的划分过程"),
        ],
    )
    assert store.has_embeddings("v1") is False

    embedder = FakeEmbedder({"归并": [1.0, 0.0, 0.0], "快速": [0.0, 1.0, 0.0]})
    counts = reindex_videos(store, embedder, ["v1"])

    assert counts == {"v1": 2}
    assert store.has_embeddings("v1") is True

    # 索引建好后，向量召回才真正开始生效
    hits = Retriever(store, embedder=embedder).search("v1", "归并")
    assert any("vector" in h.sources for h in hits)


def test_reindex_rejects_unavailable_embedder(store: Store, sample_video: Video):
    from vedioai.pipeline import reindex_videos

    class Missing:
        available = False

    with pytest.raises(RuntimeError, match="嵌入模型不可用"):
        reindex_videos(store, Missing(), ["v1"])


def test_reindex_skips_courses_without_chunks(store: Store, sample_video: Video):
    """没有文本块的课程（未完成入库）应被跳过，而不是报错。"""
    from vedioai.pipeline import reindex_videos

    counts = reindex_videos(store, FakeEmbedder({}), ["v1"])
    assert counts == {}


def test_retriever_within_chapter_filter(store: Store, sample_video: Video):
    chunks = [
        make_chunk("v1", 0, 0, 60_000, "快速排序"),
        make_chunk("v1", 1, 600_000, 660_000, "快速排序的优化"),
    ]
    store.replace_chunks("v1", chunks)
    chapters = build_chapters("v1", chunks)
    attach_parents(chunks, chapters)
    store.replace_chapters("v1", chapters)

    retriever = Retriever(store)
    # 显式限定章节时才过滤，默认不做时间偏置（保证答案可复现）
    within = chapters[0] if chapters[0].chunk_ids == ["v1-c0000"] else chapters[1]
    hits = retriever.search("v1", "快速排序", within_chapter=within)
    assert {h.chunk.chunk_id for h in hits} <= set(within.chunk_ids)


# ----------------------------------------------------------------- 上下文


def test_prefix_is_deterministic_and_has_no_volatile_content(store: Store, sample_video: Video):
    """前缀必须逐字节稳定，否则上下文缓存永不命中，成本上升一个数量级。"""
    chunks = build_chunks(
        "v1", make_segments([(0, 60_000, "快速排序"), (60_000, 120_000, "归并排序")])
    )
    chapters = build_chapters("v1", chunks)
    attach_parents(chunks, chapters)

    a = build_full_prefix(sample_video, chapters, chunks, video_summary="概述")
    b = build_full_prefix(sample_video, chapters, chunks, video_summary="概述")
    assert a == b

    # 逐字稿必须带时间戳，否则模型无法给出可点击引用
    assert "00:00" in a and "01:00" in a
    # 前缀里不允许出现「当前播放位置」这类变化量
    assert "当前播放" not in a
    assert "current" not in a.lower()


def test_outline_prefix_contains_chapters():
    video = Video(video_id="v1", path="p", title="T", duration_ms=600_000)
    segs = make_segments([(i * 300_000, i * 300_000 + 250_000, "内容" * 30) for i in range(4)])
    chunks = build_chunks("v1", segs)
    for i, c in enumerate(chunks):
        c.title = f"小标题{i}"
        c.summary = f"摘要{i}"
    chapters = build_chapters("v1", chunks)
    attach_parents(chunks, chapters)
    for i, ch in enumerate(chapters):
        ch.title = f"章节{i}"
        ch.summary = f"章节摘要{i}"

    outline = build_outline_prefix(video, chapters, chunks, video_summary="总述")
    assert "课程大纲" in outline
    assert "章节摘要0" in outline
    assert "小标题0" in outline


def test_estimate_tokens_scales():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("中" * 1600) > estimate_tokens("中" * 160)


# ----------------------------------------------------------------- 媒体


def test_plan_proxy_flags_unsupported_codecs():
    ac3 = MediaInfo(
        duration_ms=1000,
        streams=[
            StreamInfo(0, "video", "h264"),
            StreamInfo(1, "audio", "ac3"),
        ],
        format_name="matroska,webm",
    )
    need, reason = plan_proxy(ac3)
    assert need and "ac3" in reason

    good = MediaInfo(
        duration_ms=1000,
        streams=[StreamInfo(0, "video", "h264"), StreamInfo(1, "audio", "aac")],
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
    )
    need, _ = plan_proxy(good)
    assert not need


def test_pick_split_points_snaps_to_silence():
    duration = 2 * 3600 * 1000
    silences = [(i * 60_000, i * 60_000 + 1200) for i in range(120)]
    spans = pick_split_points(duration, silences, chunk_ms=20 * 60 * 1000, overlap_ms=1000)
    assert spans[0][0] == 0
    assert spans[-1][1] == duration
    assert len(spans) >= 6
    # 相邻段之间要有重叠，避免上下文在边界断开
    for a, b in zip(spans, spans[1:], strict=False):
        assert b[0] <= a[1]


def test_pick_split_points_without_silence():
    spans = pick_split_points(100_000, [], chunk_ms=30_000)
    assert spans[0][0] == 0
    assert spans[-1][1] == 100_000


def test_pick_split_points_short_audio_single_span():
    assert pick_split_points(60_000, [], chunk_ms=120_000) == [(0, 60_000)]


# ----------------------------------------------------------------- 自检


def test_selfcheck_reports_missing_credentials_without_network():
    from vedioai.config import Config
    from vedioai.selfcheck import check_asr, check_llm, check_vision

    cfg = Config()
    assert cfg.asr.ready is False
    r = check_asr(cfg, "ffmpeg")
    assert not r.ok and "未配置" in r.detail

    r = check_llm(cfg)
    assert not r.ok and "DEEPSEEK_API_KEY" in r.detail

    r = check_vision(cfg)
    assert not r.ok and "ARK_API_KEY" in r.detail


def test_asr_flash_mode_detection():
    from vedioai.config import ASRConfig
    from vedioai.ingest.asr_volc import VolcASRClient

    cfg = ASRConfig(api_key="x", resource_id="volc.bigasr.auc_turbo")
    assert VolcASRClient(cfg).is_flash is True

    cfg2 = ASRConfig(api_key="x", resource_id="volc.seedasr.auc")
    assert VolcASRClient(cfg2).is_flash is False


def test_asr_not_ready_raises_helpful_error():
    from vedioai.config import ASRConfig
    from vedioai.ingest.asr_volc import ASRError, VolcASRClient

    with pytest.raises(ASRError, match="VOLC_SPEECH_API_KEY"):
        VolcASRClient(ASRConfig())


# ----------------------------------------------------------------- 评分


def test_score_keypoints_supports_alias_arrays():
    answer = "最坏情况是 O(n^2)，解决办法是随机化选取基准元素。"
    result = score_keypoints(
        answer,
        [
            ["O(n²)", "O(n^2)"],          # 任一写法命中即可
            ["随机化", "随机选基准"],      # 命中「随机化」
            "完全没提到的概念",            # 不该命中
        ],
    )
    assert result.total == 3
    assert result.hit == 2
    assert result.missing == ["完全没提到的概念"]


def test_score_keypoints_ignores_punctuation_and_case():
    result = score_keypoints("用 Java 的 Arrays.sort 即可", ["arrays.sort"])
    assert result.hit == 1


def test_score_keypoints_ignores_extension_blockquote():
    """课外「拓展」段落不应干扰课内要点命中判定。"""
    answer = (
        "课内讲的是 adbd 提权。[时间 04:14]\n\n"
        "> **拓展**\n"
        "> 课外常有人把 chmod 777 也当成提权手段，这里不计入课内要点。\n"
    )
    stripped = strip_extension_blocks(answer)
    assert "拓展" not in stripped
    assert "chmod" not in stripped
    # 若未剥离，误命中「chmod」会让评分虚高；剥离后不应命中
    result = score_keypoints(answer, ["adbd", "chmod"])
    assert result.hit == 1
    assert result.missing == ["chmod"]


def test_normalize():
    assert normalize("今天，我们 讲：快速排序！") == "今天我们讲快速排序"


def test_score_citation_states():
    ok = score_citation([{"start_ms": 30_000}], "00:30", 90_000)
    assert ok.status == "ok" and ok.score == 1.0

    # 引用到错误位置比不引用更糟：会误导用户去跳转
    wrong = score_citation([{"start_ms": 600_000}], "00:30", 90_000)
    assert wrong.status == "wrong" and wrong.score == 0.0

    missing = score_citation([], "00:30", 90_000)
    assert missing.status == "missing" and missing.score == 0.5

    assert score_citation([], "", 90_000).status == "n/a"


def test_score_citation_tolerance():
    near = score_citation([{"start_ms": 100_000}], "00:30", 90_000)
    assert near.status == "ok"
    far = score_citation([{"start_ms": 200_000}], "00:30", 90_000)
    assert far.status == "wrong"


def test_build_citations_keeps_late_cited_positions():
    """引用截断不能按时间「留早不留晚」：结论常常落在课程末尾。

    复现 c01：模型在正文里标了 10 个位置，答案落定在最后那个（30:48）上。
    早期实现先按时间排序再截断到 8 条，恰好把最后那个位置挤掉，引用被判成
    「指错位置」（有引用但都指错，比不引用更糟）。
    """
    from vedioai.ask import AskService
    from vedioai.schema import Chunk

    def chunk(i: int) -> Chunk:
        start = i * 60_000
        return Chunk(
            chunk_id=f"v1-c{i:04d}",
            idx=i,
            start_ms=start,
            end_ms=start + 60_000,
            text=f"第 {i} 分钟",
            title=f"t{i}",
            slide_idxs=[],
        )

    chunks = [chunk(i) for i in range(32)]
    # 模型正文里出现的时间戳：前 9 个是取证过程，最后一个是结论落定处
    answer_text = "（讲师在 [30:48] 明确说一共四种）" + "".join(
        f"[{i:02d}:00]" for i in (0, 1, 2, 3, 4, 5, 6, 7, 8)
    )

    service = AskService.__new__(AskService)  # 只测纯逻辑，不建依赖
    got = service._build_citations("v1", [], chunks, [], answer_text)

    assert len(got) <= 12
    assert any(abs(c.start_ms - 30 * 60_000) <= 300_000 for c in got), (
        "时间靠后的引用被截掉了"
    )


# ------------------------------------------------- Ollama 嵌入与故障诊断


def test_ollama_gpu_failure_is_explained():
    """Ollama 的 GPU 报错必须翻译成可照做的指引。

    这个坑排查成本极高（报错文本完全不提解决办法，还容易被误判成
    「模型选错」或「选错了 cuda 库版本」），所以把「强制 CPU」的解法写进
    异常信息里，避免下次又要从头查一遍。
    """
    body = (
        '{"error":"llama-server process has terminated: exit status 0xc0000409: '
        'The system detected an overrun of a stack-based buffer in this application.: '
        'CUDA error: device kernel image is invalid"}'
    )
    msg = explain_ollama_failure(500, body, "http://127.0.0.1:11434")
    assert "OLLAMA_LLM_LIBRARY" in msg
    assert "cpu" in msg
    # 必须点明「改用 cuda_v12 没用」，否则会把人引向错误方向
    assert "cuda_v12" in msg


def test_ollama_unrelated_error_not_misdiagnosed():
    """无关报错不能被误判成 GPU 故障。"""
    assert explain_ollama_failure(500, '{"error":"model not found"}', "http://x") == ""
    assert explain_ollama_failure(404, "not found", "http://x") == ""


def test_check_embedding_skipped_when_disabled():
    """嵌入是可选组件：未启用时不该作为检查项，更不该算失败。"""
    cfg = Config()
    cfg.retrieve = RetrieveConfig(embed_backend="none")
    assert check_embedding(cfg) is None


def test_check_embedding_skipped_when_backend_unreachable():
    """后端探不通时返回 None（跳过），而不是报「失败」。

    嵌入未启用是合法状态，不该让 check --live 变红——否则每次都会误导人。
    """
    cfg = Config()
    cfg.retrieve = RetrieveConfig(embed_backend="ollama")
    cfg.retrieve.ollama_url = "http://127.0.0.1:1"  # 必然连不上
    assert check_embedding(cfg) is None


# ============================================================ 术语表（纠错层）
# 真实事故：讲师说 uiautomator，ASR 听成 "URL to meta"，这个错字穿过摘要与
# 章节笔记，最后变成「使用工具 "URL to meta" 查看 View 布局」这种通顺的错误结论。


def test_glossary_missing_file_is_noop(tmp_path: Path):
    """术语表不存在时不该报错——纠错层是可选增强，不能拖垮入库。"""
    gl = Glossary.load(tmp_path / "nope.yaml")
    text = "URL to meta 应该原样保留"
    fixed, fixes = gl.correct(text)
    assert fixed == text
    assert fixes == []


def test_glossary_loads_global_and_course_rules(tmp_path: Path):
    path = tmp_path / "g.yaml"
    path.write_text(
        "corrections:\n"
        '  - wrong: ["URL to meta"]\n'
        "    right: uiautomator\n"
        "courses:\n"
        '  "abc":\n'
        "    corrections:\n"
        '      - wrong: ["D Y L"]\n'
        "        right: UIAutomator\n",
        encoding="utf-8",
    )
    gl_abc = Glossary.load(path, "abc")
    fixed, _ = gl_abc.correct("URL to meta 和 D Y L")
    assert "uiautomator" in fixed
    assert "UIAutomator" in fixed

    # 别的课程不该拿到 abc 的规则
    gl_other = Glossary.load(path, "other")
    fixed2, _ = gl_other.correct("URL to meta 和 D Y L")
    assert "uiautomator" in fixed2
    assert "D Y L" in fixed2


def test_glossary_replace_is_idempotent(tmp_path: Path):
    """重复执行必须幂等：repair 会被跑很多次，第二次不该再改动。"""
    path = tmp_path / "g.yaml"
    path.write_text(
        'corrections:\n  - wrong: ["URL to meta"]\n    right: uiautomator\n',
        encoding="utf-8",
    )
    gl = Glossary.load(path)
    once, fixes1 = gl.correct("工具 URL to meta 的用法")
    twice, fixes2 = gl.correct(once)
    assert fixes1 and sum(f.count for f in fixes1) == 1
    assert twice == once
    assert fixes2 == []


def test_glossary_counts_all_occurrences(tmp_path: Path):
    path = tmp_path / "g.yaml"
    path.write_text(
        'corrections:\n  - wrong: ["URL to meta"]\n    right: uiautomator\n',
        encoding="utf-8",
    )
    gl = Glossary.load(path)
    _, fixes = gl.correct("URL to meta ... URL to meta ... URL to meta")
    assert sum(f.count for f in fixes) == 3


def test_glossary_flag_spaced_letters():
    """空格字母序列（D Y L）是转写噪音的典型形态，必须报出来。"""
    gl = Glossary.load(None)
    kinds = {f.kind for f in gl.flag("内部使用的是 D Y L two meter 工具类")}
    assert "spaced_letters" in kinds


def test_glossary_flag_inconsistent_spelling():
    """同一文件两种拼写，必有一处错。"""
    gl = Glossary.load(None)
    findings = gl.flag("见 jeb_winco_monitor.bat 与 jeb_wincon_monitor.bat")
    assert any(f.kind == "inconsistent" for f in findings)


def test_glossary_does_not_flag_numbered_variants():
    """Hooks2 与 Hooks3 是真实存在的不同类，不是拼错。

    这是被本课真实数据打出来的误报：把它们当成「拼写不一致」会让人去改
    本来正确的东西，比漏报更有害。
    """
    gl = Glossary.load(None)
    findings = gl.flag("Hooks2.afterHookedMethod 和 Hooks3.afterHookedMethod 都调了")
    assert [f for f in findings if f.kind == "inconsistent"] == []


def test_glossary_flag_quoted_phrase():
    """引号包住的英文短语，像是把听错的词当成了专有名词（"URL to meta"）。"""
    gl = Glossary.load(None)
    findings = gl.flag('使用工具 "URL to meta" 查看布局结构')
    assert any(f.kind == "quoted_phrase" and "URL to meta" in f.term for f in findings)


def test_glossary_trusted_term_not_flagged():
    """白名单里的术语不该被误报。"""
    gl = Glossary.load(None)
    findings = gl.flag('使用工具 "uiautomatorviewer" 查看控件')
    assert [f for f in findings if f.kind == "quoted_phrase"] == []


def test_glossary_flag_lists_suspects_from_file(tmp_path: Path):
    """不确定的词只标记、不替换——这是本层的核心纪律。"""
    path = tmp_path / "g.yaml"
    path.write_text(
        "suspects:\n  - \"D Y L two meter\"\n", encoding="utf-8"
    )
    gl = Glossary.load(path)
    text = "内部使用的是 D Y L two meter 工具类"
    fixed, fixes = gl.correct(text)
    assert fixed == text, "suspects 不该被自动替换"
    assert fixes == []
    assert any(f.kind == "listed" for f in gl.flag(text))


def test_render_findings_is_markdown_table():
    gl = Glossary.load(None)
    findings = gl.flag("用的是 D Y L two meter")
    md = render_findings(findings, Path("vedioai.glossary.yaml"))
    assert md.startswith("## 待人工确认的术语")
    assert "| 术语 |" in md
    assert "vedioai.glossary.yaml" in md


def test_render_findings_empty_is_blank():
    assert render_findings([], None) == ""


def test_project_glossary_is_loadable_and_catches_real_case():
    """项目自带的术语表必须能被解析，且真的能修掉这次的真实错字。

    这条测试是回归护栏：如果谁把 vedioai.glossary.yaml 改坏或删掉规则，
    已经修好的 "URL to meta" 会悄悄回来。
    """
    path = ROOT / "vedioai.glossary.yaml"
    if not path.exists():
        pytest.skip("项目术语表不存在")
    gl = Glossary.load(path, "7aba99cf879083b7")
    fixed, fixes = gl.correct("然后这个是一个叫 URL to meta 的，可以看到 view 的布局结构")
    assert "URL to meta" not in fixed
    assert "uiautomator" in fixed
    assert sum(f.count for f in fixes) == 1


# ---------------------------------------------------------------- 课件 OCR 分辨率
# 背景：早期版本把代表帧压到 960 宽再喂 OCR，把 "uiautomatorviewer.bat" 读成
# "uiautomatoniewer.bet"、"BASE+MD5" 读成 "BASE+MDS"、"录制设置" 读成 "爱制设置"。
# 实测同一帧只改喂图分辨率（960 → 1924），命中率从 58%/77% 提到 92%/100%，
# 而 OCR 耗时只涨 2~16%。下面几条就是钉住「别再把 OCR 的图缩掉」。


def test_slide_config_ocrs_at_native_resolution_by_default():
    cfg = SlideConfig()
    assert cfg.detect_width == 960, "变化检测那一遍要便宜，960 够用"
    assert cfg.ocr_width == 0, "0 = 原始分辨率；OCR 默认绝不能缩图"


def test_as_object_coerces_list_wrapped_response():
    """模型把对象包进数组时不能崩。

    真实事故：章节摘要那一步模型返回了 [...]，而 extract_json 为了稳健会同时
    尝试 {...} 与 [...]，于是拿到 list。老代码直接 .get() 抛 AttributeError，
    且发生在 chunks 已落库之后——一次可降级的失败被升级成整入库崩溃。
    """
    from vedioai.llm.client import LLMError
    from vedioai.summarize import _as_object

    assert _as_object({"title": "t"}, where="x") == {"title": "t"}
    assert _as_object([{"title": "t"}], where="x") == {"title": "t"}
    assert _as_object([1, "x", {"summary": "s"}], where="x") == {"summary": "s"}
    with pytest.raises(LLMError):
        _as_object([], where="x")
    with pytest.raises(LLMError):
        _as_object([1, 2], where="x")
    with pytest.raises(LLMError):
        _as_object("不是对象", where="x")


def test_waterfill_never_exceeds_budget():
    """注水法必须守住 sum(上限) <= budget。

    这条不是形式主义：前缀总长上限就靠它兑现。早期实现只统计「谁还没满」而
    不从剩余预算里扣减，算例 ([10,10,100], budget=60) 会放出 120 —— 超一倍，
    前缀直接突破 max_chars，而且没人会发现。
    """
    from vedioai.context import _waterfill

    cases = [
        ([10, 10, 100], 60),
        ([10, 10, 100], 500),
        ([0, 5], 100),
        ([1, 2, 3], 0),
        ([5, 5, 5], 4),
        ([100, 200, 300], 90),
    ]
    for sizes, budget in cases:
        limits = _waterfill(sizes, budget)
        assert sum(limits) <= budget, f"{sizes} budget={budget} 超预算：{limits}"
        assert all(l <= s for l, s in zip(limits, sizes)), "上限不该超过需求"

    # 预算足够时应当足额满足，不做无谓截断
    assert _waterfill([10, 10, 100], 500) == [10, 10, 100]


def test_prefix_keeps_slide_text_instead_of_cutting_at_400_chars():
    """课件文字不能被固定 400 字上限砍掉。

    真实事故：高清 OCR 后每块课件从百余字涨到数千字（本课中位 2637、最大 11843），
    而前缀仍按 400 字/块截断——169893 字只剩 21169 字，丢 88%。答案往往就在被
    砍掉的那段里（属性窗口里填的路径、某个控件的 resource-id），表现是模型答
    「材料中没有提到」。这种失败比答错更难发现：听起来像「课程没讲」。
    """
    from vedioai.context import build_full_prefix
    from vedioai.schema import Chapter, Chunk, Video, VideoStatus

    video = Video(
        video_id="v1", path="x.mp4", title="t", duration_ms=600_000, status=VideoStatus.READY
    )
    # 一块超长课件文字，答案藏在第 1500 字附近（远超旧的 400 字上限）
    filler = "界面元素 " * 300  # 1500 字
    needle = "F:\\studioSdk\\tools\\bin"
    long_ocr = filler + needle + " 属性窗口 起始位置"
    chunks = [
        Chunk(
            chunk_id="v1-c0000",
            idx=0,
            start_ms=0,
            end_ms=10_000,
            text="讲师打开属性窗口查看起始位置",
            ocr_text=long_ocr,
        )
    ]
    chapters = [
        Chapter(
            chapter_id="v1-h0", idx=0, start_ms=0, end_ms=10_000, title="t", summary="s", chunk_ids=["v1-c0000"]
        )
    ]

    prefix = build_full_prefix(video, chapters, chunks, max_chars=200_000)
    assert needle in prefix, "超出 400 字的课件内容也必须进前缀"
    assert len(prefix) <= 200_000, "前缀仍须守住总长上限"


def test_prefix_truncates_when_course_is_too_long():
    """总长不够时必须按上限收手，且明确告知模型材料被截断。"""
    from vedioai.context import build_full_prefix
    from vedioai.schema import Chunk, Video, VideoStatus

    video = Video(
        video_id="v1", path="x.mp4", title="t", duration_ms=600_000, status=VideoStatus.READY
    )
    chunks = [
        Chunk(chunk_id=f"v1-c{i:04d}", idx=i, start_ms=i * 1000, end_ms=i * 1000 + 1000,
              text="正文" * 500, ocr_text="课件" * 2000)
        for i in range(20)
    ]
    prefix = build_full_prefix(video, [], chunks, max_chars=20_000)
    assert len(prefix) <= 20_000 + 20, "不得突破上限（留出截断提示的余量）"
    assert "已截断" in prefix, "截断必须告知模型，否则它会以为材料完整"


def test_notes_does_not_overwrite_good_doc_when_llm_fails(tmp_path: Path):
    """多数章节生成失败时，不得用降级文档覆盖已有的完整笔记。

    真实事故：DeepSeek 余额耗尽（HTTP 402）导致 23 章全部失败，但代码照常写盘——
    一份 74154 字、带时间戳的完整笔记被 28160 字的降级版（章节笔记退化成章节摘要）
    覆盖。data/ 在 .gitignore 里，没有版本历史可回滚，好内容当场丢失。
    这种 bug 的共同点是把可降级的失败升级成不可逆损失，且留下的文件看起来正常。
    """
    from vedioai.llm.client import LLMError
    from vedioai.notes import NotesService

    class FailingLLM:
        def chat(self, *a, **kw):
            raise LLMError("HTTP 402: Insufficient Balance")

        def close(self):
            pass

    cfg = Config()
    cfg.data_dir = tmp_path / "data"
    store = Store(cfg.db_path)
    vid = "v1"
    store.upsert_video(
        Video(video_id=vid, path="x.mp4", title="t", duration_ms=120_000, status=VideoStatus.READY)
    )
    chunks = [make_chunk(vid, i, i * 10_000, i * 10_000 + 10_000, f"第{i}段正文") for i in range(8)]
    chapter_ids = [f"{vid}-h{i:04d}" for i in range(4)]
    # 块的 parent_id 决定它归到哪一章——不设的话章节取不到内容，整份文档只剩开头
    for i, chunk in enumerate(chunks):
        chunk.parent_id = chapter_ids[i // 2]
    store.replace_chunks(vid, chunks)
    store.replace_chapters(
        vid,
        [
            Chapter(
                chapter_id=chapter_ids[i],
                idx=i,
                start_ms=i * 20_000,
                end_ms=i * 20_000 + 20_000,
                title=f"第{i}章",
                summary=f"第{i}章的摘要",
                chunk_ids=[chunks[2 * i].chunk_id, chunks[2 * i + 1].chunk_id],
            )
            for i in range(4)
        ],
    )

    # 先放一份「良品」旧文档
    out_dir = cfg.library_dir / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    good = out_dir / "notes.md"
    good.write_text("这是一份完整的旧笔记，含时间戳 [12:34] 与要点", encoding="utf-8")

    service = NotesService(cfg, store, FailingLLM())
    service.generate(vid)

    assert good.read_text(encoding="utf-8") == "这是一份完整的旧笔记，含时间戳 [12:34] 与要点", (
        "章节全部失败时不得覆盖已有文档"
    )
    assert not (out_dir / "notes.md.bak").exists(), "没写盘就不该留备份"

    # --force 时才允许覆盖（并留下备份）
    service.generate(vid, force=True)
    assert "这是一份完整的旧笔记" in (out_dir / "notes.md.bak").read_text(encoding="utf-8")
    assert "第0章的摘要" in good.read_text(encoding="utf-8"), "降级版应含摘要兜底内容"
    store.close()


def test_eval_run_marked_invalid_when_most_questions_error():
    """大面积调用失败时，报告必须自我标记为无效。

    真实事故：余额耗尽（HTTP 402）导致 40 题里 39 题报错，总分 0.025，
    但报告照常生成、照常打印。留下的 summary.json 与一次真实回退的测量格式完全
    一样——后来的人拿它做基线，会以为系统坏了，去排查根本不是原因的地方。
    """
    from vedioai.eval_runner import QuestionOutcome, render_report, summarize

    def outcome(qid: str, error: str) -> QuestionOutcome:
        from vedioai.eval_runner import CitationResult, KeypointResult

        o = QuestionOutcome(
            qid=qid,
            qtype="factual",
            question="q",
            score=0.0,
            keypoints=KeypointResult(total=1, hit=0),
            citation=CitationResult(status="missing", score=0.5),
            answer="",
            error=error,
        )
        return o

    outcomes = [outcome(f"f{i:02d}", "调用失败：HTTP 402 Insufficient Balance") for i in range(10)]
    summary = summarize(outcomes, None)
    assert len(summary["failures"]) == 10

    # 复现 main 里的判定逻辑
    errored = summary.get("failures") or []
    assert len(errored) >= max(1, len(outcomes) // 2), "全部报错必须触发无效标记"

    summary["invalid_run"] = True
    summary["invalid_reason"] = "10/10 题调用失败"
    report = render_report(summary, outcomes)
    assert "本次运行无效" in report
    assert report.index("本次运行无效") < report.index("## 总览"), "警示必须在报告最顶部"


def test_prefix_keeps_tail_of_multiline_slide_text():
    """多行课件文字不能丢掉尾部。

    真实事故：预算按**原始长度**（换行算 1 字符）分配，而 _compact 会先把换行换成
    " / "（1→3 字符）再按同一个 limit 截断。于是每块必然超限、尾巴被默默砍掉，
    可分配器还报告「没有块被截断」——出错时不自知。实测让「4，反编译工具字符串搜素」
    这类块尾内容消失，而它正是某道评估题的答案。
    """
    from vedioai.context import build_full_prefix
    from vedioai.schema import Chunk, Video, VideoStatus

    video = Video(
        video_id="v1", path="x.mp4", title="t", duration_ms=60_000, status=VideoStatus.READY
    )
    tail_marker = "4，反编译工具字符串搜素"
    # 行数要够多：换行越多，压缩后比原始长出的部分越多，旧实现丢得越狠
    lines = [f"第{i}行内容" for i in range(60)] + [tail_marker]
    chunks = [
        Chunk(
            chunk_id="v1-c0000",
            idx=0,
            start_ms=0,
            end_ms=10_000,
            text="正文",
            ocr_text="\n".join(lines),
        )
    ]

    # 先按超大上限建一次，得到「完整展开后」的真实长度
    full = build_full_prefix(video, [], chunks, max_chars=10**9)
    assert tail_marker in full

    # 上限刚好等于完整长度：此时一个字符都不该丢
    prefix = build_full_prefix(video, [], chunks, max_chars=len(full) + 10)
    assert tail_marker in prefix, "上限够用时，多行课件的最后一行必须完整保留"
    assert len(prefix) <= len(full) + 10, "不得突破上限"


def test_prefix_never_exceeds_max_chars_with_many_chunks():
    """块数多、每块都很长时，也必须守住上限（省略号占位也要算）。"""
    from vedioai.context import build_full_prefix
    from vedioai.schema import Chunk, Video, VideoStatus

    video = Video(
        video_id="v1", path="x.mp4", title="t", duration_ms=600_000, status=VideoStatus.READY
    )
    chunks = [
        Chunk(
            chunk_id=f"v1-c{i:04d}",
            idx=i,
            start_ms=i * 1000,
            end_ms=i * 1000 + 1000,
            text="正文" * 50,
            ocr_text="\n".join(f"第{j}行课件文字" for j in range(40)),
        )
        for i in range(30)
    ]
    for limit in (5_000, 20_000, 60_000):
        prefix = build_full_prefix(video, [], chunks, max_chars=limit)
        assert len(prefix) <= limit, f"max_chars={limit} 被突破：{len(prefix)}"


def test_chapter_transcript_keeps_speech_when_slide_text_overflows():
    """课件文字太长时，只能削课件，不能把这一块的转写一起丢掉。

    旧实现是整块 `break`：刚好把预算撑破的那一块连转写一起消失，笔记里看不出
    少了老师的一段原话——静默数据丢失。
    """
    from vedioai.notes import CHAPTER_CHAR_BUDGET, NotesService
    from vedioai.schema import Chunk

    chunks = [
        Chunk(chunk_id="c1", idx=0, start_ms=0, end_ms=1000, text="开头", ocr_text=""),
        Chunk(
            chunk_id="c2",
            idx=1,
            start_ms=1000,
            end_ms=2000,
            text="老师原话：这里必须开权限",
            ocr_text="课件文字" * 20_000,  # 远超单章预算
        ),
    ]
    out = NotesService._chapter_transcript(chunks)
    assert "老师原话：这里必须开权限" in out, "转写必须保住"
    assert "本章内容过长" in out, "削了就要留痕"
    assert len(out) <= CHAPTER_CHAR_BUDGET + 40, f"仍应守住预算：{len(out)}"


def test_chapter_transcript_no_false_truncation_notice():
    """内容放得下时不得出现「已截断」——假告警会让模型以为材料缺失。"""
    from vedioai.notes import NotesService
    from vedioai.schema import Chunk

    chunks = [
        Chunk(
            chunk_id="c1",
            idx=0,
            start_ms=0,
            end_ms=1000,
            text="正文",
            ocr_text="课件\n第二行",
        )
    ]
    out = NotesService._chapter_transcript(chunks)
    assert "已截断" not in out
    assert "课件" in out and "第二行" in out and "正文" in out


def test_guide_keeps_only_paragraphs_anchored_to_real_timestamps():
    """导读里核不上时间点的段落必须整段丢弃。

    这是一道防「课程外内容」的机械兜底。参考过的一份同类产品把老师写在 SD 卡上的
    日志改写成「Android 10+ 应写入应用私有目录」的官方推荐——与课程内容正好相反。
    提示词拦不住这类事，所以要求每段都以 `> 时间 MM:SS` 结尾并核对锚点：
    课程之外的内容拿不到合法时间点，因此无法活下来。
    """
    from vedioai.notes import _time_anchors, _verify_guide
    from vedioai.schema import Chapter, Chunk

    chapters = [
        Chapter(chapter_id="h1", idx=0, start_ms=0, end_ms=600_000, title="甲", summary=""),
        Chapter(
            chapter_id="h2", idx=1, start_ms=960_000, end_ms=1_800_000, title="乙", summary=""
        ),
    ]
    chunks = [
        Chunk(chunk_id="c1", idx=0, start_ms=55_000, end_ms=60_000, text="正文"),
        Chunk(chunk_id="c2", idx=1, start_ms=1_820_000, end_ms=1_830_000, text="正文"),
    ]
    anchors = _time_anchors(chapters, chunks)
    assert anchors == [0, 55_000, 960_000, 1_820_000]

    text = """## 学习导读

### 真在课程里的
描述。
> 时间 00:55

### 时间点编的
描述。
> 时间 25:00

### 讲的是课程外的事
Android 10+ 应该改用应用私有目录。
> 时间 40:00

### 时间格式都不合法的
描述。
> 时间 99:99

### 没有任何锚点
这段不该留下。
"""
    body, kept, dropped = _verify_guide(text, anchors)
    assert kept == 1
    # 25:00 / 40:00 是「合法格式但核不上」→ 记入 dropped；
    # 99:99 连 MM:SS 都不是，按正文处理，同样进不了正文（见下面的断言）。
    assert dropped == 2
    assert "真在课程里的" in body
    for gone in ("时间点编的", "讲的是课程外的事", "时间格式都不合法的", "没有任何锚点"):
        assert gone not in body
    assert "Android 10+ 应该改用应用私有目录" not in body
    # 标题不该被重复带进来（调用方自己加）
    assert "学习导读" not in body


def test_guide_does_not_mistake_prose_for_a_citation():
    """正文里出现「时间」二字、或出现比例这类数字，不能被误判成时间标注。"""
    from vedioai.notes import _guide_citation

    assert _guide_citation("时间管理很重要。") is None
    assert _guide_citation("画面比例 16:9，这里是讲解。") is None
    assert _guide_citation("> 时间 00:55") == "00:55"
    assert _guide_citation("> **时间 00:55**") == "00:55"


def test_guide_drops_invalid_time_but_keeps_valid_on_same_line():
    """一行里既有合法又有非法时间点时，只保留合法的那个。"""
    from vedioai.notes import _verify_guide

    anchors = [0, 55_000, 1_820_000]
    text = "### 小节\n描述。\n> 时间 00:55、25:00\n"
    body, kept, dropped = _verify_guide(text, anchors)
    assert kept == 1 and dropped == 0
    assert "00:55" in body and "25:00" not in body


def test_guide_is_omitted_when_nothing_can_be_anchored():
    """整篇都核不上时返回空，调用方据此略去整个导读段落。"""
    from vedioai.notes import _verify_guide

    text = "## 学习导读\n\n### 甲\n描述。\n> 时间 40:00\n"
    body, kept, dropped = _verify_guide(text, [0, 55_000])
    assert body == "" and kept == 0 and dropped == 1


def test_guide_flags_terms_absent_from_material():
    """导读里出现原始材料没有的英文词时，要能被识别出来（只告警，不改写）。

    时间锚点保证「这段话指向真实位置」，但管不了段内的**术语**是不是外加的。
    本课导读就混进过一个 `Method Tracing`——视频里老师只说「方法追踪」。
    """
    from vedioai.notes import _unverified_terms

    material = "老师用 DDMS 做方法追踪，看到 Encrypt.md5 的调用栈。"
    guide = (
        "### 甲\n"
        "用 `DDMS` 对进程做方法追踪（Method Tracing），定位到 `Encrypt.md5`。\n"
        "> 时间 00:55\n"
    )
    odd = _unverified_terms(guide, material)
    assert "Tracing" in odd
    # 材料里有的词不该被误报
    assert "DDMS" not in odd
    assert "Encrypt.md5" not in odd
    assert "方法追踪" not in odd


def test_summary_without_outline_strips_duplicated_chapter_list():
    """剥掉全课摘要末尾自带的「大纲：」逐章列表。

    那个列表与紧随其后的章节表格列的是同一批章节（实测标题 23/23 一致、
    描述相似度 0.60），而且更差——没有时间。摘要 4542 字里 4190 字是它。
    """
    from vedioai.context import summary_without_outline

    summary = (
        "本课程围绕反编译展开。\n\n"
        "大纲：\n"
        "- 定位代码关键的方法：反编译后不知道代码在哪\n"
        "- 加密算法分析与DDMS方法追踪：讲 Base64 和 AES\n"
    )
    out = summary_without_outline(summary)
    assert out == "本课程围绕反编译展开。"
    assert "大纲" not in out


def test_summary_without_outline_keeps_prose_after_marker():
    """「大纲」后面若有正文（不只是列表项），宁可留着重复也不误删正文。"""
    from vedioai.context import summary_without_outline

    summary = (
        "摘要正文。\n\n"
        "大纲：\n"
        "- 第一条\n"
        "这段是结论，不是列表项，必须保住。\n"
    )
    out = summary_without_outline(summary)
    assert out == summary, "拿不准就不动"
    assert "必须保住" in out


def test_summary_without_outline_noop_without_marker():
    from vedioai.context import summary_without_outline

    assert summary_without_outline("只有一段摘要，没有列表。") == "只有一段摘要，没有列表。"
    assert summary_without_outline("") == ""


def test_prefix_does_not_repeat_chapters_from_summary_outline():
    """前缀里同一章不该出现两遍：摘要自带的列表要剥掉，只留结构那一次。"""
    from vedioai.context import build_full_prefix, build_outline_prefix
    from vedioai.schema import Chapter, Chunk, Video, VideoStatus

    video = Video(
        video_id="v1", path="x.mp4", title="课", duration_ms=600_000, status=VideoStatus.READY
    )
    chapters = [
        Chapter(chapter_id="h1", idx=0, start_ms=0, end_ms=300_000, title="甲章", summary="甲"),
        Chapter(
            chapter_id="h2", idx=1, start_ms=300_000, end_ms=600_000, title="乙章", summary="乙"
        ),
    ]
    chunks = [
        Chunk(chunk_id="c1", idx=0, start_ms=0, end_ms=1000, text="正文", parent_id="h1"),
        Chunk(chunk_id="c2", idx=1, start_ms=300_000, end_ms=301_000, text="正文", parent_id="h2"),
    ]
    summary = "总述。\n\n大纲：\n- 甲章：讲甲\n- 乙章：讲乙\n"

    for fn in (build_full_prefix, build_outline_prefix):
        out = fn(video, chapters, chunks, video_summary=summary)
        # 摘要里那份列表要消失（注意 build_outline_prefix 的标题本身含「课程大纲：」，
        # 所以只能断言列表项本身不在了）
        assert "- 甲章：讲甲" not in out
        assert "- 乙章：讲乙" not in out
        assert "总述。" in out
        assert out.count("甲章") == 1, f"{fn.__name__} 里甲章出现了 {out.count('甲章')} 次"


def test_collect_context_separates_ground_truth_from_summaries(store: Store):
    """摘要（LLM 写的）不能进证据池，否则等于用幻觉校验幻觉。

    真实事故：转写里的错字 "D Y L two meter" 被上一轮 LLM 在摘要里
    「合理化」成了 "Dalvik Debug Monitor"。若摘要能当证据，模型只要引用
    这句幻觉就能拿到「依据已核实」的章，循环论证被包装成证据。
    """
    from vedioai.glossary import collect_context

    vid = "v1"
    term = "D Y L two meter"
    store.upsert_video(
        Video(video_id=vid, path="x.mp4", title="t", duration_ms=60000, status=VideoStatus.READY)
    )
    store.replace_segments(
        vid, [Segment(idx=0, start_ms=0, end_ms=1000, text=f"它用的是一个 {term} 工具类")]
    )
    store.replace_slides(
        vid,
        [Slide(idx=0, start_ms=0, end_ms=2000, image_path="a.jpg", ocr_text=f"画面写着 {term}")],
    )
    store.replace_chunks(
        vid,
        [
            make_chunk(
                vid, 0, 0, 2000, "转写正文",
                ocr="课件里写着 " + term,
                slide_idxs=[0],
            )
        ],
    )
    # 摘要里是 LLM 自己「合理化」出来的另一个词——绝不能当证据。
    # 尾部加一段独有内容，避免与转写片段因尾部相同而被去重掉。
    summary_marker = "第二章摘要独有尾注"
    store.replace_chapters(
        vid,
        [
            Chapter(
                chapter_id="v1-h0",
                idx=0,
                start_ms=0,
                end_ms=2000,
                title="t",
                summary=f"{term} 就是上课时说的词。" + summary_marker * 4,
                chunk_ids=[],
            )
        ],
    )

    contexts, ground_truth = collect_context(store, vid, term)
    assert contexts, "应该收集到上下文"
    assert any("章节摘要" in c for c in contexts), "摘要仍可作为线索展示给模型"
    assert any(summary_marker in c for c in contexts), "带独有尾注的摘要条目不应被去重吞掉"

    truth_blob = "\n".join(ground_truth)
    assert "它用的是一个" in truth_blob, "原始转写属于一手材料"
    assert "课件里写着" in truth_blob or "画面写着" in truth_blob, "画面 OCR 属于一手材料"
    # 关键：只有 LLM 摘要里才有的内容，绝不能进证据池
    assert summary_marker not in truth_blob, "摘要不能当证据，否则等于用幻觉校验幻觉"


def test_extract_frame_native_omits_scale_filter(monkeypatch, tmp_path):
    """width=None 必须不带 scale 滤镜——缩图正是 OCR 读错小字的根因。"""
    from vedioai.ingest import media

    calls: list[list[str]] = []
    monkeypatch.setattr(media, "_run", lambda cmd: calls.append(list(cmd)))

    media.extract_frame("ffmpeg", tmp_path / "in.mp4", tmp_path / "a.jpg", 1000)
    media.extract_frame("ffmpeg", tmp_path / "in.mp4", tmp_path / "b.jpg", 1000, width=None)
    media.extract_frame("ffmpeg", tmp_path / "in.mp4", tmp_path / "c.jpg", 1000, width=1280)

    assert "scale=960:-2" in calls[0], "默认仍按 960 抽（预览/缩略图用）"
    assert not any("scale" in part for part in calls[1]), "原生分辨率不该带 scale"
    assert "scale=1280:-2" in calls[2]
    assert all("-q:v" in cmd for cmd in calls), "文字细笔画对 JPEG 压缩敏感，质量要给足"


def test_slide_detection_and_ocr_use_separate_widths(monkeypatch, tmp_path):
    """变化检测用 detect_width，但 OCR 那张必须按 ocr_width（0=原生）来抽。"""
    from vedioai.ingest import slides as S

    frames = [
        S._Frame(
            ms=0,
            path=tmp_path / "f_000001.jpg",
            gray=np.zeros((36, 64), np.float32),
            phash="0" * 16,
        ),
        S._Frame(
            ms=2000,
            path=tmp_path / "f_000002.jpg",
            gray=np.full((36, 64), 255.0, np.float32),
            phash="f" * 16,
        ),
    ]
    sample_widths: list[int] = []
    monkeypatch.setattr(
        S, "extract_frames", lambda *a, **kw: (sample_widths.append(kw["width"]), frames)[1]
    )

    ocr_widths: list[object] = []
    ocr_paths: list[Path] = []

    def fake_extract(ffmpeg, src, dst, at_ms, width=960, **kw):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
        ocr_widths.append(width)
        return dst

    monkeypatch.setattr(S, "extract_frame", fake_extract)
    monkeypatch.setattr(S, "_get_ocr", lambda: object())
    monkeypatch.setattr(
        S, "_ocr_image", lambda ocr, path: (ocr_paths.append(path), "课件文字")[1]
    )

    cfg = SlideConfig(detect_width=640, ocr_width=0, sample_interval_ms=2000)
    got = S.detect_slides("ffmpeg", tmp_path / "v.mp4", 4000, tmp_path / "out", cfg)

    assert sample_widths == [640], "变化检测必须用 detect_width"
    assert len(got) == 2, "灰度过半的相邻帧应判为换页"
    assert ocr_widths == [None, None], "ocr_width=0 表示原始分辨率，不能缩图"
    assert len(ocr_paths) == 2, "OCR 要读高清帧，不是采样帧"
    assert all(str(p).endswith(".jpg") for p in ocr_paths)
    assert all(s.ocr_text == "课件文字" for s in got)


def test_ocr_disabled_keeps_cheap_sample_frame(monkeypatch, tmp_path):
    """不做 OCR 时不该白抽高清帧，直接留采样小图。"""
    from vedioai.ingest import slides as S

    frame = S._Frame(
        ms=0, path=tmp_path / "f_000001.jpg", gray=np.zeros((36, 64), np.float32), phash="0" * 16
    )
    monkeypatch.setattr(S, "extract_frames", lambda *a, **kw: [frame])
    monkeypatch.setattr(S, "extract_frame", lambda *a, **kw: pytest.fail("不该抽高清帧"))
    monkeypatch.setattr(S, "_get_ocr", lambda: pytest.fail("OCR 关闭时不该加载引擎"))

    cfg = SlideConfig(ocr_enabled=False, sample_interval_ms=2000)
    got = S.detect_slides("ffmpeg", tmp_path / "v.mp4", 4000, tmp_path / "out", cfg)

    assert len(got) == 1
    assert got[0].image_path == str(frame.path)
    assert got[0].ocr_text == ""


def test_reocr_slides_refreshes_text_and_prunes_old_frames(monkeypatch, tmp_path):
    """回补要换掉旧图与旧文字，并清掉不再被引用的旧高清帧。"""
    from vedioai.ingest import slides as S

    hd = tmp_path / "hd"
    hd.mkdir()
    stale = hd / "s_000009.jpg"
    stale.write_bytes(b"old")

    target = [Slide(idx=0, start_ms=1000, end_ms=2000, image_path="old.jpg", ocr_text="旧文字")]

    widths: list[object] = []

    def fake_extract(ffmpeg, src, dst, at_ms, width=960, **kw):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
        widths.append(width)
        return dst

    monkeypatch.setattr(S, "extract_frame", fake_extract)
    monkeypatch.setattr(S, "_get_ocr", lambda: object())
    monkeypatch.setattr(S, "_ocr_image", lambda ocr, path: "新文字")

    out = S.reocr_slides("ffmpeg", tmp_path / "v.mp4", target, tmp_path, width=None)

    assert [s.ocr_text for s in out] == ["新文字"]
    assert out[0].image_path.endswith("s_000000.jpg")
    assert widths == [None], "回补也要用原始分辨率"
    assert not stale.exists(), "不再被引用的旧高清帧应被清理"
    assert Path(out[0].image_path).exists(), "新帧必须留下"


def test_representative_ms_uses_span_middle():
    """代表帧取区间中点：换页瞬间那帧常常是空白/半渲染，取起点会丢正文。"""
    from vedioai.ingest.slides import representative_ms

    assert representative_ms(30 * 60_000 + 54_000, 31 * 60_000 + 22_000) == 31 * 60_000 + 8_000
    assert representative_ms(1000, 3000) == 2000
    # 退化区间（end<=start）不能算出越界时刻
    assert representative_ms(5000, 5000) == 5000
    assert representative_ms(5000, 4000) == 5000


def test_detect_slides_extracts_representative_frame_at_span_middle(monkeypatch, tmp_path):
    """变化检测命中的那一帧常是切换瞬间；高清代表帧要取区间中点而非起点。"""
    import numpy as np

    from vedioai.ingest import slides as S
    from vedioai.config import SlideConfig

    # 三帧：0ms 暗（切换瞬间，无文字），2000ms 亮（正文渲染出来），4000ms 暗（换页）
    def frame(ms, value):
        gray = np.full((36, 64), float(value), np.float32)
        path = tmp_path / f"f_{ms:06d}.jpg"
        path.write_bytes(b"x")
        return S._Frame(ms=ms, path=path, gray=gray, phash="0" * 16)

    frames = [frame(0, 0), frame(2000, 255), frame(4000, 0)]
    monkeypatch.setattr(S, "extract_frames", lambda *a, **kw: frames)

    at_times: list[int] = []

    def fake_extract(ffmpeg, src, dst, at_ms, width=960, **kw):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
        at_times.append(at_ms)
        return dst

    monkeypatch.setattr(S, "extract_frame", fake_extract)
    monkeypatch.setattr(S, "_get_ocr", lambda: object())
    monkeypatch.setattr(S, "_ocr_image", lambda ocr, path: "课件文字")

    cfg = SlideConfig(sample_interval_ms=2000)
    got = S.detect_slides("ffmpeg", tmp_path / "v.mp4", 6000, tmp_path / "out", cfg)

    assert [s.start_ms for s in got] == [0, 2000, 4000]
    # 第一页区间 [0,2000)，中点 1000；第三页是最后一页，区间 [4000,6000)，中点 5000
    assert at_times == [1000, 3000, 5000], "代表帧不能固定取区间起点"


# ------------------------------------------------------------------ 用量账本


def test_ledger_charges_peak_and_off_peak_differently():
    """峰谷分时定价：同一个调用放在上午和晚上，价钱差一倍。

    官方是工作日 9:00–12:00、14:00–18:00 为高峰、其余（含周末）为闲时。
    如果不分时段取价，账要么长期高估一倍、要么长期低估一半。
    """
    from datetime import datetime, timedelta, timezone

    bj = timezone(timedelta(hours=8))
    # 2026-09-25 是周五，2026-09-26 是周六
    off = datetime(2026, 9, 25, 22, 0, tzinfo=bj)
    peak = datetime(2026, 9, 25, 10, 0, tzinfo=bj)
    weekend = datetime(2026, 9, 26, 10, 0, tzinfo=bj)

    assert ledger.is_peak(peak) is True
    assert ledger.is_peak(off) is False
    assert ledger.is_peak(weekend) is False, "周末整天都按闲时计价"

    # 边界：12:00 与 18:00 已经是闲时
    assert ledger.is_peak(datetime(2026, 9, 25, 12, 0, tzinfo=bj)) is False
    assert ledger.is_peak(datetime(2026, 9, 25, 18, 0, tzinfo=bj)) is False

    kw = dict(prompt_tokens=100_000, cached_tokens=100_000, completion_tokens=1_000)
    cheap = ledger.estimate_cost("deepseek-flash", at=off, **kw)
    pricey = ledger.estimate_cost("deepseek-flash", at=peak, **kw)
    assert pricey == pytest.approx(cheap * 2)


def test_ledger_counts_cached_input_at_the_cheap_rate():
    """缓存命中与未命中的单价差 50 倍，不能混为一谈。

    这是本项目最核心的成本杠杆（前缀必须稳定就是为它服务的）。如果记账把
    命中数当未命中算，账会虚高 50 倍，用户会误以为方案不可行。
    """
    all_hit = ledger.estimate_cost(
        "deepseek-flash", prompt_tokens=1_000_000, cached_tokens=1_000_000
    )
    all_miss = ledger.estimate_cost(
        "deepseek-flash", prompt_tokens=1_000_000, cached_tokens=0
    )
    assert all_hit == pytest.approx(0.02)   # 闲时命中 ¥0.02/M
    assert all_miss == pytest.approx(1.0)    # 闲时未命中 ¥1/M


def test_ledger_returns_none_for_unknown_model_instead_of_zero():
    """查不到价目的模型必须返回 None（=「不知道」），不能返回 0。

    返回 0 会让总账看起来是准确的，实际却漏掉了一部分花销——
    这比不记账更危险，因为它会让人放心地下结论。
    """
    assert ledger.estimate_cost("some-unknown-model", prompt_tokens=1000) is None
    assert ledger.estimate_cost("", prompt_tokens=1000) is None


def test_ledger_cached_tokens_cannot_exceed_prompt_tokens():
    """上游字段异常（命中数 > 输入数）时应封顶，不能算出负数未命中。"""
    cost = ledger.estimate_cost(
        "deepseek-flash", prompt_tokens=1000, cached_tokens=999_999
    )
    assert cost == pytest.approx(1000 * 0.02 / 1_000_000)


def test_store_usage_summary_separates_priced_from_unpriced(store: Store, sample_video: Video):
    """汇总必须把「未计价次数」暴露出来，否则总账会被误认为是全量。"""
    store.record_usage(
        kind="ask", model="deepseek-flash", video_id="v1",
        prompt_tokens=100_000, cached_tokens=90_000, completion_tokens=500,
        cost_yuan=0.02,
    )
    store.record_usage(
        kind="notes", model="deepseek-flash", video_id="v1",
        prompt_tokens=200_000, cached_tokens=0, completion_tokens=3_000,
        cost_yuan=0.31, peak=True,
    )
    # 视觉模型走火山方舟，没有价目表
    store.record_usage(
        kind="ask", model="doubao-seed", video_id="v1",
        prompt_tokens=5_000, cost_yuan=None,
    )
    store.record_usage(
        kind="asr", model="volc.bigasr.auc_turbo", video_id="v1",
        audio_ms=1_800_000, cost_yuan=0.4,
    )

    s = store.usage_summary("v1")
    assert s["calls"] == 4
    assert s["unpriced_calls"] == 1, "没有价目的那次必须被标出来"
    assert s["cost_yuan"] == pytest.approx(0.73)
    assert s["prompt_tokens"] == 305_000
    assert s["audio_ms"] == 1_800_000

    kinds = {r["kind"]: r for r in s["by_kind"]}
    assert set(kinds) == {"ask", "notes", "asr"}
    assert kinds["asr"]["cost"] == pytest.approx(0.4)

    # 单课视图不需要按课程拆分；全库视图才有
    assert s["by_video"] == []
    full = store.usage_summary()
    assert [v["video_id"] for v in full["by_video"]] == ["v1"]


def test_store_record_usage_never_raises_on_failure(store: Store):
    """记账是旁路：写库失败绝不能把一次已经付费的调用变成失败。

    这里把连接换成必然抛错的桩，验证 record_usage 吞掉异常而不是往外抛。
    """
    class Boom:
        def execute(self, *a, **kw):
            raise sqlite3.OperationalError("database is locked")

        def commit(self):
            raise sqlite3.OperationalError("database is locked")

    real = store.conn
    store.conn = Boom()  # type: ignore[assignment]
    try:
        store.record_usage(kind="ask", model="m", video_id="v1", cost_yuan=1.0)
    finally:
        store.conn = real


def test_usage_scope_tags_calls_with_purpose_and_course():
    """用途与课程靠 contextvar 传递；嵌套时退出要恢复到外层。"""
    assert ledger.current_kind() == "llm"
    with ledger.usage_scope("ask", "v1"):
        assert ledger.current_kind() == "ask"
        assert ledger.current_video_id() == "v1"
        with ledger.usage_scope("summarize", "v2"):
            assert ledger.current_kind() == "summarize"
            assert ledger.current_video_id() == "v2"
        assert ledger.current_kind() == "ask", "退出内层要恢复外层，不能串味"
        assert ledger.current_video_id() == "v1"
    assert ledger.current_kind() == "llm"
    assert ledger.current_video_id() == ""


def test_ledger_recorder_writes_what_the_client_reports(store: Store):
    """记账回调要把客户端上报的用量原样落库，并按模型取价。"""
    from vedioai.llm.client import Reply, Usage

    record = ledger.recorder(store)
    with ledger.usage_scope("notes", "v1"):
        record(Reply(
            text="x",
            usage=Usage(prompt_tokens=1_000_000, cached_tokens=1_000_000, completion_tokens=0),
            model="deepseek-flash",
        ))

    s = store.usage_summary("v1")
    assert s["calls"] == 1
    assert s["by_kind"][0]["kind"] == "notes"
    assert s["cost_yuan"] == pytest.approx(0.02)


def test_client_swallows_usage_callback_failures():
    """记账回调抛错时，客户端必须吞掉，不能让调用方看到异常。

    保护点在客户端（`_notify_usage`），不在 Store：Store 只挡 sqlite 错误，
    而回调里可能出任何错。一次调用已经计过费了，绝不能因为记账失败就报错。
    """
    from vedioai.llm.client import LLMClient, Reply, Usage

    def boom(reply):
        raise RuntimeError("disk full")

    client = LLMClient("k", "http://example.invalid", "m", on_usage=boom)
    client._notify_usage(Reply(text="x", usage=Usage(prompt_tokens=1), model="m"))


def test_backfill_asr_usage_is_grounded_and_idempotent(store: Store, sample_video: Video):
    """回填转写用量：只补「确实跑过转写」的课，且重复执行不翻倍。

    回填的依据是事实而非猜测——库里有转写句说明 ASR 真跑过、真按音频时长
    计过费，时长就在 videos 表里。没有转写的课程不能凭空记一笔，
    否则这个账本会变成编造的数字，比不记还糟。
    """
    from vedioai.cli import _backfill_asr_usage

    # sample_video 在夹具里没有转写句 → 不该被记
    assert store.get_segments("v1") == []
    assert _backfill_asr_usage(store) == 0
    assert store.usage_summary()["calls"] == 0

    store.replace_segments("v1", make_segments([(0, 1000, "这一节我们讲快速排序")]))
    assert _backfill_asr_usage(store) == 1

    s = store.usage_summary("v1")
    assert s["calls"] == 1
    assert s["by_kind"][0]["kind"] == "asr"
    assert s["audio_ms"] == 600_000, "记的是音频时长，不是猜的数字"
    # 汇总接口把金额四舍五入到 4 位小数，容差要跟着这个精度走
    assert s["cost_yuan"] == pytest.approx(
        600_000 / 3_600_000 * ledger.ASR_YUAN_PER_HOUR, abs=1e-4
    )

    # 再跑一次不能重复记账
    assert _backfill_asr_usage(store) == 0
    assert store.usage_summary("v1")["calls"] == 1


# ------------------------------------------------------------------ 会话


def test_session_crud_and_messages(store: Store, sample_video: Video):
    """一门课可有多个命名会话；消息按时间正序；归档后默认列表隐藏。"""
    a = store.create_session("v1", title="期中复习")
    b = store.create_session("v1")
    assert a["session_id"] != b["session_id"]
    assert a["title"] == "期中复习"
    assert b["title"] == ""

    store.add_message(a["session_id"], "user", "什么是快速排序？")
    store.add_message(
        a["session_id"],
        "assistant",
        "一种分治排序。",
        meta={"intent": "local", "citations": []},
    )
    store.add_message(b["session_id"], "user", "作业第三题怎么做？")

    msgs = store.get_messages(a["session_id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["meta"]["intent"] == "local"

    listed = store.list_sessions("v1")
    assert len(listed) == 2
    ids = {s["session_id"] for s in listed}
    assert ids == {a["session_id"], b["session_id"]}
    untitled = next(s for s in listed if s["session_id"] == b["session_id"])
    assert "作业" in untitled["display_title"]

    store.update_session(a["session_id"], archived=True)
    assert len(store.list_sessions("v1")) == 1
    assert len(store.list_sessions("v1", include_archived=True)) == 2

    assert store.delete_session(a["session_id"]) is True
    assert store.get_messages(a["session_id"]) == [], "级联删除必须清掉消息"


def test_build_question_puts_history_after_prefix_marker():
    """多轮历史必须出现在「用户问题」之前、作为问题侧可变部分。

    这是保住前缀缓存的硬约束：历史绝不能进稳定前缀。
    """
    from vedioai.ask import AskService

    history = [
        {"role": "user", "content": "什么是 ptrace？"},
        {"role": "assistant", "content": "一种进程跟踪机制。"},
    ]
    text = AskService._build_question(
        None,  # type: ignore[arg-type]
        "那 attach 呢？",
        None,
        None,
        None,
        history=history,
    )
    assert "【此前对话】" in text
    assert "什么是 ptrace？" in text
    assert text.index("【此前对话】") < text.index("用户问题：那 attach 呢？")


def test_ask_with_session_persists_and_feeds_history(store: Store, sample_video: Video):
    """带 session_id 提问：落库双方消息，第二轮能看见第一轮。"""
    from vedioai.ask import AskService
    from vedioai.config import Config
    from vedioai.llm.client import Reply, Usage
    from vedioai.schema import Chunk

    store.replace_chunks(
        "v1",
        [
            Chunk(
                chunk_id="v1-c0000",
                idx=0,
                start_ms=0,
                end_ms=10_000,
                text="快速排序是一种分治算法",
                ocr_text="",
            )
        ],
    )
    session = store.create_session("v1")
    seen_questions: list[str] = []

    class FakeClient:
        def ask(self, prefix, question, **kw):
            seen_questions.append(question)
            return Reply(
                text=f"答：{(question.split('用户问题：')[-1])[:20]}",
                usage=Usage(prompt_tokens=100, cached_tokens=80, completion_tokens=10),
                model="deepseek-flash",
            )

        def close(self):
            pass

    service = AskService(Config(), store, FakeClient(), vision=None)

    a1 = service.ask("v1", "什么是快速排序？", session_id=session["session_id"])
    assert a1.session_id == session["session_id"]
    assert a1.history_turns == 0
    msgs = store.get_messages(session["session_id"])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user" and "快速排序" in msgs[0]["content"]
    assert "快速排序" in (store.get_session(session["session_id"])["title"] or "")

    a2 = service.ask("v1", "那它最坏复杂度呢？", session_id=session["session_id"])
    assert a2.history_turns == 1
    assert "【此前对话】" in seen_questions[1]
    assert "什么是快速排序？" in seen_questions[1]
    assert "那它最坏复杂度呢？" in seen_questions[1]
    assert "【此前对话】" not in seen_questions[0]


def test_ask_rejects_session_from_another_video(store: Store, sample_video: Video):
    from vedioai.ask import AskService
    from vedioai.config import Config
    from vedioai.schema import Chunk, Video, VideoStatus

    store.replace_chunks(
        "v1",
        [Chunk(chunk_id="v1-c0000", idx=0, start_ms=0, end_ms=1000, text="x", ocr_text="")],
    )
    other = Video(
        video_id="v2", path="D:/b.mp4", title="别的课",
        duration_ms=1000, status=VideoStatus.READY,
    )
    store.upsert_video(other)
    session = store.create_session("v2", title="别课的会话")

    service = AskService(Config(), store, client=None, vision=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="不属于"):
        service.ask("v1", "你好", session_id=session["session_id"])


def test_qa_prompts_allow_labeled_extension_but_keep_grounding():
    """问答允许「拓展」，但课内结论仍须 grounding；摘要提示词不许拓展。"""
    from vedioai.llm import prompts

    assert "> **拓展**" in prompts.QA_LOCAL
    assert "> **拓展**" in prompts.QA_GLOBAL
    assert "课程材料中没有提到" in prompts.QA_LOCAL
    assert "不得**再标" in prompts.QA_LOCAL or "不得" in prompts.QA_LOCAL
    # 入库摘要仍是纯材料，不能掺拓展，否则笔记会把外部知识写进课内文档
    assert "不要补充外部知识" in prompts.CHUNK_SUMMARY
    assert "拓展" not in prompts.SYSTEM_TUTOR
    assert "拓展" in prompts.SYSTEM_QA


def test_segments_to_webvtt_and_search(store: Store, sample_video: Video):
    from vedioai.schema import Segment
    from vedioai.subtitles import segments_to_webvtt

    segs = [
        Segment(idx=0, start_ms=0, end_ms=1500, text="快速排序的平均复杂度"),
        Segment(idx=1, start_ms=2000, end_ms=3500, text="归并排序需要额外空间"),
    ]
    store.replace_segments("v1", segs)
    vtt = segments_to_webvtt(segs)
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt
    assert "快速排序" in vtt

    hits = store.search_segments("v1", "归并排序", top_k=5)
    assert hits and hits[0]["idx"] == 1
    assert "归并" in hits[0]["text"]


def test_append_course_correction_writes_auto_glossary(tmp_path: Path):
    from vedioai.glossary import Glossary, append_course_correction

    main = tmp_path / "vedioai.glossary.yaml"
    main.write_text("version: 1\ncorrections: []\n", encoding="utf-8")
    auto = append_course_correction(main, "vid1", "肉的", "Root", reason="界面")
    assert auto.name.endswith(".auto.yaml")
    gl = Glossary.load(main, video_id="vid1")
    fixed, fixes = gl.correct("肉的权限")
    assert fixed == "Root权限"
    assert fixes and fixes[0].right == "Root"


def test_series_groups_by_parent_dir():
    from vedioai.series import series_id_for, series_label

    assert series_id_for(r"D:\courses\Android\a.mp4") == "Android"
    assert series_label("_ungrouped") == "未分组"

