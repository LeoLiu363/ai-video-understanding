"""不依赖网络与 FFmpeg 的单元测试：存储、分段、检索、上下文、评分。

这些是「改一处怕碰坏另一处」最容易出问题的部分，所以先钉住。

运行：
    .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.run_eval import normalize, score_citation, score_keypoints  # noqa: E402
from vedioai.context import build_full_prefix, build_outline_prefix, estimate_tokens  # noqa: E402
from vedioai.ingest.media import MediaInfo, StreamInfo, pick_split_points, plan_proxy  # noqa: E402
from vedioai.ingest.segment import attach_parents, build_chapters, build_chunks  # noqa: E402
from vedioai.config import RetrieveConfig  # noqa: E402
from vedioai.retrieve import Retriever  # noqa: E402
from vedioai.schema import Segment, Slide, Video, VideoStatus, hms_to_ms, ms_to_hms  # noqa: E402
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
