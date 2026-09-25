"""混合检索：向量召回 + FTS5 关键词，RRF 融合，可选重排。

说明两点刻意的设计选择：

1. **时间偏置默认关闭**。旧项目按「当前播放位置」给检索结果加权，这会让同一个
   问题在不同时刻得到不同答案，不可复现，也无法评估。现在它退化为一个显式的
   UI 开关「仅在本章检索」，默认关。

2. **单课问答不依赖检索**。主路径是长上下文直答；检索用于多集课程库降本，
   以及给视觉型问题定位证据图。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import RetrieveConfig
from .embedding import EmbedderLike, Reranker
from .schema import Chapter, Chunk
from .store import Store

log = logging.getLogger(__name__)

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float
    sources: list[str]

    @property
    def start_ms(self) -> int:
        return self.chunk.start_ms


class Retriever:
    def __init__(
        self,
        store: Store,
        *,
        embedder: EmbedderLike | None = None,
        reranker: Reranker | None = None,
        cfg: RetrieveConfig | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = cfg or RetrieveConfig()

    def search(
        self,
        video_id: str,
        query: str,
        *,
        top_k: int | None = None,
        time_bias_ms: int | None = None,
        within_chapter: Chapter | None = None,
    ) -> list[Hit]:
        top_k = top_k or self.cfg.top_k
        pool = max(top_k * 4, 20)

        ranked: dict[str, list[int]] = {}
        sources: dict[str, set[str]] = {}

        kw = self.store.search_keyword(video_id, query, pool)
        for rank, (chunk_id, _score) in enumerate(kw):
            ranked.setdefault(chunk_id, []).append(rank)
            sources.setdefault(chunk_id, set()).add("keyword")

        if self.embedder and self.embedder.available:
            try:
                vecs = self.embedder.encode([query])
            except Exception as exc:  # noqa: BLE001
                # 外部嵌入服务（Ollama）可能中途挂掉。此时不能连带把整个问答弄崩——
                # 但必须留下日志，否则「向量召回失效」会变成静默降级。
                log.warning("向量编码失败（%s），本次只用关键词检索：%s", type(exc).__name__, exc)
                vecs = None
            if vecs is not None and len(vecs):
                for rank, (chunk_id, _sim) in enumerate(
                    self.store.search_vector(video_id, vecs[0], pool)
                ):
                    ranked.setdefault(chunk_id, []).append(rank)
                    sources.setdefault(chunk_id, set()).add("vector")

        if not ranked:
            # 检索完全没命中时，退化为「最近章节的前若干块」，
            # 而不是返回空——空结果会让用户以为课程没讲。
            return self._fallback(video_id, top_k, within_chapter)

        fused = {
            cid: sum(1.0 / (RRF_K + r) for r in ranks) for cid, ranks in ranked.items()
        }
        chunks = {c.chunk_id: c for c in self.store.get_chunks(video_id)}

        if within_chapter is not None:
            allowed = set(within_chapter.chunk_ids)
            fused = {cid: s for cid, s in fused.items() if cid in allowed}

        if self.cfg.time_bias and time_bias_ms is not None:
            for cid, score in list(fused.items()):
                chunk = chunks.get(cid)
                if not chunk:
                    continue
                # 距离越近加成越高，但幅度刻意压得很小
                distance = abs(chunk.start_ms - time_bias_ms)
                fused[cid] = score * (1.0 + 0.15 / (1.0 + distance / 600_000))

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:pool]
        hits = [
            Hit(chunk=chunks[cid], score=score, sources=sorted(sources.get(cid, [])))
            for cid, score in ordered
            if cid in chunks
        ]

        hits = self._rerank(query, hits)
        return hits[:top_k]

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not (self.reranker and self.reranker.available) or len(hits) <= 1:
            return hits
        candidates = hits[: max(self.cfg.rerank_top_n * 6, 20)]
        try:
            scores = self.reranker.score(query, [h.chunk.combined_text for h in candidates])
        except Exception:  # noqa: BLE001
            return hits
        for hit, score in zip(candidates, scores, strict=False):
            hit.score = float(score)
            hit.sources.append("rerank")
        candidates.sort(key=lambda h: -h.score)
        return candidates + hits[len(candidates) :]

    def _fallback(
        self, video_id: str, top_k: int, chapter: Chapter | None
    ) -> list[Hit]:
        chunks = self.store.get_chunks(video_id)
        if chapter is not None:
            allowed = set(chapter.chunk_ids)
            picked = [c for c in chunks if c.chunk_id in allowed]
        else:
            # 取开头若干块：通常是课程目标与目录，比随机取好
            picked = chunks
        picked = picked[: max(3, min(top_k, 5))]
        return [Hit(chunk=c, score=0.0, sources=["fallback"]) for c in picked]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)
