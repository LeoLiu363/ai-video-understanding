"""分层摘要树：块要点 → 章节摘要 → 全课摘要。

这是「总结整门课」和「跨段聚合问答」能成立的前提。
如果只在提问时临时 map-reduce，会有三个问题：每次都重算、结果不稳定、慢；
而且全局型问题没有可召回的结构，必然漏项。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .llm import prompts
from .llm.client import LLMClient, LLMError
from .schema import Chapter, Chunk, Video, ms_to_hms
from .store import Store

log = logging.getLogger(__name__)


@dataclass
class Concept:
    term: str
    definition: str
    start_ms: int = 0


@dataclass
class SummaryResult:
    video_summary: str = ""
    video_title: str = ""
    concepts: list[Concept] = field(default_factory=list)
    # 降级信息：摘要只是锦上添花，失败不该让整次入库作废。
    # 但这些数字必须带出来，否则「静默降级」会被误当成「一切正常」。
    failed_chunks: int = 0
    failed_chapters: int = 0
    video_summary_failed: bool = False

    @property
    def degraded(self) -> bool:
        return bool(self.failed_chunks or self.failed_chapters or self.video_summary_failed)

    def describe(self) -> str:
        if not self.degraded:
            return ""
        parts = []
        if self.failed_chunks:
            parts.append(f"{self.failed_chunks} 个片段")
        if self.failed_chapters:
            parts.append(f"{self.failed_chapters} 个章节")
        if self.video_summary_failed:
            parts.append("全课摘要")
        return "摘要生成部分失败（" + "、".join(parts) + "），转写与检索不受影响。可重跑入库补齐。"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


# 结构化抽取的 token 预算。这些是大批量的高频调用（一门课几十次），
# 给够即可，不要抠。配合 chat_json 的「截断自动翻倍重试」，不会再出现
# 因为思考吃光预算而返回空内容的情况。
CHUNK_SUMMARY_TOKENS = 2048
CHAPTER_SUMMARY_TOKENS = 2048
VIDEO_SUMMARY_TOKENS = 4096


def summarize_chunks(
    client: LLMClient,
    chunks: list[Chunk],
    *,
    workers: int = 4,
    progress=None,
) -> tuple[list[Concept], int]:
    """为每个语义块生成小标题、要点与术语。

    返回 (术语列表, 失败片段数)。
    """
    concepts: list[Concept] = []
    failures = 0

    def work(chunk: Chunk) -> tuple[Chunk, list[Concept], bool]:
        body = _truncate(chunk.combined_text, 6000)
        messages = [
            {"role": "system", "content": prompts.SYSTEM_TUTOR},
            {"role": "user", "content": f"{prompts.CHUNK_SUMMARY}\n\n---\n{body}"},
        ]
        try:
            data, _ = client.chat_json(messages, max_tokens=CHUNK_SUMMARY_TOKENS)
        except LLMError as exc:
            log.warning("片段摘要失败 %s: %s", chunk.chunk_id, exc)
            return chunk, [], False
        chunk.title = str(data.get("title") or "").strip()[:40]
        chunk.summary = str(data.get("summary") or "").strip()
        local: list[Concept] = []
        for item in data.get("concepts") or []:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term") or "").strip()
            definition = str(item.get("definition") or "").strip()
            if term and definition:
                local.append(Concept(term=term, definition=definition, start_ms=chunk.start_ms))
        return chunk, local, True

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, c) for c in chunks]
        for future in as_completed(futures):
            chunk, local, ok = future.result()
            if not ok:
                failures += 1
            concepts.extend(local)
            done += 1
            if progress:
                progress(done, len(chunks), f"片段摘要 {done}/{len(chunks)}")
    return concepts, failures


def summarize_chapters(
    client: LLMClient,
    chapters: list[Chapter],
    chunks: list[Chunk],
    *,
    workers: int = 4,
    progress=None,
) -> int:
    """为一章生成标题与摘要（输入是该章的片段要点，不是逐字稿）。返回失败章节数。"""
    by_id = {c.chunk_id: c for c in chunks}
    failures = 0

    def work(chapter: Chapter) -> bool:
        lines = []
        for cid in chapter.chunk_ids:
            chunk = by_id.get(cid)
            if not chunk:
                continue
            label = chunk.title or ms_to_hms(chunk.start_ms)
            lines.append(f"- [{ms_to_hms(chunk.start_ms)}] {label}：{chunk.summary or _truncate(chunk.text, 200)}")
        body = "\n".join(lines) or "（本章无可用要点）"
        messages = [
            {"role": "system", "content": prompts.SYSTEM_TUTOR},
            {"role": "user", "content": f"{prompts.CHAPTER_SUMMARY}\n\n---\n{body}"},
        ]
        try:
            data, _ = client.chat_json(messages, max_tokens=CHAPTER_SUMMARY_TOKENS)
        except LLMError as exc:
            log.warning("章节摘要失败 %s: %s", chapter.chapter_id, exc)
            return False
        chapter.title = str(data.get("title") or "").strip()[:40]
        summary = str(data.get("summary") or "").strip()
        points = [str(p).strip() for p in (data.get("key_points") or []) if str(p).strip()]
        if points:
            summary += "\n\n要点：\n" + "\n".join(f"- {p}" for p in points)
        chapter.summary = summary
        return True

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, ch) for ch in chapters]
        for future in as_completed(futures):
            if not future.result():
                failures += 1
            done += 1
            if progress:
                progress(done, len(chapters), f"章节摘要 {done}/{len(chapters)}")
    return failures


def summarize_video(
    client: LLMClient,
    video: Video,
    chapters: list[Chapter],
    chunks: list[Chunk],
) -> SummaryResult:
    """全课摘要。输入是章节摘要层，不是逐字稿——这就是「分层」的意义。"""
    lines: list[str] = [f"课程原名：{video.title}", f"总时长：{ms_to_hms(video.duration_ms)}"]
    for ch in chapters:
        span = f"{ms_to_hms(ch.start_ms)}-{ms_to_hms(ch.end_ms)}"
        lines.append(f"\n## {ch.title or f'第 {ch.idx + 1} 章'} [{span}]")
        lines.append(ch.summary or "（无摘要）")
    body = "\n".join(lines)

    messages = [
        {"role": "system", "content": prompts.SYSTEM_TUTOR},
        {"role": "user", "content": f"{prompts.VIDEO_SUMMARY}\n\n---\n{body}"},
    ]
    result = SummaryResult()
    try:
        data, _ = client.chat_json(messages, max_tokens=VIDEO_SUMMARY_TOKENS)
    except LLMError as exc:
        # 关键：这一步失败**绝不能**让整次入库作废。
        # 转写、课件、分段、索引都已经是有效成果，其中转写还已经付过费。
        log.warning("全课摘要失败（不影响转写与检索）：%s", exc)
        result.video_summary_failed = True
        return result

    result.video_title = str(data.get("title") or "").strip()
    result.video_summary = str(data.get("summary") or "").strip()

    outline = data.get("outline") or []
    if outline:
        extra = []
        for item in outline:
            if not isinstance(item, dict):
                continue
            chapter = str(item.get("chapter") or "").strip()
            points = [str(p).strip() for p in (item.get("points") or []) if str(p).strip()]
            if chapter:
                extra.append(f"- {chapter}" + ("：" + "；".join(points) if points else ""))
        if extra:
            result.video_summary += "\n\n大纲：\n" + "\n".join(extra)

    for item in data.get("concepts") or []:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        definition = str(item.get("definition") or "").strip()
        if term and definition:
            result.concepts.append(Concept(term=term, definition=definition))

    return result


def build_summary_tree(
    client: LLMClient,
    store: Store,
    video: Video,
    chunks: list[Chunk],
    chapters: list[Chapter],
    *,
    progress=None,
) -> SummaryResult:
    """完整跑一遍摘要树并落库。

    任何一层失败都只降级、不抛出——摘要失败不该让已经付过费的转写作废。
    失败信息通过 SummaryResult 带出，由调用方展示，不做静默吞掉。
    """
    if progress:
        progress(0, 1, "生成片段要点")
    chunk_concepts, failed_chunks = summarize_chunks(client, chunks, progress=progress)

    if progress:
        progress(0, 1, "生成章节摘要")
    failed_chapters = summarize_chapters(client, chapters, chunks, progress=progress)

    if progress:
        progress(0, 1, "生成全课摘要")
    result = summarize_video(client, video, chapters, chunks)
    result.failed_chunks = failed_chunks
    result.failed_chapters = failed_chapters

    for chunk in chunks:
        store.update_chunk_summary(video.video_id, chunk.chunk_id, chunk.title, chunk.summary)
    store.replace_chapters(video.video_id, chapters)
    if result.video_summary:
        store.set_video_summary(video.video_id, result.video_summary)

    # 术语表：同名合并，取首次出现位置
    merged: dict[str, Concept] = {}
    for concept in chunk_concepts + result.concepts:
        if concept.term not in merged:
            merged[concept.term] = concept
    result.concepts = list(merged.values())
    return result
