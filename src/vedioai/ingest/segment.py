"""语义分段：句级转写 → 语义块 → 章节。

刻意不用固定窗口硬切（30–90 秒），那会把概念从中间切断。这里用
「停顿间隙 + 时长上下界」组合，尽量落在自然边界上。
"""

from __future__ import annotations

from .slides import slides_in_range
from ..schema import Chapter, Chunk, Segment, Slide

# 块时长上下界
CHUNK_MIN_MS = 30_000
CHUNK_MAX_MS = 90_000
# 段落之间超过该间隙，视为自然边界
CHUNK_GAP_MS = 1_500

# 章节：更大的停顿，或时长上限
CHAPTER_GAP_MS = 6_000
CHAPTER_MAX_MS = 600_000


def build_chunks(
    video_id: str,
    segments: list[Segment],
    slides: list[Slide] | None = None,
) -> list[Chunk]:
    if not segments:
        return []

    slides = slides or []
    groups: list[list[Segment]] = []
    current: list[Segment] = [segments[0]]

    for seg in segments[1:]:
        prev = current[-1]
        gap = seg.start_ms - prev.end_ms
        span = seg.end_ms - current[0].start_ms
        long_enough = span >= CHUNK_MIN_MS
        if (long_enough and gap >= CHUNK_GAP_MS) or span >= CHUNK_MAX_MS:
            groups.append(current)
            current = [seg]
        else:
            current.append(seg)
    groups.append(current)

    chunks: list[Chunk] = []
    for idx, group in enumerate(groups):
        start_ms = group[0].start_ms
        end_ms = group[-1].end_ms
        text = "\n".join(_with_speaker(s) for s in group)

        slide_idxs = slides_in_range(slides, start_ms, end_ms)
        ocr_text = "\n".join(
            slides[i].ocr_text for i in slide_idxs if slides[i].ocr_text
        ).strip()

        chunks.append(
            Chunk(
                chunk_id=f"{video_id}-c{idx:04d}",
                idx=idx,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                ocr_text=ocr_text,
                slide_idxs=slide_idxs,
            )
        )
    return chunks


def _with_speaker(seg: Segment) -> str:
    if seg.speaker and seg.speaker not in ("0", "1", ""):
        return f"[{seg.speaker}] {seg.text}"
    return seg.text


def build_chapters(video_id: str, chunks: list[Chunk]) -> list[Chapter]:
    """把语义块聚成章节。章节目的是给长上下文直答提供结构，并支撑全局聚合问答。"""
    if not chunks:
        return []

    groups: list[list[Chunk]] = []
    current: list[Chunk] = [chunks[0]]
    for chunk in chunks[1:]:
        prev = current[-1]
        gap = chunk.start_ms - prev.end_ms
        span = chunk.end_ms - current[0].start_ms
        if gap >= CHAPTER_GAP_MS or span >= CHAPTER_MAX_MS:
            groups.append(current)
            current = [chunk]
        else:
            current.append(chunk)
    groups.append(current)

    chapters: list[Chapter] = []
    for idx, group in enumerate(groups):
        chapters.append(
            Chapter(
                chapter_id=f"{video_id}-ch{idx:03d}",
                idx=idx,
                start_ms=group[0].start_ms,
                end_ms=group[-1].end_ms,
                title="",
                summary="",
                chunk_ids=[c.chunk_id for c in group],
            )
        )
    return chapters


def attach_parents(chunks: list[Chunk], chapters: list[Chapter]) -> None:
    """建立 parent-child 关系：命中子块时把父章节一并返回。"""
    by_chunk: dict[str, str] = {}
    for ch in chapters:
        for cid in ch.chunk_ids:
            by_chunk[cid] = ch.chapter_id
    for chunk in chunks:
        chunk.parent_id = by_chunk.get(chunk.chunk_id)
