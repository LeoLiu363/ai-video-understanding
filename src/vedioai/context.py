"""构建稳定前缀（给长上下文直答用）。

**这里唯一的硬约束：前缀必须逐字节稳定。**

因为成本模型建立在上下文缓存命中上：整稿约 4-5 万 token，问 30 个问题就要重发
30 次，命中缓存后输入成本降到 ¥0.05–0.10/M，不命中则要 ¥1.5–3/M，差 6–30 倍。

所以：
- 不要把「当前播放进度」「当前时间」等变化量放进前缀；
- 不要在前缀里插入随机 ID；
- 需要播放位置上下文的，放到问题那一侧（见 ask.build_question）。
"""

from __future__ import annotations

from .schema import Chapter, Chunk, Video, ms_to_hms


def build_full_prefix(
    video: Video,
    chapters: list[Chapter],
    chunks: list[Chunk],
    *,
    video_summary: str = "",
    max_chars: int = 200_000,
) -> str:
    """整稿前缀：课程元信息 + 章节摘要 + 逐字稿（带时间戳）。

    短视频/长视频都用它——2 小时中文课全稿只有约 3 万字，完全塞得进 1M 窗口，
    分块检索反而是最弱的一档。
    """
    parts: list[str] = []
    parts.append(f"# 课程材料：{video.title or '未命名课程'}")
    parts.append(f"总时长：{ms_to_hms(video.duration_ms)}")

    if video_summary:
        parts.append("\n## 全课摘要\n" + video_summary.strip())

    if chapters:
        parts.append("\n## 章节结构")
        for ch in chapters:
            head = f"- [{ms_to_hms(ch.start_ms)}] "
            head += ch.title or f"第 {ch.idx + 1} 章"
            if ch.summary:
                head += f"：{ch.summary.strip()}"
            parts.append(head)

    parts.append("\n## 逐字稿")
    for chunk in chunks:
        parts.append(f"\n【{ms_to_hms(chunk.start_ms)}】")
        if chunk.ocr_text:
            parts.append(f"（课件）{_compact(chunk.ocr_text)}")
        parts.append(chunk.text)

    text = "\n".join(parts)
    if len(text) > max_chars:
        # 极端长的课程才截断，且明确告知模型，避免它以为材料完整
        text = text[:max_chars] + "\n\n（材料过长已截断）"
    return text


def build_outline_prefix(
    video: Video,
    chapters: list[Chapter],
    chunks: list[Chunk],
    *,
    video_summary: str = "",
) -> str:
    """大纲前缀：只给章节摘要与片段要点，用于全局聚合型问题。

    「这门课一共讲了几种 X」这类问题，靠 top-k 召回结构上就不可能答全，
    必须走这一层。
    """
    parts: list[str] = [f"# 课程大纲：{video.title or '未命名课程'}"]
    parts.append(f"总时长：{ms_to_hms(video.duration_ms)}")
    if video_summary:
        parts.append("\n## 全课摘要\n" + video_summary.strip())

    by_parent: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        if chunk.parent_id:
            by_parent.setdefault(chunk.parent_id, []).append(chunk)

    for ch in chapters:
        title = ch.title or f"第 {ch.idx + 1} 章"
        parts.append(f"\n## {title}　[{ms_to_hms(ch.start_ms)} - {ms_to_hms(ch.end_ms)}]")
        if ch.summary:
            parts.append(ch.summary.strip())
        for chunk in by_parent.get(ch.chapter_id, []):
            label = chunk.title or ms_to_hms(chunk.start_ms)
            line = f"- [{ms_to_hms(chunk.start_ms)}] {label}"
            if chunk.summary:
                line += f"：{chunk.summary.strip()}"
            parts.append(line)
    return "\n".join(parts)


def build_citation_prefix(chunks: list[Chunk]) -> str:
    """只给命中的片段，用于多集课程库场景（单课不用）。"""
    parts = ["# 检索到的课程片段"]
    for chunk in chunks:
        parts.append(f"\n【{ms_to_hms(chunk.start_ms)} - {ms_to_hms(chunk.end_ms)}】")
        parts.append(chunk.text)
        if chunk.ocr_text:
            parts.append(f"（课件）{_compact(chunk.ocr_text)}")
    return "\n".join(parts)


def _compact(text: str, limit: int = 400) -> str:
    text = " / ".join(line.strip() for line in text.splitlines() if line.strip())
    return text if len(text) <= limit else text[:limit] + "…"


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。中文约 1 token ≈ 0.7 字，这里按字符数保守估。"""
    return max(1, int(len(text) / 1.6))
