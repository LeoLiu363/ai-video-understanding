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

import re

from .schema import Chapter, Chunk, Video, ms_to_hms


def _waterfill(sizes: list[int], budget: int) -> list[int]:
    """把 ``budget`` 按需分配给 ``sizes``，返回每一项的上限（注水法）。

    为什么不是简单均分：均分会让小项浪费配额、大项照样被截。做法是逐轮抬高
    「水位」——当前水位之下的项按需求足额满足，剩下的预算再由还在挨截的项平分。
    对长度差异很大的项（本课课件文字从几十字到一万多字都有），这能显著提高
    实际保留量。

    正确性要求：``sum(result) <= budget``。这一点必须守住，否则前缀会突破
    max_chars。所以每轮都要从 remaining 里真实扣减，不能只看「谁还没满」。

    必须是确定性的：前缀要逐字节稳定才能命中上下文缓存，所以这里不能有任何
    随机或依赖输入顺序之外的东西。
    """
    n = len(sizes)
    if n == 0 or budget <= 0:
        return [0] * n
    limits = [0] * n
    remaining = budget
    active = list(range(n))
    while active:
        share = remaining // len(active)
        if share <= 0:
            break
        below = [i for i in active if sizes[i] <= share]
        if not below:
            # 没人能在当前水位被满足，就把水位定在这里、预算分完
            for i in active:
                limits[i] = share
            break
        for i in below:
            limits[i] = sizes[i]
            remaining -= sizes[i]
        below_set = set(below)
        active = [i for i in active if i not in below_set]
    return limits


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
        # 剥掉摘要自带的「大纲：」列表：下面紧接着就按章节列一遍，且带时间。
        parts.append("\n## 全课摘要\n" + summary_without_outline(video_summary).strip())

    if chapters:
        parts.append("\n## 章节结构")
        for ch in chapters:
            head = f"- [{ms_to_hms(ch.start_ms)}] "
            head += ch.title or f"第 {ch.idx + 1} 章"
            if ch.summary:
                head += f"：{ch.summary.strip()}"
            parts.append(head)

    # ---------------------------------------------------------- 逐字稿
    #
    # 课件文字（OCR）不能按固定字数截断。高清 OCR 之后每块课件可达数千字，
    # 固定上限会把绝大部分内容砍掉——实测本课 169893 字只剩 21169 字（丢 88%），
    # 而答案常常就在被砍掉的那段里（讲师在属性窗口里填的路径、某个控件的
    # resource-id），表现是模型答「材料中没有提到」。这类失败比答错更难发现：
    # 它听起来像「课程没讲」，用户不会去怀疑。
    #
    # 改成把前缀的剩余空间按需分配给各块课件文字（_waterfill）：转写一个字不砍，
    # 课件文字在总长上限内尽量保住。这样前缀总长仍守 max_chars。
    #
    # 关键细节：尺寸必须按**压缩后**的长度算，不能按原始长度。
    # _compact 会把换行换成 " / "（1 字符 → 3 字符）再按同一个 limit 截断，
    # 所以「按原始长度分配、按压缩长度执行」必然每块都超限、尾巴被默默砍掉。
    # 实测这一处让「4，反编译工具字符串搜素」这类位于块尾的内容消失，
    # 而分配器自己还报告「没有块被截断」——出错时不自知，最难查。
    compacted = [_compact(c.ocr_text, 10**9) if c.ocr_text else "" for c in chunks]
    ocr_sizes = [len(t) for t in compacted]

    # 预算要把每一项都算进去：元信息、章节、逐字稿骨架、每块课件前的「（课件）」
    # 标记，以及 join 时每个元素之间的换行。
    base = sum(len(p) for p in parts) + len("\n## 逐字稿\n")
    per_chunk_fixed = sum(len(f"\n【{ms_to_hms(c.start_ms)}】") + len(c.text) for c in chunks)
    ocr_markers = 4 * sum(1 for t in compacted if t)  # "（课件）"
    n_parts = len(parts) + 1 + sum(2 + (1 if t else 0) for t in compacted)
    join_sep = max(0, n_parts - 1)

    budget = max(0, max_chars - base - per_chunk_fixed - ocr_markers - join_sep)
    ocr_limits = _waterfill(ocr_sizes, budget)

    parts.append("\n## 逐字稿")
    for chunk, body, ocr_limit in zip(chunks, compacted, ocr_limits):
        parts.append(f"\n【{ms_to_hms(chunk.start_ms)}】")
        if body and ocr_limit > 0:
            # 省略号也要占位：否则每截断一块就多出 1 字符，累加起来突破上限，
            # 触发末尾兜底截断——那会把最后几块整段砍掉，正是要避免的事。
            shown = body if len(body) <= ocr_limit else body[: max(0, ocr_limit - 1)] + "…"
            parts.append(f"（课件）{shown}")
        parts.append(chunk.text)

    text = "\n".join(parts)
    if len(text) > max_chars:
        # 兜底。上面的分配已经按 max_chars 算过，正常不该走到这里；
        # 真走到了也明确告知模型，避免它以为材料完整。
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
        # 同上：下面会逐章展开，摘要里的「大纲：」列表是重复的。
        parts.append("\n## 全课摘要\n" + summary_without_outline(video_summary).strip())

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


# 存库摘要里「大纲：」这一行（容忍 ## 前缀与加粗）
_SUMMARY_OUTLINE_MARK = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*大纲\s*(?:\*\*)?\s*[:：]\s*$")
_SUMMARY_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.、)）])\s*\S")


def summary_without_outline(summary: str) -> str:
    """去掉全课摘要末尾自带的「大纲：」逐章列表。

    存库的 video_summary 由 VIDEO_SUMMARY 提示词生成，那个提示词同时要求
    `"summary"` 和 `"outline"`——而 outline 是让模型把**刚给它的章节摘要再列
    一遍**，属于信息复用而非推导。于是摘要里 92% 的字数（实测 4542 字中的
    4190 字）是这份列表，而所有调用方紧接着就会把同样的章节再列一遍：

    - notes.md：后面紧跟「课程大纲」表格（标题 23/23 完全一致，描述相似度 0.60）
    - build_full_prefix：「## 章节结构」
    - build_outline_prefix：逐章展开
    - `vedioai show` / `/api/library`：分别打印章节列表

    而且重复的那一份更差——它没有时间。所以这里在渲染时剥掉，不动库里数据，
    零 LLM 成本，已有课程立即生效。

    只在「标记行之后全是列表项或空行」时才截断：万一模型把大纲写在摘要中间，
    宁可留着重复，也不要误删正文。
    """
    if not summary:
        return summary
    lines = summary.splitlines()
    for i, line in enumerate(lines):
        if not _SUMMARY_OUTLINE_MARK.match(line):
            continue
        rest = [item for item in lines[i + 1 :] if item.strip()]
        if rest and all(_SUMMARY_ITEM.match(item) for item in rest):
            return "\n".join(lines[:i]).rstrip()
    return summary


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。中文约 1 token ≈ 0.7 字，这里按字符数保守估。"""
    return max(1, int(len(text) / 1.6))
