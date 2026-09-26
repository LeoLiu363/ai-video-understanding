"""问答服务。

两条通路（不是只做 top-k）：
- **局部事实型** → 整稿长上下文直答。2 小时中文课全稿约 3 万字，塞得进 1M 窗口，
  证据完整、无需召回。
- **全局聚合型**（「这门课一共讲了几种 X」「老师对 Y 的态度有变化吗」）→ 用全局
  提示词，并在材料超过预算时改用「大纲前缀」；这类问题靠 top-k 结构上就答不全。
- **视觉型**（「老师说的那张图第三步是什么」）→ 附上当时的课件图交多模态模型。

变化的量（播放进度）只放在问题一侧，绝不进前缀，否则缓存全废。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .context import build_full_prefix, build_outline_prefix, estimate_tokens
from .embedding import EmbedderLike, Reranker
from . import ledger
from .llm import prompts
from .llm.client import LLMClient, Usage
from .retrieve import Retriever
from .schema import Chapter, Chunk, Video, ms_to_hms
from .store import Store

log = logging.getLogger(__name__)

_TS_PATTERN = re.compile(r"\[(?:时间\s*)?(\d{1,2}:\d{2}(?::\d{2})?)\]")
_GLOBAL_HINTS = ("一共", "几种", "哪些", "总结", "概括", "整体", "全部", "整个课程", "贯穿", "对比", "比较", "有变化")
_VISUAL_HINTS = ("这张图", "画面", "板书", "ppt", "幻灯", "图里", "图表", "写的什么", "屏幕")

# 回答里最多带多少条可跳转引用。给得比 8 宽一些：跨段问题会在多个位置取证，
# 卡得太紧就会把「结论落定」的那一段（通常时间靠后）截掉。
_MAX_CITATIONS = 12

# 前缀超过这个 token 预算才退化到大纲前缀
PREFIX_TOKEN_BUDGET = 400_000
# 多轮历史挂在「问题」侧（前缀之后），保住上下文缓存。
# 只取最近若干条，避免历史把窗口吃光；偶数 = 完整的「问+答」对。
HISTORY_MESSAGE_LIMIT = 12
# 单条历史消息写入提示词时的正文上限（字符）。
HISTORY_MSG_CHARS = 800


@dataclass
class Citation:
    start_ms: int
    end_ms: int
    text: str
    slide_idxs: list[int] = field(default_factory=list)
    chapter_title: str = ""

    @property
    def label(self) -> str:
        return ms_to_hms(self.start_ms)

    def to_dict(self) -> dict:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "label": self.label,
            "text": self.text[:400],
            "slide_idxs": self.slide_idxs,
            "chapter_title": self.chapter_title,
        }


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    intent: str
    usage: Usage
    prefix_tokens: int
    used_outline: bool
    images: list[str] = field(default_factory=list)
    session_id: str = ""
    history_turns: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "intent": self.intent,
            "citations": [c.to_dict() for c in self.citations],
            "images": self.images,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "cached_tokens": self.usage.cached_tokens,
                "cache_hit_rate": round(self.usage.cache_hit_rate, 3),
            },
            "prefix_tokens": self.prefix_tokens,
            "used_outline": self.used_outline,
            "session_id": self.session_id,
            "history_turns": self.history_turns,
        }


class AskService:
    def __init__(
        self,
        cfg: Config,
        store: Store,
        client: LLMClient,
        vision: LLMClient | None = None,
        *,
        embedder: EmbedderLike | None = None,
        reranker: Reranker | None = None,
    ):
        self.cfg = cfg
        self.store = store
        self.client = client
        self.vision = vision
        # 关键：必须把嵌入/重排模型交给检索器，否则向量召回与重排永远不会生效
        # （入库阶段算好的向量会被白白闲置）。两者为 None 时自动退化为关键词检索。
        self.retriever = Retriever(
            store, embedder=embedder, reranker=reranker, cfg=cfg.retrieve
        )
        # 前缀缓存：入库完成后内容不变，可安全复用，这是命中上下文缓存的前提
        self._prefix_cache: dict[str, tuple[str, int, bool]] = {}

    # ------------------------------------------------------------------ 入口

    def ask(
        self,
        video_id: str,
        question: str,
        *,
        current_ms: int | None = None,
        within_chapter: Chapter | None = None,
        top_k: int | None = None,
        session_id: str | None = None,
        persist: bool = True,
    ) -> Answer:
        video = self.store.get_video(video_id)
        if video is None:
            raise ValueError(f"未找到课程 {video_id}")

        chapters = self.store.get_chapters(video_id)
        chunks = self.store.get_chunks(video_id)
        if not chunks:
            raise ValueError("该课程尚未完成入库（没有转写内容）")

        # 会话校验：必须属于本课。空 session_id = 匿名单轮（CLI / 评估用）。
        session = None
        history: list[dict] = []
        if session_id:
            session = self.store.get_session(session_id)
            if session is None:
                raise ValueError(f"会话不存在：{session_id}")
            if session["video_id"] != video_id:
                raise ValueError("会话不属于该课程")
            # 取最近 N 条作为多轮上下文；本轮问题还没入库，所以全是「上一轮及更早」
            history = self.store.get_messages(session_id, limit=HISTORY_MESSAGE_LIMIT)

        intent = self._classify(question)

        prefix, prefix_tokens, used_outline = self._get_prefix(video_id, video, chapters, chunks)
        prompt = prompts.QA_GLOBAL if intent == "global" else prompts.QA_LOCAL
        full_prompt = f"{prompt}\n\n---\n{prefix}"

        # 检索仍只看本问——历史里的旧话题不应劫持本轮召回
        hits = self.retriever.search(
            video_id, question, top_k=top_k, time_bias_ms=current_ms, within_chapter=within_chapter
        )

        images: list[str] = []
        reply = None
        # 标注「用途 + 课程」供记账归类。记账回调挂在客户端出口上，读的就是
        # 这里设置的 contextvar（见 ledger.recorder）。
        with ledger.usage_scope("ask", video_id):
            if intent == "visual" and self.vision is not None:
                image_paths = self._evidence_images(video_id, hits)
                if image_paths:
                    question_text = self._build_question(
                        question, current_ms, within_chapter, None, history=history
                    )
                    reply = self.vision.ask_with_images(full_prompt, question_text, image_paths)
                    images = [str(p) for p in image_paths]

            if reply is None:
                question_text = self._build_question(
                    question, current_ms, within_chapter, hits, history=history
                )
                reply = self.client.ask(full_prompt, question_text)

        citations = self._build_citations(video_id, chapters, chunks, hits, reply.text)
        answer = Answer(
            text=reply.text,
            citations=citations,
            intent=intent,
            usage=reply.usage,
            prefix_tokens=prefix_tokens,
            used_outline=used_outline,
            images=images,
            session_id=session_id or "",
            history_turns=sum(1 for m in history if m["role"] == "user"),
        )

        # 落库：先用户后助手。评估 / CLI 默认也可以带 session；persist=False 留给干跑。
        if session is not None and persist:
            self.store.add_message(session_id, "user", question)
            self.store.add_message(
                session_id,
                "assistant",
                reply.text,
                meta={
                    "intent": intent,
                    "citations": [c.to_dict() for c in citations],
                    "usage": {
                        "prompt_tokens": reply.usage.prompt_tokens,
                        "completion_tokens": reply.usage.completion_tokens,
                        "cached_tokens": reply.usage.cached_tokens,
                        "cache_hit_rate": round(reply.usage.cache_hit_rate, 3),
                    },
                    "prefix_tokens": prefix_tokens,
                    "images": images,
                },
            )
            # 首问自动起名：只在标题仍空时写一次，之后用户改名不被覆盖
            if not (session.get("title") or "").strip():
                self.store.update_session(session_id, title=_auto_title(question))

        return answer

    def ask_stream(
        self,
        video_id: str,
        question: str,
        *,
        current_ms: int | None = None,
        within_chapter: Chapter | None = None,
        top_k: int | None = None,
        session_id: str | None = None,
        persist: bool = True,
    ):
        """流式问答。产量：

        - ``("meta", dict)``：intent / session 等元信息（先发）
        - ``("delta", str)``：正文增量
        - ``("done", Answer)``：完整 Answer（含 citations / usage），最后一项

        视觉题仍走非流式（图文接口流式不稳定），一次性产出全文再 ``done``。
        消息前缀结构与 ``ask`` 完全一致。
        """
        video = self.store.get_video(video_id)
        if video is None:
            raise ValueError(f"未找到课程 {video_id}")

        chapters = self.store.get_chapters(video_id)
        chunks = self.store.get_chunks(video_id)
        if not chunks:
            raise ValueError("该课程尚未完成入库（没有转写内容）")

        session = None
        history: list[dict] = []
        if session_id:
            session = self.store.get_session(session_id)
            if session is None:
                raise ValueError(f"会话不存在：{session_id}")
            if session["video_id"] != video_id:
                raise ValueError("会话不属于该课程")
            history = self.store.get_messages(session_id, limit=HISTORY_MESSAGE_LIMIT)

        intent = self._classify(question)
        prefix, prefix_tokens, used_outline = self._get_prefix(video_id, video, chapters, chunks)
        prompt = prompts.QA_GLOBAL if intent == "global" else prompts.QA_LOCAL
        full_prompt = f"{prompt}\n\n---\n{prefix}"
        hits = self.retriever.search(
            video_id, question, top_k=top_k, time_bias_ms=current_ms, within_chapter=within_chapter
        )

        yield (
            "meta",
            {
                "intent": intent,
                "session_id": session_id or "",
                "prefix_tokens": prefix_tokens,
                "used_outline": used_outline,
            },
        )

        images: list[str] = []
        reply = None

        with ledger.usage_scope("ask", video_id):
            if intent == "visual" and self.vision is not None:
                image_paths = self._evidence_images(video_id, hits)
                if image_paths:
                    question_text = self._build_question(
                        question, current_ms, within_chapter, None, history=history
                    )
                    reply = self.vision.ask_with_images(full_prompt, question_text, image_paths)
                    images = [str(p) for p in image_paths]
                    if reply.text:
                        yield ("delta", reply.text)

            if reply is None:
                question_text = self._build_question(
                    question, current_ms, within_chapter, hits, history=history
                )
                for kind, payload in self.client.ask_stream(full_prompt, question_text):
                    if kind == "delta":
                        yield ("delta", payload)
                    elif kind == "done":
                        reply = payload

        if reply is None:
            raise RuntimeError("流式问答未收到完成事件")

        citations = self._build_citations(video_id, chapters, chunks, hits, reply.text)
        answer = Answer(
            text=reply.text,
            citations=citations,
            intent=intent,
            usage=reply.usage,
            prefix_tokens=prefix_tokens,
            used_outline=used_outline,
            images=images,
            session_id=session_id or "",
            history_turns=sum(1 for m in history if m["role"] == "user"),
        )

        if session is not None and persist:
            self.store.add_message(session_id, "user", question)
            self.store.add_message(
                session_id,
                "assistant",
                reply.text,
                meta={
                    "intent": intent,
                    "citations": [c.to_dict() for c in citations],
                    "usage": {
                        "prompt_tokens": reply.usage.prompt_tokens,
                        "completion_tokens": reply.usage.completion_tokens,
                        "cached_tokens": reply.usage.cached_tokens,
                        "cache_hit_rate": round(reply.usage.cache_hit_rate, 3),
                    },
                    "prefix_tokens": prefix_tokens,
                    "images": images,
                },
            )
            if not (session.get("title") or "").strip():
                self.store.update_session(session_id, title=_auto_title(question))

        yield ("done", answer)

    def warm(self, video_id: str) -> int:
        """预生成前缀，让第一次提问就命中缓存。"""
        video = self.store.get_video(video_id)
        if video is None:
            return 0
        chapters = self.store.get_chapters(video_id)
        chunks = self.store.get_chunks(video_id)
        _, tokens, _ = self._get_prefix(video_id, video, chapters, chunks)
        return tokens

    # --------------------------------------------------------------- 内部实现

    def _get_prefix(
        self, video_id: str, video: Video, chapters: list[Chapter], chunks: list[Chunk]
    ) -> tuple[str, int, bool]:
        cached = self._prefix_cache.get(video_id)
        if cached:
            return cached

        summary = self.store.get_video_summary(video_id)
        full = build_full_prefix(video, chapters, chunks, video_summary=summary)
        tokens = estimate_tokens(full)

        if tokens <= PREFIX_TOKEN_BUDGET:
            result = (full, tokens, False)
        else:
            # 只有超长课程才退化到大纲前缀
            outline = build_outline_prefix(video, chapters, chunks, video_summary=summary)
            result = (outline, estimate_tokens(outline), True)

        self._prefix_cache[video_id] = result
        return result

    @staticmethod
    def _classify(question: str) -> str:
        """先做便宜的关键词判断。注意：即便判成 local，整稿也在上下文里，
        所以误判的代价很小——全局提示词只是让模型更倾向枚举完整。"""
        q = question.lower()
        if any(h in q for h in _VISUAL_HINTS):
            return "visual"
        if any(h in q for h in _GLOBAL_HINTS):
            return "global"
        return "local"

    def _build_question(
        self,
        question: str,
        current_ms: int | None,
        within_chapter: Chapter | None,
        hits: list | None,
        *,
        history: list[dict] | None = None,
    ) -> str:
        """问题侧才放变化内容。

        多轮历史必须挂在这里（前缀之后），绝不能塞进稳定前缀——否则每次对话
        都会打穿 DeepSeek 的上下文缓存，成本立刻升一个数量级。
        """
        parts: list[str] = []
        if history:
            parts.append("【此前对话】（仅供理解指代与追问，课程事实仍以材料为准）")
            for msg in history:
                role = "用户" if msg["role"] == "user" else "助手"
                body = (msg.get("content") or "").strip()
                if len(body) > HISTORY_MSG_CHARS:
                    body = body[: HISTORY_MSG_CHARS - 1] + "…"
                parts.append(f"{role}：{body}")
            parts.append("")  # 空行分隔本问
        parts.append(f"用户问题：{question}")
        if current_ms is not None:
            parts.append(f"（用户当前播放到 {ms_to_hms(current_ms)}，仅作参考，不代表问题范围）")
        if within_chapter is not None:
            parts.append(f"（用户限定只在本章回答：{within_chapter.title or within_chapter.chapter_id}）")
        if hits:
            spans = "、".join(
                f"{ms_to_hms(h.chunk.start_ms)}-{ms_to_hms(h.chunk.end_ms)}" for h in hits[:5]
            )
            parts.append(f"（以下是检索到的可能相关位置，供你优先核对：{spans}）")
        return "\n".join(parts)

    def _evidence_images(self, video_id: str, hits: list) -> list[Path]:
        wanted: list[int] = []
        for hit in hits:
            for idx in hit.chunk.slide_idxs:
                if idx not in wanted:
                    wanted.append(idx)
            if len(wanted) >= self.cfg.vision.max_images:
                break
        slides = {s.idx: s for s in self.store.get_slides(video_id)}
        out: list[Path] = []
        for idx in wanted[: self.cfg.vision.max_images]:
            slide = slides.get(idx)
            if slide and Path(slide.image_path).exists():
                out.append(Path(slide.image_path))
        return out

    def _build_citations(
        self,
        video_id: str,
        chapters: list[Chapter],
        chunks: list[Chunk],
        hits: list,
        answer_text: str,
    ) -> list[Citation]:
        """引用来源有两个：模型自己在正文里标的时间戳，以及检索命中的片段。

        前者让用户能核对「它说的这句话在哪」，后者保证即使模型忘了标时间，
        也能给出可跳转的位置。
        """
        chapter_of: dict[str, Chapter] = {}
        for ch in chapters:
            for cid in ch.chunk_ids:
                chapter_of[cid] = ch

        # 正文里出现的时间戳 → 找到最接近的块
        cited_ms: list[int] = []
        for match in _TS_PATTERN.finditer(answer_text or ""):
            cited_ms.append(_parse_ts(match.group(1)))

        picked: list[Chunk] = []
        seen: set[str] = set()

        for ms in cited_ms:
            chunk = _nearest_chunk(chunks, ms)
            if chunk and chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                picked.append(chunk)

        for hit in hits:
            if hit.chunk.chunk_id not in seen and len(picked) < _MAX_CITATIONS:
                seen.add(hit.chunk.chunk_id)
                picked.append(hit.chunk)

        # 先截断、再按时间排序。list 的顺序就是优先级：前面是模型自己在正文里
        # 标的时间戳（它标在哪就说明在哪几处找过答案），后面才是检索命中的兜底。
        # 早期实现先排序后截断，等于按时间「留早不留晚」——模型把结论落在课程
        # 末尾时（c01 的笔记在 [30:48] 才写全 4 种方法），这个位置反而被挤掉，
        # 引用就被判成指错位置了。
        picked = picked[:_MAX_CITATIONS]
        picked.sort(key=lambda c: c.start_ms)
        return [
            Citation(
                start_ms=c.start_ms,
                end_ms=c.end_ms,
                text=c.text,
                slide_idxs=c.slide_idxs,
                chapter_title=(chapter_of.get(c.chunk_id).title if chapter_of.get(c.chunk_id) else ""),
            )
            for c in picked
        ]


def _parse_ts(text: str) -> int:
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3], parts[-2], parts[-1]
    return ((h * 60 + m) * 60 + s) * 1000


def _nearest_chunk(chunks: list[Chunk], ms: int) -> Chunk | None:
    if not chunks:
        return None
    best = min(chunks, key=lambda c: 0 if c.start_ms <= ms <= c.end_ms else abs(c.start_ms - ms))
    return best


def _auto_title(question: str, limit: int = 28) -> str:
    """用首问给会话起个可读的名字。"""
    text = (question or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text or "新对话"
    return text[: limit - 1] + "…"
