"""学习文档生成。

关键点：**不要一次让模型写完全片**。走分层 map-reduce——
章节逐字稿 → 章节笔记 → 汇总成学习文档。这样 2 小时的课也能稳定产出，
而不是被截断或开始胡编。

每一层都带时间锚点，导出的 Markdown 里可以直接跳回视频位置。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .context import estimate_tokens
from .glossary import Glossary, render_findings
from .llm import prompts
from .llm.client import LLMClient, LLMError, Usage
from .schema import Chapter, Chunk, Video, ms_to_hms
from .store import Store

log = logging.getLogger(__name__)

# 单章送给模型的逐字稿上限，超长时截断并提示
CHAPTER_CHAR_BUDGET = 60_000


@dataclass
class NotesResult:
    markdown: str
    path: Path | None = None
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict:
        return {
            "markdown": self.markdown,
            "path": str(self.path) if self.path else None,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "cached_tokens": self.usage.cached_tokens,
            },
        }


class NotesService:
    def __init__(self, cfg, store: Store, client: LLMClient):
        self.cfg = cfg
        self.store = store
        self.client = client
        # 术语表用于在导出文档末尾标出可疑术语（只标记，不擅自替换）
        self.glossary = Glossary.load(getattr(cfg, "glossary_path", None))

    def generate(
        self,
        video_id: str,
        *,
        save: bool = True,
        progress=None,
    ) -> NotesResult:
        video = self.store.get_video(video_id)
        if video is None:
            raise ValueError(f"未找到课程 {video_id}")

        chapters = self.store.get_chapters(video_id)
        chunks = self.store.get_chunks(video_id)
        if not chunks:
            raise ValueError("该课程尚未完成入库")

        usage = Usage()
        by_chapter: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            key = chunk.parent_id or "_orphan"
            by_chapter.setdefault(key, []).append(chunk)

        # ---- 开头：全课概览
        if progress:
            progress(0, len(chapters) + 1, "生成课程概览")
        head = self._header(video, chapters, chunks, usage)

        # ---- 每章一节
        sections: list[str] = []
        for i, chapter in enumerate(chapters, start=1):
            if progress:
                progress(i, len(chapters) + 1, f"生成第 {i}/{len(chapters)} 章笔记")
            body_chunks = by_chapter.get(chapter.chapter_id, [])
            if not body_chunks:
                continue
            sections.append(self._chapter_section(chapter, body_chunks, usage))

        # ---- 术语表
        concepts = self._concepts_table(chunks)

        markdown = "\n\n".join(filter(None, [head, *sections, concepts]))

        # ---- 待人工确认的术语
        # 放在最后而不是插在正文里：不打断阅读，但保证「可能听错的词」不会被
        # 当成事实悄悄留在文档里。只标记，不擅自替换。
        findings = self.glossary.flag(markdown)
        if findings:
            gpath = getattr(self.cfg, "glossary_path", None)
            markdown += "\n\n" + render_findings(
                findings, Path(gpath) if gpath else None
            )
            log.info("文档末尾标记待确认术语 %d 处", len(findings))

        path = None
        if save:
            out_dir = self.cfg.library_dir / video_id
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "notes.md"
            path.write_text(markdown, encoding="utf-8")
            log.info("学习文档已写入 %s", path)

        return NotesResult(markdown=markdown, path=path, usage=usage)

    # --------------------------------------------------------------- 各段生成

    def _header(
        self,
        video: Video,
        chapters: list[Chapter],
        chunks: list[Chunk],
        usage: Usage,
    ) -> str:
        summary = self.store.get_video_summary(video.video_id)
        lines = [
            f"# {video.title or '课程学习笔记'}",
            "",
            f"> 总时长 {ms_to_hms(video.duration_ms)}　·　"
            f"{len(chapters)} 章　·　{len(chunks)} 个片段　·　"
            f"导出时间 {datetime.now():%Y-%m-%d %H:%M}",
            "",
        ]
        if summary:
            lines += [summary.strip(), ""]

        if chapters:
            lines += ["## 课程大纲", "", "| 章节 | 时间 | 要点 |", "|---|---|---|"]
            for chapter in chapters:
                title = chapter.title or f"第 {chapter.idx + 1} 章"
                first_line = (chapter.summary or "").strip().splitlines()
                brief = first_line[0] if first_line else ""
                lines.append(f"| {title} | {ms_to_hms(chapter.start_ms)} | {brief} |")
            lines.append("")

        if summary:
            # 有全课摘要就不必再花一次调用；否则让模型补一段「你将学到」
            return "\n".join(lines)

        body = self._chapter_digest(chapters, chunks)
        messages = [
            {"role": "system", "content": prompts.SYSTEM_TUTOR},
            {"role": "user", "content": f"{prompts.NOTES_VIDEO_INTRO}\n\n---\n{body}"},
        ]
        try:
            reply = self.client.chat(messages, max_tokens=2000)
            usage.merge(reply.usage)
            lines += [reply.text, ""]
        except LLMError as exc:
            log.warning("生成课程开头失败：%s", exc)
        return "\n".join(lines)

    def _chapter_section(
        self,
        chapter: Chapter,
        chunks: list[Chunk],
        usage: Usage,
    ) -> str:
        title = chapter.title or f"第 {chapter.idx + 1} 章"
        span = f"{ms_to_hms(chapter.start_ms)} - {ms_to_hms(chapter.end_ms)}"

        transcript = self._chapter_transcript(chunks)
        messages = [
            {"role": "system", "content": prompts.SYSTEM_TUTOR},
            {"role": "user", "content": f"{prompts.NOTES_CHAPTER}\n\n---\n{transcript}"},
        ]
        try:
            reply = self.client.chat(messages, max_tokens=6000)
            usage.merge(reply.usage)
            note = reply.text
        except LLMError as exc:
            log.warning("生成章节笔记失败 %s：%s", chapter.chapter_id, exc)
            # 失败时至少保留原文，不要把这一章丢掉
            note = chapter.summary or transcript[:3000]

        note = _strip_duplicate_heading(note, title)
        return f"\n---\n\n## {chapter.idx + 1}. {title}\n\n`{span}`\n\n{note.strip()}"

    # --------------------------------------------------------------- 组装辅助

    @staticmethod
    def _chapter_transcript(chunks: list[Chunk]) -> str:
        parts: list[str] = []
        total = 0
        for chunk in sorted(chunks, key=lambda c: c.start_ms):
            head = f"\n【{ms_to_hms(chunk.start_ms)}】"
            if chunk.ocr_text:
                head += f"\n（课件）{chunk.ocr_text}"
            body = f"{head}\n{chunk.text}"
            total += len(body)
            if total > CHAPTER_CHAR_BUDGET:
                parts.append("\n（本章内容过长，已截断）")
                break
            parts.append(body)
        return "\n".join(parts)

    @staticmethod
    def _chapter_digest(chapters: list[Chapter], chunks: list[Chunk]) -> str:
        by_id = {c.chunk_id: c for c in chunks}
        lines: list[str] = []
        for chapter in chapters:
            title = chapter.title or f"第 {chapter.idx + 1} 章"
            lines.append(f"\n## {title}　[{ms_to_hms(chapter.start_ms)}]")
            lines.append(chapter.summary or "（无摘要）")
            for cid in chapter.chunk_ids[:20]:
                chunk = by_id.get(cid)
                if chunk and chunk.summary:
                    lines.append(f"- [{ms_to_hms(chunk.start_ms)}] {chunk.title}：{chunk.summary}")
        return "\n".join(lines)

    def _concepts_table(self, chunks: list[Chunk]) -> str:
        """术语表从「块标题」里回收，避免再花一次模型调用。"""
        terms: dict[str, str] = {}
        for chunk in chunks:
            for line in (chunk.summary or "").splitlines():
                line = line.strip()
                if not line or "：" not in line:
                    continue
                term, _, definition = line.partition("：")
                term = term.strip(" -•*")
                if 2 <= len(term) <= 20 and term not in terms and definition.strip():
                    terms[term] = definition.strip()
        if not terms:
            return ""
        rows = ["## 术语表", "", "| 术语 | 说明 |", "|---|---|"]
        for term, definition in list(terms.items())[:60]:
            rows.append(f"| {term} | {definition[:200]} |")
        return "\n".join(rows)

    def estimate_cost_tokens(self, video_id: str) -> int:
        chapters = self.store.get_chapters(video_id)
        chunks = self.store.get_chunks(video_id)
        return sum(
            estimate_tokens(self._chapter_transcript(by_chapter))
            for by_chapter in _group_by_chapter(chapters, chunks).values()
        )


def _group_by_chapter(chapters: list[Chapter], chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    out: dict[str, list[Chunk]] = {ch.chapter_id: [] for ch in chapters}
    for chunk in chunks:
        if chunk.parent_id in out:
            out[chunk.parent_id].append(chunk)
    return out


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")


def _strip_duplicate_heading(text: str, title: str) -> str:
    """模型常自己加一级标题，会和我们在外层加的重名，去掉一次。"""
    lines = text.strip().splitlines()
    if lines and _HEADING.match(lines[0]) and title in lines[0]:
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines)
