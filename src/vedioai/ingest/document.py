"""文档课程：把 Markdown / 纯文本切成 Chapter + Chunk。

不走 ASR / 播放代理。虚拟时间轴按字符估算，兼容现有引用字段。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..schema import Chapter, Chunk

DOC_EXTENSIONS = {".md", ".txt", ".markdown"}

# 约 20 字/秒的阅读节奏，用来填 start_ms/end_ms（仅作锚点，不是真实时长）
_MS_PER_CHAR = 50
_CHUNK_TARGET_CHARS = 1200
_CHUNK_MAX_CHARS = 2400

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class ParsedDocument:
    title: str
    text: str
    chapters: list[Chapter]
    chunks: list[Chunk]
    duration_ms: int


def is_document_path(path: Path | str) -> bool:
    return Path(path).suffix.lower() in DOC_EXTENSIONS


def list_document_files(paths: list[Path] | None = None, folder: Path | None = None) -> list[Path]:
    """收集待入库文档路径（去重、只保留存在的文件）。"""
    found: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        p = p.expanduser().resolve()
        if not p.is_file() or not is_document_path(p):
            return
        key = str(p).casefold()
        if key in seen:
            return
        seen.add(key)
        found.append(p)

    for raw in paths or []:
        add(Path(raw))
    if folder:
        root = Path(folder).expanduser().resolve()
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file() and is_document_path(p):
                    add(p)
    return found


def load_document_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_document(
    text: str,
    video_id: str,
    *,
    title: str = "",
) -> ParsedDocument:
    """按 Markdown 标题分章；章内按段落聚合成块。无标题则整篇一章。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("文档为空")

    display_title = (title or "").strip() or _guess_title(text) or "未命名文档"
    sections = _split_sections(text, fallback_title=display_title)

    chapters: list[Chapter] = []
    chunks: list[Chunk] = []
    cursor_ms = 0
    chunk_idx = 0

    for ch_i, (sec_title, sec_body) in enumerate(sections):
        body = sec_body.strip()
        if not body and not sec_title:
            continue
        parts = _split_chunk_texts(body or sec_title)
        if not parts:
            parts = [sec_title or "（空）"]

        ch_start = cursor_ms
        chunk_ids: list[str] = []
        for part in parts:
            span = max(1_000, len(part) * _MS_PER_CHAR)
            start_ms = cursor_ms
            end_ms = cursor_ms + span
            cid = f"{video_id}-c{chunk_idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    idx=chunk_idx,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=part,
                    title="",
                    summary="",
                )
            )
            chunk_ids.append(cid)
            chunk_idx += 1
            cursor_ms = end_ms

        chapters.append(
            Chapter(
                chapter_id=f"{video_id}-ch{ch_i:03d}",
                idx=ch_i,
                start_ms=ch_start,
                end_ms=max(ch_start + 1, cursor_ms),
                title=sec_title[:80],
                summary="",
                chunk_ids=chunk_ids,
            )
        )

    if not chunks:
        raise ValueError("未能从文档中切出任何段落")

    # parent 关系
    by_chunk = {cid: ch.chapter_id for ch in chapters for cid in ch.chunk_ids}
    for c in chunks:
        c.parent_id = by_chunk.get(c.chunk_id)

    return ParsedDocument(
        title=display_title,
        text=text,
        chapters=chapters,
        chunks=chunks,
        duration_ms=cursor_ms,
    )


def _guess_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _HEADING_RE.match(line)
        if m:
            return m.group(2).strip()
        return line.lstrip("#").strip()[:80]
    return ""


def _split_sections(text: str, *, fallback_title: str) -> list[tuple[str, str]]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(fallback_title, text)]

    sections: list[tuple[str, str]] = []
    # 标题前的序言
    preface = text[: matches[0].start()].strip()
    if preface:
        sections.append(("导言", preface))

    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((title or f"第 {i + 1} 节", body))
    return sections


def _split_chunk_texts(body: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paras:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for p in paras:
        if size and size + len(p) > _CHUNK_MAX_CHARS:
            chunks.append("\n\n".join(buf))
            buf, size = [p], len(p)
            continue
        buf.append(p)
        size += len(p)
        if size >= _CHUNK_TARGET_CHARS:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks
