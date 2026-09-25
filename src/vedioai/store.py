"""SQLite 存储层。

设计要点：
- 转写、课件 OCR、章节、摘要全部落库，一份 SQLite 文件即可备份。
- 关键词检索用 FTS5 + jieba 分词。**不要**沿用旧项目那种用 contains 近似词频的
  「BM25」——中文无空格场景下那等于坏掉的 BM25。
- 向量检索：块数只有 10²–10³ 量级，优先用 sqlite-vec；扩展不可用时退化为
  numpy 暴力检索（这个规模下 <10ms，完全够用）。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from collections.abc import Iterable
from pathlib import Path

import jieba
import numpy as np

from .schema import Chapter, Chunk, Segment, Slide, Video, VideoStatus

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    duration_ms  INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'pending',
    size_bytes   INTEGER NOT NULL DEFAULT 0,
    proxy_path   TEXT,
    error        TEXT,
    video_summary TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS segments (
    video_id  TEXT NOT NULL,
    idx       INTEGER NOT NULL,
    start_ms  INTEGER NOT NULL,
    end_ms    INTEGER NOT NULL,
    text      TEXT NOT NULL,
    speaker   TEXT,
    words     TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (video_id, idx)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT NOT NULL,
    video_id   TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    text       TEXT NOT NULL,
    ocr_text   TEXT NOT NULL DEFAULT '',
    title      TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL DEFAULT '',
    parent_id  TEXT,
    slide_idxs TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (video_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS chapters (
    chapter_id TEXT NOT NULL,
    video_id   TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    title      TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL DEFAULT '',
    chunk_ids  TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (video_id, chapter_id)
);

CREATE TABLE IF NOT EXISTS slides (
    video_id   TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    ocr_text   TEXT NOT NULL DEFAULT '',
    phash      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (video_id, idx)
);

-- 关键词检索：body 存 jieba 分词后的空格串
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    video_id UNINDEXED,
    body,
    tokenize = 'unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    video_id UNINDEXED,
    idx UNINDEXED,
    body,
    tokenize = 'unicode61'
);

-- 向量：不依赖扩展，BLOB 存 float32，检索时 numpy 暴力算
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_video  ON chunks(video_id, idx);
CREATE INDEX IF NOT EXISTS idx_chapter_video ON chapters(video_id, idx);
CREATE INDEX IF NOT EXISTS idx_slide_video   ON slides(video_id, idx);
CREATE INDEX IF NOT EXISTS idx_seg_video     ON segments(video_id, idx);
"""


def tokenize_zh(text: str) -> str:
    """jieba 分词后用空格连接，喂给 FTS5 的 unicode61 分词器。

    FTS5 内置分词器不切中文，必须先自己切好。
    """
    if not text:
        return ""
    tokens = [t.strip() for t in jieba.cut_for_search(text) if t.strip()]
    return " ".join(tokens)


def query_terms(text: str) -> list[str]:
    """把用户问题切成检索词，去重并丢弃单字虚词噪声。"""
    seen: list[str] = []
    for token in jieba.cut_for_search(text or ""):
        token = token.strip()
        if len(token) < 2:
            continue
        if token not in seen:
            seen.append(token)
    return seen


def _pack(vec: np.ndarray) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec.astype(np.float32).tolist())


def _unpack(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"<{n}f", blob), dtype=np.float32)


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ 视频

    def upsert_video(self, video: Video) -> None:
        self.conn.execute(
            """
            INSERT INTO videos (video_id, path, title, duration_ms, status,
                                size_bytes, proxy_path, error)
            VALUES (:video_id, :path, :title, :duration_ms, :status,
                    :size_bytes, :proxy_path, :error)
            ON CONFLICT(video_id) DO UPDATE SET
                path=excluded.path,
                title=excluded.title,
                duration_ms=excluded.duration_ms,
                status=excluded.status,
                size_bytes=excluded.size_bytes,
                proxy_path=excluded.proxy_path,
                error=excluded.error,
                updated_at=datetime('now')
            """,
            video.to_row(),
        )
        self.conn.commit()

    def set_status(self, video_id: str, status: VideoStatus, error: str | None = None) -> None:
        self.conn.execute(
            "UPDATE videos SET status=?, error=?, updated_at=datetime('now') WHERE video_id=?",
            (status.value, error, video_id),
        )
        self.conn.commit()

    def set_video_summary(self, video_id: str, summary: str) -> None:
        self.conn.execute(
            "UPDATE videos SET video_summary=?, updated_at=datetime('now') WHERE video_id=?",
            (summary, video_id),
        )
        self.conn.commit()

    def get_video(self, video_id: str) -> Video | None:
        row = self.conn.execute("SELECT * FROM videos WHERE video_id=?", (video_id,)).fetchone()
        if not row:
            return None
        return Video(
            video_id=row["video_id"],
            path=row["path"],
            title=row["title"],
            duration_ms=row["duration_ms"],
            status=VideoStatus(row["status"]),
            size_bytes=row["size_bytes"],
            proxy_path=row["proxy_path"],
            error=row["error"],
        )

    def get_video_summary(self, video_id: str) -> str:
        row = self.conn.execute(
            "SELECT video_summary FROM videos WHERE video_id=?", (video_id,)
        ).fetchone()
        return row["video_summary"] if row else ""

    def list_videos(self) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT v.*,
                   (SELECT COUNT(*) FROM chapters c WHERE c.video_id = v.video_id) AS chapter_count,
                   (SELECT COUNT(*) FROM chunks k WHERE k.video_id = v.video_id)   AS chunk_count,
                   (SELECT COUNT(*) FROM slides s WHERE s.video_id = v.video_id)   AS slide_count
            FROM videos v
            ORDER BY v.updated_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def find_by_path(self, path: str) -> Video | None:
        row = self.conn.execute("SELECT video_id FROM videos WHERE path=?", (path,)).fetchone()
        return self.get_video(row["video_id"]) if row else None

    # ------------------------------------------------------------------ 分段

    def replace_segments(self, video_id: str, segments: list[Segment]) -> None:
        self.conn.execute("DELETE FROM segments WHERE video_id=?", (video_id,))
        self.conn.execute("DELETE FROM segments_fts WHERE video_id=?", (video_id,))
        for seg in segments:
            self.conn.execute(
                "INSERT INTO segments(video_id, idx, start_ms, end_ms, text, speaker, words) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    video_id,
                    seg.idx,
                    seg.start_ms,
                    seg.end_ms,
                    seg.text,
                    seg.speaker,
                    json.dumps(seg.words, ensure_ascii=False),
                ),
            )
            self.conn.execute(
                "INSERT INTO segments_fts(video_id, idx, body) VALUES(?,?,?)",
                (video_id, seg.idx, tokenize_zh(seg.text)),
            )
        self.conn.commit()

    def get_segments(self, video_id: str) -> list[Segment]:
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE video_id=? ORDER BY idx", (video_id,)
        ).fetchall()
        return [
            Segment(
                idx=r["idx"],
                start_ms=r["start_ms"],
                end_ms=r["end_ms"],
                text=r["text"],
                speaker=r["speaker"],
                words=json.loads(r["words"] or "[]"),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ 语义块

    def replace_chunks(self, video_id: str, chunks: list[Chunk]) -> None:
        self.conn.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
        self.conn.execute("DELETE FROM chunks_fts WHERE video_id=?", (video_id,))
        for c in chunks:
            self.conn.execute(
                """
                INSERT INTO chunks(chunk_id, video_id, idx, start_ms, end_ms, text,
                                   ocr_text, title, summary, parent_id, slide_idxs)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    c.chunk_id,
                    video_id,
                    c.idx,
                    c.start_ms,
                    c.end_ms,
                    c.text,
                    c.ocr_text,
                    c.title,
                    c.summary,
                    c.parent_id,
                    json.dumps(c.slide_idxs),
                ),
            )
            self.conn.execute(
                "INSERT INTO chunks_fts(chunk_id, video_id, body) VALUES(?,?,?)",
                (c.chunk_id, video_id, tokenize_zh(c.combined_text)),
            )
        self.conn.commit()

    def update_chunk_summary(self, video_id: str, chunk_id: str, title: str, summary: str) -> None:
        self.conn.execute(
            "UPDATE chunks SET title=?, summary=? WHERE video_id=? AND chunk_id=?",
            (title, summary, video_id, chunk_id),
        )
        self.conn.commit()

    def update_chunk_texts(self, video_id: str, chunk_id: str, text: str, ocr_text: str) -> None:
        """纠错后重建该块的检索索引（局部重嵌入）。"""
        self.conn.execute(
            "UPDATE chunks SET text=?, ocr_text=? WHERE video_id=? AND chunk_id=?",
            (text, ocr_text, video_id, chunk_id),
        )
        self.conn.execute(
            "UPDATE chunks_fts SET body=? WHERE video_id=? AND chunk_id=?",
            (tokenize_zh(f"{text}\n【课件】{ocr_text}" if ocr_text else text), video_id, chunk_id),
        )
        self.conn.commit()

    def get_chunks(self, video_id: str) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE video_id=? ORDER BY idx", (video_id,)
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    @staticmethod
    def _row_to_chunk(r: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=r["chunk_id"],
            idx=r["idx"],
            start_ms=r["start_ms"],
            end_ms=r["end_ms"],
            text=r["text"],
            ocr_text=r["ocr_text"],
            title=r["title"],
            summary=r["summary"],
            parent_id=r["parent_id"],
            slide_idxs=json.loads(r["slide_idxs"] or "[]"),
        )

    # ------------------------------------------------------------------ 章节

    def replace_chapters(self, video_id: str, chapters: list[Chapter]) -> None:
        self.conn.execute("DELETE FROM chapters WHERE video_id=?", (video_id,))
        for ch in chapters:
            self.conn.execute(
                """
                INSERT INTO chapters(chapter_id, video_id, idx, start_ms, end_ms,
                                     title, summary, chunk_ids)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    ch.chapter_id,
                    video_id,
                    ch.idx,
                    ch.start_ms,
                    ch.end_ms,
                    ch.title,
                    ch.summary,
                    json.dumps(ch.chunk_ids),
                ),
            )
        self.conn.commit()

    def get_chapters(self, video_id: str) -> list[Chapter]:
        rows = self.conn.execute(
            "SELECT * FROM chapters WHERE video_id=? ORDER BY idx", (video_id,)
        ).fetchall()
        return [
            Chapter(
                chapter_id=r["chapter_id"],
                idx=r["idx"],
                start_ms=r["start_ms"],
                end_ms=r["end_ms"],
                title=r["title"],
                summary=r["summary"],
                chunk_ids=json.loads(r["chunk_ids"] or "[]"),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ 幻灯片

    def replace_slides(self, video_id: str, slides: list[Slide]) -> None:
        self.conn.execute("DELETE FROM slides WHERE video_id=?", (video_id,))
        for s in slides:
            self.conn.execute(
                "INSERT INTO slides(video_id, idx, start_ms, end_ms, image_path, ocr_text, phash) "
                "VALUES(?,?,?,?,?,?,?)",
                (video_id, s.idx, s.start_ms, s.end_ms, s.image_path, s.ocr_text, s.phash),
            )
        self.conn.commit()

    def get_slides(self, video_id: str) -> list[Slide]:
        rows = self.conn.execute(
            "SELECT * FROM slides WHERE video_id=? ORDER BY idx", (video_id,)
        ).fetchall()
        return [
            Slide(
                idx=r["idx"],
                start_ms=r["start_ms"],
                end_ms=r["end_ms"],
                image_path=r["image_path"],
                ocr_text=r["ocr_text"],
                phash=r["phash"],
            )
            for r in rows
        ]

    def slide_at(self, video_id: str, ms: int) -> list[Slide]:
        rows = self.conn.execute(
            "SELECT * FROM slides WHERE video_id=? AND start_ms<=? AND end_ms>=? ORDER BY idx",
            (video_id, ms, ms),
        ).fetchall()
        return [
            Slide(
                idx=r["idx"],
                start_ms=r["start_ms"],
                end_ms=r["end_ms"],
                image_path=r["image_path"],
                ocr_text=r["ocr_text"],
                phash=r["phash"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ 向量

    def upsert_embeddings(self, video_id: str, items: Iterable[tuple[str, np.ndarray]]) -> None:
        rows = []
        for chunk_id, vec in items:
            rows.append((chunk_id, video_id, len(vec), _pack(vec)))
        self.conn.executemany(
            "INSERT INTO embeddings(chunk_id, video_id, dim, vec) VALUES(?,?,?,?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
            rows,
        )
        self.conn.commit()

    def has_embeddings(self, video_id: str) -> bool:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM embeddings WHERE video_id=?", (video_id,)
        ).fetchone()
        return bool(row and row["n"])

    def search_vector(self, video_id: str, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """numpy 暴力余弦检索。10²–10³ 块规模下无需 ANN 索引。"""
        rows = self.conn.execute(
            "SELECT chunk_id, vec FROM embeddings WHERE video_id=?", (video_id,)
        ).fetchall()
        if not rows:
            return []
        q = query_vec.astype(np.float32)
        mat = np.vstack([_unpack(r["vec"]) for r in rows])

        # 维度守卫：换过嵌入模型（例如 bge-m3 1024 维 → bge-small-zh 512 维）时，
        # 旧向量与新查询向量维度不同。若不拦住，mat @ q 会直接抛异常让整个问答崩掉。
        # 但也不能静默跳过——那等于向量召回悄悄失效，属于我们一直在防的那类 bug。
        if mat.shape[1] != q.shape[0]:
            log.warning(
                "向量维度不匹配：库中 %d 维，当前模型 %d 维。"
                "多半是换过嵌入模型，请执行 `vedioai reindex` 重建索引；"
                "本次已跳过向量召回，仅用关键词。",
                mat.shape[1],
                q.shape[0],
            )
            return []

        denom = (np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0))
        denom[denom == 0] = 1.0
        sims = (mat @ q) / denom
        order = np.argsort(-sims)[:top_k]
        return [(rows[i]["chunk_id"], float(sims[i])) for i in order]

    # ------------------------------------------------------------------ 关键词

    def search_keyword(self, video_id: str, query: str, top_k: int) -> list[tuple[str, float]]:
        terms = query_terms(query)
        if not terms:
            return []
        match_expr = " OR ".join(terms)
        try:
            rows = self.conn.execute(
                """
                SELECT chunk_id, bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ? AND video_id = ?
                ORDER BY score
                LIMIT ?
                """,
                (match_expr, video_id, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        # FTS5 的 bm25() 越小越相关，转成越大越相关
        return [(r["chunk_id"], -float(r["score"])) for r in rows]

    def close(self) -> None:
        self.conn.close()
