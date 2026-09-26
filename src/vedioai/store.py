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

SCHEMA_VERSION = 2

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

-- 用量账本。每次 LLM 调用与每次转写都记一行。
-- cost_yuan 允许为 NULL：表示「这个模型没有价目，算不出钱」。
-- 必须与 0 区分开——0 是「免费」，NULL 是「不知道」，混起来总账就是错的。
CREATE TABLE IF NOT EXISTS usage_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    at                TEXT NOT NULL DEFAULT (datetime('now')),
    video_id          TEXT NOT NULL DEFAULT '',
    kind              TEXT NOT NULL DEFAULT 'llm',   -- ask / notes / summarize / glossary / asr
    model             TEXT NOT NULL DEFAULT '',
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    cached_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    audio_ms          INTEGER NOT NULL DEFAULT 0,
    cost_yuan         REAL,
    peak              INTEGER NOT NULL DEFAULT 0
);

-- 问答会话。一门课可以有多个命名会话（「期中复习」「作业答疑」）。
-- archived=1 表示归档：默认列表不显示，但仍可按 ID 打开。
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    archived    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 会话消息。role = user | assistant。
-- meta 存助手侧的 citations / usage / intent 等 JSON，用户消息通常是 {}。
CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    meta        TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_video  ON chunks(video_id, idx);
CREATE INDEX IF NOT EXISTS idx_chapter_video ON chapters(video_id, idx);
CREATE INDEX IF NOT EXISTS idx_slide_video   ON slides(video_id, idx);
CREATE INDEX IF NOT EXISTS idx_seg_video     ON segments(video_id, idx);
CREATE INDEX IF NOT EXISTS idx_usage_video   ON usage_log(video_id, at);
CREATE INDEX IF NOT EXISTS idx_usage_at      ON usage_log(at);
CREATE INDEX IF NOT EXISTS idx_session_video ON chat_sessions(video_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_msg_session   ON chat_messages(session_id, id);
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

    def search_segments(
        self, video_id: str, query: str, top_k: int = 20
    ) -> list[dict]:
        """课内句级关键词搜索（segments_fts）。

        返回 [{idx, start_ms, end_ms, text, score}, ...]，按相关度降序。
        供界面搜索面板与跳转使用；与 search_keyword（块级）互补。
        """
        terms = query_terms(query)
        if not terms:
            return []
        match_expr = " OR ".join(terms)
        try:
            rows = self.conn.execute(
                """
                SELECT s.idx, s.start_ms, s.end_ms, s.text,
                       bm25(segments_fts) AS score
                FROM segments_fts
                JOIN segments s
                  ON s.video_id = segments_fts.video_id AND s.idx = segments_fts.idx
                WHERE segments_fts MATCH ? AND segments_fts.video_id = ?
                ORDER BY score
                LIMIT ?
                """,
                (match_expr, video_id, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {
                "idx": int(r["idx"]),
                "start_ms": int(r["start_ms"]),
                "end_ms": int(r["end_ms"]),
                "text": r["text"] or "",
                "score": -float(r["score"]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ 用量账

    def record_usage(
        self,
        *,
        kind: str,
        model: str = "",
        video_id: str = "",
        prompt_tokens: int = 0,
        cached_tokens: int = 0,
        completion_tokens: int = 0,
        audio_ms: int = 0,
        cost_yuan: float | None = None,
        peak: bool = False,
    ) -> None:
        """记一笔用量。

        **这个方法绝不抛异常**：记账是旁路，不能因为它失败而让一次已经付过费的
        转写或一次已经拿到答案的问答前功尽弃。失败只丢一行账，并留下日志。
        """
        try:
            self.conn.execute(
                """
                INSERT INTO usage_log (video_id, kind, model, prompt_tokens,
                                       cached_tokens, completion_tokens, audio_ms,
                                       cost_yuan, peak)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id, kind, model, prompt_tokens, cached_tokens,
                    completion_tokens, audio_ms, cost_yuan, 1 if peak else 0,
                ),
            )
            self.conn.commit()
        except sqlite3.Error as exc:  # 磁盘满 / 库锁 / 表被改坏
            log.warning("用量记账失败（不影响主流程）：%s", exc)

    def usage_summary(self, video_id: str | None = None) -> dict:
        """汇总用量。video_id 省略 = 全库。

        返回的 ``unpriced_calls`` 很关键：它表示有多少次调用没查不到价目，
        也就是**总花费被低估了多少次**。界面必须把它显示出来，否则用户会
        把「算出来的数」当成「真实的数」。
        """
        where, params = "", []
        if video_id:
            where, params = "WHERE video_id = ?", [video_id]

        total = self.conn.execute(
            f"""
            SELECT COUNT(*) AS calls,
                   COALESCE(SUM(prompt_tokens), 0)     AS prompt_tokens,
                   COALESCE(SUM(cached_tokens), 0)     AS cached_tokens,
                   COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                   COALESCE(SUM(audio_ms), 0)          AS audio_ms,
                   COALESCE(SUM(cost_yuan), 0)         AS cost,
                   SUM(CASE WHEN cost_yuan IS NULL THEN 1 ELSE 0 END) AS unpriced
            FROM usage_log {where}
            """,
            params,
        ).fetchone()

        def rows(sql: str) -> list[dict]:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

        by_kind = rows(
            f"""
            SELECT kind,
                   COUNT(*) AS calls,
                   COALESCE(SUM(prompt_tokens), 0)     AS prompt_tokens,
                   COALESCE(SUM(cached_tokens), 0)     AS cached_tokens,
                   COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                   COALESCE(SUM(cost_yuan), 0)         AS cost
            FROM usage_log {where}
            GROUP BY kind ORDER BY cost DESC
            """
        )
        by_model = rows(
            f"""
            SELECT model, kind, COUNT(*) AS calls,
                   COALESCE(SUM(cost_yuan), 0) AS cost
            FROM usage_log {where}
            GROUP BY model, kind ORDER BY cost DESC
            """
        )
        # 单课视图下不需要「按课程拆分」（只有一行），所以只在全库视图里算。
        by_video: list[dict] = []
        if not video_id:
            by_video = [
                dict(r)
                for r in self.conn.execute(
                    """
                    SELECT u.video_id,
                           COALESCE(v.title, u.video_id) AS title,
                           COUNT(*) AS calls,
                           COALESCE(SUM(u.cost_yuan), 0) AS cost
                    FROM usage_log u
                    LEFT JOIN videos v ON v.video_id = u.video_id
                    GROUP BY u.video_id ORDER BY cost DESC
                    """
                ).fetchall()
            ]
        by_day = rows(
            f"""
            SELECT date(at, 'localtime') AS day, COUNT(*) AS calls,
                   COALESCE(SUM(cost_yuan), 0) AS cost
            FROM usage_log {where}
            GROUP BY day ORDER BY day DESC LIMIT 30
            """
        )

        return {
            "video_id": video_id or "",
            "calls": total["calls"],
            "unpriced_calls": total["unpriced"] or 0,
            "cost_yuan": round(total["cost"] or 0.0, 4),
            "prompt_tokens": total["prompt_tokens"],
            "cached_tokens": total["cached_tokens"],
            "completion_tokens": total["completion_tokens"],
            "audio_ms": total["audio_ms"],
            "by_kind": by_kind,
            "by_model": by_model,
            "by_video": by_video,
            "by_day": by_day,
        }

    # ------------------------------------------------------------------ 会话

    def create_session(self, video_id: str, title: str = "") -> dict:
        """新建会话。session_id 用短 hex，够本地用、也好读。"""
        import secrets

        session_id = secrets.token_hex(8)
        self.conn.execute(
            """
            INSERT INTO chat_sessions (session_id, video_id, title)
            VALUES (?, ?, ?)
            """,
            (session_id, video_id, (title or "").strip()),
        )
        self.conn.commit()
        return self.get_session(session_id)  # type: ignore[return-value]

    def get_session(self, session_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, video_id: str, *, include_archived: bool = False) -> list[dict]:
        """按最近活跃排序。默认隐藏已归档。"""
        sql = """
            SELECT s.*,
                   (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.session_id) AS message_count,
                   (SELECT content FROM chat_messages m
                    WHERE m.session_id = s.session_id AND m.role = 'user'
                    ORDER BY m.id LIMIT 1) AS first_question
            FROM chat_sessions s
            WHERE s.video_id = ?
        """
        params: list = [video_id]
        if not include_archived:
            sql += " AND s.archived = 0"
        sql += " ORDER BY s.updated_at DESC, s.created_at DESC"
        rows = self.conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            # 没起名时用首问当展示标题，但不写回库——用户改名才能真正落盘
            if not d.get("title") and d.get("first_question"):
                d["display_title"] = _short_title(d["first_question"])
            else:
                d["display_title"] = d.get("title") or "新对话"
            out.append(d)
        return out

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        archived: bool | None = None,
        touch: bool = False,
    ) -> dict | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        fields: list[str] = []
        params: list = []
        if title is not None:
            fields.append("title = ?")
            params.append(title.strip())
        if archived is not None:
            fields.append("archived = ?")
            params.append(1 if archived else 0)
        if touch or fields:
            fields.append("updated_at = datetime('now')")
        if not fields:
            return session
        params.append(session_id)
        self.conn.execute(
            f"UPDATE chat_sessions SET {', '.join(fields)} WHERE session_id = ?",
            params,
        )
        self.conn.commit()
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """硬删除会话及其消息（FK CASCADE）。"""
        cur = self.conn.execute(
            "DELETE FROM chat_sessions WHERE session_id = ?", (session_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        meta: dict | None = None,
    ) -> dict:
        cur = self.conn.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, meta)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, json.dumps(meta or {}, ensure_ascii=False)),
        )
        # 任何新消息都把会话顶到列表最前
        self.conn.execute(
            "UPDATE chat_sessions SET updated_at = datetime('now') WHERE session_id = ?",
            (session_id,),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM chat_messages WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return _message_row(row)

    def get_messages(self, session_id: str, *, limit: int | None = None) -> list[dict]:
        """按时间正序返回。limit 表示「最近 N 条」，仍按正序排好再给调用方。"""
        if limit is None:
            rows = self.conn.execute(
                """
                SELECT * FROM chat_messages
                WHERE session_id = ? ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        else:
            # 先倒序取最近 N 条，再翻正——这样多轮上下文取「尾巴」时不需要全表扫描
            rows = self.conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM chat_messages
                    WHERE session_id = ? ORDER BY id DESC LIMIT ?
                ) ORDER BY id
                """,
                (session_id, limit),
            ).fetchall()
        return [_message_row(r) for r in rows]

    def close(self) -> None:
        self.conn.close()


def _message_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["meta"] = json.loads(d.get("meta") or "{}")
    except json.JSONDecodeError:
        d["meta"] = {}
    return d


def _short_title(text: str, limit: int = 28) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text or "新对话"
    return text[: limit - 1] + "…"
