"""课内自测：选择题生成与判分。

先出选择（好判、省 token），提交后再揭晓答案与讲解。
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from . import ledger
from .llm import prompts
from .llm.client import LLMClient, LLMError
from .schema import ms_to_hms
from .store import Store

log = logging.getLogger(__name__)

DEFAULT_COUNT = 8
MAX_COUNT = 20
MIN_COUNT = 3


def _quiz_material(store: Store, video_id: str, *, budget_chars: int = 14000) -> str:
    """拼出题用的课内材料：优先章节摘要，不足再补片段要点/正文。"""
    video = store.get_video(video_id)
    if video is None:
        raise ValueError("课程不存在")
    lines: list[str] = [
        f"课程：{video.title}",
        f"总时长：{ms_to_hms(video.duration_ms)}",
    ]
    summary = store.get_video_summary(video_id) or ""
    if summary.strip():
        lines.append("\n## 全课摘要\n" + summary.strip()[:3000])

    chapters = store.get_chapters(video_id)
    chunk_by_id = {c.chunk_id: c for c in store.get_chunks(video_id)}
    used = 0
    for ch in chapters:
        title = (ch.title or f"第 {ch.idx + 1} 章").strip()
        block = f"\n## {title} [{ms_to_hms(ch.start_ms)}-{ms_to_hms(ch.end_ms)}]\n"
        if (ch.summary or "").strip():
            block += ch.summary.strip()[:800] + "\n"
        else:
            # 摘要缺失时用块正文凑一点，否则空材料出题会胡编
            parts = []
            for cid in ch.chunk_ids[:4]:
                c = chunk_by_id.get(cid)
                if not c:
                    continue
                tip = (c.summary or c.title or c.text or "").strip()
                if tip:
                    parts.append(f"- [{ms_to_hms(c.start_ms)}] {tip[:220]}")
            if parts:
                block += "\n".join(parts) + "\n"
        if used + len(block) > budget_chars:
            break
        lines.append(block)
        used += len(block)

    text = "\n".join(lines).strip()
    if len(text) < 80:
        raise ValueError("课程材料过少，请先完成入库与摘要后再出题")
    return text


def _normalize_questions(raw: Any, *, count: int) -> list[dict]:
    if not isinstance(raw, dict):
        raise LLMError("测验 JSON 根节点必须是对象")
    items = raw.get("questions")
    if not isinstance(items, list) or not items:
        raise LLMError("测验 JSON 缺少 questions 数组")

    out: list[dict] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        stem = str(item.get("stem") or "").strip()
        opts_raw = item.get("options") or {}
        if not isinstance(opts_raw, dict):
            continue
        options = {
            k: str(opts_raw.get(k) or "").strip()
            for k in ("A", "B", "C", "D")
        }
        if not stem or any(not options[k] for k in options):
            continue
        answer = str(item.get("answer") or "").strip().upper()
        if answer not in options:
            continue
        try:
            start_ms = int(item.get("start_ms") or 0)
        except (TypeError, ValueError):
            start_ms = 0
        start_ms = max(0, start_ms)
        out.append(
            {
                "id": str(item.get("id") or f"q{i + 1}").strip() or f"q{i + 1}",
                "stem": stem,
                "options": options,
                "answer": answer,
                "explanation": str(item.get("explanation") or "").strip(),
                "start_ms": start_ms,
            }
        )
        if len(out) >= count:
            break
    if len(out) < min(count, MIN_COUNT):
        raise LLMError(f"有效题目不足（得到 {len(out)} 道，期望 {count}）")
    # 规范化 id
    for i, q in enumerate(out, 1):
        q["id"] = f"q{i}"
    return out


def generate_quiz(
    client: LLMClient,
    store: Store,
    video_id: str,
    *,
    count: int = DEFAULT_COUNT,
) -> dict:
    """生成一套选择题并落库。返回含答案的完整卷（服务端保存用）。"""
    count = max(MIN_COUNT, min(MAX_COUNT, int(count or DEFAULT_COUNT)))
    material = _quiz_material(store, video_id)
    prompt = prompts.QUIZ_GENERATE.format(count=count)
    messages = [
        {"role": "system", "content": prompts.SYSTEM_TUTOR},
        {"role": "user", "content": f"{prompt}\n\n---\n{material}"},
    ]
    with ledger.usage_scope("quiz", video_id):
        data, _ = client.chat_json(messages, max_tokens=4096)
    questions = _normalize_questions(data, count=count)
    quiz_id = secrets.token_hex(8)
    video = store.get_video(video_id)
    payload = {
        "quiz_id": quiz_id,
        "video_id": video_id,
        "title": (video.title if video else "") or "",
        "count": len(questions),
        "questions": questions,
    }
    store.save_quiz(quiz_id, video_id, payload)
    return payload


def public_quiz(payload: dict) -> dict:
    """发给前端作答用的卷子：不含标准答案与讲解。"""
    qs = []
    for q in payload.get("questions") or []:
        qs.append(
            {
                "id": q["id"],
                "stem": q["stem"],
                "options": q["options"],
                "start_ms": q.get("start_ms") or 0,
            }
        )
    return {
        "quiz_id": payload["quiz_id"],
        "video_id": payload["video_id"],
        "title": payload.get("title") or "",
        "count": len(qs),
        "questions": qs,
    }


def grade_quiz(payload: dict, answers: dict[str, str]) -> dict:
    """判分并附带每题讲解。"""
    answers = {str(k): str(v or "").strip().upper() for k, v in (answers or {}).items()}
    details = []
    correct_n = 0
    for q in payload.get("questions") or []:
        qid = q["id"]
        chosen = answers.get(qid, "")
        ok = chosen == q["answer"]
        if ok:
            correct_n += 1
        details.append(
            {
                "id": qid,
                "stem": q["stem"],
                "options": q["options"],
                "chosen": chosen,
                "answer": q["answer"],
                "correct": ok,
                "explanation": q.get("explanation") or "",
                "start_ms": q.get("start_ms") or 0,
            }
        )
    total = len(details)
    return {
        "quiz_id": payload["quiz_id"],
        "video_id": payload["video_id"],
        "total": total,
        "correct": correct_n,
        "score": round(correct_n / total * 100.0, 1) if total else 0.0,
        "details": details,
    }
