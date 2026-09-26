"""课程系列：按源文件父目录自动分组，支持系列内跨课问答。

单课前缀已经能塞进 DeepSeek 1M 窗口；系列内多课若全塞会爆炸，
因此跨课路径改为「各课摘要前缀 + 跨课检索命中块」。
"""

from __future__ import annotations

from pathlib import Path

from vedioai.schema import ms_to_hms
from vedioai.store import Store


def series_id_for(path: str | Path) -> str:
    """源路径的父目录名作为系列 ID；根目录文件归入 ``_ungrouped``。"""
    p = Path(path)
    parent = p.parent.name.strip() if p.parent else ""
    return parent or "_ungrouped"


def series_label(series_id: str) -> str:
    if series_id == "_ungrouped":
        return "未分组"
    return series_id


def annotate_library_items(items: list[dict]) -> list[dict]:
    """给课程列表项加上 series_id / series_label。"""
    out = []
    for item in items:
        sid = series_id_for(item.get("path") or "")
        out.append({**item, "series_id": sid, "series_label": series_label(sid)})
    return out


def list_series(store: Store) -> list[dict]:
    """按系列聚合课程库。"""
    groups: dict[str, list[dict]] = {}
    for v in store.list_videos():
        v = dict(v)
        v.pop("video_summary", None)
        sid = series_id_for(v.get("path") or "")
        groups.setdefault(sid, []).append(
            {
                **v,
                "duration_label": ms_to_hms(v.get("duration_ms") or 0),
                "series_id": sid,
                "series_label": series_label(sid),
            }
        )
    result = []
    for sid, vids in groups.items():
        result.append(
            {
                "series_id": sid,
                "label": series_label(sid),
                "video_count": len(vids),
                "videos": sorted(vids, key=lambda x: (x.get("title") or "").lower()),
            }
        )
    result.sort(key=lambda g: (g["series_id"] == "_ungrouped", g["label"].lower()))
    return result


def build_series_prefix(store: Store, video_ids: list[str], *, max_summary_chars: int = 1200) -> str:
    """各课标题 + 截断后的全课摘要，组成跨课稳定前缀。"""
    parts = ["【课程系列摘要】", "下面是本系列每门课的浓缩摘要，跨课问题请先看这里。", ""]
    for vid in video_ids:
        video = store.get_video(vid)
        if video is None:
            continue
        title = video.title or vid
        summary = (store.get_video_summary(vid) or "").strip()
        if len(summary) > max_summary_chars:
            summary = summary[:max_summary_chars].rstrip() + "…"
        parts.append(f"## 《{title}》（video_id={vid}）")
        parts.append(summary or "（尚无全课摘要）")
        parts.append("")
    return "\n".join(parts)


def search_across(
    store: Store, video_ids: list[str], query: str, *, per_video: int = 5
) -> list[dict]:
    """跨课关键词检索：每课取 top per_video 块，合并后带课名。"""
    hits: list[dict] = []
    for vid in video_ids:
        video = store.get_video(vid)
        title = (video.title if video else None) or vid
        for chunk_id, score in store.search_keyword(vid, query, per_video):
            chunk = next((c for c in store.get_chunks(vid) if c.chunk_id == chunk_id), None)
            if chunk is None:
                continue
            hits.append(
                {
                    "video_id": vid,
                    "title": title,
                    "chunk_id": chunk_id,
                    "start_ms": chunk.start_ms,
                    "end_ms": chunk.end_ms,
                    "text": (chunk.text or "")[:240],
                    "score": score,
                    "label": ms_to_hms(chunk.start_ms),
                }
            )
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[: max(8, per_video * 2)]
