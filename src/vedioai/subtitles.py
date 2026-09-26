"""WebVTT 字幕：由句级转写 segments 生成，供播放器 <track> 与转写面板共用。"""

from __future__ import annotations

from vedioai.schema import Segment


def _vtt_timestamp(ms: int) -> str:
    """毫秒 → WebVTT 时间戳 ``HH:MM:SS.mmm``。"""
    if ms < 0:
        ms = 0
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"


def segments_to_webvtt(segments: list[Segment]) -> str:
    """把 segments 序列化成 WebVTT 文本（UTF-8）。

    空句跳过；若 end <= start，给 1.5 秒默认时长，避免播放器丢 cue。
    """
    lines = ["WEBVTT", ""]
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        start = int(seg.start_ms or 0)
        end = int(seg.end_ms or 0)
        if end <= start:
            end = start + 1500
        # WebVTT 正文里的换行要保留为多行 cue；箭头与时间戳语法需转义极少见字符
        safe = text.replace("-->", "→")
        lines.append(f"{_vtt_timestamp(start)} --> {_vtt_timestamp(end)}")
        lines.append(safe)
        lines.append("")
    return "\n".join(lines)
