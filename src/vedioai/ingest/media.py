"""FFmpeg 封装：探测、播放代理、抽音频、静音检测与切段。

两个必须做对的点（否则第一周就会踩坑）：

1. **播放代理**：Chromium 不支持 AC3/EAC3，HEVC 只在有硬件解码时可用，
   MKV/FLV/RMVB/WMV 不可靠。所以入库时统一转成 H.264/AAC 的 MP4。
   同编码时直接 `-c copy` remux，几乎零成本。

2. **音频规格**：抽成 16kHz 单声道低码率 mp3，既够 ASR 用，又能让 2 小时课
   的 base64 体积控制在几十 MB 以内（base64 会放大约 33%）。
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Chromium/Electron 能稳定播放的编码。之外的都要转。
VIDEO_OK = {"h264", "vp8", "vp9", "av1"}
AUDIO_OK = {"aac", "mp3", "opus", "vorbis", "flac"}


class FFmpegError(RuntimeError):
    pass


@dataclass
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str
    channels: int = 0
    sample_rate: int = 0


@dataclass
class MediaInfo:
    duration_ms: int
    streams: list[StreamInfo]
    format_name: str = ""

    @property
    def video(self) -> StreamInfo | None:
        return next((s for s in self.streams if s.codec_type == "video"), None)

    @property
    def audio(self) -> StreamInfo | None:
        return next((s for s in self.streams if s.codec_type == "audio"), None)


def _run(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise FFmpegError(f"命令失败（exit {proc.returncode}）: {' '.join(cmd[:3])}...\n{tail}")
    return proc


def probe(ffprobe: str, src: Path) -> MediaInfo:
    proc = _run(
        [
            ffprobe,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(src),
        ]
    )
    data = json.loads(proc.stdout or "{}")
    duration_ms = int(float((data.get("format") or {}).get("duration") or 0) * 1000)
    streams = []
    for s in data.get("streams") or []:
        streams.append(
            StreamInfo(
                index=int(s.get("index", 0)),
                codec_type=str(s.get("codec_type") or ""),
                codec_name=str(s.get("codec_name") or ""),
                channels=int(s.get("channels") or 0),
                sample_rate=int(s.get("sample_rate") or 0),
            )
        )
    return MediaInfo(
        duration_ms=duration_ms,
        streams=streams,
        format_name=str((data.get("format") or {}).get("format_name") or ""),
    )


def plan_proxy(info: MediaInfo) -> tuple[bool, str]:
    """返回 (是否需要重新生成, 原因)。"""
    v, a = info.video, info.audio
    if v is None:
        return False, "无视频轨"
    if v.codec_name not in VIDEO_OK:
        return True, f"视频编码 {v.codec_name} Chromium 不可靠"
    if a is not None and a.codec_name not in AUDIO_OK:
        return True, f"音频编码 {a.codec_name} Chromium 不支持（如 AC3/EAC3）"
    if info.format_name and "mp4" not in info.format_name:
        return True, f"容器 {info.format_name} 不可靠"
    return False, "已兼容，可直出"


def build_proxy(
    ffmpeg: str,
    src: Path,
    dst: Path,
    *,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    crf: int = 23,
    preset: str = "veryfast",
) -> tuple[bool, str]:
    """生成浏览器可播放的 MP4。同编码时只 remux，不重编码。

    返回 (是否重新编码, 说明)。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = probe(_ffprobe_for(ffmpeg), src)
    v, a = info.video, info.audio
    if v is None:
        raise FFmpegError(f"{src.name} 没有视频轨，无法生成播放代理")

    copy_video = v.codec_name in VIDEO_OK
    copy_audio = a is None or a.codec_name in AUDIO_OK

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    cmd += ["-c:v", "copy"] if copy_video else ["-c:v", video_codec, "-crf", str(crf), "-preset", preset]
    if a is None:
        cmd += ["-an"]
    elif copy_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", audio_codec, "-b:a", "128k"]
    # faststart 让 <video> 可以边下边播
    cmd += ["-movflags", "+faststart", str(dst)]

    _run(cmd)
    detail = "remux（未重编码）" if (copy_video and copy_audio) else "已转码为 H.264/AAC"
    return (not copy_video or not copy_audio), detail


def _ffprobe_for(ffmpeg: str) -> str:
    """从 ffmpeg 路径推出同目录的 ffprobe。"""
    p = Path(ffmpeg)
    candidate = p.with_name("ffprobe" + (p.suffix or ".exe"))
    if candidate.exists():
        return str(candidate)
    return "ffprobe"


def extract_audio(
    ffmpeg: str,
    src: Path,
    dst: Path,
    *,
    sample_rate: int = 16000,
    bitrate: str = "32k",
) -> Path:
    """抽成 16kHz 单声道 mp3。低码率是为了控制 base64 上传体积。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "libmp3lame",
            "-b:a", bitrate,
            str(dst),
        ]
    )
    return dst


_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)")


def detect_silences(
    ffmpeg: str,
    src: Path,
    *,
    noise_db: float = -35.0,
    min_silence_s: float = 0.6,
) -> list[tuple[int, int]]:
    """返回静音区间 [(start_ms, end_ms)]，用于切段时不切断句子。"""
    proc = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-nostats",
            "-i", str(src),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
            "-f", "null", "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    starts = [float(m) for m in _SILENCE_START.findall(proc.stderr or "")]
    ends = [float(m) for m in _SILENCE_END.findall(proc.stderr or "")]
    out: list[tuple[int, int]] = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else s + 1.0
        out.append((int(s * 1000), int(e * 1000)))
    return out


def pick_split_points(
    duration_ms: int,
    silences: list[tuple[int, int]],
    *,
    chunk_ms: int,
    overlap_ms: int = 1000,
) -> list[tuple[int, int]]:
    """把整段音频切成若干 (start_ms, end_ms)，切点优先落在静音中点。

    返回的区间带 overlap，避免上下文在边界被割裂。
    """
    if duration_ms <= chunk_ms:
        return [(0, duration_ms)]

    # 静音中点作为候选切点
    candidates = sorted((s + e) // 2 for s, e in silences)
    # 兜底：没有任何静音时用等分点
    if not candidates:
        candidates = [chunk_ms]

    targets = list(range(chunk_ms, duration_ms, chunk_ms))
    points: list[int] = []
    for target in targets:
        window = max(5000, chunk_ms // 6)
        near = [c for c in candidates if abs(c - target) <= window]
        point = min(near, key=lambda c: abs(c - target)) if near else target
        # 保证单调且与上一个切点留有间隔
        if not points or point - points[-1] >= 30_000:
            points.append(point)

    bounds = [0] + points + [duration_ms]
    spans: list[tuple[int, int]] = []
    for i in range(len(bounds) - 1):
        start = max(0, bounds[i] - (overlap_ms if i > 0 else 0))
        end = min(duration_ms, bounds[i + 1])
        if end - start > 5_000:
            spans.append((start, end))
    return spans


def slice_audio(ffmpeg: str, src: Path, dst: Path, start_ms: int, end_ms: int) -> Path:
    """按毫秒区间裁出一段音频（重编码以保证切点精确）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_ms / 1000:.3f}",
            "-i", str(src),
            "-t", f"{(end_ms - start_ms) / 1000:.3f}",
            "-c:a", "libmp3lame", "-b:a", "32k", "-ac", "1", "-ar", "16000",
            str(dst),
        ]
    )
    return dst


def extract_frame(
    ffmpeg: str,
    src: Path,
    dst: Path,
    at_ms: int,
    width: int | None = 960,
    *,
    quality: int = 2,
) -> Path:
    """抽取指定时刻的帧。用于课件图。

    ``width=None`` 表示**保持原始分辨率**，这是 OCR 该用的值。

    为什么 OCR 不该缩图：RapidOCR 内部按 ``limit_side_len=736,
    limit_type=min`` 工作——短边不足 736 时它会**向上插值放大**。所以喂
    960 宽的图，等于我们先丢掉一半像素、它再把模糊插值猜回来，成本没省，
    精度白丢。实测同一帧只有喂图分辨率不同：96 缩略图把
    "uiautomatorviewer.bat" 读成 "uiautomatoniewer.bet"、把 "BASE+MD5"
    读成 "BASE+MDS"，原始分辨率全对，耗时只多 2~16%。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{at_ms / 1000:.3f}",
        "-i", str(src),
        "-frames:v", "1",
    ]
    if width:
        cmd += ["-vf", f"scale={width}:-2"]
    # 文字细笔画对 JPEG 压缩很敏感，默认质量给足（2 最好、31 最差）
    cmd += ["-q:v", str(quality), str(dst)]
    _run(cmd)
    return dst

