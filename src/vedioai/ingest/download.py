"""从 URL 拉取视频（直链 / B 站 / YouTube 等）。

依赖 yt-dlp；B 站大会员或需登录内容可导入浏览器导出的 Netscape Cookie。
Cookie 只存本机 data/cookies/，不进版本库、不上云。
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

log = logging.getLogger(__name__)

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
# 无协议时也认常见站点（避免用户只粘 BV 页路径）
_HOST_RE = re.compile(
    r"^(?:www\.)?(?:"
    r"bilibili\.com|b23\.tv|youtube\.com|youtu\.be|"
    r"youtu\.be|m\.bilibili\.com"
    r")/",
    re.IGNORECASE,
)
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass
class DownloadResult:
    path: Path
    title: str
    url: str
    source_id: str
    extractor: str = ""


def looks_like_url(text: str) -> bool:
    """判断输入更像视频链接，而不是本机路径。"""
    t = (text or "").strip().strip('"').strip("'")
    if not t or _WIN_DRIVE_RE.match(t) or t.startswith("\\\\"):
        return False
    if _URL_RE.match(t):
        return True
    if _HOST_RE.match(t):
        return True
    return False


def normalize_url(text: str) -> str:
    t = (text or "").strip().strip('"').strip("'")
    if not t:
        raise ValueError("空链接")
    if not _URL_RE.match(t):
        t = "https://" + t.lstrip("/")
    return t


def source_id_for_url(url: str) -> str:
    """同一规范化 URL → 稳定 video_id（与本地路径哈希体系分开）。"""
    canon = _canonicalize_url(normalize_url(url))
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]


def _canonicalize_url(url: str) -> str:
    p = urlparse(url)
    # 去掉 fragment；保留 query（B 站分 P 靠 p=）
    return urlunparse(
        (p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/") or "/", p.params, p.query, "")
    )


def cookies_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "cookies"


def cookies_file(data_dir: Path) -> Path:
    return cookies_dir(data_dir) / "cookies.txt"


def cookie_status(data_dir: Path) -> dict:
    path = cookies_file(data_dir)
    if not path.exists():
        return {"present": False, "path": str(path), "bytes": 0, "lines": 0}
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [
        ln
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return {
        "present": True,
        "path": str(path),
        "bytes": path.stat().st_size,
        "lines": len(lines),
    }


def save_cookies(data_dir: Path, content: str | bytes) -> dict:
    """写入 Netscape cookies.txt。内容由用户从浏览器扩展导出。"""
    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="replace")
    else:
        text = content
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
    if not _looks_like_netscape_cookies(text):
        raise ValueError(
            "不像 Netscape cookies.txt（需要浏览器扩展导出的 Cookie 文件，"
            "例如「Get cookies.txt LOCALLY」）"
        )
    dest = cookies_file(data_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8", newline="\n")
    try:
        dest.chmod(0o600)
    except OSError:
        pass
    return cookie_status(data_dir)


def clear_cookies(data_dir: Path) -> None:
    path = cookies_file(data_dir)
    if path.exists():
        path.unlink()


def _looks_like_netscape_cookies(text: str) -> bool:
    if "# Netscape" in text or "# HTTP Cookie File" in text:
        return True
    # 至少一行：domain \\t flag \\t path \\t …（≥6 列）
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) >= 6 and parts[0]:
            return True
    return False


def ytdlp_available() -> bool:
    try:
        import yt_dlp  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_ffmpeg_location(ffmpeg: str | None = None) -> str | None:
    """把配置里的 ffmpeg 命令/路径解析成 yt-dlp 要的「所在目录」。

    配置常写 ``ffmpeg``（靠 PATH），不能原样塞给 ``ffmpeg_location``，
    否则 yt-dlp 会当成相对目录，报「ffmpeg is not installed」。
    """
    candidates = [ffmpeg] if ffmpeg else []
    candidates.extend(["ffmpeg", "ffmpeg.exe"])
    seen: set[str] = set()
    for raw in candidates:
        if not raw or raw in seen:
            continue
        seen.add(raw)
        p = Path(raw)
        if p.is_file():
            return str(p.resolve().parent)
        found = shutil.which(raw)
        if found:
            return str(Path(found).resolve().parent)
    return None


def download_video(
    url: str,
    dest_dir: Path,
    *,
    cookies: Path | None = None,
    ffmpeg: str | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> DownloadResult:
    """下载单个视频到 dest_dir，返回本地路径与元数据。

    progress(percent_0_100, message) 可选。
    """
    if not ytdlp_available():
        raise RuntimeError("未安装 yt-dlp：pip install yt-dlp")

    import yt_dlp

    url = normalize_url(url)
    source_id = source_id_for_url(url)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 清掉旧的 media.*，避免换源后残留
    for old in dest_dir.glob("media.*"):
        try:
            old.unlink()
        except OSError:
            pass

    outtmpl = str(dest_dir / "media.%(ext)s")
    cookie_path = Path(cookies) if cookies else None
    if cookie_path and not cookie_path.exists():
        cookie_path = None

    ffmpeg_dir = resolve_ffmpeg_location(ffmpeg)
    if ffmpeg_dir is None:
        raise RuntimeError(
            "未找到 ffmpeg。B 站等站点需用它合并音视频。\n"
            "请把 ffmpeg 加到 PATH，或在 vedioai.config.yaml 的 media.ffmpeg "
            "写成可执行文件完整路径（例如 D:\\\\…\\\\bin\\\\ffmpeg.exe）。"
        )

    def hook(d: dict) -> None:
        if not progress:
            return
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            pct = (done / total * 100.0) if total else 0.0
            speed = d.get("speed")
            speed_s = f"{speed / 1024 / 1024:.1f} MB/s" if speed else ""
            progress(min(pct, 99.0), f"拉取视频 {pct:.0f}% {speed_s}".strip())
        elif status == "finished":
            progress(99.0, "拉取完成，合并音视频…")

    opts: dict = {
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        # 优先 mp4，B 站常见分离音视频由 ffmpeg 合并
        "format": "bv*+ba/b",
        "retries": 5,
        "fragment_retries": 5,
        "ffmpeg_location": ffmpeg_dir,
    }
    if cookie_path:
        opts["cookiefile"] = str(cookie_path)

    if progress:
        progress(0.0, "解析链接…")

    info: dict = {}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True) or {}
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        low = msg.lower()
        if "ffmpeg" in low and ("not installed" in low or "not found" in low):
            raise RuntimeError(
                f"下载失败：找不到可用的 ffmpeg（已尝试目录 {ffmpeg_dir}）。\n"
                "请确认该目录下有 ffmpeg.exe，或在配置里写完整路径。"
            ) from exc
        if cookie_path is None and any(
            k in low for k in ("login", "cookie", "会员", "premium", "private", "403")
        ):
            raise RuntimeError(
                f"下载失败（可能需要登录 Cookie）：{msg}\n"
                "请在界面导入浏览器导出的 cookies.txt 后重试。"
            ) from exc
        raise RuntimeError(f"下载失败：{msg}") from exc

    path = _find_downloaded_file(dest_dir)
    if path is None:
        raise RuntimeError("下载完成但未找到视频文件")

    title = (
        (info.get("title") or "").strip()
        or path.stem
        or source_id
    )
    extractor = str(info.get("extractor") or info.get("extractor_key") or "")
    log.info("已下载 %s → %s（%s）", url, path, title)
    if progress:
        progress(100.0, f"已拉取：{title[:40]}")
    return DownloadResult(
        path=path,
        title=title,
        url=url,
        source_id=source_id,
        extractor=extractor,
    )


def _find_downloaded_file(dest_dir: Path) -> Path | None:
    candidates = sorted(
        [
            p
            for p in dest_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {
                ".mp4", ".mkv", ".webm", ".flv", ".ts", ".m4a", ".mp3", ".mov"
            }
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def ensure_download_deps() -> str | None:
    """返回缺失依赖的提示；齐全则 None。"""
    if not ytdlp_available():
        return "未安装 yt-dlp：pip install yt-dlp"
    if shutil.which("ffmpeg") is None:
        # 不硬失败：直链单文件有时不需要合并；B 站多数需要
        return None
    return None
