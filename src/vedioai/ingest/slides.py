"""课件抽帧与 OCR。

为什么第一期就必须要这一步：中文网课大量信息在 PPT/板书，不在语音里。
只有转写的检索，在 PPT 类课程上会明显低于预期——而且会被误判成「期望太高」。

变化检测用「帧差 + 感知哈希」，不用场景切换：屏幕录制类课程画面几乎不变，
真正变化的是幻灯片上的文字。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import SlideConfig
from ..schema import Slide
from .media import _run  # noqa: PLC2701  同一包内复用 ffmpeg 调用封装

log = logging.getLogger(__name__)

_OCR_ENGINE = None
_OCR_TRIED = False


def _get_ocr():
    """懒加载 RapidOCR。未安装时返回 None 并提示，不阻塞入库。

    刻意不用 PaddleOCR：paddle 运行时体积大、Windows 部署麻烦，是打包地狱。
    RapidOCR + PP-OCRv5 ONNX 模型只有 22MB。
    """
    global _OCR_ENGINE, _OCR_TRIED
    if _OCR_TRIED:
        return _OCR_ENGINE
    _OCR_TRIED = True
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        _OCR_ENGINE = RapidOCR()
        log.info("RapidOCR 已加载")
    except Exception as exc:  # noqa: BLE001
        log.warning("RapidOCR 不可用（%s），将跳过课件 OCR。安装：pip install rapidocr-onnxruntime", exc)
        _OCR_ENGINE = None
    return _OCR_ENGINE


@dataclass
class _Frame:
    ms: int
    path: Path
    gray: np.ndarray
    phash: str


def _ahash(gray: np.ndarray, size: int = 8) -> str:
    """平均哈希，用于判定相邻页是否同一张。"""
    img = Image.fromarray(gray).resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{size * size // 4}x}"


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def extract_frames(
    ffmpeg: str,
    src: Path,
    out_dir: Path,
    *,
    interval_ms: int,
    width: int = 960,
    max_frames: int = 2000,
) -> list[_Frame]:
    """按固定间隔批量抽帧（一次 ffmpeg 调用，比逐帧 seek 快得多）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("f_*.jpg"):
        stale.unlink(missing_ok=True)

    fps = 1000.0 / max(interval_ms, 1)
    _run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vf", f"fps={fps:.6f},scale={width}:-2",
            "-q:v", "3",
            str(out_dir / "f_%06d.jpg"),
        ]
    )

    frames: list[_Frame] = []
    for i, path in enumerate(sorted(out_dir.glob("f_*.jpg"))):
        if i >= max_frames:
            path.unlink(missing_ok=True)
            continue
        with Image.open(path) as im:
            small = im.convert("L").resize((64, 36), Image.Resampling.BILINEAR)
            gray = np.asarray(small, dtype=np.float32)
        frames.append(
            _Frame(ms=i * interval_ms, path=path, gray=gray, phash=_ahash(np.asarray(small)))
        )
    return frames


def detect_slides(
    ffmpeg: str,
    src: Path,
    duration_ms: int,
    out_dir: Path,
    cfg: SlideConfig,
    *,
    progress=None,
) -> list[Slide]:
    """检测并去重幻灯片，返回带时间区间和 OCR 文本的 Slide 列表。

    progress(done, total, message)：OCR 单张实测 3–22 秒（随文字量增长），
    一门课上百张就是几十分钟。没有进度反馈会被误认为卡死。
    """
    if progress:
        progress(0, 0, "抽取候选帧")
    frames = extract_frames(
        ffmpeg, src, out_dir,
        interval_ms=cfg.sample_interval_ms,
        max_frames=cfg.max_slides * 6,
    )
    if not frames:
        return []

    # 判定换页：与「当前代表帧」比较（而不是上一帧），避免渐变动画被拆成多页
    groups: list[list[_Frame]] = [[frames[0]]]
    ref = frames[0]
    for frame in frames[1:]:
        diff = float(np.mean(np.abs(frame.gray - ref.gray)) / 255.0)
        changed = diff > cfg.change_ratio
        held = frame.ms - groups[-1][0].ms > cfg.max_hold_ms
        if changed or held:
            groups.append([frame])
            ref = frame
        else:
            groups[-1].append(frame)

    groups = groups[: cfg.max_slides]
    ocr = _get_ocr() if cfg.ocr_enabled else None

    slides: list[Slide] = []
    for gi, group in enumerate(groups):
        start_ms = group[0].ms
        end_ms = groups[gi + 1][0].ms if gi + 1 < len(groups) else max(duration_ms, start_ms + 1000)
        image_path = group[0].path

        ocr_text = ""
        if ocr is not None:
            if progress:
                progress(gi, len(groups), f"课件 OCR {gi}/{len(groups)}")
            ocr_text = _ocr_image(ocr, image_path)

        slides.append(
            Slide(
                idx=len(slides),
                start_ms=start_ms,
                end_ms=end_ms,
                image_path=str(image_path),
                ocr_text=ocr_text,
                phash=group[0].phash,
            )
        )

    if progress and ocr is not None:
        progress(len(groups), len(groups), f"课件 OCR {len(groups)}/{len(groups)}")

    # 清理未被选为幻灯片的采样帧，节省磁盘
    keep = {s.image_path for s in slides}
    for frame in frames:
        if str(frame.path) not in keep:
            frame.path.unlink(missing_ok=True)

    return slides


def _ocr_image(ocr, image_path: Path) -> str:
    try:
        result, _ = ocr(str(image_path))
    except Exception as exc:  # noqa: BLE001
        log.warning("OCR 失败 %s: %s", image_path.name, exc)
        return ""
    if not result:
        return ""
    lines = []
    for item in result:
        # RapidOCR 返回 [box, text, score]
        if len(item) >= 3 and item[2] is not None and float(item[2]) < 0.5:
            continue
        text = str(item[1]).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def slides_in_range(slides: list[Slide], start_ms: int, end_ms: int) -> list[int]:
    """返回与给定时间区间有交叠的幻灯片下标。"""
    out = []
    for s in slides:
        if s.start_ms < end_ms and s.end_ms > start_ms:
            out.append(s.idx)
    return out
