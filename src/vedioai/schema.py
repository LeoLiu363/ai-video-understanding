"""课程中间表示（IR）。

问答与写文档只吃这份结构化证据，不吃原始 MP4。

层级：
    Video
      └─ Chapter（章节）
           └─ Chunk（语义块，parent-child 关系）
                └─ Segment（句级转写，带 ASR 原生时间戳）
      └─ Slide（去重后的课件图 + OCR 文本）
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum


class VideoStatus(str, Enum):
    PENDING = "pending"
    PROBING = "probing"
    PROXY = "proxy"
    AUDIO = "audio"
    ASR = "asr"
    SLIDES = "slides"
    SEGMENT = "segment"
    SUMMARY = "summary"
    EMBED = "embed"
    READY = "ready"
    FAILED = "failed"

    @property
    def label(self) -> str:
        return {
            "pending": "待处理",
            "probing": "读取视频信息",
            "proxy": "生成播放代理",
            "audio": "抽取音频",
            "asr": "语音转写",
            "slides": "课件抽帧与 OCR",
            "segment": "语义分段",
            "summary": "生成分层摘要",
            "embed": "建立向量索引",
            "ready": "就绪",
            "failed": "失败",
        }[self.value]


class ContentKind(str, Enum):
    """一门课的介质类型：视频走 ASR/播放器；文档走原文阅读。"""

    VIDEO = "video"
    DOCUMENT = "document"


@dataclass
class Video:
    video_id: str
    path: str
    title: str
    duration_ms: int
    status: VideoStatus = VideoStatus.PENDING
    size_bytes: int = 0
    proxy_path: str | None = None
    error: str | None = None
    kind: ContentKind = ContentKind.VIDEO

    def to_row(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["kind"] = self.kind.value
        return d


@dataclass
class Segment:
    """句级转写。start_ms/end_ms 来自 ASR 原生对齐，不是 DTW 估算。"""

    idx: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None
    # 词级时间戳，用于「引用漂移」防护：引用时定位到具体那句，而不是整块起点
    words: list[dict] = field(default_factory=list)


@dataclass
class Chunk:
    """语义块。chunk_id 稳定，供引用与增量重建。"""

    chunk_id: str
    idx: int
    start_ms: int
    end_ms: int
    text: str
    ocr_text: str = ""
    title: str = ""
    summary: str = ""
    parent_id: str | None = None
    # 该块内出现过的幻灯片下标，供视觉型问题取证
    slide_idxs: list[int] = field(default_factory=list)

    @property
    def combined_text(self) -> str:
        """口播 + 课件的融合文本，用于检索与嵌入。"""
        if not self.ocr_text:
            return self.text
        return f"{self.text}\n【课件】{self.ocr_text}"


@dataclass
class Chapter:
    chapter_id: str
    idx: int
    start_ms: int
    end_ms: int
    title: str
    summary: str
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class Slide:
    idx: int
    start_ms: int
    end_ms: int
    image_path: str
    ocr_text: str = ""
    # 感知哈希，用于去重相邻重复页
    phash: str = ""


def chunk_to_json(chunk: Chunk) -> str:
    return json.dumps(asdict(chunk), ensure_ascii=False)


def ms_to_hms(ms: int) -> str:
    """毫秒 → HH:MM:SS，用于时间戳展示。"""
    total = max(0, int(ms)) // 1000
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def hms_to_ms(text: str) -> int:
    """HH:MM:SS 或 MM:SS → 毫秒。"""
    parts = [int(p) for p in text.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3], parts[-2], parts[-1]
    return ((h * 60 + m) * 60 + s) * 1000
