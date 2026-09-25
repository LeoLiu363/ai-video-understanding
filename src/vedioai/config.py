"""配置加载。

优先级：环境变量 > .env 文件 > vedioai.config.yaml > 内置默认值。

刻意不把密钥写进代码（旧项目最大的工程债之一）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析。已存在的环境变量优先，不覆盖。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class ASRConfig:
    """火山引擎录音文件识别。"""

    app_id: str = ""
    access_token: str = ""
    api_key: str = ""
    resource_id: str = "volc.bigasr.auc_turbo"
    # 单次提交的音频上限（字节）。超过则按静音切段后分批提交，再按偏移合并。
    # 极速版走 base64，base64 会放大约 33%，这里给的是原始音频的预算。
    max_upload_bytes: int = 24 * 1024 * 1024
    # 切成多少毫秒一段（仅在超出上传上限或用户显式要求时分段）
    chunk_ms: int = 20 * 60 * 1000
    # 分段之间的重叠，避免切在句子中间丢上下文
    overlap_ms: int = 1_000
    poll_interval_s: float = 5.0
    poll_timeout_s: float = 3600.0

    @property
    def ready(self) -> bool:
        return bool(self.api_key or (self.app_id and self.access_token))


@dataclass
class LLMConfig:
    """OpenAI 兼容的文本模型（问答 / 写文档）。"""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    # 注意：DeepSeek 的实际模型 ID 是 deepseek-flash（"V4 Flash" 是它的宣传名，
    # 不是可调用的 ID）。写 deepseek-v4-flash 会 404。
    # 可用值见 GET /models：deepseek-flash、deepseek-v4-pro
    model: str = "deepseek-flash"
    temperature: float = 0.2
    max_tokens: int = 8192
    # 整稿要作为稳定前缀才能命中上下文缓存，前缀过小不值得
    min_prefix_tokens: int = 512
    # 是否开启模型的思考（思维链）。默认关闭，理由有三：
    # 1) 思考 token 按输出计费且占用 max_tokens，容易把正文挤到截断；
    # 2) 思考模式下服务端会忽略 temperature，评估集无法复现；
    # 3) 课内问答的证据已在上下文里，主要成本是延迟，开思考会明显变慢。
    # 觉得某些难题答得不够好时可以打开对比。
    thinking: bool = False


@dataclass
class VisionConfig:
    """OpenAI 兼容的视觉模型（课件图 / 板书）。"""

    api_key: str = ""
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    model: str = "doubao-seed-2-1-turbo-260628"
    max_images: int = 3


@dataclass
class MediaConfig:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    audio_sample_rate: int = 16000
    # 转成低码率 mp3，2 小时课约 28MB，base64 后才 ~38MB，适合直传
    audio_bitrate: str = "48k"
    # 播放代理统一转成 H.264/AAC，规避 Chromium 不支持 AC3/EAC3/MKV 的问题
    proxy_video_codec: str = "libx264"
    proxy_audio_codec: str = "aac"
    proxy_crf: int = 23
    proxy_preset: str = "veryfast"


@dataclass
class SlideConfig:
    enabled: bool = True
    # 抽样间隔：每 N 毫秒取一帧做变化检测（PPT 课通常变化稀疏）
    sample_interval_ms: int = 2000
    # 帧差超过该比例视为换页
    change_ratio: float = 0.02
    # 幻灯片最长停留时间，超过则强制切一张，避免长时间无变化时丢图
    max_hold_ms: int = 120_000
    max_slides: int = 400
    ocr_enabled: bool = True
    # 变化检测用的采样宽度。这一遍要处理上千帧，必须便宜；屏幕录制类课程
    # 画面变化稀疏，960 足够判定换页。
    detect_width: int = 960
    # OCR 用的宽度。0 = 保持原始分辨率（默认，且强烈建议）。
    #
    # 为什么必须和 detect_width 分开：实测把 1924 宽的录屏压到 960 再 OCR，
    # 会把 "uiautomatorviewer.bat" 读成 "uiautomatoniewer.bet"、
    # "BASE+MD5" 读成 "BASE+MDS"、"录制设置" 读成 "爱制设置"；
    # 换成原始分辨率后这些全对，而 OCR 耗时只涨 2~16%——
    # 因为 RapidOCR 内部本来就会把短边插值放大到 736（limit_type=min），
    # 喂小图等于先丢像素、再插值猜回来，成本没省下，精度白丢。
    ocr_width: int = 0


@dataclass
class RetrieveConfig:
    top_k: int = 8
    rerank_top_n: int = 5
    # 检索时是否按当前播放位置加权。默认关闭 —— 开启会让答案不可复现。
    time_bias: bool = False
    # 嵌入后端：auto | onnx | ollama | none
    # auto：有 ONNX 权重就用 ONNX，否则尝试本机 Ollama，都不行则退化为纯关键词。
    # 之所以要 Ollama 这条路：Ollama 里的 bge-m3 是 GGUF，onnxruntime 读不了。
    embed_backend: str = "auto"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_embed_model: str = "bge-m3"


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    slides: SlideConfig = field(default_factory=SlideConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    language: str = "zh"
    # 术语表位置。None 时用项目根目录的 vedioai.glossary.yaml。
    glossary_file: Path | None = None

    @property
    def library_dir(self) -> Path:
        return self.data_dir / "library"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vedioai.db"

    @property
    def glossary_path(self) -> Path:
        """术语表路径。

        刻意放在项目根目录而不是 data/：它是需要进版本库的策展知识，
        而 data/ 是入库产物（已被 .gitignore 忽略）。
        """
        return self.glossary_file or (PROJECT_ROOT / "vedioai.glossary.yaml")

    @property
    def models_dir(self) -> Path:
        """本地 ONNX 模型根目录（嵌入 / 重排）。"""
        return self.data_dir.parent / "models"

    @property
    def embed_model_dir(self) -> Path:
        return self.models_dir / "bge-m3-onnx"

    @property
    def rerank_model_dir(self) -> Path:
        return self.models_dir / "bge-reranker-v2-m3-onnx"


def _from_yaml(cfg: Config, raw: dict) -> Config:
    for section in ("asr", "llm", "vision", "media", "slides", "retrieve"):
        block = raw.get(section) or {}
        target = getattr(cfg, section)
        for key, value in block.items():
            if hasattr(target, key):
                setattr(target, key, value)
    if raw.get("data_dir"):
        cfg.data_dir = Path(str(raw["data_dir"])).expanduser()
    if raw.get("glossary_file"):
        cfg.glossary_file = Path(str(raw["glossary_file"])).expanduser()
    if raw.get("language"):
        cfg.language = str(raw["language"])
    return cfg


def load_config(config_path: Path | None = None) -> Config:
    _load_dotenv(PROJECT_ROOT / ".env")

    cfg = Config()

    yaml_path = config_path or (PROJECT_ROOT / "vedioai.config.yaml")
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        cfg = _from_yaml(cfg, raw)

    env = os.environ

    cfg.asr.app_id = env.get("VOLC_SPEECH_APP_ID", cfg.asr.app_id)
    cfg.asr.access_token = env.get("VOLC_SPEECH_ACCESS_TOKEN", cfg.asr.access_token)
    cfg.asr.api_key = env.get("VOLC_SPEECH_API_KEY", cfg.asr.api_key)
    cfg.asr.resource_id = env.get("VOLC_ASR_RESOURCE_ID", cfg.asr.resource_id)

    cfg.llm.api_key = env.get("DEEPSEEK_API_KEY", cfg.llm.api_key)
    cfg.llm.base_url = env.get("DEEPSEEK_BASE_URL", cfg.llm.base_url)
    cfg.llm.model = env.get("DEEPSEEK_MODEL", cfg.llm.model)

    cfg.vision.api_key = env.get("ARK_API_KEY", cfg.vision.api_key)
    cfg.vision.base_url = env.get("ARK_BASE_URL", cfg.vision.base_url)
    cfg.vision.model = env.get("ARK_VISION_MODEL", cfg.vision.model)

    # 嵌入后端：让用户不必编辑 YAML 就能切换（同时也方便 CI/容器里覆盖）
    cfg.retrieve.embed_backend = env.get("VEDIOAI_EMBED_BACKEND", cfg.retrieve.embed_backend)
    cfg.retrieve.ollama_url = env.get(
        "OLLAMA_URL", env.get("OLLAMA_HOST_URL", cfg.retrieve.ollama_url)
    )
    cfg.retrieve.ollama_embed_model = env.get(
        "OLLAMA_EMBED_MODEL", cfg.retrieve.ollama_embed_model
    )

    if env.get("VEDIOAI_DATA_DIR"):
        cfg.data_dir = Path(env["VEDIOAI_DATA_DIR"]).expanduser()

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.library_dir.mkdir(parents=True, exist_ok=True)
    return cfg
