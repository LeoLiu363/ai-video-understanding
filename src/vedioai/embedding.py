"""本地嵌入（bge-m3 ONNX）与重排（bge-reranker-v2-m3 ONNX）。

两者都是**可选**的。未安装或未下载模型时，检索退化为纯 FTS5 关键词检索——
单课场景下长上下文直答是主力，检索只是辅助，所以退化不影响 MVP 可用性。

为什么不默认走云端嵌入：视频内容不该为了建索引而整批上传。本地方案在
CPU 上 bge-m3 int8 约 28–35 句/秒，质量只降约 1%，完全够用。

重排注意事项：必须用 **fp32**。int8 量化会改变 top-1 排序——而排序器量化的
恰好就是这个「第一名」。

模型文件放置位置（目录名不要改）：
    models/bge-m3-onnx/model.onnx  + models/bge-m3-onnx/tokenizer.json
    models/bge-reranker-v2-m3-onnx/model.onnx + .../tokenizer.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)


class EmbedderLike(Protocol):
    """嵌入后端接口。ONNX 与 Ollama 两种实现都满足它。

    抽出协议是为了让检索层不关心向量从哪来；以后要接别的本地推理服务，
    只要实现这两个成员即可。
    """

    @property
    def available(self) -> bool: ...

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray: ...


class Embedder:
    """bge-m3 ONNX 编码器。找不到模型则 available=False。"""

    def __init__(self, model_dir: Path, *, max_len: int = 512):
        self.model_dir = Path(model_dir)
        self.max_len = max_len
        self._session = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        model_path = self.model_dir / "model.onnx"
        tok_path = self.model_dir / "tokenizer.json"
        if not model_path.exists() or not tok_path.exists():
            log.info("未找到本地嵌入模型（%s），检索将只用关键词", self.model_dir)
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self._tokenizer = Tokenizer.from_file(str(tok_path))
            self._tokenizer.enable_truncation(max_length=self.max_len)
            log.info("已加载本地嵌入模型 %s", self.model_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("加载嵌入模型失败（%s），检索将只用关键词", exc)
            self._session = None
            self._tokenizer = None

    @property
    def available(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not self.available:
            raise RuntimeError("嵌入模型不可用")
        assert self._session is not None and self._tokenizer is not None

        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encodings = [self._tokenizer.encode(t or " ") for t in batch]
            max_len = max(len(e.ids) for e in encodings)
            ids = np.zeros((len(batch), max_len), dtype=np.int64)
            mask = np.zeros((len(batch), max_len), dtype=np.int64)
            for row, enc in enumerate(encodings):
                ids[row, : len(enc.ids)] = enc.ids
                mask[row, : len(enc.attention_mask)] = enc.attention_mask

            inputs = {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": np.zeros_like(ids),
            }
            wanted = {i.name for i in self._session.get_inputs()}
            inputs = {k: v for k, v in inputs.items() if k in wanted}
            hidden = self._session.run(None, inputs)[0]
            # 均值池化 + L2 归一化
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-6, None)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            out.append(pooled / np.clip(norms, 1e-6, None))
        return np.vstack(out) if out else np.zeros((0, 1024), dtype=np.float32)


class OllamaEmbedder:
    """通过 Ollama 的 HTTP 接口做嵌入，复用机器上已有的 GGUF 模型。

    存在的理由（实测得出的现实情况）：很多人机器上已经用 Ollama 拉过 bge-m3，
    而那是 **GGUF** 格式（文件头 47 47 55 46），onnxruntime 根本读不了。
    为了一个嵌入模型再下一份 2.3GB 的 ONNX 权重纯属浪费，而且还要额外装
    onnxruntime + tokenizers。直接复用本地推理服务省事得多。

    接口与 Embedder 完全一致，检索层无需区分来源。

    注意：Ollama 不做交叉编码重排，所以重排仍只有 ONNX 一条路（缺失就跳过）。

    常见坑（本机实测踩过）：若 Ollama 的 GPU 后端与显卡驱动不匹配，
    会报 "CUDA error: device kernel image is invalid" 导致请求 500。
    嵌入模型很小，走 CPU 完全够用——在 Ollama 服务里设 OLLAMA_LLM_LIBRARY=cpu
    即可绕开（实测 2 条文本 2.4 秒）。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "bge-m3",
        *,
        timeout: float = 180.0,
        probe_timeout: float = 3.0,
        batch_size: int = 16,
    ):
        self.base_url = base_url.rstrip("/")
        # 容忍 OLLAMA_URL=localhost:11434 这种漏写协议的写法
        if not self.base_url.startswith(("http://", "https://")):
            self.base_url = "http://" + self.base_url
        self.model = model
        self.timeout = timeout
        self.probe_timeout = probe_timeout
        self.batch_size = batch_size
        self.dim = 0
        self._reason = ""
        self._load()

    def _load(self) -> None:
        """探活并确认模型已拉取。

        刻意用短超时：本机没跑 Ollama 是常态，不能因此卡住启动。
        """
        try:
            import httpx
        except ImportError:  # pragma: no cover
            self._reason = "未安装 httpx"
            return

        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=self.probe_timeout)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            self._reason = f"Ollama 未响应（{self.base_url}）：{type(exc).__name__}"
            log.info("Ollama 嵌入不可用：%s", self._reason)
            return

        names = [m.get("name", "") for m in (resp.json().get("models") or [])]
        # 用户写 "bge-m3"，Ollama 里存的是 "bge-m3:latest"，两者都要认
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            self._reason = (
                f"Ollama 里没有模型 {self.model}（现有：{', '.join(names) or '无'}）。"
                f"请先执行：ollama pull {self.model}"
            )
            log.info("Ollama 嵌入不可用：%s", self._reason)
            return

        self._reason = ""
        log.info("已接入 Ollama 嵌入模型 %s（%s）", self.model, self.base_url)

    @property
    def available(self) -> bool:
        return not self._reason

    @property
    def reason(self) -> str:
        """不可用原因，供 check 命令原样展示。"""
        return self._reason

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not self.available:
            raise RuntimeError(f"Ollama 嵌入不可用：{self._reason}")

        import httpx

        size = batch_size or self.batch_size
        chunks: list[np.ndarray] = []
        with httpx.Client(timeout=self.timeout) as client:
            for i in range(0, len(texts), size):
                batch = [t or " " for t in texts[i : i + size]]
                resp = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": batch},
                )
                resp.raise_for_status()
                vecs = np.asarray(resp.json().get("embeddings") or [], dtype=np.float32)
                if vecs.ndim != 2 or len(vecs) != len(batch):
                    raise RuntimeError(
                        f"Ollama 返回的向量形状异常：{vecs.shape}（期望 ({len(batch)}, d)）"
                    )
                # 与 ONNX 路径保持一致：L2 归一化，让余弦相似度可直接点积
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                chunks.append(vecs / np.clip(norms, 1e-6, None))

        if not chunks:
            return np.zeros((0, self.dim or 1024), dtype=np.float32)
        out = np.vstack(chunks)
        self.dim = out.shape[1]
        return out


class Reranker:
    """bge-reranker-v2-m3 ONNX 交叉编码器。"""

    def __init__(self, model_dir: Path, *, max_len: int = 512):
        self.model_dir = Path(model_dir)
        self.max_len = max_len
        self._session = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        model_path = self.model_dir / "model.onnx"
        tok_path = self.model_dir / "tokenizer.json"
        if not model_path.exists() or not tok_path.exists():
            log.info("未找到本地重排模型（%s），将不做重排", self.model_dir)
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self._tokenizer = Tokenizer.from_file(str(tok_path))
            self._tokenizer.enable_truncation(max_length=self.max_len)
            log.info("已加载本地重排模型 %s", self.model_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("加载重排模型失败（%s）", exc)

    @property
    def available(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not self.available:
            raise RuntimeError("重排模型不可用")
        assert self._session is not None and self._tokenizer is not None

        scores: list[float] = []
        for doc in docs:
            enc = self._tokenizer.encode(query, doc)
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            inputs = {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": np.zeros_like(ids),
            }
            wanted = {i.name for i in self._session.get_inputs()}
            inputs = {k: v for k, v in inputs.items() if k in wanted}
            logits = self._session.run(None, inputs)[0]
            scores.append(float(np.asarray(logits).reshape(-1)[0]))
        return scores


def build_embedder(cfg: "Config"):
    """按配置挑选嵌入后端。

    优先级（`retrieve.embed_backend` 可强制指定某一种）：
      auto（默认）→ ONNX 权重存在就用 ONNX（最快、无外部依赖）；
                    否则若本机 Ollama 已拉取该模型，就用 Ollama；
                    都不行则返回不可用的占位对象，检索自动退化为关键词。
      onnx / ollama → 只用指定后端。
      none → 明确关闭向量召回。

    之所以把选择逻辑收在一处：调用点有三处（CLI / 服务端 / 评估），
    任何一处漏传都会让向量召回静默失效——这类 bug 不报错，只是悄悄退化，
    之前已经栽过一次。
    """
    rcfg = cfg.retrieve
    backend = getattr(rcfg, "embed_backend", "auto")

    if backend in ("auto", "onnx"):
        emb = Embedder(cfg.embed_model_dir)
        if emb.available or backend == "onnx":
            return emb
        log.info("ONNX 嵌入权重不存在，尝试改用 Ollama")

    if backend in ("auto", "ollama"):
        return OllamaEmbedder(
            getattr(rcfg, "ollama_url", "http://127.0.0.1:11434"),
            getattr(rcfg, "ollama_embed_model", "bge-m3"),
        )

    return Embedder(cfg.embed_model_dir)  # backend == "none"：必然不可用


def build_local_models(cfg: "Config"):
    """按配置构造本地嵌入与重排模型。

    统一入口，避免调用方各自推导模型路径——之前路径在三处重复，
    其中两处漏传给了检索器，等于白算向量。
    两者都可能不可用（available=False），调用方无需判断，检索会自动退化。
    """
    return (
        build_embedder(cfg),
        Reranker(cfg.rerank_model_dir),
    )
