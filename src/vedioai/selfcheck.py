"""凭证与依赖的实时自检。

为什么需要这个：最常见的失败不是「忘了填密钥」，而是「密钥填了但服务没开通」
或「资源 ID 填错」。这两种情况在真正入库到一半时才暴露，白等十几分钟。

`check --live` 会发最小的真实请求去验证：
- 火山 ASR：提交 1 秒静音，看是否返回成功码（同时验证 resource_id 是否开通）
- DeepSeek / 火山方舟：一次 max_tokens=1 的对话，验证 key 与模型名
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Config


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str

    @property
    def mark(self) -> str:
        return "OK  " if self.ok else "失败"


def check_asr(cfg: Config, ffmpeg: str) -> CheckResult:
    """验证 ASR 凭证。

    刻意提交一段**合成静音**：我们不需要它转写出内容，只需要服务走完
    「鉴权 → 资源校验 → 参数解析」这条路并回一个应用层状态码。
    合成音频永远不含语音，所以服务必然回 `20000003`（无语音）——
    这恰恰证明链路是通的。

    因此判定标准是：拿到 `20000003` 或 `20000000` 就算通过；
    `4xxxxxxx`、401、403、资源未开通才算失败。
    """
    if not cfg.asr.ready:
        return CheckResult("火山 ASR", False, "未配置密钥")

    from .ingest.asr_volc import ASRError, VolcASRClient

    with tempfile.TemporaryDirectory() as tmp:
        probe_audio = Path(tmp) / "silence.mp3"
        try:
            subprocess.run(
                [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                    "-t", "2", "-c:a", "libmp3lame", "-b:a", "32k", str(probe_audio),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001
            return CheckResult("火山 ASR", False, f"无法生成测试音频：{exc}")

        client = VolcASRClient(cfg.asr)
        try:
            result = client.transcribe_file(probe_audio)
            if result.no_speech:
                return CheckResult(
                    "火山 ASR",
                    True,
                    f"{cfg.asr.resource_id} 可用"
                    f"（鉴权、资源开通、参数解析均正常；静音样本按预期返回无语音）",
                )
            return CheckResult(
                "火山 ASR", True, f"{cfg.asr.resource_id} 可用（意外识别出了内容）"
            )
        except ASRError as exc:
            return CheckResult("火山 ASR", False, str(exc))
        except httpx.HTTPError as exc:
            return CheckResult("火山 ASR", False, f"网络异常：{exc}")
        finally:
            client.close()


def check_llm(cfg: Config) -> CheckResult:
    if not cfg.llm.api_key:
        return CheckResult("DeepSeek", False, "未配置 DEEPSEEK_API_KEY")
    return _probe_chat(
        "DeepSeek", cfg.llm.api_key, cfg.llm.base_url, cfg.llm.model
    )


def check_vision(cfg: Config) -> CheckResult:
    if not cfg.vision.api_key:
        return CheckResult("视觉模型", False, "未配置 ARK_API_KEY")
    return _probe_chat(
        "视觉模型", cfg.vision.api_key, cfg.vision.base_url, cfg.vision.model
    )


def _probe_chat(name: str, api_key: str, base_url: str, model: str) -> CheckResult:
    """一次最小对话：既验证 key，也验证模型名（模型名写错是最常见的第二种坑）。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
    except httpx.HTTPError as exc:
        return CheckResult(name, False, f"网络异常：{exc}")

    if resp.status_code == 200:
        return CheckResult(name, True, f"{model} 可用")
    if resp.status_code == 401:
        return CheckResult(name, False, "密钥无效（401）")
    if resp.status_code == 404:
        return CheckResult(name, False, f"模型名可能不存在：{model}（404）")
    if resp.status_code == 402:
        return CheckResult(name, False, "余额不足（402）")
    return CheckResult(name, False, f"HTTP {resp.status_code}：{resp.text[:200]}")


def check_embedding(cfg: Config) -> CheckResult | None:
    """真正编码一句话，验证本地嵌入后端可用。

    为什么不能只看 `/api/tags` 里有没有模型：模型「已拉取」和「能加载」是两回事。
    新版 Ollama 在旧显卡驱动上，模型名明明在列表里，一旦真正加载就崩
    （CUDA 报错）。只看列表会给出「可用」的错误结论，而实际入库时会白跑很久
    才发现向量根本没建起来。

    嵌入是可选组件，未启用时返回 None（不作为检查项，也不算失败）。
    """
    from .embedding import build_embedder

    try:
        embedder = build_embedder(cfg)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("本地嵌入", False, f"构造嵌入后端失败：{exc}")

    if not embedder.available:
        return None

    backend = type(embedder).__name__
    try:
        vecs = embedder.encode(["连通性测试"])
    except Exception as exc:  # noqa: BLE001
        return CheckResult("本地嵌入", False, f"{backend}：{exc}")

    if vecs.ndim != 2 or vecs.shape[0] != 1 or vecs.shape[1] < 1:
        return CheckResult("本地嵌入", False, f"{backend} 返回向量形状异常：{vecs.shape}")
    return CheckResult("本地嵌入", True, f"{backend} 可用（维度 {vecs.shape[1]}）")


def run_live_checks(cfg: Config) -> list[CheckResult]:
    candidates = [
        check_asr(cfg, cfg.media.ffmpeg),
        check_llm(cfg),
        check_vision(cfg),
        check_embedding(cfg),
    ]
    # None = 可选组件未启用，不作为检查项
    return [r for r in candidates if r is not None]
