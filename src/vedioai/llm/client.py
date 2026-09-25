"""OpenAI 兼容的 LLM 客户端（文本 + 视觉）。

DeepSeek 和火山方舟都是 OpenAI 兼容接口，因此共用一套客户端。

**缓存纪律（很重要）**：本项目的成本模型建立在「整稿作为稳定前缀命中上下文缓存」
之上。所以：
- 稳定内容（整份转写、章节摘要）必须放消息列表**最前**；
- 变化内容（用户问题、当前播放进度）放**最后**；
- 绝不把「当前播放到第几分钟」塞进前缀，否则缓存永远不命中，成本上升一个数量级。
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMTruncatedError(LLMError):
    """输出被 max_tokens 截断。

    对推理模型尤其容易发生：**思维链 token 也算进 max_tokens**。
    一旦思考过程吃光预算，content 会是空字符串、finish_reason 是 'length'。
    如果把它笼统报成「无法解析 JSON」，真正的原因就被藏起来了。
    """


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    def merge(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_tokens += other.cached_tokens
        self.total_tokens += other.total_tokens

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass
class Reply:
    text: str
    usage: Usage
    model: str = ""
    # 停止原因。'length' 表示被 max_tokens 截断——必须能看出来，
    # 否则会被误报成「无法解析模型返回的 JSON」。
    finish_reason: str = ""
    # 思维链长度（字符）。推理模型的思考占用 max_tokens 预算，需要可见。
    reasoning_chars: int = 0

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _read_image_data_url(path: Path, max_side: int = 1280) -> str:
    """把图片转成 data URL。过大的图先降采样，避免 token 浪费。"""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_side:
            ratio = max_side / max(im.size)
            im = im.resize((int(im.width * ratio), int(im.height * ratio)), Image.Resampling.LANCZOS)
        import io

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str):
    """从模型回复里稳健地取出 JSON（容忍 ```json 包裹和前后废话）。"""
    text = (text or "").strip()
    m = _JSON_BLOCK.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退化：抓第一个 { 到最后一个 } / [ 到最后一个 ]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"无法解析模型返回的 JSON：{text[:300]}")


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        timeout: float = 300.0,
        supports_thinking: bool = False,
        thinking: bool | None = None,
        on_usage: Callable[[Reply], None] | None = None,
    ):
        if not api_key:
            raise LLMError("LLM API Key 未配置")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # 只有 DeepSeek V4 这类支持 thinking 参数的才置 True；
        # 火山方舟（视觉）等不认识该字段，误发可能直接 400。
        self.supports_thinking = supports_thinking
        # 默认思考开关。None = 不发送该字段（用服务端默认）。
        self.thinking = thinking
        # 每次调用结束后的回调，用于记账。
        # 挂在这里（唯一出口）而不是各调用点：靠调用点自觉上报的话，
        # 漏一个就少记一笔账，而且不会报错——属于最难发现的错。
        self.on_usage = on_usage
        self._client = httpx.Client(timeout=timeout)

    # ---------------------------------------------------------------- public

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        thinking: bool | None = None,
    ) -> Reply:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        # 思考开关：默认开启的推理模型，其思维链按输出 token 计费，
        # 也占用 max_tokens。抽取类任务开思考只会更慢更贵，还容易把正文挤没。
        effective = self.thinking if thinking is None else thinking
        if effective is not None and self.supports_thinking:
            payload["thinking"] = {"type": "enabled" if effective else "disabled"}
        elif effective is not None:
            log.debug("模型 %s 不支持 thinking 参数，忽略该设置", payload["model"])

        return self._post(payload)

    def chat_json(
        self,
        messages: list[dict],
        *,
        thinking: bool | None = False,
        max_tokens: int | None = None,
        retries_on_truncation: int = 1,
        **kwargs,
    ):
        """结构化抽取。默认**关闭思考**，并在被截断时自动放大预算重试。

        关闭思考的理由（实测得出，不是臆测）：片段摘要这类任务在
        max_tokens=1024 时，推理过程会吃掉全部预算——实测失败率 50%，
        且失败样本的 finish_reason 全是 'length'、content 为空字符串。
        关掉思考后既没有截断，也省下了大量输出 token。

        另一个副作用：思考模式下 temperature 会被服务端忽略，
        而评估集依赖 temperature=0 保证可复现。关掉思考让温度重新生效。
        """
        budget = max_tokens or self.max_tokens
        last: LLMError | None = None

        for attempt in range(retries_on_truncation + 1):
            reply = self.chat(
                messages, json_mode=True, max_tokens=budget, thinking=thinking, **kwargs
            )
            if reply.truncated:
                last = LLMTruncatedError(
                    f"输出被 max_tokens 截断（budget={budget}，"
                    f"思维链 {reply.reasoning_chars} 字符，正文 {len(reply.text)} 字符）。"
                    "推理模型的思考也计入 max_tokens；可关闭思考或提高上限。"
                )
                # 预算翻倍再试。若模型仍在思考，第一次翻倍往往还不够
                budget *= 2
                continue
            return extract_json(reply.text), reply

        raise last  # type: ignore[misc]

    def ask(self, prefix: str, question: str, *, model: str | None = None) -> Reply:
        """稳定前缀 + 变化问题。前缀在前，问题在后，才能命中缓存。"""
        messages = [
            {"role": "system", "content": "你是一位严谨的课程助教，只依据提供的材料回答。"},
            {"role": "user", "content": prefix},
            {"role": "user", "content": question},
        ]
        return self.chat(messages, model=model)

    def ask_with_images(
        self,
        prefix: str,
        question: str,
        image_paths: list[Path],
        *,
        model: str | None = None,
    ) -> Reply:
        content: list[dict] = [{"type": "text", "text": f"{prefix}\n\n{question}"}]
        for path in image_paths:
            try:
                content.append({"type": "image_url", "image_url": {"url": _read_image_data_url(path)}})
            except Exception as exc:  # noqa: BLE001
                log.warning("读取图片失败 %s: %s", path, exc)
        messages = [
            {"role": "system", "content": "你是一位严谨的课程助教，只依据提供的材料和图片回答。"},
            {"role": "user", "content": content},
        ]
        return self.chat(messages, model=model)

    def close(self) -> None:
        self._client.close()

    # --------------------------------------------------------------- private

    def _post(self, payload: dict, retries: int = 3) -> Reply:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self._client.post(url, json=payload, headers=headers)
                if resp.status_code >= 400:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                reply = self._parse(data, payload["model"])
                self._notify_usage(reply)
                return reply
            except (httpx.HTTPError, LLMError) as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(2.0 * (attempt + 1))
        raise LLMError(f"调用失败（{payload['model']}）：{last_error}")

    def _notify_usage(self, reply: Reply) -> None:
        """上报一次成功调用的用量。

        记账是旁路：它失败绝不能让一次**已经计过费**的调用变成失败，
        否则用户不但花了钱，还拿不到结果，而且看不出真实原因。
        """
        if self.on_usage is None:
            return
        try:
            self.on_usage(reply)
        except Exception as exc:  # noqa: BLE001 — 旁路，任何异常都只记日志
            log.warning("用量上报失败（不影响本次调用）：%s", exc)

    @staticmethod
    def _parse(data: dict, model: str) -> Reply:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"返回中没有 choices：{str(data)[:300]}")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if isinstance(text, list):
            # 有些实现返回 content 数组
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))

        # 推理模型会另开一个 reasoning_content 字段，它是 content 为空时的关键线索
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""

        raw = data.get("usage") or {}
        cached = 0
        for key in ("prompt_cache_hit_tokens", "cached_tokens"):
            if key in raw:
                cached = int(raw.get(key) or 0)
                break
        if not cached and isinstance(raw.get("prompt_tokens_details"), dict):
            cached = int(raw["prompt_tokens_details"].get("cached_tokens") or 0)

        usage = Usage(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            cached_tokens=cached,
            total_tokens=int(raw.get("total_tokens") or 0),
        )
        return Reply(
            text=text.strip(),
            usage=usage,
            model=model,
            finish_reason=str(choice.get("finish_reason") or ""),
            reasoning_chars=len(reasoning),
        )


def from_llm_config(
    cfg,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMClient:
    """按配置构造文本模型客户端（DeepSeek）。

    集中在这里的理由：supports_thinking / thinking 这类参数如果在各调用点
    手写，漏传就会让配置静默失效——和「embedder 没传进检索器」是同一类
    bug：不报错，只是悄悄退化到默认行为。
    """
    return LLMClient(
        cfg.api_key,
        cfg.base_url,
        cfg.model,
        temperature=cfg.temperature if temperature is None else temperature,
        max_tokens=max_tokens or cfg.max_tokens,
        supports_thinking=True,
        thinking=getattr(cfg, "thinking", None),
    )
