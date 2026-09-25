"""LLM 客户端的截断处理与思考开关。

对应一次真实事故：`deepseek-flash` 是推理模型，思维链 token 计入 max_tokens。
片段摘要给 1024 预算时实测失败率 50%，失败样本的 finish_reason 全是 'length'、
content 是空字符串。而代码把它报成「无法解析模型返回的 JSON」，
真正的原因（被截断）被完全藏住，整个入库因此失败。

全部用 MockTransport，不发真实请求。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vedioai.llm.client import (  # noqa: E402
    LLMClient,
    LLMTruncatedError,
    extract_json,
)


class Recorder:
    def __init__(self, responses: list[httpx.Response]):
        self.responses = responses
        self.payloads: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.payloads.append(json.loads(request.content))
        idx = min(len(self.payloads) - 1, len(self.responses) - 1)
        return self.responses[idx]

    @property
    def calls(self) -> int:
        return len(self.payloads)


def reply_body(content: str, finish: str = "stop", reasoning: str = "") -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def make_client(recorder: Recorder, **kwargs) -> LLMClient:
    kwargs.setdefault("supports_thinking", True)
    client = LLMClient("test-key", "https://api.example.com", "deepseek-flash", **kwargs)
    client._client = httpx.Client(transport=httpx.MockTransport(recorder))
    return client


MSGS = [{"role": "user", "content": "输出 JSON"}]


# ------------------------------------------------------------- 截断处理


def test_truncation_raises_actionable_error_not_json_error():
    """被截断必须报「截断」并给出可照做的指引，而不是「无法解析 JSON」。"""
    rec = Recorder([httpx.Response(200, json=reply_body("", "length", "很长的思考" * 50))])
    client = make_client(rec)

    with pytest.raises(LLMTruncatedError) as exc:
        client.chat_json(MSGS, max_tokens=1024)

    message = str(exc.value)
    assert "截断" in message
    assert "闭环" not in message and "思维链" in message
    assert "关闭思考" in message, "应指出可照做的解法"


def test_truncation_retries_with_doubled_budget():
    """第一次被截断后应放大预算重试，而不是直接失败。"""
    rec = Recorder([
        httpx.Response(200, json=reply_body("", "length", "x" * 100)),
        httpx.Response(200, json=reply_body('{"title": "定位技巧"}')),
    ])
    client = make_client(rec)

    data, reply = client.chat_json(MSGS, max_tokens=1024)

    assert data == {"title": "定位技巧"}
    assert rec.calls == 2, "应重试一次"
    assert rec.payloads[0]["max_tokens"] == 1024
    assert rec.payloads[1]["max_tokens"] == 2048, "预算应翻倍"


def test_truncation_gives_up_after_retries_and_reports_last_budget():
    rec = Recorder([httpx.Response(200, json=reply_body("", "length", "x" * 100))])
    client = make_client(rec)

    with pytest.raises(LLMTruncatedError) as exc:
        client.chat_json(MSGS, max_tokens=512, retries_on_truncation=2)

    assert rec.calls == 3, "1 次原始 + 2 次重试"
    assert "budget=2048" in str(exc.value), "错误里应带最终预算，便于判断该给多大"


def test_reply_truncated_flag_and_reasoning_length():
    rec = Recorder([httpx.Response(200, json=reply_body("abc", "length", "思考" * 10))])
    reply = make_client(rec).chat(MSGS)

    assert reply.truncated is True
    assert reply.reasoning_chars == 20
    assert reply.finish_reason == "length"


def test_partial_json_on_truncation_is_still_reported_as_truncation():
    """截断时可能输出了半个 JSON —— 也要报截断，不能报成 JSON 语法错误。"""
    rec = Recorder([httpx.Response(200, json=reply_body('{"title": "定位', "length", "x"))])
    with pytest.raises(LLMTruncatedError):
        make_client(rec).chat_json(MSGS, max_tokens=100)


# ------------------------------------------------------------- 思考开关


def test_chat_json_disables_thinking_by_default():
    """抽取类任务默认关思考：更便宜、更快，且避免了截断这一整类问题。"""
    rec = Recorder([httpx.Response(200, json=reply_body('{"ok": 1}'))])
    make_client(rec).chat_json(MSGS)

    assert rec.payloads[0]["thinking"] == {"type": "disabled"}


def test_chat_json_can_enable_thinking_explicitly():
    rec = Recorder([httpx.Response(200, json=reply_body('{"ok": 1}'))])
    make_client(rec).chat_json(MSGS, thinking=True)

    assert rec.payloads[0]["thinking"] == {"type": "enabled"}


def test_thinking_field_omitted_when_provider_does_not_support_it():
    """火山方舟等不认识 thinking 字段，误发可能直接 400。"""
    rec = Recorder([httpx.Response(200, json=reply_body('{"ok": 1}'))])
    client = make_client(rec, supports_thinking=False)

    client.chat_json(MSGS)

    assert "thinking" not in rec.payloads[0]


def test_client_level_thinking_default_is_respected():
    """问答走 chat()，思考开关来自配置。"""
    rec = Recorder([httpx.Response(200, json=reply_body("好的"))])
    client = make_client(rec, thinking=False)
    client.chat(MSGS)
    assert rec.payloads[0]["thinking"] == {"type": "disabled"}

    rec2 = Recorder([httpx.Response(200, json=reply_body("好的"))])
    client2 = make_client(rec2, thinking=True)
    client2.chat(MSGS)
    assert rec2.payloads[0]["thinking"] == {"type": "enabled"}


def test_no_thinking_field_when_unset():
    """未显式设置时不要发送该字段，交回服务端默认。"""
    rec = Recorder([httpx.Response(200, json=reply_body("好的"))])
    make_client(rec, thinking=None).chat(MSGS)
    assert "thinking" not in rec.payloads[0]


# --------------------------------------------------------------- 解析


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('好的，结果如下：\n{"a": 1}\n希望有帮助') == {"a": 1}
    assert extract_json('[1, 2, 3]') == [1, 2, 3]
