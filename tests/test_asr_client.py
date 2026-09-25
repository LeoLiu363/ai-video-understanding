"""ASR 客户端的错误码分类与重试策略测试。

这一组测试对应一次真实事故：把「音频里没有语音」（`20000003`）当成硬错误，
导致 1) 自检永远失败、2) 课程含静音段时整次入库被误判失败、
3) 不可重试的参数错误被白重试 3 次。

全部用 httpx.MockTransport 模拟服务端，**不发真实请求**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from vedioai.config import ASRConfig  # noqa: E402
from vedioai.ingest.asr_volc import (  # noqa: E402
    NO_SPEECH_CODE,
    SUCCESS_CODE,
    ASRFatalError,
    ASRTransientError,
    VolcASRClient,
    explain_failure,
)


@pytest.fixture()
def audio_file(tmp_path: Path) -> Path:
    p = tmp_path / "a.mp3"
    p.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 400)
    return p


class Recorder:
    """记录每次请求，便于断言重试次数与请求 id 是否变化。"""

    def __init__(self, responses: list[httpx.Response]):
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        idx = min(len(self.requests) - 1, len(self.responses) - 1)
        return self.responses[idx]


def make_client(recorder: Recorder, cfg: ASRConfig | None = None) -> VolcASRClient:
    cfg = cfg or ASRConfig(api_key="test-key", resource_id="volc.bigasr.auc_turbo")
    transport = httpx.MockTransport(recorder)
    return VolcASRClient(cfg, client=httpx.Client(transport=transport))


def resp(status: int, code: str, message: str = "", body: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"X-Api-Status-Code": code, "X-Api-Message": message},
        json=body if body is not None else {},
    )


# --------------------------------------------------------------- 无语音


def test_no_speech_is_not_an_error(audio_file: Path):
    """静音音频返回 20000003 —— 正常结果，不该抛异常，也不该重试。"""
    rec = Recorder([resp(200, NO_SPEECH_CODE, "[Normal silence audio] no valid speech in audio",
                         {"audio_info": {"duration": 2124},
                          "result": {"additions": {"duration": "2124"}, "text": ""}})])
    client = make_client(rec)

    result = client.transcribe_file(audio_file)

    assert result.utterances == []
    assert result.no_speech is True
    assert result.duration_ms == 2124
    assert len(rec.requests) == 1, "无语音是正常结果，不应该重试"


def test_no_speech_true_when_result_empty_but_code_ok(audio_file: Path):
    """状态码成功但 utterances 为空，也应标记 no_speech。"""
    rec = Recorder([resp(200, SUCCESS_CODE, "OK", {"result": {"utterances": []}})])
    result = make_client(rec).transcribe_file(audio_file)
    assert result.utterances == []
    assert result.no_speech is True


# --------------------------------------------------------------- 成功


def test_success_parses_words_and_skips_invalid_timestamps(audio_file: Path):
    body = {
        "result": {
            "utterances": [
                {
                    "start_time": 480,
                    "end_time": 5880,
                    "text": "刚刚还在想",
                    "words": [
                        {"start_time": 480, "end_time": 600, "text": "刚", "confidence": 0.9},
                        # v3 里无效 token 的时间戳是 -1，必须丢掉，否则会制造假静音
                        {"start_time": -1, "end_time": -1, "text": " ", "confidence": None},
                        {"start_time": 680, "end_time": 800, "text": "才", "confidence": 0.8},
                    ],
                }
            ]
        }
    }
    rec = Recorder([resp(200, SUCCESS_CODE, "OK", body)])
    result = make_client(rec).transcribe_file(audio_file)

    assert result.no_speech is False
    assert len(result.utterances) == 1
    utt = result.utterances[0]
    assert utt.text == "刚刚还在想"
    assert [w["text"] for w in utt.words] == ["刚", "才"]


# --------------------------------------------------------- 重试策略


def test_param_error_is_fatal_and_not_retried(audio_file: Path):
    """4xxxxxxx 是客户端错误，重试没有意义 —— 必须只发一次请求。"""
    rec = Recorder([resp(200, "45000000", "code: 11500 message: error params")])
    client = make_client(rec)

    with pytest.raises(ASRFatalError) as exc:
        client.transcribe_file(audio_file)

    assert len(rec.requests) == 1, f"参数错误不该重试，实际发了 {len(rec.requests)} 次"
    assert "error params" in str(exc.value)
    assert "请求参数不合法" in str(exc.value), "应给出可照做的指引"


def test_auth_error_is_fatal(audio_file: Path):
    rec = Recorder([resp(403, "45000001", "auth failed")])
    with pytest.raises(ASRFatalError):
        make_client(rec).transcribe_file(audio_file)
    assert len(rec.requests) == 1


def test_server_error_is_retried_then_succeeds(audio_file: Path):
    """5xxxxxxx 是服务端瞬时故障，应重试并最终成功。"""
    rec = Recorder([
        resp(200, "50000000", "internal error"),
        resp(200, SUCCESS_CODE, "OK", {"result": {"utterances": [
            {"start_time": 0, "end_time": 1000, "text": "内容"}
        ]}}),
    ])
    result = make_client(rec).transcribe_file(audio_file)

    assert len(rec.requests) == 2, "服务端错误应重试"
    assert result.utterances[0].text == "内容"


def test_http_5xx_is_retried(audio_file: Path):
    rec = Recorder([
        httpx.Response(503, json={"error": "unavailable"}),
        resp(200, SUCCESS_CODE, "OK", {"result": {"utterances": [
            {"start_time": 0, "end_time": 1000, "text": "恢复"}
        ]}}),
    ])
    result = make_client(rec).transcribe_file(audio_file)
    assert len(rec.requests) == 2
    assert result.utterances[0].text == "恢复"


def test_retries_exhausted_raises_transient(audio_file: Path):
    rec = Recorder([resp(200, "50000000", "internal error")])
    with pytest.raises(ASRTransientError):
        make_client(rec).transcribe_file(audio_file)
    assert len(rec.requests) == 3, "服务端错误应重试满 3 次"


def test_requests_are_oversized_rejected_before_network(tmp_path: Path):
    cfg = ASRConfig(api_key="k", max_upload_bytes=10)
    big = tmp_path / "big.mp3"
    big.write_bytes(b"x" * 100)
    rec = Recorder([resp(200, SUCCESS_CODE, "OK")])

    with pytest.raises(ASRFatalError, match="超过单次上传上限"):
        make_client(rec, cfg).transcribe_file(big)
    assert not rec.requests, "体积校验应在发请求之前完成"


def test_url_mode_keeps_same_request_id_between_submit_and_query():
    """异步路径下 request_id 就是任务 ID，提交与查询必须一致。"""
    submitted: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        rid = request.headers["X-Api-Request-Id"]
        if "submit" in str(request.url):
            submitted["id"] = rid
            return resp(200, SUCCESS_CODE, "OK")
        # query
        assert rid == submitted["id"], "查询必须复用提交时的任务 ID"
        return resp(200, SUCCESS_CODE, "OK", {"result": {"utterances": [
            {"start_time": 0, "end_time": 500, "text": "好"}
        ]}})

    cfg = ASRConfig(api_key="k", resource_id="volc.seedasr.auc", poll_interval_s=0.01)
    client = VolcASRClient(cfg, client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = client.transcribe_file(Path("unused.mp3"), audio_url="https://example.com/a.mp3")

    assert result.utterances[0].text == "好"


# --------------------------------------------------------------- 指引


def test_explain_failure_gives_actionable_hints():
    assert "请求参数不合法" in explain_failure("45000000", "error params")
    assert "鉴权失败" in explain_failure("45000001", "unauthorized")
    assert "配额" in explain_failure("45000011", "qps exceed limit")
    # 认不出的错误至少要原样带出服务端信息，不要吞掉
    assert "45000999" in explain_failure("45000999", "unknown thing")


def test_missing_credentials_message_is_actionable():
    from vedioai.ingest.asr_volc import ASRError

    with pytest.raises(ASRError, match="VOLC_SPEECH_API_KEY"):
        VolcASRClient(ASRConfig())
