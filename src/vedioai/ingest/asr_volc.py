"""火山引擎「录音文件识别」客户端。

接口选择（这里很容易踩坑，务必看清）：

- **标准版** `volc.seedasr.auc`（0.8 元/小时）：只接受 `audio.url`（公网可访问链接），
  必须 submit + query 轮询。想用它处理本地文件，得先上传到 TOS 拿公网 URL。
- **极速版** `volc.bigasr.auc_turbo`：`/recognize/flash` 接口，支持 `audio.data`
  （本地文件 base64），**同步直接返回结果，无需轮询**。
- **闲时版** `volc.bigasr.auc_idle`：更便宜，但同样只支持 url。

因为本项目的主场景是**本地课程视频**，默认走极速版 + base64，开箱即用、无需 TOS。
如果你已经有对象存储，把 resource_id 换成标准版并传入 url 即可。

注意：不要把这类接口和「音频理解 / 多模态音频通读」混为一谈——后者有时长上限，
而且时间戳是模型生成的，不是对齐出来的。

## 关于错误码的一个关键认知

`20000003`（`no valid speech in audio`）**不是错误**：它表示服务成功解析了请求，
只是音频里没检测到语音。静音段、纯音乐、无人说话都会出现它。

把它当错误处理会踩两个坑：
1. 自检时提交合成音频（永远不含语音）必然"失败"，看不出凭证其实是好的；
2. 真实课程里遇到静音段或纯音乐开场，整次入库会被误判为失败。

重试策略同理：**只在传输层异常和服务端瞬时故障上重试**。4xxxxxxx 是客户端错误
（参数非法、鉴权失败、资源未开通），重试一百次也一样，只会白烧时间和额度。
"""

from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..config import ASRConfig
from ..schema import Segment

FLASH_ENDPOINT = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
SUBMIT_ENDPOINT = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
QUERY_ENDPOINT = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"

SUCCESS_CODE = "20000000"
# 请求合法，但音频里没有语音。这是正常结果，不是失败。
NO_SPEECH_CODE = "20000003"
# 异步任务仍在处理中
PROCESSING_CODES = frozenset({"20000001", "20000002"})

# 只在传输层与服务端故障上重试
RETRYABLE_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3


class ASRError(RuntimeError):
    """转写失败。"""


class ASRFatalError(ASRError):
    """不可重试：请求参数、鉴权、资源开通等客户端问题。"""


class ASRTransientError(ASRError):
    """可重试：网络抖动、服务端 5xx、限流。"""


@dataclass
class Utterance:
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None
    words: list[dict] = field(default_factory=list)


@dataclass
class TranscribeResult:
    """一次转写的结果。

    `no_speech` 单独标出来，让上层能区分「没语音」和「转写为空但没报错」。
    """

    utterances: list[Utterance] = field(default_factory=list)
    no_speech: bool = False
    duration_ms: int = 0

    @property
    def text(self) -> str:
        return "".join(u.text for u in self.utterances)


def explain_failure(code: str, message: str) -> str:
    """把服务端返回的错误码/消息翻译成能直接照做的指引。

    刻意只按**消息关键字**给提示，不硬编码一张码表——码表我无法逐条核实，
    而消息文本是服务端自己给的，更可靠。
    """
    detail = f"{code} {message}".strip()
    low = detail.lower()

    hints: list[tuple[tuple[str, ...], str]] = [
        (("error params", "invalid param", "param"),
         "请求参数不合法。若是首次配置，先确认 VOLC_ASR_RESOURCE_ID 与实际开通的服务一致。"),
        (("no permission", "not granted", "unauthorized", "auth", "forbidden"),
         "无权限或鉴权失败：检查 API Key / Access Token 是否正确、是否已过期、"
         "以及该服务是否已在控制台开通。"),
        (("quota", "exceed", "limit", "qps"),
         "超出配额或并发限制，稍后重试即可；批量入库时请降低并发。"),
        (("not found", "no such"),
         "资源不存在：确认 VOLC_ASR_RESOURCE_ID 是否拼写正确。"),
    ]
    for keywords, hint in hints:
        if any(k in low for k in keywords):
            return f"{detail}\n     → {hint}"
    return detail


class VolcASRClient:
    def __init__(
        self,
        cfg: ASRConfig,
        timeout: float = 600.0,
        client: httpx.Client | None = None,
    ):
        if not cfg.ready:
            raise ASRError(
                "火山 ASR 凭证未配置。请在 .env 中设置 VOLC_SPEECH_API_KEY"
                "（或老版控制台的 VOLC_SPEECH_APP_ID + VOLC_SPEECH_ACCESS_TOKEN）。"
            )
        self.cfg = cfg
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    # ---------------------------------------------------------------- public

    @property
    def is_flash(self) -> bool:
        return "flash" in self.cfg.resource_id or self.cfg.resource_id.endswith("auc_turbo")

    def transcribe_file(
        self, audio_path: Path, audio_url: str | None = None
    ) -> TranscribeResult:
        """转写单个音频文件，返回 utterances（毫秒，相对该文件起点）。

        传 audio_url 走标准版/闲时版的 URL 模式；否则走极速版 base64。
        """
        if audio_url:
            return self._transcribe_via_url(audio_url)
        return self._transcribe_via_base64(audio_path)

    def transcribe_spans(self, spans: list[tuple[int, int]], *, slice_fn) -> list[Segment]:
        """按 (start_ms, end_ms) 区间分批转写，并把时间戳按偏移还原成全片绝对时间。

        slice_fn(start_ms, end_ms, index) -> Path 由调用方提供，避免本模块依赖 FFmpeg。
        """
        segments: list[Segment] = []
        idx = 0
        for i, (start, end) in enumerate(spans):
            part = slice_fn(start, end, i)
            result = self.transcribe_file(part)
            for utt in result.utterances:
                # 关键：每段的相对时间戳 + 该段在全片中的偏移
                abs_start = start + utt.start_ms
                abs_end = start + utt.end_ms
                # 分段之间有 overlap，去重：丢掉完全落在上一段已覆盖区域内的句子
                if segments and abs_end <= segments[-1].end_ms:
                    continue
                words = [
                    {
                        "start_ms": start + int(w.get("start_time", 0)),
                        "end_ms": start + int(w.get("end_time", 0)),
                        "text": w.get("text", ""),
                        "confidence": w.get("confidence"),
                    }
                    for w in utt.words
                ]
                segments.append(
                    Segment(
                        idx=idx,
                        start_ms=abs_start,
                        end_ms=abs_end,
                        text=utt.text,
                        speaker=utt.speaker,
                        words=words,
                    )
                )
                idx += 1
        return segments

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # --------------------------------------------------------------- private

    def _headers(self, request_id: str) -> dict[str, str]:
        headers = {
            "X-Api-Resource-Id": self.cfg.resource_id,
            "X-Api-Request-Id": request_id,
            "X-Api-Sequence": "-1",
            "Content-Type": "application/json",
        }
        if self.cfg.api_key:
            # 新版控制台只需要 X-Api-Key
            headers["X-Api-Key"] = self.cfg.api_key
        else:
            headers["X-Api-App-Key"] = self.cfg.app_id
            headers["X-Api-Access-Key"] = self.cfg.access_token
        return headers

    def _request_body(self, audio: dict) -> dict:
        return {
            "user": {"uid": self.cfg.app_id or "vedioai"},
            "audio": audio,
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                # 语义顺滑会把口语顺成书面语，转写阶段不要，避免改动词句影响引用
                "enable_ddc": False,
                "show_utterances": True,
                "enable_speaker_info": False,
            },
        }

    def _transcribe_via_base64(self, audio_path: Path) -> TranscribeResult:
        size = audio_path.stat().st_size
        if size > self.cfg.max_upload_bytes:
            raise ASRFatalError(
                f"{audio_path.name} 体积 {size / 1e6:.1f}MB 超过单次上传上限 "
                f"{self.cfg.max_upload_bytes / 1e6:.1f}MB，应先切段"
            )
        payload = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        body = self._request_body({"data": payload, "format": "mp3"})

        for attempt in range(1, MAX_ATTEMPTS + 1):
            # 每次尝试用新的 request_id：重试是新的独立请求
            request_id = str(uuid.uuid4())
            try:
                resp = self._client.post(
                    FLASH_ENDPOINT, json=body, headers=self._headers(request_id)
                )
            except httpx.HTTPError as exc:
                if attempt == MAX_ATTEMPTS:
                    raise ASRTransientError(f"极速版转写失败（网络）：{exc}") from exc
                time.sleep(1.5 * attempt)
                continue

            kind, detail = self._categorize(resp)
            if kind == "ok":
                return self._parse(resp.json())
            if kind == "no_speech":
                # 正常结果：音频里没有语音
                data = _safe_json(resp)
                return TranscribeResult(no_speech=True, duration_ms=_duration_of(data))
            if kind == "transient" and attempt < MAX_ATTEMPTS:
                time.sleep(1.5 * attempt)
                continue
            if kind == "transient":
                raise ASRTransientError(f"极速版转写失败：{explain_failure(*detail)}")
            # fatal：重试无意义，立刻给出可照做的指引
            raise ASRFatalError(f"极速版转写失败：{explain_failure(*detail)}")

        raise ASRTransientError("极速版转写失败：重试次数已用尽")

    def _transcribe_via_url(self, audio_url: str) -> TranscribeResult:
        # 提交与查询必须共用同一个 request_id（它就是任务 ID）
        request_id = str(uuid.uuid4())
        body = self._request_body({"url": audio_url, "format": "mp3"})

        try:
            resp = self._client.post(
                SUBMIT_ENDPOINT, json=body, headers=self._headers(request_id)
            )
        except httpx.HTTPError as exc:
            raise ASRTransientError(f"提交转写任务失败（网络）：{exc}") from exc

        kind, detail = self._categorize(resp)
        if kind in ("fatal", "transient") and kind != "no_speech":
            raise (ASRFatalError if kind == "fatal" else ASRTransientError)(
                f"提交转写任务失败：{explain_failure(*detail)}"
            )

        deadline = time.monotonic() + self.cfg.poll_timeout_s
        while time.monotonic() < deadline:
            time.sleep(self.cfg.poll_interval_s)
            try:
                query = self._client.post(
                    QUERY_ENDPOINT, json={}, headers=self._headers(request_id)
                )
            except httpx.HTTPError:
                # 轮询期的网络抖动不该中断整个任务
                continue

            code = query.headers.get("X-Api-Status-Code", "")
            if code == SUCCESS_CODE:
                return self._parse(query.json())
            if code == NO_SPEECH_CODE:
                return TranscribeResult(no_speech=True, duration_ms=_duration_of(_safe_json(query)))
            if code in PROCESSING_CODES:
                continue
            qkind, qdetail = self._categorize(query)
            if qkind == "transient":
                continue
            raise ASRFatalError(f"转写任务失败：{explain_failure(*qdetail)}")
        raise ASRTransientError("转写轮询超时")

    @staticmethod
    def _categorize(resp: httpx.Response) -> tuple[str, tuple[str, str]]:
        """判定这次响应该怎么处理。

        返回 (kind, (code, message))，kind ∈ ok / no_speech / processing / transient / fatal。
        """
        code = resp.headers.get("X-Api-Status-Code", "")
        message = resp.headers.get("X-Api-Message", "")

        if code == SUCCESS_CODE:
            return "ok", (code, message)
        if code == NO_SPEECH_CODE:
            return "no_speech", (code, message)
        if code in PROCESSING_CODES:
            return "processing", (code, message)

        if resp.status_code in RETRYABLE_HTTP:
            return "transient", (code or f"HTTP {resp.status_code}", message or resp.text[:200])
        # 应用层 5xxxxxxx 视为服务端问题，可重试
        if code.startswith("5"):
            return "transient", (code, message)
        # 应用层 4xxxxxxx 是客户端问题，重试无意义
        if code.startswith("4"):
            return "fatal", (code, message)
        if resp.status_code >= 400:
            return "fatal", (f"HTTP {resp.status_code}", resp.text[:300])
        if not code:
            # HTTP 200 但没有状态码，无法判断，按可重试处理
            return "transient", ("HTTP 200 无 X-Api-Status-Code", resp.text[:200])
        return "fatal", (code, message)

    @staticmethod
    def _parse(data: dict) -> TranscribeResult:
        result = data.get("result") or data
        utterances = result.get("utterances") or []
        out: list[Utterance] = []
        for u in utterances:
            text = (u.get("text") or "").strip()
            if not text:
                continue
            additions = u.get("additions") or {}
            words = []
            for w in u.get("words") or []:
                start = int(w.get("start_time", -1))
                end = int(w.get("end_time", -1))
                # v3 结果里无效 token（常见为空格）时间戳是 -1，不过滤会制造假静音
                if start < 0 or end < 0:
                    continue
                words.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "text": w.get("text", ""),
                        "confidence": w.get("confidence"),
                    }
                )
            out.append(
                Utterance(
                    start_ms=int(u.get("start_time", 0)),
                    end_ms=int(u.get("end_time", 0)),
                    text=text,
                    speaker=additions.get("speaker"),
                    words=words,
                )
            )
        return TranscribeResult(
            utterances=out,
            no_speech=not out,
            duration_ms=_duration_of(data),
        )


def _safe_json(resp: httpx.Response) -> dict:
    try:
        return resp.json()
    except Exception:  # noqa: BLE001
        return {}


def _duration_of(data: dict) -> int:
    info = data.get("audio_info") or {}
    try:
        return int(info.get("duration") or 0)
    except (TypeError, ValueError):
        return 0
