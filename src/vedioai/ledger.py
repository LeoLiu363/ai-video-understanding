"""用量记账：记录每次 LLM / ASR 调用的 token 与估算花费。

**为什么要有这个模块**

在这之前，全项目没有任何地方记录「花了多少钱」：

- `summarize.py` 直接 `data, _ = client.chat_json(...)` 把 usage 丢掉了；
- `notes.py` 算了用量，但只塞进返回值，没落库；
- 数据库 19 张表里没有用量表。

于是「这门课一共花了多少钱」只能靠翻日志和评估报告倒推，而评估报告
只覆盖问答、不覆盖摘要与文档。这个模块把用量变成一等数据。

**设计上的两个关键选择**

1. **记账挂在 `LLMClient.chat()` 这唯一的出口上**，而不是各调用点自己上报。
   理由和 `from_llm_config` 一样：靠每个调用点自觉，漏一个就静默少记一笔，
   而且不会报错。挂出口等于「不可能漏」。

2. **花费按调用发生时的价目估算并存下来，而不是查询时才算。**
   官方经常调价（2026-09-10 就调过一次），历史记录应当反映当时的价钱，
   否则回看旧账会得出错误结论。token 原样保留，所以随时可以按新价重算。

**关于估价精度**：官方是**峰谷分时定价**（北京时间工作日 9:00–12:00、
14:00–18:00 为高峰，价格翻倍；其余时间含周末为闲时）。所以同一个调用
在上午和晚上跑，价钱差一倍。这里按实际发生时刻取价。

价目表查不到对应模型时，`estimate_cost` 返回 None，库里记为「未计价」，
界面会显示「有 N 次调用未计价」——**宁可显示「不知道」，也不要假装是 0**，
否则用户会以为总账是准的。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- 价目表

@dataclass(frozen=True)
class Price:
    """一元 / 百万 token 的单价。"""

    hit: float   # 输入：命中上下文缓存
    miss: float  # 输入：未命中
    out: float   # 输出


# 来源：DeepSeek 官方定价页（api-docs.deepseek.com/quick_start/pricing），
# 2026-09-10 12:00 起生效的 flash 系列调价。单位：元 / 百万 token。
#
# 注意 deepseek-v4-pro 官方以美元标价，这里按 1 USD ≈ 7.2 CNY 折算，
# 只作估算——要精确请用控制台账单。
PRICES: dict[str, dict[str, Price]] = {
    "deepseek-flash": {
        "off_peak": Price(hit=0.02, miss=1.0, out=4.0),
        "peak": Price(hit=0.04, miss=2.0, out=8.0),
    },
    "deepseek-v4-pro": {
        "off_peak": Price(hit=0.16, miss=4.75, out=14.26),
        "peak": Price(hit=0.32, miss=9.50, out=28.52),
    },
}

# 思考模式（思维链）按输出 token 计费，费用归到同一价目即可，无需单列。

BEIJING = timezone(timedelta(hours=8))


def is_peak(at: datetime | None = None) -> bool:
    """是否处于高峰计费时段。

    高峰 = 北京时间**工作日** 9:00–12:00 与 14:00–18:00；其余（含整个周末）
    按闲时计价。这里显式换算到北京时间，而不是用本机时区——本机时区一变，
    估价就会静默错一倍。
    """
    at = at or datetime.now(timezone.utc)
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    bj = at.astimezone(BEIJING)
    if bj.weekday() >= 5:  # 周六日
        return False
    hour = bj.hour
    return (9 <= hour < 12) or (14 <= hour < 18)


def estimate_cost(
    model: str,
    *,
    prompt_tokens: int = 0,
    cached_tokens: int = 0,
    completion_tokens: int = 0,
    at: datetime | None = None,
) -> float | None:
    """估算一次调用的花费（元）。返回 None = 该模型没有价目，无法计价。"""
    tiers = PRICES.get(model)
    if not tiers:
        return None
    price = tiers["peak" if is_peak(at) else "off_peak"]
    # 缓存命中数不应超过输入总数；真超了就说明上游字段有问题，按输入总数封顶，
    # 免得算出「命中比输入还多」这种负数未命中的账。
    cached = max(0, min(cached_tokens, prompt_tokens))
    miss = max(0, prompt_tokens - cached)
    return (
        cached * price.hit + miss * price.miss + completion_tokens * price.out
    ) / 1_000_000


# 录音文件识别按音频时长计费。标准版 0.8 元/小时；极速版单价需查控制台，
# 这里用标准版价作为保守估算（两者同档，且官方给 20 小时免费额度）。
ASR_YUAN_PER_HOUR = 0.8


def estimate_asr_cost(audio_ms: int) -> float:
    """按音频时长估算转写花费（元）。"""
    return audio_ms / 3_600_000 * ASR_YUAN_PER_HOUR


# ------------------------------------------------------- 调用用途（用于归类）

# 记账需要知道「这次调用是为了什么」——问答、摘要、还是写文档。用 contextvar
# 而不是给 chat() 加参数：加参数要改 ask / ask_with_images / chat_json 四个
# 方法及全部调用点，而且以后新增调用点还会漏。
#
# 必须注意：contextvar 不跨线程传递。所有 LLM 调用都发生在任务线程里，所以
# 作用域要在**该线程内**设置——因此 usage_scope 包在 Service 的入口
# （AskService.ask / NotesService.generate 等）内，而不是包在提交任务的
# 外层，否则子线程读到的是默认值。
_kind: ContextVar[str] = ContextVar("usage_kind", default="llm")
_video: ContextVar[str] = ContextVar("usage_video_id", default="")


@contextmanager
def usage_scope(kind: str, video_id: str = "") -> Iterator[None]:
    """标注这段代码里的 LLM 调用属于哪门课、用来做什么。"""
    k = _kind.set(kind)
    v = _video.set(video_id)
    try:
        yield
    finally:
        _kind.reset(k)
        _video.reset(v)


def current_kind() -> str:
    return _kind.get()


def current_video_id() -> str:
    return _video.get()


# ------------------------------------------------------------------- 记账回调

def recorder(store) -> Callable[[object], None]:
    """构造记账回调，挂到 ``LLMClient.on_usage`` 上。

    挂在客户端的**唯一出口**上，所以 ask / summarize / notes / glossary 全部
    自动被记，新增调用点也不会漏记。
    """

    def _record(reply) -> None:
        usage = getattr(reply, "usage", None)
        if usage is None:
            return
        model = getattr(reply, "model", "") or ""
        at = datetime.now(timezone.utc)
        cost = estimate_cost(
            model,
            prompt_tokens=usage.prompt_tokens,
            cached_tokens=usage.cached_tokens,
            completion_tokens=usage.completion_tokens,
            at=at,
        )
        if cost is None:
            log.debug("模型 %s 无价目，本次调用记为「未计价」", model)
        store.record_usage(
            kind=current_kind(),
            model=model,
            video_id=current_video_id(),
            prompt_tokens=usage.prompt_tokens,
            cached_tokens=usage.cached_tokens,
            completion_tokens=usage.completion_tokens,
            # None 会被原样存成 NULL，界面据此显示「未计价」而不是 ¥0
            cost_yuan=cost,
            peak=is_peak(at),
        )

    return _record


def attach(client, store):
    """给 LLM 客户端挂上记账。store 为 None 时原样返回（便于脚本/测试不带库）。"""
    if client is not None and store is not None:
        client.on_usage = recorder(store)
    return client
