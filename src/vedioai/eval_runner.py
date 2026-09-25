"""评估集运行器。

自动打分两项：要点命中率、引用正确性（规则见 SCORING.md）。
输出落到 evals/runs/<时间戳>/：
    answers.jsonl   每题原始回答，便于人眼复查
    summary.json    机器可读分数，用于跨版本对比
    report.md       可读结论

关键用法：
    vedioai eval evals/questions.yaml                 # 跑全部题
    vedioai eval evals/questions.yaml --only visual   # 只跑某一类
    vedioai eval --scaffold <video_id>                # 生成带真实章节的题目骨架
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from vedioai.ask import AskService
from vedioai.embedding import build_local_models
from vedioai.llm.client import LLMClient, from_llm_config
from vedioai.notes import NotesService
from vedioai.schema import hms_to_ms, ms_to_hms
from vedioai.store import Store

log = logging.getLogger(__name__)

_NORM_RE = re.compile(r"[\s，。、；：？！,.;:?!\"'“”‘’()（）\[\]【】《》<>·—\-_/\\|]+")
_PUNCT_KEEP = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff^+\-=<>()]+")


def normalize(text: str) -> str:
    """归一化：去掉标点与空白，统一小写。中文技术术语常有多余空格与标点。"""
    text = (text or "").lower()
    text = _NORM_RE.sub("", text)
    return _PUNCT_KEEP.sub("", text)


@dataclass
class KeypointResult:
    total: int = 0
    hit: int = 0
    missing: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.hit / self.total if self.total else 0.0


def score_keypoints(answer: str, keypoints: list) -> KeypointResult:
    """要点命中率。

    每个要点可以是：
      - 字符串：答案里出现该串即命中
      - 数组：命中其中任一写法即命中（技术名词写法多样，不要因此判错）
    """
    result = KeypointResult()
    norm_answer = normalize(answer)
    for item in keypoints or []:
        alternatives = item if isinstance(item, list) else [item]
        alternatives = [str(a) for a in alternatives if str(a).strip()]
        if not alternatives:
            continue
        result.total += 1
        if any(normalize(alt) in norm_answer for alt in alternatives):
            result.hit += 1
        else:
            result.missing.append(alternatives[0])
    return result


@dataclass
class CitationResult:
    status: str = "n/a"  # ok | wrong | missing | n/a
    score: float = 1.0

    def to_dict(self) -> dict:
        return {"status": self.status, "score": self.score}


def score_citation(
    citations: list[dict], expect_time: str, tolerance_ms: int
) -> CitationResult:
    if not expect_time:
        return CitationResult(status="n/a", score=1.0)
    if not citations:
        # 没引用不算错，但也不算对——单独统计，避免被平均值掩盖
        return CitationResult(status="missing", score=0.5)
    expect_ms = hms_to_ms(expect_time)
    for cite in citations:
        if abs(int(cite.get("start_ms", -10**9)) - expect_ms) <= tolerance_ms:
            return CitationResult(status="ok", score=1.0)
    # 有引用但都指错位置，比不引用更糟：会误导用户
    return CitationResult(status="wrong", score=0.0)


@dataclass
class QuestionOutcome:
    qid: str
    qtype: str
    question: str
    score: float
    keypoints: KeypointResult
    citation: CitationResult
    answer: str
    intent: str = ""
    cached_tokens: int = 0
    prompt_tokens: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.qid,
            "type": self.qtype,
            "question": self.question,
            "score": round(self.score, 4),
            "keypoint_hit": round(self.keypoints.rate, 4),
            "keypoints_missing": self.keypoints.missing,
            "citation": self.citation.to_dict(),
            "intent": self.intent,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "answer": self.answer,
            "error": self.error,
        }


def run_questions(
    questions: list[dict],
    service: AskService,
    video_id: str,
    *,
    only: str | None = None,
    progress=None,
) -> list[QuestionOutcome]:
    outcomes: list[QuestionOutcome] = []
    active = [
        q
        for q in questions
        if (q.get("question") or "").strip() and (not only or q.get("type") == only)
    ]
    for i, q in enumerate(active, start=1):
        if progress:
            progress(i, len(active), f"{q.get('id')} {q.get('question', '')[:40]}")
        qtype = q.get("type") or "factual"
        try:
            answer = service.ask(video_id, q["question"])
            kp = score_keypoints(answer.text, q.get("expect_keypoints") or [])
            cite = score_citation(
                [c.to_dict() for c in answer.citations],
                q.get("expect_time") or "",
                int(q.get("tolerance_ms") or 90_000),
            )
            has_time = bool(q.get("expect_time"))
            score = (0.7 * kp.rate + 0.3 * cite.score) if has_time else kp.rate
            outcomes.append(
                QuestionOutcome(
                    qid=q.get("id", f"q{i}"),
                    qtype=qtype,
                    question=q["question"],
                    score=score,
                    keypoints=kp,
                    citation=cite,
                    answer=answer.text,
                    intent=answer.intent,
                    prompt_tokens=answer.usage.prompt_tokens,
                    cached_tokens=answer.usage.cached_tokens,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("题目 %s 失败：%s", q.get("id"), exc)
            outcomes.append(
                QuestionOutcome(
                    qid=q.get("id", f"q{i}"),
                    qtype=qtype,
                    question=q["question"],
                    score=0.0,
                    keypoints=KeypointResult(),
                    citation=CitationResult(),
                    answer="",
                    error=str(exc),
                )
            )
    return outcomes


def summarize(outcomes: list[QuestionOutcome], notes_hit: float | None) -> dict:
    by_type: dict[str, list[float]] = {}
    for o in outcomes:
        by_type.setdefault(o.qtype, []).append(o.score)

    with_time = [o for o in outcomes if o.citation.status != "n/a"]
    summary = {
        "count": len(outcomes),
        "overall": round(statistics.fmean(o.score for o in outcomes), 4) if outcomes else 0.0,
        "by_type": {
            t: round(statistics.fmean(scores), 4) for t, scores in sorted(by_type.items())
        },
        "no_citation_rate": round(
            sum(1 for o in with_time if o.citation.status == "missing") / len(with_time), 4
        )
        if with_time
        else 0.0,
        "wrong_citation_rate": round(
            sum(1 for o in with_time if o.citation.status == "wrong") / len(with_time), 4
        )
        if with_time
        else 0.0,
        "failures": [o.qid for o in outcomes if o.error],
        "gate_pass": {},
    }
    if notes_hit is not None:
        summary["notes_hit"] = round(notes_hit, 4)

    gates = {
        "factual": 0.80,
        "cross": 0.65,
        "global": 0.65,
        "visual": 0.60,
        "overall": 0.70,
    }
    for key, threshold in gates.items():
        value = summary["by_type"].get(key) if key != "overall" else summary["overall"]
        if value is not None:
            summary["gate_pass"][key] = {"value": value, "threshold": threshold, "pass": value >= threshold}
    summary["gate_pass"]["wrong_citation_rate"] = {
        "value": summary["wrong_citation_rate"],
        "threshold": 0.05,
        "pass": summary["wrong_citation_rate"] <= 0.05,
    }
    return summary


def render_report(summary: dict, outcomes: list[QuestionOutcome], cfg=None) -> str:
    lines = [
        "# 评估报告",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
    ]
    if summary.get("invalid_run"):
        # 放在最顶部：报告一旦生成就会被引用，不能让它看起来像一次有效测量
        lines += [
            "> ⚠️ **本次运行无效，分数不可用于对比。**",
            f"> {summary.get('invalid_reason', '')}",
            "",
        ]
    lines += [
        "## 总览",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 题量 | {summary['count']} |",
        f"| 总分 | {summary['overall']} |",
    ]
    for qtype, value in summary["by_type"].items():
        lines.append(f"| {qtype} | {value} |")
    if "notes_hit" in summary:
        lines.append(f"| 笔记要点命中 | {summary['notes_hit']} |")
    lines += [
        f"| 未引用比例 | {summary['no_citation_rate']} |",
        f"| 引用错位比例 | {summary['wrong_citation_rate']} |",
        "",
        "## 门槛检查",
        "",
        "| 项目 | 实测 | 门槛 | 结论 |",
        "|---|---|---|---|",
    ]
    for name, gate in summary["gate_pass"].items():
        mark = "通过" if gate["pass"] else "**未通过**"
        lines.append(f"| {name} | {gate['value']} | {gate['threshold']} | {mark} |")

    weak = [o for o in outcomes if o.score < 0.5]
    if weak:
        lines += ["", "## 低分题（分数 < 0.5）", ""]
        for o in weak:
            lines.append(f"### {o.qid} `{o.qtype}` · {o.score:.2f}")
            lines.append(f"- 问题：{o.question}")
            if o.keypoints.missing:
                lines.append(f"- 未命中要点：{'、'.join(o.keypoints.missing)}")
            lines.append(f"- 引用：{o.citation.status}")
            if o.error:
                lines.append(f"- 错误：{o.error}")
            lines.append(f"- 回答节选：{o.answer[:300].replace(chr(10), ' ')}")
            lines.append("")

    failures = [o for o in outcomes if o.error]
    if failures:
        lines += ["", "## 调用失败", ""]
        for o in failures:
            lines.append(f"- {o.qid}: {o.error}")

    lines += ["", "## 逐题明细", "", "| 题号 | 类型 | 分数 | 要点命中 | 引用 |", "|---|---|---|---|---|"]
    for o in outcomes:
        lines.append(
            f"| {o.qid} | {o.qtype} | {o.score:.2f} | {o.keypoints.hit}/{o.keypoints.total} | {o.citation.status} |"
        )
    return "\n".join(lines)


def main(args, cfg) -> int:
    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"找不到评估集：{questions_path}")
        return 1

    data = yaml.safe_load(questions_path.read_text(encoding="utf-8")) or {}
    course = data.get("course") or {}
    video_id = course.get("video_id") or ""
    if not video_id:
        print("评估集里没有填 course.video_id。先跑 `vedioai list` 查 video_id。")
        return 1

    store = Store(cfg.db_path)
    if store.get_video(video_id) is None:
        print(f"课程库中没有 {video_id}")
        return 1

    only = getattr(args, "only", None)

    # 先确认真的有题可跑，再去要密钥：题目没填时给出的是「去填题」，
    # 而不是让人以为缺密钥
    active = [
        q
        for q in (data.get("questions") or [])
        if (q.get("question") or "").strip() and (not only or q.get("type") == only)
    ]
    total_slots = len(data.get("questions") or [])
    if not active:
        scope = f"类型 {only} 下" if only else "整份评估集里"
        print(f"{scope}没有任何已填写的题目（共 {total_slots} 个空位）。")
        print("先出题：一边看视频一边填 question / expect_keypoints / expect_time。")
        print("可以跑 `vedioai eval --scaffold <video_id>` 生成带真实章节的骨架。")
        print("评分标准见 evals/SCORING.md。")
        return 1

    # 评估必须可复现，所以温度取 0；同时关闭思考（思考模式下服务端会忽略
    # temperature，评估分数会飘，无法用同一套题做前后对比）。
    llm = from_llm_config(cfg.llm, temperature=0.0)
    llm.thinking = False
    vision = None
    if cfg.vision.api_key:
        vision = LLMClient(cfg.vision.api_key, cfg.vision.base_url, cfg.vision.model)
    _embedder, _reranker = build_local_models(cfg)
    service = AskService(
        cfg, store, llm, vision, embedder=_embedder, reranker=_reranker
    )

    def progress(i, total, message):
        print(f"\r[{i}/{total}] {message[:70]:<70}", end="", flush=True)

    print(f"跑评估集：{course.get('title') or video_id}（{questions_path.name}）")
    print(f"共 {len(active)} 题（空位 {total_slots - len(active)} 个已跳过）")
    outcomes = run_questions(
        data.get("questions") or [], service, video_id, only=only, progress=progress
    )
    print()

    notes_hit = None
    expected_notes = data.get("notes_expected") or []
    if expected_notes and not only:
        print("生成学习文档并评估…")
        notes_service = NotesService(cfg, store, llm)
        try:
            result = notes_service.generate(video_id, save=True)
            notes_hit = score_keypoints(result.markdown, expected_notes).rate
            run_dir = Path(getattr(args, "out", "evals/runs"))
            run_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            print(f"文档生成失败：{exc}")

    summary = summarize(outcomes, notes_hit)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(getattr(args, "out", "evals/runs")) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answers.jsonl").write_text(
        "\n".join(json.dumps(o.to_dict(), ensure_ascii=False) for o in outcomes),
        encoding="utf-8",
    )

    # 大面积调用失败时，这次的分数毫无意义，不能当成一次有效测量。
    #
    # 真实事故：DeepSeek 余额耗尽（HTTP 402），40 题里 39 题报错，总分 0.025，
    # 但报告照常生成、照常打印。留下的 summary.json 与一份真实回退的测量在格式上
    # 完全一样——后来的人拿它做基线对比，会以为系统坏了，去排查根本不是原因的地方。
    #
    # 所以：仍然落盘（保留现场供排查），但把结论标成无效，让人一眼看出别用。
    errored = summary.get("failures") or []
    if outcomes and len(errored) >= max(1, len(outcomes) // 2):
        summary["invalid_run"] = True
        summary["invalid_reason"] = (
            f"{len(errored)}/{len(outcomes)} 题调用失败（多为额度/网络问题），本次分数不可用于对比"
        )
        log.warning("本次评估 %d/%d 题失败，标记为无效运行", len(errored), len(outcomes))

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_report(summary, outcomes), encoding="utf-8")

    print(f"\n总分 {summary['overall']}　by_type {summary['by_type']}")
    if summary.get("invalid_run"):
        print(f"⚠️  本次运行无效：{summary.get('invalid_reason', '')}")
    if notes_hit is not None:
        print(f"笔记要点命中 {summary['notes_hit']}")
    print(f"未引用 {summary['no_citation_rate']}　引用错位 {summary['wrong_citation_rate']}")
    failed = [k for k, v in summary["gate_pass"].items() if not v["pass"]]
    print(f"未过门槛：{failed if failed else '无'}")
    print(f"报告：{out_dir / 'report.md'}")

    llm.close()
    store.close()
    return 0


# --------------------------------------------------------------------- 骨架


def scaffold(video_id: str, cfg, out_path: Path) -> int:
    """按课程真实章节生成 40 题骨架，省掉手敲时间点的工作。"""
    store = Store(cfg.db_path)
    video = store.get_video(video_id)
    if video is None:
        print(f"课程库中没有 {video_id}")
        return 1
    chapters = store.get_chapters(video_id)
    chunks = store.get_chunks(video_id)

    picks = [_pick_marker(chapters, i) for i in range(10)]
    chapter_hint = "\n".join(
        f"#   {ch.idx + 1:>2}. [{ms_to_hms(ch.start_ms)}] {ch.title or '（无标题）'}"
        for ch in chapters
    )
    chunk_hint = "\n".join(
        f"#   [{ms_to_hms(c.start_ms)}] {c.title or c.text[:40].replace(chr(10), ' ')}"
        for c in chunks[:40]
    )

    template = f"""# 自动生成的评估集骨架
# 课程：{video.title}（{video_id}）
# 时长：{ms_to_hms(video.duration_ms)}
#
# 章节分布：
{chapter_hint}
#
# 前若干片段（供挑题目落点）：
{chunk_hint}
#
# 请把每题填满：question / expect_keypoints / expect_time
# expect_keypoints 用课程里出现的具体名词与数字，不要写「讲得清楚」这类无法判定的描述。
# 评分标准见 evals/SCORING.md

version: 1

course:
  video_id: "{video_id}"
  title: "{video.title}"
  notes: ""

notes_expected: []

questions:
"""
    for qtype, prefix, tol in (
        ("factual", "f", 90000),
        ("cross", "c", 300000),
        ("visual", "v", 120000),
        ("global", "g", 600000),
    ):
        template += f"  # ===== {qtype} ×10\n"
        for i in range(1, 11):
            marker = picks[i - 1] if qtype in ("factual", "visual") else ""
            template += (
                f'  - {{ id: {prefix}{i:02d}, type: {qtype}, question: "", '
                f'expect_keypoints: [], expect_time: "{marker}", tolerance_ms: {tol} }}\n'
            )
        template += "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(template, encoding="utf-8")
    print(f"已生成骨架：{out_path}")
    print(f"章节 {len(chapters)} 个，片段 {len(chunks)} 个，已预填 10 个常见落点时间。")
    store.close()
    return 0


def _pick_marker(chapters, index: int) -> str:
    """均匀取章节起点作为题目落点提示。"""
    if not chapters:
        return ""
    step = max(1, len(chapters) // 10)
    pick = chapters[min(index * step, len(chapters) - 1)]
    return ms_to_hms(pick.start_ms)
