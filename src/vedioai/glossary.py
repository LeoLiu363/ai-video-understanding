"""术语表：转写纠错 + 可疑术语标注。

为什么需要这一层
----------------
ASR 会把专有名词听错。本课的真实例子：

    讲师说 uiautomator  →  转写成 "URL to meta" / "URL to meter"

这类错字会一路穿过切块、摘要、章节笔记，最后变成「读起来很像事实」的错误结论——
比明显乱码危险得多，因为用户不会去怀疑一个通顺的句子。

文档质量的天花板不在模型，而在转写里那几个被听错的名词。

本模块只做两件事
----------------
1. ``correct()``：按术语表把**已知**错听确定性替换成正确写法（不改语义）；
2. ``flag()``：找出「可疑但不确定」的术语，交给人工确认。

判断原则：宁可标注，也不擅自替换。改错了比不改更糟。所以
``vedioai.glossary.yaml`` 里刻意分成 ``corrections``（确定，自动替换）
与 ``suspects``（不确定，只报不改）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- 数据结构


@dataclass(frozen=True)
class Correction:
    """一条确定的错听 → 正确写法。``wrong`` 可写多种变体。"""

    wrong: tuple[str, ...]
    right: str
    reason: str = ""


@dataclass
class Fix:
    """一次实际发生的替换，用于记账与展示。"""

    wrong: str
    right: str
    reason: str
    count: int


@dataclass
class Finding:
    """一处「可疑但未替换」，需要人工确认。"""

    term: str
    kind: str
    hint: str
    count: int = 1


@dataclass
class Glossary:
    corrections: list[Correction] = field(default_factory=list)
    suspects: list[str] = field(default_factory=list)
    trusted: list[str] = field(default_factory=list)
    # 实际载入了哪几个文件（手写表 + 自动表），便于排查「规则为什么没生效」
    sources: list[Path] = field(default_factory=list)

    # ------------------------------------------------------------ 加载

    @classmethod
    def load(cls, path: Path | None, video_id: str = "") -> Glossary:
        """读术语表。文件不存在时返回空表（不报错，纠错层是可选增强）。

        会同时读取手写表与同目录下的 ``*.auto.yaml``（机器判定产物）。
        两者分开存放：手写那份的注释是策展成果，不能让机器覆写。
        """
        gl = cls()
        gl.trusted = list(_DEFAULT_TRUSTED)
        gl.sources = []

        for candidate in (path, _auto_path(path) if path is not None else None):
            if candidate is None or not Path(candidate).exists():
                continue
            gl.sources.append(Path(candidate))
            gl._merge_file(Path(candidate), video_id)

        # 长串优先替换，避免短串先命中把长串切碎
        gl.corrections.sort(key=lambda c: max(len(w) for w in c.wrong), reverse=True)
        return gl

    def _merge_file(self, path: Path, video_id: str) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        def add_corrections(items) -> None:
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                right = str(item.get("right") or "").strip()
                wrong = item.get("wrong") or []
                if isinstance(wrong, str):
                    wrong = [wrong]
                wrong = tuple(str(w).strip() for w in wrong if str(w).strip())
                if right and wrong:
                    self.corrections.append(
                        Correction(wrong=wrong, right=right, reason=str(item.get("reason") or ""))
                    )

        add_corrections(raw.get("corrections"))

        course = (raw.get("courses") or {}).get(video_id) or {}
        add_corrections(course.get("corrections"))

        self.suspects += [str(s) for s in (raw.get("suspects") or [])]
        self.suspects += [str(s) for s in (course.get("suspects") or [])]
        self.trusted += [str(s) for s in (raw.get("trusted") or [])]
        self.trusted += [str(s) for s in (course.get("trusted") or [])]

    # ------------------------------------------------------------ 纠错

    def correct(self, text: str) -> tuple[str, list[Fix]]:
        """返回（纠正后的文本, 实际发生的替换列表）。"""
        if not text:
            return text, []
        fixes: list[Fix] = []
        result = text
        for corr in self.corrections:
            total = 0
            for wrong in corr.wrong:
                n = result.count(wrong)
                if n:
                    result = result.replace(wrong, corr.right)
                    total += n
            if total:
                fixes.append(Fix(wrong=corr.wrong[0], right=corr.right, reason=corr.reason, count=total))
        return result, fixes

    def correct_many(self, texts: list[str]) -> tuple[list[str], list[Fix]]:
        """批量纠错，并把同名替换合并计数（便于汇总日志）。"""
        merged: dict[tuple[str, str], Fix] = {}
        out: list[str] = []
        for text in texts:
            fixed, fixes = self.correct(text)
            out.append(fixed)
            for f in fixes:
                key = (f.wrong, f.right)
                if key in merged:
                    merged[key].count += f.count
                else:
                    merged[key] = Fix(f.wrong, f.right, f.reason, f.count)
        return out, list(merged.values())

    # ------------------------------------------------------------ 标注

    def flag(self, text: str) -> list[Finding]:
        """找出可疑术语。只报不改。

        四类信号：
          spaced_letters  形如 "D Y L" 的空格字母序列——转写噪音的典型形态
          quoted_phrase   引号包住的英文短语，像是把听错的词当成了专有名词
          inconsistent    同一实体的两种拼写（jeb_wincon_ / jeb_winco_）
          listed          术语表 suspects 里显式列出的
        """
        if not text:
            return []
        findings: list[Finding] = []
        trusted = {t.lower() for t in self.trusted}

        # --- 1. 空格分隔的单字母序列
        for m in _SPACED_LETTERS.finditer(text):
            term = m.group(0)
            findings.append(
                Finding(term, "spaced_letters", "疑似转写噪音，请对照视频确认原词")
            )

        # --- 2. 引号里的英文短语
        for m in _QUOTED.finditer(text):
            term = m.group(1).strip()
            if term.lower() in trusted:
                continue
            words = re.findall(r"[A-Za-z]+", term)
            # 只关心「像自然语言短语」的：含空格 + 至少一个纯小写单词
            if " " in term and any(w.islower() for w in words):
                findings.append(
                    Finding(term, "quoted_phrase", "被当成专有名词引用，请确认是否为听错的词")
                )

        # --- 3. 同实体多种拼写
        findings.extend(_inconsistent_identifiers(text))

        # --- 4. 显式列出的suspect
        for s in self.suspects:
            n = text.count(s)
            if n:
                findings.append(Finding(s, "listed", "术语表标记为待确认", count=n))

        return _dedupe(findings)


# ------------------------------------------------------------------ 正则

# "D Y L two meter" 里的 "D Y L"；要求至少 3 个单字母，避免 "A B" 之类误报
_SPACED_LETTERS = re.compile(r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b")
_QUOTED = re.compile(r"[\"“]([A-Za-z][A-Za-z0-9 _./\\+-]{2,})[\"”]")
# 像文件名或下划线标识符
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[_.][A-Za-z0-9]+)+")

# 已确认无误的术语，避免误报
_DEFAULT_TRUSTED = (
    "uiautomator",
    "uiautomatorviewer",
    "DDMS",
    "ADB",
    "jadx-gui",
    "jeb",
    "Xposed",
    "EdXposed",
    "BuildProp Enhancer",
    "AES",
    "MD5",
    "Base64",
    "IV",
    "Cipher",
    "CLogUtils",
    "InfiniteLog",
    "AndroidManifest",
    "OkHttp",
    "Logcat",
    "SDK",
)


def _skeleton(token: str) -> str:
    """归一化标识符：去掉分隔符并小写，用于发现同类拼写。"""
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _spelling_variant(a: str, b: str) -> bool:
    """判断两个标识符是否「同一个词、两种拼写」。

    只在差异是**字母**时成立。数字不同不算拼写差异——``Hooks2`` 与 ``Hooks3``
    是真实存在的两个不同类（本课 Hook 类从 Hooks2 编到 Hooks14），
    把它们当成「拼写不一致」是误报，会让人去改本来对的东西。
    """
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = [(x, y) for x, y in zip(a, b) if x != y]
        if len(diffs) != 1:
            return False
        return all(c.isalpha() for pair in diffs for c in pair)
    short, long_word = (a, b) if la < lb else (b, a)
    i = j = 0
    extra = ""
    while i < len(short) and j < len(long_word):
        if short[i] != long_word[j]:
            if extra:
                return False
            extra = long_word[j]
            j += 1
            continue
        i += 1
        j += 1
    if not extra:
        extra = long_word[j:]
    return bool(extra) and extra.isalpha()


def _inconsistent_identifiers(text: str) -> list[Finding]:
    """找出「几乎同名但拼写不同」的标识符，通常是同一工具的两种写法。"""
    counts: dict[str, int] = {}
    for m in _IDENT.finditer(text):
        token = m.group(0)
        counts[token] = counts.get(token, 0) + 1
    tokens = list(counts)
    if len(tokens) < 2:
        return []

    by_skeleton: dict[str, list[str]] = {}
    for t in tokens:
        by_skeleton.setdefault(_skeleton(t), []).append(t)

    findings: list[Finding] = []
    skeletons = list(by_skeleton)
    seen: set[tuple[str, str]] = set()
    for i, s1 in enumerate(skeletons):
        for s2 in skeletons[i + 1 :]:
            if not _spelling_variant(s1, s2):
                continue
            for t1 in by_skeleton[s1]:
                for t2 in by_skeleton[s2]:
                    if t1 == t2:
                        continue
                    key = tuple(sorted((t1, t2)))
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append(
                        Finding(
                            f"{t1} / {t2}",
                            "inconsistent",
                            "同一实体两种拼写，必有一处是错的",
                            count=counts[t1] + counts[t2],
                        )
                    )
    return findings


def _dedupe(findings: list[Finding]) -> list[Finding]:
    merged: dict[tuple[str, str], Finding] = {}
    for f in findings:
        key = (f.term, f.kind)
        if key in merged:
            merged[key].count += f.count
        else:
            merged[key] = f
    ordered = sorted(merged.values(), key=lambda f: (-f.count, f.kind, f.term))

    # 抑制冗余的 listed：若该词已出现在其它 finding 的 term 里（如 "D Y L" 已被
    # spaced_letters 报过、jeb_wincon 已被 inconsistent 报过），就不再单独重复一条。
    others = [f.term for f in ordered if f.kind != "listed"]
    return [
        f
        for f in ordered
        if not (f.kind == "listed" and any(f.term in t for t in others))
    ]


# ------------------------------------------------------------------ 术语判定
# 有些词在文本侧永远无法确定（"D Y L two meter" 到底念的什么？），只能放弃；
# 但也有些词能靠**交叉证据**定下来——最典型的是课件 OCR。
# OCR 直接取自画面，不会「听错」；语音会。所以当同一位置 OCR 写出了
# 正确写法、而语音转写是错的，答案就在那里。
#
# 本节的职责：把上下文交给文本模型，让它给出**结论 + 依据 + 置信度**，
# 并强制它「证据不足就说不知道」。


@dataclass
class Verdict:
    """对一个待确认词的判定结论。"""

    term: str
    correct: str = ""
    confidence: str = "low"
    evidence: str = ""
    reason: str = ""
    action: str = "uncertain"
    # 依据是否真的出现在我们给的上下文里（防模型编造依据）
    evidence_verified: bool = False
    contexts_used: int = 0

    @property
    def applicable(self) -> bool:
        """能否安全地写成一条自动纠正规则。

        要求同时满足：动作是 replace、有正确写法、置信度不是 low、
        且依据能在给定上下文里逐字找到。任一不满足都只能交人工。
        """
        return (
            self.action == "replace"
            and bool(self.correct.strip())
            and self.confidence in ("high", "medium")
            and self.evidence_verified
            and self.correct != self.term
        )


def _dedup_key(text: str) -> str:
    """去重键：保留中文与字母数字，只归一化空白与标点。

    不能复用 ``_skeleton``——它按 ``[^a-z0-9]`` 清洗，会把**中文全部丢掉**。
    那样一来，凡是包含同一个英文术语的片段，键都会退化成同一个英文残片
    （如都变成 "dyltwometer"），互相碰撞、被误判为重复而丢弃。
    实测这个 bug 会让模型每次只看到一条上下文，判定质量凭空下降。
    """
    return re.sub(r"[\s\u3000]+", "", text).lower()


def collect_context(
    store,
    video_id: str,
    term: str,
    *,
    window: int = 110,
    limit: int = 8,
) -> tuple[list[str], set[str]]:
    """收集某个词在材料里出现的上下文。

    返回 ``(带来源标注的上下文列表, 可作证据的原文集合)``。

    第二个返回值很重要：**摘要类文本不能当证据**。摘要由 LLM 生成，它可能已经
    把错字「合理化」成了另一个词（实测 "D Y L two meter" 就被摘要写成了
    "Dalvik Debug Monitor"）。拿 LLM 的产物去校验 LLM 的判断是循环论证，
    看着像有依据，其实依据本身就是幻觉。所以只有画面 OCR 与原始转写
    ——这两类一手材料——才能用于逐字核对。
    """
    from .schema import ms_to_hms

    found: list[tuple[int, str, str]] = []  # (优先级, 带来源标注的整条, 片段正文)
    ground_truth: list[str] = []

    def add(priority: int, label: str, text: str, ms: int | None = None, *, trusted: bool) -> None:
        if not text:
            return
        start = 0
        while True:
            i = text.find(term, start)
            if i < 0:
                break
            lo = max(0, i - window)
            hi = min(len(text), i + len(term) + window)
            snippet = text[lo:hi].replace("\n", " ")
            stamp = f"[{ms_to_hms(ms)}] " if ms is not None else ""
            found.append((priority, f"{label} {stamp}…{snippet}…", snippet))
            if trusted:
                ground_truth.append(snippet)
            start = i + len(term)

    # 一手材料：画面 OCR 与原始转写。OCR 取自画面不会听错；转写会被听错但可核对。
    for slide in store.get_slides(video_id):
        add(0, "课件OCR", slide.ocr_text or "", slide.start_ms, trusted=True)
    for chunk in store.get_chunks(video_id):
        add(0, "片段内课件", chunk.ocr_text or "", chunk.start_ms, trusted=True)
    # 二手材料：摘要由 LLM 写的，只能提供线索，不能当证据
    for chapter in store.get_chapters(video_id):
        add(1, "章节摘要", chapter.summary or "", chapter.start_ms, trusted=False)
    for chunk in store.get_chunks(video_id):
        add(1, "片段摘要", chunk.summary or "", chunk.start_ms, trusted=False)
        add(1, "片段正文", chunk.text or "", chunk.start_ms, trusted=True)
    add(1, "全课摘要", store.get_video_summary(video_id) or "", trusted=False)
    for segment in store.get_segments(video_id):
        add(2, "转写", segment.text or "", segment.start_ms, trusted=True)

    # 去重（同一句话在摘要与正文里会重复出现），按可信度优先。
    # 键取**片段正文**而不是整条标注文本——标注里的来源名会让同一句话看起来不同。
    seen: set[str] = set()
    out: list[str] = []
    for _, text, snippet in sorted(found, key=lambda item: item[0]):
        key = _dedup_key(snippet)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out, set(ground_truth)


def adjudicate_suspects(
    glossary: Glossary,
    store,
    client,
    video_id: str,
    *,
    terms: list[str] | None = None,
    window: int = 110,
    limit: int = 8,
) -> list[Verdict]:
    """让文本模型依据上下文判定待确认术语。

    ``terms`` 省略时用术语表里的 suspects。只会返回**有上下文**的词——
    上下文为空的词无需问模型，问了也只能得到猜测。
    """
    from .llm import prompts

    targets = terms if terms is not None else list(glossary.suspects)
    verdicts: list[Verdict] = []

    for term in targets:
        contexts, ground_truth = collect_context(store, video_id, term, window=window, limit=limit)
        if not contexts:
            continue
        body = "\n".join(f"- {c}" for c in contexts)
        messages = [
            {"role": "system", "content": "你是一位严谨的转写校对员，只依据给定上下文判断，不猜测。"},
            {"role": "user", "content": prompts.TERM_ADJUDICATION.format(term=term, contexts=body)},
        ]
        try:
            data, _reply = client.chat_json(messages, max_tokens=800)
        except Exception as exc:  # noqa: BLE001
            log.warning("判定 %s 失败：%s", term, exc)
            continue

        verdict = Verdict(
            term=term,
            correct=str(data.get("correct") or "").strip(),
            confidence=str(data.get("confidence") or "low").strip().lower(),
            evidence=str(data.get("evidence") or "").strip(),
            reason=str(data.get("reason") or "").strip(),
            action=str(data.get("action") or "uncertain").strip().lower(),
            contexts_used=len(contexts),
        )
        # 反幻觉：模型的「依据」必须能在**一手材料**里逐字找到。
        #
        # 这里刻意只用 ground_truth（画面 OCR + 原始转写），而不是全部上下文：
        # 摘要与全课摘要都是 LLM 写的，可能已经把错字「合理化」成了另一个词。
        # 实测 "D Y L two meter" 的摘要里就写着 "Dalvik Debug Monitor"——
        # 那是上一轮 LLM 的产物。若拿它当证据，模型只需引用一句幻觉就能获得
        # 「依据已核实」的印章，等于把循环论证包装成证据。宁可判不了。
        if verdict.evidence:
            norm_ev = _skeleton(verdict.evidence)
            haystack = _skeleton("\n".join(ground_truth))
            verdict.evidence_verified = bool(norm_ev) and norm_ev in haystack
        if verdict.action == "replace" and not verdict.evidence_verified:
            verdict.action = "uncertain"
            verdict.reason = (
                verdict.reason + "（依据未能在画面 OCR/原始转写中逐字核实，降级为待人工）"
            ).strip()

        verdicts.append(verdict)

    return verdicts


def render_verdicts(verdicts: list[Verdict]) -> str:
    """把判定结论渲染成可读表格。"""
    if not verdicts:
        return "没有可判定的待确认术语（可能材料里找不到它们的上下文）。"
    lines = [
        "| 待确认词 | 判定结论 | 置信度 | 动作 | 依据 |",
        "|---|---|---|---|---|",
    ]
    for v in verdicts:
        correct = v.correct or "—"
        ev = (v.evidence or "").replace("|", "\\|")
        if len(ev) > 70:
            ev = ev[:70] + "…"
        reason = (v.reason or "").replace("|", "\\|")
        cell = f"{ev}<br>{reason}" if reason else ev
        lines.append(
            f"| `{v.term}` | `{correct}` | {v.confidence} | {v.action} | {cell} |"
        )
    return "\n".join(lines)


AUTO_GLOSSARY_HEADER = """# 自动生成的术语纠正（机器判定，可随时删除重建）
#
# 由 `vedioai adjudicate --apply` 生成，依据是课件 OCR 与转写上下文。
# 与 vedioai.glossary.yaml 分开存放的理由：
#   - 手写那份是人类策展的，注释宝贵，不能让机器覆写；
#   - 这份可以随时删掉重跑，不影响人工积累。
#
# 想否定某条结论：直接从下面删掉即可，不会被重新加回来（除非再跑一次 --apply）。
"""


def write_auto_glossary(path: Path, verdicts: list[Verdict]) -> tuple[Path, int]:
    """把可安全采纳的判定写入自动术语表。返回（路径, 写入条数）。

    已存在的条目会按 ``wrong`` 合并去重，因此反复执行不会累积重复。
    """
    existing: dict = {}
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    corrections: list[dict] = list(existing.get("corrections") or [])
    by_wrong = {tuple(c.get("wrong") or []): c for c in corrections}

    written = 0
    for v in verdicts:
        if not v.applicable:
            continue
        key = (v.term,)
        entry = {
            "wrong": [v.term],
            "right": v.correct,
            "reason": (f"文本模型依据上下文判定（置信度 {v.confidence}）：{v.reason}")[:300],
        }
        if key in by_wrong:
            by_wrong[key].update(entry)
        else:
            corrections.append(entry)
            by_wrong[key] = entry
        written += 1

    payload = {"version": 1, "corrections": corrections}
    text = AUTO_GLOSSARY_HEADER + yaml.safe_dump(
        payload, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    return path, written


def _auto_path(path: Path) -> Path:
    """``vedioai.glossary.yaml`` → ``vedioai.glossary.auto.yaml``。"""
    return path.with_name(f"{path.stem}.auto{path.suffix}")


def append_course_correction(
    path: Path,
    video_id: str,
    wrong: str,
    right: str,
    *,
    reason: str = "界面标记",
) -> Path:
    """把一条课内纠错写入自动术语表的 ``courses.<video_id>.corrections``。

    不写入手写 ``vedioai.glossary.yaml``（注释是策展成果，机器不得覆写）。
    同 wrong 已存在则更新 right/reason。返回实际写入的 auto 路径。
    """
    wrong = (wrong or "").strip()
    right = (right or "").strip()
    video_id = (video_id or "").strip()
    if not wrong or not right or not video_id:
        raise ValueError("wrong / right / video_id 都不能为空")
    if wrong == right:
        raise ValueError("纠正前后相同，无需写入")

    auto = _auto_path(path)
    existing: dict = {"version": 1, "corrections": [], "courses": {}}
    if auto.exists():
        loaded = yaml.safe_load(auto.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            existing.update(loaded)

    courses = existing.setdefault("courses", {})
    if not isinstance(courses, dict):
        courses = {}
        existing["courses"] = courses
    course = courses.setdefault(video_id, {})
    if not isinstance(course, dict):
        course = {}
        courses[video_id] = course
    items: list = list(course.get("corrections") or [])
    updated = False
    for item in items:
        if not isinstance(item, dict):
            continue
        w = item.get("wrong") or []
        if isinstance(w, str):
            w = [w]
        if wrong in [str(x).strip() for x in w]:
            item["wrong"] = [wrong]
            item["right"] = right
            item["reason"] = reason
            updated = True
            break
    if not updated:
        items.append({"wrong": [wrong], "right": right, "reason": reason})
    course["corrections"] = items

    # 保留全局 corrections（adjudicate 写入的），只额外挂上 courses
    payload = {
        "version": int(existing.get("version") or 1),
        "corrections": list(existing.get("corrections") or []),
        "courses": courses,
    }
    text = AUTO_GLOSSARY_HEADER + yaml.safe_dump(
        payload, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    auto.write_text(text, encoding="utf-8", newline="\n")
    return auto


# ------------------------------------------------------------------ 便捷函数


def load(path: Path | None, video_id: str = "") -> Glossary:
    return Glossary.load(path, video_id)


def apply_to_segments(segments, glossary: Glossary):
    """对 Segment 列表做纠错，就地替换 text。返回（segments, fixes）。"""
    texts, fixes = glossary.correct_many([s.text for s in segments])
    for seg, text in zip(segments, texts):
        seg.text = text
    return segments, fixes


def render_findings(findings: list[Finding], path: Path | None = None) -> str:
    """把可疑术语渲染成 Markdown 小节，附在笔记末尾。"""
    if not findings:
        return ""
    lines = [
        "## 待人工确认的术语",
        "",
        "> 以下术语**未做自动替换**：它们可能是转写听错，也可能本来就是对的。",
        "> 自动改错了比不改更危险，所以这里只标记出来，请对照视频原声确认。",
        "",
        "| 术语 | 类型 | 说明 | 出现次数 |",
        "|---|---|---|---|",
    ]
    kind_label = {
        "spaced_letters": "转写噪音",
        "quoted_phrase": "疑为听错",
        "inconsistent": "拼写不一致",
        "listed": "已标记",
    }
    for f in findings:
        term = f.term.replace("|", "\\|")
        lines.append(
            f"| `{term}` | {kind_label.get(f.kind, f.kind)} | {f.hint} | {f.count} |"
        )
    lines.append("")
    if path is not None:
        lines.append(f"本次所用术语表：`{path}`")
    return "\n".join(lines)
