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

    # ------------------------------------------------------------ 加载

    @classmethod
    def load(cls, path: Path | None, video_id: str = "") -> Glossary:
        """读术语表。文件不存在时返回空表（不报错，纠错层是可选增强）。"""
        gl = cls()
        gl.trusted = list(_DEFAULT_TRUSTED)
        if path is None or not Path(path).exists():
            return gl

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

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
                    gl.corrections.append(
                        Correction(wrong=wrong, right=right, reason=str(item.get("reason") or ""))
                    )

        add_corrections(raw.get("corrections"))

        course = (raw.get("courses") or {}).get(video_id) or {}
        add_corrections(course.get("corrections"))

        gl.suspects = [str(s) for s in (raw.get("suspects") or [])]
        gl.suspects += [str(s) for s in (course.get("suspects") or [])]
        gl.trusted += [str(s) for s in (raw.get("trusted") or [])]
        gl.trusted += [str(s) for s in (course.get("trusted") or [])]

        # 长串优先替换，避免短串先命中把长串切碎
        gl.corrections.sort(key=lambda c: max(len(w) for w in c.wrong), reverse=True)
        return gl

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
