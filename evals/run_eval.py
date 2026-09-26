"""评估运行器（兼容入口）。

真正的实现在 `vedioai.eval_runner`，因为它是随包安装的一部分——
这样 `vedioai eval` 在任何工作目录下都能跑。

本文件保留的原因：让人可以直接 `python evals/run_eval.py` 调试，
以及让 `evals/` 目录自带一份可读的实现位置。

用法仍以 CLI 为准：
    vedioai eval evals/questions.yaml
    vedioai eval --scaffold <video_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接以脚本方式运行
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from vedioai.eval_runner import (  # noqa: E402,F401
    QuestionOutcome,
    KeypointResult,
    CitationResult,
    main,
    normalize,
    render_report,
    run_questions,
    scaffold,
    score_citation,
    score_keypoints,
    strip_extension_blocks,
    summarize,
)

if __name__ == "__main__":
    import argparse

    from vedioai.config import load_config

    parser = argparse.ArgumentParser(description="跑评估集")
    parser.add_argument("questions", nargs="?", default="evals/questions.yaml")
    parser.add_argument("--out", default="evals/runs")
    parser.add_argument("--only", choices=["factual", "cross", "visual", "global"])
    parser.add_argument("--scaffold", metavar="VIDEO_ID")
    parser.add_argument("-v", "--verbose", action="store_true")
    ns = parser.parse_args()

    cfg = load_config()
    if ns.scaffold:
        out = Path("evals") / f"questions.{ns.scaffold}.yaml"
        raise SystemExit(scaffold(ns.scaffold, cfg, out))
    raise SystemExit(main(ns, cfg))
