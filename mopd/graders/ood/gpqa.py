"""GPQA-Diamond scoring — strict (official) and general (permissive) side by side.

`strict` is openai/simple-evals verbatim: ONE regex, no fallback, a miss is a miss. That is the
number comparable with published GPQA results, and it stays the headline `score`.

`general` adds the other shapes a model writes its verdict in. It exists because the strict
number conflates two things: whether the model knows the answer, and whether it ends its reply
in the requested format. Measured on this study's frozen generations, the base OT3 model wrote
35.5% of its answers as `\boxed{C}` (its SFT data is math traces) while the MOPD checkpoints
wrote `Answer: X` 89-95% of the time; accuracy among parsed samples was 48.7-51.9% for all of
them. Reporting both keeps the comparable number AND shows the science signal.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from third_party.simple_evals.gpqa_official import gpqa_extract

#: Permissive verdict patterns, each capturing the option letter. Applied to the whole response
#: and the LAST match wins (models restate their verdict; the final one is the answer) — the
#: same "last verdict" convention MMLU-Pro's official cascade uses.
GENERAL_PATTERNS = (
    # `Answer: C`, `Answer: **C**`, `Answer: \$D\$`, `Answer: (C)` — the official pattern is
    # stricter than this: it dies on the `*` or the backslash right after the colon.
    r"(?i)answers?[ \t]*[:\-][ \t]*[\\$\*\(\[\s]*([A-D])\b",
    # \boxed{C}, \boxed{\text{C}}, \boxed{(C)}
    r"\\boxed\{\s*(?:\\(?:text|mathrm|mathbf)\s*\{\s*)?\(?\**([A-D])\**\)?\s*\}?\s*\}",
    # **Answer:** C  /  **Answer** C
    r"(?i)\*\*\s*answers?\s*\*{0,2}[\s:\-]*\(?([A-D])\)?\b",
    # the answer is C / final answer = C
    r"(?i)(?:the\s+)?(?:final\s+|correct\s+|best\s+)?answers?\s*(?:is|are|=|would\s+be)\s*[\*\(\[\s]*([A-D])\b",
    # C is the correct answer
    r"(?i)\b([A-D])\)?\s+is\s+the\s+(?:correct|right|best)\s+(?:answer|option|choice)\b",
    # option C is correct
    r"(?i)option\s*\(?([A-D])\)?\s+is\s+(?:correct|right|the\s+answer)\b",
)
_GENERAL = tuple(re.compile(p) for p in GENERAL_PATTERNS)


def gpqa_extract_general(text: str) -> Optional[str]:
    """The letter the model meant, under any of the observed verdict shapes.

    The official extractor is tried first; only if it finds nothing do the permissive patterns
    run, and the last match across the response wins. Degenerate repetition and replies that
    name no letter stay unparsed.
    """
    t = text or ""
    got = gpqa_extract(t)
    if got is not None:
        return got
    best = None
    best_pos = -1
    for rx in _GENERAL:
        for m in rx.finditer(t):
            if m.start() >= best_pos:
                best_pos, best = m.start(), m.group(1).upper()
    return best


def score_gpqa(rows: Sequence[Dict[str, Any]], samples: Sequence[Sequence[str]]) -> Dict[str, Any]:
    per: List[List[bool]] = []
    tot = tot_gen = 0.0
    unparsed = unparsed_gen = 0
    n_samples = 0
    by_dom: Dict[str, List[float]] = {}
    for r, outs in zip(rows, samples):
        gold = r["answer"]
        c, cg = [], []
        for o in outs:
            got = gpqa_extract(o)
            gotg = gpqa_extract_general(o)
            if got is None:
                unparsed += 1
            if gotg is None:
                unparsed_gen += 1
            c.append(bool(got is not None and got == gold))
            cg.append(bool(gotg is not None and gotg == gold))
        n_samples += len(outs)
        per.append(c)
        acc = sum(c) / max(len(c), 1)
        tot += acc
        tot_gen += sum(cg) / max(len(cg), 1)
        by_dom.setdefault(r.get("high_level_domain") or "?", []).append(acc)
    n = max(len(rows), 1)
    k = len(samples[0]) if samples else 0
    return {
        "score": 100.0 * tot / n, "metric": "avg@%d (strict, simple-evals)" % k,
        "n": len(rows), "accuracy": tot / n,
        "score_general": 100.0 * tot_gen / n,
        "unparsed_frac": unparsed / max(n_samples, 1),
        "unparsed_frac_general": unparsed_gen / max(n_samples, 1),
        "by_domain": {d: 100.0 * sum(v) / len(v) for d, v in sorted(by_dom.items())},
        "higher_is_better": True,
        "per_row": per,
    }
