"""Build labeled equivalence pairs to validate an LLM judge for Nemotron math rows.

Ground-truth strategy: for PARSEABLE expected answers Math-Verify itself provides the
label (and sympy generates equivalent alternate forms = hard positives); for PROSE
expected answers (the judge's actual job) we use identity pairs (label=equal) and
cross-row pairs (label=different). A good judge must pass both populations.

  PYTHONPATH=. python nemo/judge_eval/build_pairs.py \
      --rlvr /path/to/Nemotron-RL-Ultra-restored/rlvr1.jsonl \
      --out nemo/judge_eval/pairs.jsonl --n-parseable 150 --n-prose 150 --seed 42

Pair kinds:
  pos_identity       candidate == expected (must judge equal)
  pos_equivalent     sympy-derived alternate form of a parseable expected (must judge equal)
  neg_cross          another row's expected answer, Math-Verify-confirmed different (parseable)
                     or plain different prose (prose rows)
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Optional

MATH_AGENT = "math_with_judge_simple_agent"


def strip_math_delimiters(s: str) -> str:
    """== official LibraryJudgeMathResourcesServer._strip_math_delimiters."""
    s = s.strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1].strip()
    return s


def make_verifier():
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    return math_metric(gold_extraction_target=(LatexExtractionConfig(),),
                      pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()))


def library_verify(verifier, expected: str, candidate: str) -> Optional[bool]:
    """True/False per Math-Verify; None if gold unparseable/errors (prose)."""
    try:
        gold = "\\boxed{" + strip_math_delimiters(expected) + "}"
        score, extracted = verifier([gold], ["\\boxed{" + strip_math_delimiters(candidate) + "}"])
        if extracted is None:
            return None
        return bool(score >= 0.5)
    except BaseException:
        return None


def alternate_form(expected: str) -> Optional[str]:
    """A numerically equivalent alternate rendering (numeric answers only), or None.
    Kept sympy-light (plain Rational; no latex parsing) so it runs in any venv."""
    import sympy
    s = strip_math_delimiters(expected).replace(",", "")
    try:
        e = sympy.Rational(s)
        if e.is_Integer:
            return f"{int(e)}.0" if abs(int(e)) < 10**6 else None
        v = float(e)
        out = f"{v:.6f}".rstrip("0").rstrip(".")
        return out if out and "e" not in out and out != s else None
    except BaseException:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlvr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-parseable", type=int, default=150)
    ap.add_argument("--n-prose", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    with open(args.rlvr) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("agent_ref") or {}).get("name") != MATH_AGENT:
                continue
            q, e = (r.get("question") or "").strip(), (r.get("expected_answer") or "").strip()
            if q and e:
                rows.append({"question": q, "expected_answer": e})
    print(f"math rows with question+answer: {len(rows)}")
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    verifier = make_verifier()
    # NOTE: identity self-verification passes even for prose (the extracted latex string
    # matches itself), so parseable-vs-prose must be a STRUCTURAL call: short, single-token-ish
    # answers are Math-Verify's home turf; multi-word/sentence answers are the judge's job.
    def is_prose(e: str) -> bool:
        s = strip_math_delimiters(e)
        words = s.split()
        return len(words) > 6 or len(s) > 80 or "\n" in s or s.rstrip().endswith(".") and len(words) > 3

    parseable, prose = [], []
    for r in rows:
        if len(parseable) >= args.n_parseable * 3 and len(prose) >= args.n_prose * 3:
            break
        if is_prose(r["expected_answer"]):
            prose.append(r)
        elif library_verify(verifier, r["expected_answer"], r["expected_answer"]) is True:
            parseable.append(r)
    print(f"classified: parseable={len(parseable)} prose={len(prose)} (scanned subset)")

    pairs = []

    def add(row, cand, label, kind):
        pairs.append({"question": row["question"], "expected_answer": row["expected_answer"],
                      "candidate": cand, "label": label, "kind": kind})

    import re

    def perturb(s: str) -> Optional[str]:
        """HARD negative: same surface form, one value/keyword changed (different by
        construction)."""
        kw = [("odd", "even"), ("even", "odd"), ("increasing", "decreasing"),
              ("positive", "negative"), ("maximum", "minimum"), ("greater", "less")]
        m = list(re.finditer(r"\d+", s))
        if m:
            i = m[-1]
            old = i.group()
            new = str(int(old) + 1)
            return s[: i.start()] + new + s[i.end():]
        low = s.lower()
        for a, b in kw:
            if a in low:
                idx = low.index(a)
                return s[:idx] + b + s[idx + len(a):]
        return None

    for r in parseable[: args.n_parseable]:
        add(r, r["expected_answer"], 1, "pos_identity_parseable")
        alt = alternate_form(r["expected_answer"])
        if alt and library_verify(verifier, r["expected_answer"], alt) is True:
            add(r, alt, 1, "pos_equivalent")
        add(r, "The final answer is " + strip_math_delimiters(r["expected_answer"]) + ".", 1,
            "pos_reformat_parseable")
        p = perturb(r["expected_answer"])
        if p and library_verify(verifier, r["expected_answer"], p) is False:
            add(r, p, 0, "neg_perturb_parseable")
        for other in rng.sample(parseable, 8):
            if library_verify(verifier, r["expected_answer"], other["expected_answer"]) is False:
                add(r, other["expected_answer"], 0, "neg_cross_parseable")
                break
    for r in prose[: args.n_prose]:
        add(r, r["expected_answer"], 1, "pos_identity_prose")
        add(r, "After working through the problem: " + r["expected_answer"], 1, "pos_reformat_prose")
        p = perturb(r["expected_answer"])
        if p:
            add(r, p, 0, "neg_perturb_prose")
        other = rng.choice(prose)
        if other["expected_answer"] != r["expected_answer"]:
            add(r, other["expected_answer"], 0, "neg_cross_prose")

    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print("pairs by kind:", dict(Counter(p["kind"] for p in pairs)))
    print(f"wrote {len(pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
