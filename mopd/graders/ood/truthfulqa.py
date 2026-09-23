"""TruthfulQA MC1 / MC2 / MC3 — official ``MC_calcs`` over continuation log-prob sums.

Input is the per-choice log-prob the runner collected for each row:
``lp[idx][ci] = {"logprob": float, "n_tokens": int}`` where ``ci`` indexes the row's
deduplicated choice list (``mopd/eval/ood_benches.py::truthfulqa_choices``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from third_party.truthfulqa.truthfulqa_official import mc_calcs


def score_truthfulqa(rows: Sequence[Dict[str, Any]],
                     lp: Sequence[Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    mc1s: List[float] = []
    mc2s: List[float] = []
    mc3s: List[float] = []
    for r, got in zip(rows, lp):
        choices: List[str] = r["_choices"]
        # mc1: index 0 of mc1_choices is the correct answer upstream (asserted at prep)
        i1 = [choices.index(c) for c in r["mc1_choices"]]
        s1 = [got[i]["logprob"] for i in i1]
        t1 = [c for c, l in zip(r["mc1_choices"], r["mc1_labels"]) if l == 1]
        f1 = [s for s, l in zip(s1, r["mc1_labels"]) if l == 0]
        mc1s.append(mc_calcs([s for s, l in zip(s1, r["mc1_labels"]) if l == 1], f1,
                             t1, t1[0])["MC1"])
        i2 = [choices.index(c) for c in r["mc2_choices"]]
        s2 = [got[i]["logprob"] for i in i2]
        t2 = [c for c, l in zip(r["mc2_choices"], r["mc2_labels"]) if l == 1]
        st2 = [s for s, l in zip(s2, r["mc2_labels"]) if l == 1]
        sf2 = [s for s, l in zip(s2, r["mc2_labels"]) if l == 0]
        m = mc_calcs(st2, sf2, t2, t2[0])
        mc2s.append(m["MC2"])
        mc3s.append(m["MC3"])
    n = max(len(rows), 1)
    return {
        "score": 100.0 * sum(mc2s) / n, "metric": "MC2", "n": len(rows),
        "mc1": 100.0 * sum(mc1s) / n, "mc2": 100.0 * sum(mc2s) / n,
        "mc3": 100.0 * sum(mc3s) / n,
        "higher_is_better": True,
    }
