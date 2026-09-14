"""Benchmark registry: where the rows live, how to prompt, how to score.

data layout (built by ``python -m mopd.data.prep_eval`` on the cpu-instance):
  data/eval/aime24/test.jsonl   {id, problem, answer, year, contest, problem_idx, source}
  data/eval/aime25/test.jsonl
  data/eval/aime26/test.jsonl
  data/eval/lcb_v6/test.jsonl   full LiveCodeBench release_v6 rows (1055; private tests embedded)
  data/eval/ifeval/test.jsonl   {key, prompt, instruction_id_list, kwargs}
  data/eval/ifbench/test.jsonl  {key, prompt, instruction_id_list, kwargs}
Each dir has manifest.json (source ids, revision, sha256, row count).
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from mopd.common.io import data_root, iter_jsonl
from mopd.eval import prompts as P
from mopd.graders import code_grader, if_grader, math_grader


# ----------------------------------------------------------------------------- loaders
def _load(name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = os.path.join(data_root(), "eval", name, "test.jsonl")
    rows = []
    for r in iter_jsonl(path):
        rows.append(r)
        if limit and len(rows) >= limit:
            break
    return rows


def _load_lcb(name: str, limit: Optional[int] = None, start_date: str = "", end_date: str = "") -> List[Dict[str, Any]]:
    """Optional contest_date window like the LCB harness ``--start_date/--end_date``
    (inclusive on both ends, datetime compare)."""
    rows = []
    sd = _dt.datetime.fromisoformat(start_date) if start_date else None
    ed = _dt.datetime.fromisoformat(end_date) if end_date else None
    for r in iter_jsonl(os.path.join(data_root(), "eval", name, "test.jsonl")):
        d = _dt.datetime.fromisoformat(r["contest_date"])
        if sd and d < sd:
            continue
        if ed and d > ed:
            continue
        rows.append(r)
        if limit and len(rows) >= limit:
            break
    return rows


# ----------------------------------------------------------------------------- scorers
def score_math(rows, samples: List[List[str]]) -> Dict[str, Any]:
    """avg@n accuracy (paper: AIME avg@32) + per-row pass counts."""
    per = []
    tot = 0.0
    for r, outs in zip(rows, samples):
        c = [math_grader.grade(o, str(r["answer"])) for o in outs]
        per.append(c)
        tot += sum(c) / max(len(c), 1)
    n = max(len(rows), 1)
    k = len(samples[0]) if samples else 0
    return {"score": 100.0 * tot / n, "metric": f"avg@{k}", "n": len(rows), "per_row": per,
            "maj_pass": 100.0 * sum(1 for c in per if sum(c) > len(c) / 2) / n}


def score_if(bench: str):
    def _score(rows, samples: List[List[str]]) -> Dict[str, Any]:
        k = len(samples[0]) if samples else 1
        accs = []
        per_rows = None
        for j in range(k):
            res = if_grader.score_benchmark(bench, rows, [s[j] for s in samples])
            accs.append(res)
            per_rows = res["per_example"]
        out = {m: sum(a[m] for a in accs) / k for m in
               ("prompt_level_strict_acc", "prompt_level_loose_acc", "inst_level_strict_acc", "inst_level_loose_acc")}
        # headline: IFEval -> prompt-level strict ; IFBench paper -> prompt-level loose
        out["score"] = out["prompt_level_strict_acc"] if bench == "ifeval" else out["prompt_level_loose_acc"]
        out["metric"] = "prompt_strict" if bench == "ifeval" else "prompt_loose"
        out["n"] = len(rows)
        out["per_row"] = [[e["prompt_strict"], e["prompt_loose"]] for e in per_rows] if per_rows else []
        return out
    return _score


def score_lcb(rows, samples: List[List[str]], num_workers: Optional[int] = None) -> Dict[str, Any]:
    """LCB pass@1 = mean over (problem, sample) of all-tests-pass; per-difficulty breakdown."""
    items, index = [], []
    for i, (r, outs) in enumerate(zip(rows, samples)):
        io = code_grader.lcb_row_to_input_output(r)
        for j, o in enumerate(outs):
            items.append({"input_output": io, "code": code_grader.extract_code(o)})
            index.append((i, j))
    graded = code_grader.grade_batch(items, num_workers=num_workers) if items else []
    per = [[False] * len(outs) for outs in samples]
    for (i, j), g in zip(index, graded):
        per[i][j] = bool(g["passed"])
    n = max(len(rows), 1)
    tot = sum(sum(c) / max(len(c), 1) for c in per)
    by_diff: Dict[str, List[float]] = {}
    for r, c in zip(rows, per):
        by_diff.setdefault(r.get("difficulty", "?"), []).append(sum(c) / max(len(c), 1))
    k = len(samples[0]) if samples else 0
    return {"score": 100.0 * tot / n, "metric": f"pass@1(avg@{k})", "n": len(rows), "per_row": per,
            "by_difficulty": {d: 100.0 * sum(v) / len(v) for d, v in by_diff.items()},
            "no_code_block": sum(1 for it in items if not it["code"].strip())}


# ----------------------------------------------------------------------------- registry
class Bench:
    def __init__(self, name, kind, load, messages, score, default_n, max_tokens):
        self.name, self.kind, self.load, self.messages, self.score = name, kind, load, messages, score
        self.default_n, self.max_tokens = default_n, max_tokens


def _aime(name):
    return Bench(name, "math", lambda limit=None, **kw: _load(name, limit),
                 lambda r, style="qwen", **kw: P.math_messages(r["problem"], style),
                 score_math, default_n=8, max_tokens=16384)


BENCHMARKS: Dict[str, Bench] = {
    "aime24": _aime("aime24"),
    "aime25": _aime("aime25"),
    "aime26": _aime("aime26"),
    "lcb_v6": Bench("lcb_v6", "code",
                    lambda limit=None, start_date="", end_date="", **kw: _load_lcb("lcb_v6", limit, start_date, end_date),
                    lambda r, **kw: P.code_messages(r["question_content"], r.get("starter_code") or "", system=kw.get("lcb_system", False)),
                    score_lcb, default_n=1, max_tokens=16384),
    "ifeval": Bench("ifeval", "if", lambda limit=None, **kw: _load("ifeval", limit),
                    lambda r, **kw: P.if_messages(r["prompt"]), score_if("ifeval"), default_n=1, max_tokens=16384),
    "ifbench": Bench("ifbench", "if", lambda limit=None, **kw: _load("ifbench", limit),
                     lambda r, **kw: P.if_messages(r["prompt"]), score_if("ifbench"), default_n=1, max_tokens=16384),
}

DOMAIN_OF = {"aime24": "math", "aime25": "math", "aime26": "math", "lcb_v6": "code", "ifeval": "if", "ifbench": "if"}
DEFAULT_SET = ["aime24", "aime25", "aime26", "lcb_v6", "ifeval", "ifbench"]
