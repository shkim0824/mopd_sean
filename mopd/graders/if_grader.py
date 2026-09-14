"""Instruction-following grading, three official registries, unchanged:

* ``ifeval``   — Google ``instruction_following_eval`` (25 ids)      -> IFEval benchmark
* ``ifbench``  — allenai/IFBench ``ifbench`` package (25 + 58 ids)   -> IFBench benchmark
* ``ifevalg``  — allenai/open-instruct ``IFEvalG`` (25 + 29 ids)     -> IF-RLVR TRAIN reward
                 (the registry that ``allenai/IF_multi_constraints_upto5`` ground truths refer to)

Benchmark metrics reproduce ``evaluation_lib.test_instruction_following_{strict,loose}``
exactly (loose = 8 response variants: raw, ``*`` stripped, first/last/both lines
removed, each also ``*``-stripped; an instruction passes if ANY variant passes).
Reported numbers: prompt-level strict/loose, instruction-level strict/loose.
IFEval convention = prompt-level strict; IFBench paper convention = prompt-level loose.

The thinking block is removed before checking (IFBench README: "for thinking
models ... we then process the output to extract the answer without the
reasoning chains"; open-instruct ``IFEvalVerifier`` does ``remove_thinking_section``).

Training reward (``ifevalg_reward``) = open-instruct ``IFEvalVerifier`` semantics:
fraction of constraints whose STRICT check passes on the de-thinked answer
(``all_or_nothing=True`` gives the paper-text variant: 1.0 iff every constraint passes).
"""
from __future__ import annotations

import ast
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from mopd.graders.math_grader import strip_thinking

# offline nltk assets (punkt_tab, stopwords, tagger) — staged by env/setup_mopd_env.sbatch
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
os.environ.setdefault("NLTK_DATA", os.path.join(_REPO, "env", "nltk_data"))

_REGISTRIES: Dict[str, Dict[str, Any]] = {}


def registry(name: str) -> Dict[str, Any]:
    """Lazy import so unit tests that only need one registry stay light."""
    if name in _REGISTRIES:
        return _REGISTRIES[name]
    if name == "ifeval":
        from third_party.ifeval_google import instructions_registry as r
    elif name == "ifbench":
        from third_party.ifbench import instructions_registry as r
    elif name == "ifevalg":
        from third_party.ifevalg import instructions_registry as r
    else:
        raise KeyError(name)
    _REGISTRIES[name] = r.INSTRUCTION_DICT
    return _REGISTRIES[name]


def _clean_kwargs(kw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # HF parquet rows carry every possible kwarg key with null -> drop the nulls
    # (lm-eval-harness ``{k: v for k, v in kwargs.items() if v}``; open-instruct
    # ``if v is not None``). We use ``is not None`` so a legitimate 0 / False survives.
    return {k: v for k, v in (kw or {}).items() if v is not None}


def _loose_variants(response: str) -> List[str]:
    r = response.split("\n")
    remove_first = "\n".join(r[1:]).strip()
    remove_last = "\n".join(r[:-1]).strip()
    remove_both = "\n".join(r[1:-1]).strip()
    revised = response.replace("*", "")
    return [
        response, revised, remove_first, remove_last, remove_both,
        remove_first.replace("*", ""), remove_last.replace("*", ""), remove_both.replace("*", ""),
    ]


def _check_list(reg: Dict[str, Any], prompt: str, ids: Sequence[str], kwargs_list: Sequence[Optional[Dict[str, Any]]],
                response: str, loose: bool) -> List[bool]:
    variants = _loose_variants(response) if loose else [response]
    out: List[bool] = []
    for index, iid in enumerate(ids):
        cls = reg[iid]
        inst = cls(iid)
        kw = _clean_kwargs(kwargs_list[index] if index < len(kwargs_list) else None)
        inst.build_description(**kw)
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=prompt)
        ok = False
        for v in variants:
            if v.strip() and inst.check_following(v):
                ok = True
                break
        out.append(ok)
    return out


def evaluate_example(bench: str, prompt: str, instruction_id_list: Sequence[str],
                     kwargs_list: Sequence[Optional[Dict[str, Any]]], response: str,
                     strip_think: bool = True) -> Dict[str, Any]:
    """One prompt -> {strict: [bool per instr], loose: [...], prompt_strict, prompt_loose}."""
    reg = registry(bench)
    resp = strip_thinking(response) if strip_think else (response or "")
    strict = _check_list(reg, prompt, instruction_id_list, kwargs_list, resp, loose=False)
    loose = _check_list(reg, prompt, instruction_id_list, kwargs_list, resp, loose=True)
    return {"strict": strict, "loose": loose,
            "prompt_strict": all(strict), "prompt_loose": all(loose)}


def score_benchmark(bench: str, rows: Sequence[Dict[str, Any]], responses: Sequence[str]) -> Dict[str, Any]:
    """rows: {prompt, instruction_id_list, kwargs}; returns the 4 official numbers (in %)."""
    assert len(rows) == len(responses)
    ps = pl = 0
    is_ = il = it = 0
    per = []
    for row, resp in zip(rows, responses):
        e = evaluate_example(bench, row["prompt"], row["instruction_id_list"], row["kwargs"], resp)
        ps += e["prompt_strict"]
        pl += e["prompt_loose"]
        is_ += sum(e["strict"])
        il += sum(e["loose"])
        it += len(e["strict"])
        per.append(e)
    n = max(len(rows), 1)
    return {
        "prompt_level_strict_acc": 100.0 * ps / n,
        "prompt_level_loose_acc": 100.0 * pl / n,
        "inst_level_strict_acc": 100.0 * is_ / max(it, 1),
        "inst_level_loose_acc": 100.0 * il / max(it, 1),
        "n": len(rows), "n_instructions": it, "per_example": per,
    }


# ----------------------------------------------------------------------------- training reward
def parse_ground_truth(gt: Any) -> Dict[str, Any]:
    """IF_multi_constraints_upto5.ground_truth is a python-literal string:
    "[{'instruction_id': [...], 'kwargs': [None, {...}]}]"."""
    if isinstance(gt, str):
        try:
            gt = ast.literal_eval(gt)
        except Exception:
            gt = json.loads(gt)
    if isinstance(gt, list):
        gt = gt[0]
    if isinstance(gt, str):
        gt = json.loads(gt)
    return gt


def ifevalg_reward(response: str, ground_truth: Any, all_or_nothing: bool = False) -> float:
    """open-instruct ``IFEvalVerifier``: strict checks on the de-thinked answer,
    reward = fraction passed (or 1.0 iff all passed)."""
    gt = parse_ground_truth(ground_truth)
    ids = gt["instruction_id"]
    kws = gt.get("kwargs") or [None] * len(ids)
    answer = strip_thinking(response)
    if not (response or "").strip() or not answer.strip():
        return 0.0
    reg = registry("ifevalg")
    passed = _check_list(reg, "", ids, kws, answer, loose=False)
    if all_or_nothing:
        return 1.0 if all(passed) else 0.0
    return sum(passed) / max(len(passed), 1)
