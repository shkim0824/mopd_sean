"""Grader for OpenThoughts3-1.2M problems (math / code auto-gradeable; science reference-only).

The released dataset has NO ground truth; grading uses the sidecar built by
``mopd.data.build_ot3_verify`` (join back to OpenMathInstruct-2 / code_contests / TACO / APPS
through the row-aligned mlfoundations-dev superset — see that module's docstring). Rows are
matched by qhash = sha1 of the whitespace-normalised question text, so the grader works on any
subset/shuffle of the data without carrying indices around.

Mechanics reuse the existing OFFICIAL-code graders, sandbox included:
  math    -> mopd.graders.math_grader.grade  (last \\boxed + Math-Verify) vs the joined
             `expected_answer` — same extraction contract as the OpenThoughts/evalchemy stack
  code    -> mopd.graders.code_grader        (LiveCodeBench harness; stdin/stdout + fn_name
             call-based; Landlock+seccomp sandboxed children) vs joined test cases
  science -> NOT machine-gradeable: returns the textbook reference for an (optional,
             caller-supplied) LLM judge
  stackexchange_codegolf / stackexchange-physics / unknown -> gradeable=False

Verdict dict: {"gradeable": bool, "kind": "math|code|science_ref|none", "passed": bool|None,
"detail": ...}. ``passed`` is None whenever gradeable is False.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from mopd.graders import code_grader, math_grader

DEFAULT_SIDECAR = "data/train/ot3_verify"


def norm_q(text: str) -> str:
    return " ".join((text or "").split())


def qhash(text: str) -> str:
    return hashlib.sha1(norm_q(text).encode("utf-8")).hexdigest()


class OT3Grader:
    def __init__(self, sidecar_dir: str = DEFAULT_SIDECAR, max_tests: Optional[int] = None,
                 code_timeout: int = code_grader.DEFAULT_TIMEOUT):
        """max_tests: cap tests per code problem at grade time (None = all; RL rewards may
        want ~15-20 for throughput, eval should keep all)."""
        self.dir = sidecar_dir
        self.max_tests = max_tests
        self.code_timeout = code_timeout
        self.math: Dict[str, str] = {}
        self.code: Dict[str, Dict[str, Any]] = {}
        self.science: Dict[str, Dict[str, Any]] = {}
        with open(os.path.join(sidecar_dir, "math_answers.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                self.math[r["qhash"]] = r["answer"]
        with gzip.open(os.path.join(sidecar_dir, "code_tests.jsonl.gz"), "rt") as f:
            for line in f:
                r = json.loads(line)
                self.code[r["qhash"]] = r["input_output"]
        p = os.path.join(sidecar_dir, "science_refs.jsonl")
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    self.science[r["qhash"]] = r

    # ------------------------------------------------------------------ lookup
    def lookup(self, question: str) -> Dict[str, Any]:
        h = qhash(question)
        if h in self.math:
            return {"kind": "math", "qhash": h, "gold": self.math[h]}
        if h in self.code:
            return {"kind": "code", "qhash": h, "input_output": self.code[h]}
        if h in self.science:
            return {"kind": "science_ref", "qhash": h, "reference": self.science[h]}
        return {"kind": "none", "qhash": h}

    def _cap(self, io: Dict[str, Any]) -> Dict[str, Any]:
        if self.max_tests and len(io["inputs"]) > self.max_tests:
            return {"inputs": io["inputs"][: self.max_tests], "outputs": io["outputs"][: self.max_tests],
                    "fn_name": io.get("fn_name")}
        return io

    # ------------------------------------------------------------------ grading
    def grade(self, question: str, response: str) -> Dict[str, Any]:
        """Grade ONE model response to an OT3 question."""
        info = self.lookup(question)
        kind = info["kind"]
        if kind == "math":
            ok = math_grader.grade(response, info["gold"])
            return {"gradeable": True, "kind": kind, "passed": bool(ok),
                    "detail": {"gold": info["gold"], "pred": math_grader.extract_answer(response)}}
        if kind == "code":
            out = code_grader.grade_response(self._cap(info["input_output"]), response,
                                             timeout=self.code_timeout)
            return {"gradeable": True, "kind": kind, "passed": bool(out["passed"]),
                    "detail": {"n_tests": len(info["input_output"]["inputs"]), "meta": out.get("meta", {})}}
        if kind == "science_ref":
            return {"gradeable": False, "kind": kind, "passed": None,
                    "detail": {"reference": info["reference"]}}
        return {"gradeable": False, "kind": "none", "passed": None, "detail": {}}

    def grade_batch(self, questions: Sequence[str], responses: Sequence[str],
                    num_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Batch: math/science graded inline (fast); code fanned out through
        code_grader.grade_batch (per-program sandboxed subprocesses)."""
        assert len(questions) == len(responses)
        out: List[Optional[Dict[str, Any]]] = [None] * len(questions)
        code_items, code_pos = [], []
        for i, (q, r) in enumerate(zip(questions, responses)):
            info = self.lookup(q)
            if info["kind"] == "code":
                code_items.append({"input_output": self._cap(info["input_output"]),
                                   "code": code_grader.extract_code(r)})
                code_pos.append(i)
            else:
                out[i] = self.grade(q, r)
        if code_items:
            graded = code_grader.grade_batch(code_items, num_workers=num_workers,
                                             timeout=self.code_timeout)
            for i, g in zip(code_pos, graded):
                out[i] = {"gradeable": True, "kind": "code", "passed": bool(g["passed"]),
                          "detail": {"meta": g.get("meta", {})}}
        return out  # type: ignore

    def coverage(self) -> Dict[str, int]:
        return {"math": len(self.math), "code": len(self.code), "science_ref": len(self.science)}
