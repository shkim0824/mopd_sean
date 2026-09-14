"""Verifiable rewards for the three domains (TRL ``reward_funcs`` signature).

Each function receives the batch as keyword lists (``prompts``, ``completions``
plus every extra dataset column) and returns one float per completion.

  math : 1.0 iff last \\boxed{} == answer (math-verify), else 0.0
  code : 1.0 iff the LAST fenced program passes EVERY test (LCB harness), else 0.0
  if   : fraction of constraints passed (open-instruct IFEvalVerifier) — or all-or-nothing

All three strip the <think> block first and give 0 to unfinished thinking
(no ``</think>``) when ``require_finished_thinking`` — a truncated rollout must
not be rewarded for a boxed number that appeared inside its reasoning.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from mopd.graders import code_grader, if_grader, math_grader


def _texts(completions) -> List[str]:
    out = []
    for c in completions:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list) and c and isinstance(c[0], dict):
            out.append(c[-1].get("content", ""))
        else:
            out.append(str(c))
    return out


class DomainReward:
    """Callable reward with per-domain routing, a rollout log and running stats."""

    __name__ = "domain_reward"

    def __init__(self, log_dir: Optional[str] = None, code_workers: int = 16, code_timeout: int = 6,
                 if_all_or_nothing: bool = False, require_finished_thinking: bool = True, thinking: bool = True):
        self.code_workers = code_workers
        self.code_timeout = code_timeout
        self.if_all_or_nothing = if_all_or_nothing
        self.require_finished = require_finished_thinking and thinking
        self.log_path = os.path.join(log_dir, "rollouts.jsonl") if log_dir else None
        self._lock = threading.Lock()
        self._stats: Dict[str, List[float]] = {}
        self._lens: List[int] = []
        self._unfinished = 0
        self._n = 0
        if self.log_path:
            os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------ per-domain
    def _math(self, texts, answers) -> List[float]:
        return [1.0 if math_grader.grade(t, str(a)) else 0.0 for t, a in zip(texts, answers)]

    def _code(self, texts, tests) -> List[float]:
        items = []
        for t, io in zip(texts, tests):
            io = json.loads(io) if isinstance(io, str) else io
            items.append({"input_output": io, "code": code_grader.extract_code(t)})
        graded = code_grader.grade_batch(items, num_workers=self.code_workers, timeout=self.code_timeout)
        return [1.0 if g["passed"] else 0.0 for g in graded]

    def _if(self, texts, gts) -> List[float]:
        return [if_grader.ifevalg_reward(t, gt, all_or_nothing=self.if_all_or_nothing) for t, gt in zip(texts, gts)]

    # ------------------------------------------------------------------ trl entry
    def __call__(self, prompts=None, completions=None, **kw) -> List[float]:
        texts = _texts(completions)
        n = len(texts)
        domains = kw.get("domain") or ["math"] * n
        rewards = [0.0] * n
        finished = [("</think>" in t) or (not self.require_finished) for t in texts]
        groups: Dict[str, List[int]] = {}
        for i, d in enumerate(domains):
            groups.setdefault(d, []).append(i)
        for d, idx in groups.items():
            sub = [texts[i] if finished[i] else "" for i in idx]
            if d == "math":
                r = self._math(sub, [kw["answer"][i] for i in idx])
            elif d == "code":
                r = self._code(sub, [kw["tests"][i] for i in idx])
            elif d == "if":
                r = self._if(sub, [kw["ground_truth"][i] for i in idx])
            else:
                raise ValueError(f"unknown domain {d}")
            for i, v in zip(idx, r):
                rewards[i] = float(v) if finished[i] else 0.0
        self._record(prompts, texts, domains, rewards, finished, kw)
        return rewards

    def _record(self, prompts, texts, domains, rewards, finished, kw):
        with self._lock:
            for d, r in zip(domains, rewards):
                self._stats.setdefault(d, []).append(r)
            self._unfinished += sum(1 for f in finished if not f)
            self._n += len(texts)
            if self.log_path:
                with open(self.log_path, "a") as f:
                    for i in range(len(texts)):
                        f.write(json.dumps({"t": time.time(), "domain": domains[i], "reward": rewards[i],
                                            "finished_thinking": finished[i], "id": (kw.get("id") or [None] * len(texts))[i],
                                            "len_chars": len(texts[i]), "answer_head": math_grader.strip_thinking(texts[i])[:300]},
                                           ensure_ascii=False) + "\n")

    def flush_stats(self) -> Dict[str, float]:
        with self._lock:
            out = {f"reward/{d}": (sum(v) / len(v)) for d, v in self._stats.items() if v}
            out["reward/unfinished_thinking_frac"] = self._unfinished / max(self._n, 1)
            self._stats = {}
            self._unfinished = 0
            self._n = 0
        return out
