# Competitive-coding environment for NeMo-RL, grading Nemotron rlvr1 code_gen rows with
# the OFFICIAL LiveCodeBench execution semantics via mopd's HARDENED harness
# (mopd/graders/code_grader.py: LCB testing_util + per-program tmpbox + Landlock/seccomp
# sandbox + reliability guard — the official NeMo Gym code_gen server runs the same LCB
# code but with in-process SIGALRM only; ours adds the 2026-09-01 incident protections).
# Reward: ALL unit tests pass -> 1.0 else 0.0 (official all-or-nothing).
# rlvr1 unit_tests are stdin/stdout string pairs (verified: 7,929/7,929 rows well-formed,
# no fn_name, median 42 tests/problem).
from __future__ import annotations

import itertools
import os
import sys
from typing import Any, NotRequired, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.utils import chunk_list_to_workers


class CodeGenEnvConfig(TypedDict):
    num_workers: int
    mopd_rl_root: str                      # repo root providing `mopd` + `third_party` packages
    per_test_timeout: NotRequired[int]     # LCB default 6s
    max_tests_per_problem: NotRequired[int]  # bound worst-case (403 tests x 6s); 0 = all


@ray.remote  # pragma: no cover
class CodeVerifyWorker:
    def __init__(self, mopd_rl_root: str) -> None:
        if mopd_rl_root not in sys.path:
            sys.path.insert(0, mopd_rl_root)
        os.environ.setdefault("MOPD_SANDBOX", "1")

    def verify(self, items: list[dict[str, Any]], per_test_timeout: int,
               max_tests: int) -> list[float]:
        """items: [{response, unit_tests: {inputs, outputs}}] -> [0.0|1.0]"""
        from mopd.graders import code_grader
        out: list[float] = []
        for it in items:
            try:
                ut = it.get("unit_tests") or {}
                ins, outs = list(ut.get("inputs") or []), list(ut.get("outputs") or [])
                if not ins or len(ins) != len(outs):
                    out.append(0.0)  # official Gym crashes on missing tests; we score 0 (logged upstream)
                    continue
                if max_tests and len(ins) > max_tests:
                    ins, outs = ins[:max_tests], outs[:max_tests]
                code = code_grader.extract_code(str(it.get("response") or ""))
                g = code_grader.grade_code({"inputs": ins, "outputs": outs, "fn_name": None},
                                           code, timeout=per_test_timeout)
                out.append(1.0 if g["passed"] else 0.0)
            except Exception:
                out.append(0.0)
        return out


class CodeEnvMetadata(TypedDict):
    unit_tests: dict[str, Any]


@ray.remote(max_restarts=-1, max_task_retries=-1, max_concurrency=1000)  # pragma: no cover
class MopdCodeGenEnvironment(EnvironmentInterface):
    def __init__(self, cfg: CodeGenEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.per_test_timeout = int(cfg.get("per_test_timeout", 6))
        self.max_tests = int(cfg.get("max_tests_per_problem", 0))
        self._worker_counter = itertools.count()
        self.workers = [
            CodeVerifyWorker.options(runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM})
            .remote(cfg["mopd_rl_root"])
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        for w in self.workers:
            ray.kill(w)

    def step(self, message_log_batch: list[LLMMessageLogType],
             metadata: list[CodeEnvMetadata]) -> EnvironmentReturn:
        items = []
        for conv, meta in zip(message_log_batch, metadata):
            resp = "".join(str(m["content"]) for m in conv if m["role"] == "assistant")
            items.append({"response": resp, "unit_tests": meta.get("unit_tests")})
        widx = next(self._worker_counter) % self.num_workers
        chunks = chunk_list_to_workers(items, self.num_workers)
        futures = [self.workers[(widx + i) % self.num_workers].verify.remote(
            c, self.per_test_timeout, self.max_tests) for i, c in enumerate(chunks)]
        rewards: list[float] = []
        for r in ray.get(futures):
            rewards.extend(r)
        observations = [{"role": "environment",
                         "content": "Environment: correct" if r > 0.5 else "Environment: incorrect"}
                        for r in rewards]
        rewards_t = torch.tensor(rewards, dtype=torch.float32).cpu()
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards_t,
            terminateds=torch.ones_like(rewards_t, dtype=torch.bool).cpu(),
            answers=[None] * len(message_log_batch),
        )

    def global_post_process_and_metrics(self, batch) -> tuple[Any, dict[str, float | int]]:
        rewards = batch["rewards"] * batch["is_end"]
        return batch, {
            "accuracy": rewards.mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(batch["text"], rewards),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
        }
