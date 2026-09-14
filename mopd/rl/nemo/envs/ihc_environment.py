# IH-Challenge (safety) environment for NeMo-RL: reward = the row's own Python grader applied to the defender's
# post-thinking answer (1.0 = the higher-priority instruction was followed despite the injected attack, else 0.0).
# Deterministic, no LLM judge, no <think> format gate (recipe RECIPES.md §2.5). Graders run in forked children with a
# real 5 s timeout inside worker actors (mopd.graders.domains.ihc_grader).
from __future__ import annotations

import itertools
from typing import Any, NotRequired, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.utils import chunk_list_to_workers


class IHCEnvConfig(TypedDict):
    num_workers: int
    timeout: NotRequired[float]


class IHCEnvMetadata(TypedDict):
    grader_code: str
    attack: str
    task_type: NotRequired[str]


@ray.remote  # pragma: no cover
class IHCGradeWorker:
    def __init__(self, timeout: float):
        from mopd.graders.domains.ihc_grader import grade_batch
        self._grade = grade_batch
        self.timeout = timeout

    def grade(self, items: list[dict[str, Any]]) -> list[float]:
        return self._grade(items, timeout=self.timeout)


@ray.remote(max_restarts=-1, max_task_retries=-1, max_concurrency=1000)  # pragma: no cover
class MopdIHCEnvironment(EnvironmentInterface):
    def __init__(self, cfg: IHCEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self._counter = itertools.count()
        runtime_env = {"py_executable": PY_EXECUTABLES.SYSTEM}
        self.workers = [IHCGradeWorker.options(runtime_env=runtime_env).remote(float(cfg.get("timeout", 5.0)))
                        for _ in range(self.num_workers)]

    def shutdown(self) -> None:
        for w in self.workers:
            ray.kill(w)

    def step(self, message_log_batch: list[LLMMessageLogType], metadata: list[IHCEnvMetadata]) -> EnvironmentReturn:
        items = []
        for conv, meta in zip(message_log_batch, metadata):
            resp = "".join(str(m["content"]) for m in conv if m["role"] == "assistant")
            items.append({"response": resp, "grader_code": meta["grader_code"], "attack": meta.get("attack", "")})
        widx = next(self._counter) % self.num_workers
        chunks = chunk_list_to_workers(items, self.num_workers)
        futures = [self.workers[(widx + i) % self.num_workers].grade.remote(c) for i, c in enumerate(chunks)]
        rewards: list[float] = []
        for r in ray.get(futures):
            rewards.extend(r)
        observations = [{"role": "environment", "content": "Environment: correct" if r > 0.5 else "Environment: incorrect"}
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
