# Math environment with LLM-judge fallback for NeMo-RL, porting the OFFICIAL NeMo Gym
# resources_servers/math_with_judge logic (the verifier the Nemotron-RL-Ultra blends were
# built for) onto nemo_rl.environments.math_environment scaffolding:
#   1) Math-Verify library check with the Gym's delimiter-stripping:
#      gold = "\boxed{" + strip_math_delimiters(expected) + "}"
#   2) if library_reward <= 0.5: bidirectional Arena-Hard equivalence judge
#      (order-1 expected-vs-generated must say [[A=B]]; then order-2 swapped must
#      also say [[A=B]] -> reward 1.0; anything else -> 0.0; malformed -> not-equal)
# Judge transport: OpenAI-compatible /v1/chat/completions on a vLLM server we run on a
# dedicated node (config: judge_base_url/judge_model). Judge input = extracted answer if
# available else the generated text with the <think> block stripped and tail-truncated
# (deviation from Gym, which passes the full text: our rollouts are 32k thinking traces
# and the judge context is 8k).
from __future__ import annotations

import contextlib
import io
import itertools
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NotRequired, Optional, TypedDict

import ray
import torch
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.math_environment import BaseMathEnvironment
from nemo_rl.environments.utils import chunk_list_to_workers

# verbatim from NeMo Gym resources_servers/math_with_judge/app.py
JUDGE_SYSTEM_MESSAGE = """Please act as an impartial judge and evaluate the equivalence of the solutions given by two AI assistants to the mathematical problem displayed below. You will be given AI assistant A's answer and AI assistant B's answer. Your job is to evaluate whether assistant A's answer is equivalent to assistant B's answer.

Consider the mathematical equivalence of the AI assistants' answers above all other considerations. If the problem requests special formatting instructions, you may disregard any formatting considerations when evaluating the answers -- consider only mathematical equivalence.

After evaluating both answers for equivalence, you must output only one of the following choices as your final verdict with a label:

1.  The AI assistants' answers are equivalent: [[A=B]]
2.  The AI assistants' answers are different: [[A!=B]]

Example output: "My final verdict is different [[A!=B]]"."""
JUDGE_PROMPT_TEMPLATE = (
    "<|Problem|>\n{question}\n\n<|Start of Assistant A's Answer|>\n{first_answer}"
    "\n<|End of Assistant A's Answer|>\n\n<|Start of Assistant B's Answer|>\n"
    "{second_answer}\n<|End of Assistant B's Answer|>"
)
EQ_LABEL, NEQ_LABEL = "[[A=B]]", "[[A!=B]]"


class MathWithJudgeEnvConfig(TypedDict):
    num_workers: int
    judge_base_url: str            # e.g. http://<judge-node>:8231/v1
    judge_model: str               # served-model-name
    should_use_judge: NotRequired[bool]
    judge_max_tokens: NotRequired[int]
    judge_timeout_seconds: NotRequired[float]
    judge_concurrency: NotRequired[int]
    judge_answer_max_chars: NotRequired[int]
    stop_strings: NotRequired[list[str] | None]


def strip_math_delimiters(s: str) -> str:
    """== Gym LibraryJudgeMathResourcesServer._strip_math_delimiters."""
    s = s.strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1].strip()
    return s


def parse_judge_verdict(text: str) -> bool:
    """== Gym label parsing: first label wins; malformed -> not-equal."""
    e, n = text.find(EQ_LABEL), text.find(NEQ_LABEL)
    if e < 0:
        return False
    if n < 0:
        return True
    return e < n


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(devnull_out), contextlib.redirect_stderr(devnull_err):
        yield


@ray.remote  # pragma: no cover
class GymStyleVerifyWorker:
    """Math-Verify worker with the Gym's expected-answer delimiter stripping."""

    def __init__(self) -> None:
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )

    def verify(self, pred_responses: list[str], ground_truths: list[str]
               ) -> tuple[list[float], list[Optional[str]]]:
        scores: list[float] = []
        extracted: list[Optional[str]] = []
        for response, gt in zip(pred_responses, ground_truths):
            try:
                gold = "\\boxed{" + strip_math_delimiters(gt) + "}"
                with _mute_output():
                    ret_score, ext = self.verify_func([gold], [response])
                ans = None
                if ext is not None and len(ext) == 2:
                    gold_ext, pred_ext = ext
                    from math_verify import grader
                    for pred in pred_ext:
                        try:
                            if any(grader.verify(g, pred) for g in gold_ext):
                                ans = pred
                                break
                        except BaseException:
                            continue
                    else:
                        ans = pred_ext[0] if pred_ext else None
                scores.append(float(ret_score))
                extracted.append(str(ans) if ans is not None else None)
            except BaseException:
                scores.append(0.0)
                extracted.append(None)
        return scores, extracted


class MathWithJudgeMetadata(TypedDict):
    ground_truth: str
    extracted_answer: NotRequired[str | None]
    library_reward: NotRequired[float]
    used_judge: NotRequired[bool]


@ray.remote(max_restarts=-1, max_task_retries=-1, max_concurrency=1000)  # pragma: no cover
class MathWithJudgeEnvironment(BaseMathEnvironment):
    WORKER_CLASS_DICT = {"math_with_judge": GymStyleVerifyWorker}

    def __init__(self, cfg: MathWithJudgeEnvConfig):
        self.cfg = dict(cfg)
        self.cfg.setdefault("verifier_type", "math_with_judge")
        self.num_workers = cfg["num_workers"]
        self._worker_counter = itertools.count()
        self.workers = [
            GymStyleVerifyWorker.options(runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}).remote()
            for _ in range(self.num_workers)
        ]
        # "file:///path" -> the URL is (re-)read from that file at every step, so a
        # babysitter-restarted judge job (new node/hostname) picks up transparently.
        self._judge_url_cfg = cfg["judge_base_url"]
        try:
            self.judge_base_url = self._resolve_judge_url()
        except Exception:  # e.g. judge disabled and the URL file does not exist yet
            self.judge_base_url = ""
        self.judge_model = cfg["judge_model"]
        self.should_use_judge = bool(cfg.get("should_use_judge", True))
        self.judge_max_tokens = int(cfg.get("judge_max_tokens", 4096))
        self.judge_timeout = float(cfg.get("judge_timeout_seconds", 600.0))
        self.judge_concurrency = int(cfg.get("judge_concurrency", 16))
        self.judge_answer_max_chars = int(cfg.get("judge_answer_max_chars", 4000))
        import requests
        self._session = requests.Session()
        self._session.trust_env = False  # never route judge calls through the corp proxy

    # ------------------------------------------------------------------ judge
    def _resolve_judge_url(self) -> str:
        u = self._judge_url_cfg
        if u.startswith("file://"):
            with open(u[len("file://"):]) as f:
                u = f.read().strip()
        return u.rstrip("/")

    def _judge_once(self, question: str, first: str, second: str) -> bool:
        r = self._session.post(
            self.judge_base_url + "/chat/completions",
            json={"model": self.judge_model, "temperature": 0.0,
                  "max_tokens": self.judge_max_tokens,
                  "messages": [
                      {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
                      {"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(
                          question=question, first_answer=first, second_answer=second)}],
                  "chat_template_kwargs": {"enable_thinking": False}},
            timeout=self.judge_timeout,
        )
        r.raise_for_status()
        return parse_judge_verdict(r.json()["choices"][0]["message"]["content"] or "")

    def _judge_pair(self, question: str, expected: str, generated: str) -> float:
        """Official bidirectional protocol; any transport error -> 0.0 (conservative)."""
        try:
            if not self._judge_once(question, expected, generated):
                return 0.0
            return 1.0 if self._judge_once(question, generated, expected) else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ step
    def step(self, message_log_batch: list[LLMMessageLogType],
             metadata: list[MathWithJudgeMetadata],
             return_extracted_answer: bool = False) -> EnvironmentReturn:
        responses, questions = [], []
        for conversation in message_log_batch:
            responses.append("".join(str(m["content"]) for m in conversation
                                     if m["role"] == "assistant"))
            user_msgs = [str(m["content"]) for m in conversation if m["role"] == "user"]
            questions.append(user_msgs[0] if user_msgs else "")
        ground_truths = [g["ground_truth"] for g in metadata]

        widx = next(self._worker_counter) % self.num_workers
        chunks_r = chunk_list_to_workers(responses, self.num_workers)
        chunks_g = chunk_list_to_workers(ground_truths, self.num_workers)
        futures = [self.workers[(widx + i) % self.num_workers].verify.remote(cr, cg)
                   for i, (cr, cg) in enumerate(zip(chunks_r, chunks_g))]
        lib_scores: list[float] = []
        lib_extracted: list[Optional[str]] = []
        for s, e in ray.get(futures):
            lib_scores.extend(s)
            lib_extracted.extend(e)

        rewards = list(lib_scores)
        used_judge = [False] * len(rewards)
        if self.should_use_judge:
            need = [i for i, s in enumerate(lib_scores) if s <= 0.5]
            if need:
                try:
                    self.judge_base_url = self._resolve_judge_url()
                except Exception:
                    pass
                def run(i: int) -> tuple[int, float]:
                    ans = lib_extracted[i]
                    if not ans:
                        ans = strip_thinking(responses[i]).strip()[-self.judge_answer_max_chars:]
                    return i, self._judge_pair(questions[i], ground_truths[i], ans)
                with ThreadPoolExecutor(max_workers=self.judge_concurrency) as ex:
                    for i, r in ex.map(run, need):
                        rewards[i] = r
                        used_judge[i] = True

        observations = [
            {"role": "environment",
             "content": "Environment: correct" if r > 0.5 else "Environment: incorrect"}
            for r in rewards
        ]
        out_meta = []
        for i, m in enumerate(metadata):
            m2 = dict(m)
            m2["extracted_answer"] = lib_extracted[i]
            m2["library_reward"] = lib_scores[i]
            m2["used_judge"] = used_judge[i]
            out_meta.append(m2)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).cpu()
        done_t = torch.ones_like(rewards_t, dtype=torch.bool).cpu()
        next_stop_strings = [None] * len(message_log_batch)
        return EnvironmentReturn(
            observations=observations,
            metadata=out_meta,
            next_stop_strings=next_stop_strings,
            rewards=rewards_t,
            terminateds=done_t,
            answers=lib_extracted,
        )
