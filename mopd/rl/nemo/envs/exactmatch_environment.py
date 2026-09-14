"""Deterministic exact-match RL environment for the med/law/fin domain pools.
Modeled on mopd.rl.nemo.envs.math_with_judge_environment (same NeMo-RL interfaces),
WITHOUT any judge: rewards are pure machine verification, dispatched by the
ground-truth format:
  - single letter A-J        -> option-letter match (MedQA/MedMCQA/CaseHOLD/MBE)
  - Yes/No                   -> yes-no match (housing_qa)
  - numeric                  -> DocMath-Eval official compare_two_numbers
                                (0.15% rel. tolerance + scale rescues; vendored)
  - anything else            -> normalized exact string match (fallback)
Self-contained on purpose: ray workers only need this single file on PYTHONPATH.
"""
from __future__ import annotations

import itertools
import math
import re
from typing import Optional, TypedDict

import ray
import torch

from nemo_rl.environments.utils import chunk_list_to_workers
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.environments.math_environment import BaseMathEnvironment
from nemo_rl.data.interfaces import LLMMessageLogType

# ----------------------------------------------------------------- extraction
THINK_CLOSE = "</think>"
_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANS_LINE = re.compile(r"(?:answer)\s*(?:is)?\s*[:\-]?\s*\(?([A-Ja-j])\)?(?![a-zA-Z])",
                       re.IGNORECASE)
_LONE_LETTER = re.compile(r"(?<![A-Za-z])([A-J])(?![A-Za-z])")
_YNM = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_NUM = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?%?")


def strip_thinking(text: str) -> str:
    return text.split(THINK_CLOSE)[-1] if THINK_CLOSE in text else text


def extract_letter(response: str, n_options: int = 10) -> Optional[str]:
    valid = {chr(ord("A") + i) for i in range(n_options)}
    text = strip_thinking(response).strip()
    for m in reversed(_BOXED.findall(text)):
        c = m.strip().strip("()").upper()
        if c in valid:
            return c
    for m in reversed(list(_ANS_LINE.finditer(text))):
        c = m.group(1).upper()
        if c in valid:
            return c
    cands = [c.upper() for c in _LONE_LETTER.findall(text[-200:]) if c.upper() in valid]
    return cands[-1] if cands else None


def extract_yesno(response: str) -> Optional[str]:
    text = strip_thinking(response)
    for m in reversed(_BOXED.findall(text)):
        if m.strip().lower() in ("yes", "no"):
            return m.strip().lower()
    hits = [w.lower() for w in _YNM.findall(text)]
    return hits[-1] if hits else None


def _clean_number(tok: str) -> Optional[float]:
    tok = tok.strip().replace("$", "").replace(",", "")
    neg = tok.startswith("(") and tok.endswith(")")
    tok = tok.strip("()").rstrip("%")
    try:
        v = float(tok)
    except ValueError:
        return None
    return -v if neg else v


def extract_number(response: str) -> Optional[float]:
    text = strip_thinking(response)
    for m in reversed(_BOXED.findall(text)):
        for tok in reversed(_NUM.findall(m)):
            v = _clean_number(tok)
            if v is not None:
                return v
    segs = re.split(r"(?:answer|answer is|final answer)\s*[:\-]?", text,
                    flags=re.IGNORECASE)
    for seg in reversed(segs[1:] or []):
        for tok in _NUM.findall(seg[:120]):
            v = _clean_number(tok)
            if v is not None:
                return v
    toks = _NUM.findall(text[-300:])
    for tok in reversed(toks):
        v = _clean_number(tok)
        if v is not None:
            return v
    return None


# ---------------- DocMath-Eval official comparator (VENDORED VERBATIM, MIT) ----
def _dm_within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.0015
    return gt - eps <= pred <= gt + eps


def _dm_round_up_to_decimal(number, decimals):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def _dm_compare_two_numbers(p, gt):
    if not isinstance(p, (int, float)) or isinstance(p, bool):
        return False
    v1, v2 = max(abs(gt), abs(p)), min(abs(gt), abs(p))
    if (v1 != 0 and v2 != 0) and int(math.log10(v1 / v2)) == math.log10(v1 / v2):
        return True
    if v2 <= v1 / 50 and _dm_within_eps(pred=v2 * 100, gt=v1):
        return True
    elif v2 <= v1 / 500 and _dm_within_eps(pred=v2 * 1000, gt=v1):
        return True
    elif v2 <= v1 / 50000 and _dm_within_eps(pred=v2 * 100000, gt=v1):
        return True
    if _dm_round_up_to_decimal(v1, 3) == _dm_round_up_to_decimal(v2, 3):
        return True
    return _dm_within_eps(pred=p, gt=gt)


# ----------------------------------------------------------------- verification
_LETTER_GT = re.compile(r"^[A-J]$")


def verify_one(response: str, gt: str) -> tuple[float, Optional[str]]:
    gt = str(gt).strip()
    if _LETTER_GT.match(gt.upper()) and len(gt) == 1:
        pred = extract_letter(response)
        return (1.0 if pred == gt.upper() else 0.0), pred
    if gt.lower() in ("yes", "no"):
        pred = extract_yesno(response)
        return (1.0 if pred == gt.lower() else 0.0), pred
    gv = _clean_number(gt)
    if gv is not None:
        pv = extract_number(response)
        ok = pv is not None and _dm_compare_two_numbers(pv, gv)
        return (1.0 if ok else 0.0), (str(pv) if pv is not None else None)
    tail = strip_thinking(response).strip().lower()
    norm = re.sub(r"[^a-z0-9 ]", "", gt.lower())
    return (1.0 if norm and norm in re.sub(r"[^a-z0-9 ]", "", tail[-300:]) else 0.0), None


@ray.remote(max_restarts=-1, max_task_retries=-1)
class ExactMatchVerifyWorker:
    def verify(self, pred_responses: list[str], ground_truths: list[str]):
        scores, extracted = [], []
        for r, g in zip(pred_responses, ground_truths):
            s, e = verify_one(r, g)
            scores.append(s)
            extracted.append(e)
        return scores, extracted


class DomainExactMatchEnvConfig(TypedDict):
    num_workers: int


class DomainExactMatchMetadata(TypedDict):
    ground_truth: str


@ray.remote(max_restarts=-1, max_task_retries=-1, max_concurrency=1000)  # pragma: no cover
class DomainExactMatchEnvironment(BaseMathEnvironment):
    WORKER_CLASS_DICT = {"domain_exactmatch": ExactMatchVerifyWorker}

    def __init__(self, cfg: DomainExactMatchEnvConfig):
        self.cfg = dict(cfg)
        self.num_workers = cfg["num_workers"]
        self._worker_counter = itertools.count()
        self.workers = [
            ExactMatchVerifyWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}).remote()
            for _ in range(self.num_workers)
        ]

    def step(self, message_log_batch: list[LLMMessageLogType],
             metadata: list[DomainExactMatchMetadata],
             return_extracted_answer: bool = False) -> EnvironmentReturn:
        responses = ["".join(str(m["content"]) for m in conv if m["role"] == "assistant")
                     for conv in message_log_batch]
        ground_truths = [g["ground_truth"] for g in metadata]

        widx = next(self._worker_counter) % self.num_workers
        chunks_r = chunk_list_to_workers(responses, self.num_workers)
        chunks_g = chunk_list_to_workers(ground_truths, self.num_workers)
        futures = [self.workers[(widx + i) % self.num_workers].verify.remote(cr, cg)
                   for i, (cr, cg) in enumerate(zip(chunks_r, chunks_g))]
        rewards: list[float] = []
        extracted: list[Optional[str]] = []
        for s, e in ray.get(futures):
            rewards.extend(s)
            extracted.extend(e)

        observations = [
            {"role": "environment",
             "content": "Environment: correct" if r > 0.5 else "Environment: incorrect"}
            for r in rewards
        ]
        out_meta = []
        for i, m in enumerate(metadata):
            m2 = dict(m)
            m2["extracted_answer"] = extracted[i]
            m2["library_reward"] = rewards[i]
            out_meta.append(m2)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).cpu()
        done_t = torch.ones_like(rewards_t, dtype=torch.bool).cpu()
        return EnvironmentReturn(
            observations=observations,
            metadata=out_meta,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards_t,
            terminateds=done_t,
            answers=extracted,
        )
