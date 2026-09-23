"""Phase-3 OOD benchmarks: loaders, prompts and per-bench protocol.

Four benchmarks, frozen by ``mopd/data/prep_ood_eval.py`` into ``data/eval/<name>/``:

  gpqa           198 GPQA-Diamond MCQs      generation, avg@4, simple-evals prompt+regex
  harmbench      300 HarmBench behaviors    generation, greedy, official 13B classifier
  sycophancy  30,051 Anthropic rows         log-prob comparison of the answer options
  truthfulqa_mc  817 TruthfulQA MC          log-prob, official QA_PRIMER + MC_calcs

Two request kinds, both served by one vLLM engine per GPU (``mopd/eval/run_all.py``):
  ``gen``  sample n completions
  ``lp``   score fixed continuations (prompt_logprobs) — no generation at all

All generative benchmarks sample exactly like the training rollouts (one preset,
``registry.EVAL_SAMPLING``); deviations from a benchmark's published decoding setting are
documented per benchmark in ``docs/EVAL_ALL.md``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from mopd.common.io import data_root, iter_jsonl
from third_party.simple_evals.gpqa_official import gpqa_prompt
from third_party.truthfulqa.truthfulqa_official import qa_context, qa_continuation


def _path(name: str) -> str:
    return os.path.join(data_root(), "eval", name, "test.jsonl")


def _load(name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    for r in iter_jsonl(_path(name)):
        rows.append(r)
        if limit and len(rows) >= limit:
            break
    return rows


# ----------------------------------------------------------------------------- gpqa
def load_gpqa(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """The frozen conductor rows carry the choices/answer inside ``gpqa_row``."""
    out = []
    for r in _load("gpqa", limit):
        aux = json.loads(r["gpqa_row"])
        assert aux["answer"] == r["ground_truth"], "frozen gpqa row is inconsistent"
        out.append({"id": aux["id"], "question": aux["question"], "choices": aux["choices"],
                    "answer": aux["answer"], "subdomain": aux.get("subdomain", ""),
                    "high_level_domain": aux.get("high_level_domain", "")})
    return out


def gpqa_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"role": "user", "content": gpqa_prompt(row["question"], row["choices"])}]


# ----------------------------------------------------------------------------- harmbench
def load_harmbench(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    return _load("harmbench", limit)


def harmbench_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Official DirectRequest test case as the single user turn; no system prompt
    (HarmBench uses each model's own template with its default system message, and the
    mopd chat contract has none)."""
    return [{"role": "user", "content": row["test_case"]}]


# ----------------------------------------------------------------------------- sycophancy
def load_sycophancy(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    return _load("sycophancy", limit)


def sycophancy_specs(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ci 0 = the sycophantic answer, then every non-matching answer.

    The question already ends with "Answer:"; the official wrapper continues the
    assistant turn with "I believe the best answer is" / "I believe the better option is",
    and the scored continuation is the raw " (X)" string from the dataset.
    """
    choices = [row["matching"]] + list(row["not_matching"])
    return [{"mode": "chat", "messages": [{"role": "user", "content": row["question"]}],
             "thinking": False, "assistant_prefix": row["assistant_prefix"],
             "cont": c} for c in choices]


# ----------------------------------------------------------------------------- truthfulqa
def load_truthfulqa(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = _load("truthfulqa_mc", limit)
    for r in rows:
        seen: Dict[str, int] = {}
        for c in list(r["mc1_choices"]) + list(r["mc2_choices"]):
            if c not in seen:
                seen[c] = len(seen)
        r["_choices"] = list(seen)      # dedup: mc1 answers usually reappear in mc2
    return rows


def truthfulqa_specs(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    ctx = qa_context(row["question"])
    return [{"mode": "raw", "text": ctx, "thinking": False, "assistant_prefix": "",
             "cont": qa_continuation(c)} for c in row["_choices"]]

# ----------------------------------------------------------------------------- safety (phase 3-2)
def load_xstest(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """250 safe + 200 unsafe contrast prompts (XSTest, NAACL 2024)."""
    return _load("xstest", limit)


def xstest_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """The bare prompt as the single user turn -- XSTest ships no wrapper, and adding one would
    change exactly what the test measures (how the model reacts to the prompt itself)."""
    return [{"role": "user", "content": row["prompt"]}]


def load_strongreject(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """313 forbidden prompts (StrongREJECT, arXiv 2402.10260)."""
    return _load("strongreject", limit)


def strongreject_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"role": "user", "content": row["prompt"]}]


def load_safetybench(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """11,435 safety multiple-choice questions, 7 categories (SafetyBench, ACL 2024)."""
    return _load("safetybench", limit)


def safetybench_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Official zero-shot EN template from code/evaluate_base.py:
    'Question: {q}\nOptions:\n(A) ...\nAnswer:' -- kept verbatim, including the trailing
    'Answer:' with no space."""
    letters = ["(A)", "(B)", "(C)", "(D)"]
    opts = "".join("%s %s\n" % (letters[i], o) for i, o in enumerate(row["options"]))
    return [{"role": "user", "content": "Question: %s\nOptions:\n%sAnswer:"
                                        % (row["question"].strip(), opts)}]

def load_sorrybench(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """440 class-balanced unsafe instructions over 44 topics (SORRY-Bench, ICLR 2025)."""
    return _load("sorrybench", limit)


def sorrybench_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"role": "user", "content": row["prompt"]}]
