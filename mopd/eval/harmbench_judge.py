"""HarmBench judge step — the official ``cais/HarmBench-Llama-2-13b-cls`` verdicts.

    python -m mopd.eval.harmbench_judge --out outputs/eval_all/<TAG> \
        [--cls $S/models/HarmBench-Llama-2-13b-cls] [--num-tokens 512] [--tp 1]

Runs inside the SAME Slurm job as the generation (one extra engine per job, not per
benchmark), between ``run_all --merge-only`` and ``run_all --aggregate-only``.  Reads
``harmbench.gen.jsonl``, writes ``harmbench.labels.jsonl``; if it fails, the aggregate
step still scores every other benchmark and reports harmbench as unlabelled.

Faithful to ``evaluate_completions.py``:
  * the judged text is clipped to ``--num-tokens`` (512) CLASSIFIER tokens,
    ``truncation_side="right"`` — done here with the classifier's own tokenizer
  * ``LLAMA2_CLS_PROMPT`` / ``prompt_contextual`` for behaviors tagged ``context``
  * ``SamplingParams(temperature=0.0, max_tokens=1)``, verdict ``yes``/``no``, else -1
  * over-long classifier prompts are truncated from the LEFT (upstream sets
    ``cls.llm_engine.tokenizer.tokenizer.truncation_side="left"``; vLLM's own
    ``truncate_prompt_tokens`` is the same left truncation)
The one deviation, documented in mopd/graders/ood/harmbench.py: the clip is applied to the
model's post-</think> ANSWER, because a thinking model's first 512 tokens are reasoning.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from mopd.common.io import read_jsonl
from mopd.eval.ood_benches import load_harmbench
from mopd.graders.ood.harmbench import harmbench_answer, judge_prompt

DEFAULT_CLS = os.path.join(os.environ.get("MOPD_MODELS")
                           or os.path.join(os.environ.get("MOPD_REPO", "."), "models"),
                           "HarmBench-Llama-2-13b-cls")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cls", default=DEFAULT_CLS)
    ap.add_argument("--num-tokens", type=int, default=512)
    ap.add_argument("--tp", type=int, default=1)
    # the classifier's own config says max_position_embeddings=2048, and upstream
    # passes no max_model_len at all -> vLLM uses exactly that.
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    gen_path = os.path.join(a.out, "harmbench.gen.jsonl")
    if not os.path.exists(gen_path):
        print("[judge] no %s -- nothing to judge" % gen_path)
        return 0
    rows = load_harmbench(limit=a.limit)
    gens = read_jsonl(gen_path)
    assert len(gens) == len(rows), "%d generations for %d behaviors" % (len(gens), len(rows))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.cls)
    tok.truncation_side = "right"          # official: clip the generation from the right

    prompts: List[str] = []
    index: List[Dict[str, Any]] = []
    for r, g in zip(rows, gens):
        for j, s in enumerate(g["samples"]):
            ans = harmbench_answer(s["text"])
            # VERBATIM upstream call (it keeps add_special_tokens at its default, so the
            # BOS counts toward max_length and decode(skip_special_tokens=True) drops it)
            ids = tok.encode(ans, max_length=a.num_tokens, truncation=True)
            clipped = tok.decode(ids, skip_special_tokens=True)
            prompts.append(judge_prompt(r, clipped))
            index.append({"id": r["id"], "sample": j, "answer_tokens": len(ids)})

    llm = LLM(model=a.cls, tensor_parallel_size=a.tp, dtype="bfloat16",
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem_util,
              trust_remote_code=True, enforce_eager=(a.tp > 2))
    params = SamplingParams(temperature=0.0, max_tokens=1,
                            truncate_prompt_tokens=a.max_model_len - 8)
    outs = llm.generate(prompts, params)

    from third_party.harmbench.harmbench_official import parse_classifier_label

    tmp = os.path.join(a.out, "harmbench.labels.jsonl.tmp")
    n_yes = n_bad = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for meta, o in zip(index, outs):
            verdict = o.outputs[0].text
            label = parse_classifier_label(verdict)
            n_yes += int(label == 1)
            n_bad += int(label == -1)
            f.write(json.dumps({**meta, "label": label, "verdict": verdict},
                               ensure_ascii=False) + "\n")
    os.replace(tmp, os.path.join(a.out, "harmbench.labels.jsonl"))
    print("[judge] %d completions: %d yes, %d unparseable -> ASR %.2f%%"
          % (len(index), n_yes, n_bad, 100.0 * n_yes / max(len(index), 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
