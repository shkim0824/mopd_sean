"""Domain benchmark eval runner (offline vLLM, DP workers over row shards).

Worker mode (one process per GPU, launched by the sbatch script):
  python -m mopd.eval.domains.run_domain_eval --model M --benchmarks a,b --out O \
      --shard-rank R --num-shards N [--limit K] [--max-tokens 32768] [--temperature 0.6]
Aggregate mode (after all shards exist):
  python -m mopd.eval.domains.run_domain_eval --aggregate-only --benchmarks a,b --out O

Sampling (2026-09-16) = the training rollouts: T=1.0 (``--temperature``), top_p 1.0, top_k
disabled; chat template with enable_thinking=True. Shards are resume-safe (skip existing).
NOTE every ``-T1`` dom3 directory on disk predates this and was produced with top_p 0.95 /
top_k 20, so its numbers are not comparable cell-by-cell with a fresh run.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from mopd.eval.domains.benches import BENCHES


def build_rows(names: List[str], limit: int | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for n in names:
        bench_rows = BENCHES[n](limit=limit) if limit else BENCHES[n]()
        for r in bench_rows:
            r["bench"] = n
        rows.extend(bench_rows)
    return rows


def worker(args) -> None:
    rows = build_rows(args.benchmarks.split(","), args.limit)
    mine = rows[args.shard_rank::args.num_shards]
    out_f = os.path.join(args.out, f"shard{args.shard_rank:03d}.jsonl")
    if os.path.exists(out_f + ".done"):
        print(f"[eval-worker {args.shard_rank}] already done"); return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              tensor_parallel_size=args.tp, gpu_memory_utilization=0.9, trust_remote_code=True)
    # training-matched sampling (2026-09-16): top_p 1.0, top_k disabled. Before that this
    # runner used top_p 0.95 / top_k 20, which is what every `-T1` dom3 dir on disk holds.
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        max_tokens=args.max_tokens, seed=args.seed)
    prompts = [tok.apply_chat_template(r.get("messages") or [{"role": "user", "content": r["prompt"]}],
                                       add_generation_prompt=True, tokenize=False,
                                       enable_thinking=True) for r in mine]   # ihc rows carry system+user messages
    outs = llm.generate(prompts, sp)
    tmp = out_f + ".tmp"
    with open(tmp, "w") as f:
        for r, o in zip(mine, outs):
            f.write(json.dumps({"id": r["id"], "bench": r["bench"],
                                "response": o.outputs[0].text,
                                "finish": o.outputs[0].finish_reason},
                               ensure_ascii=False) + "\n")
    os.replace(tmp, out_f)
    open(out_f + ".done", "w").write("ok")
    print(f"[eval-worker {args.shard_rank}] wrote {len(mine)} rows")


def aggregate(args) -> None:
    import glob
    from mopd.graders.domains import graders as G
    names = args.benchmarks.split(",")
    rows = build_rows(names, args.limit)
    meta = {r["id"]: r for r in rows}
    resp: Dict[str, Dict[str, Any]] = {}
    for f in sorted(glob.glob(os.path.join(args.out, "shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            resp[r["id"]] = r
    metrics: Dict[str, Any] = {}
    by_bench: Dict[str, List[str]] = defaultdict(list)
    for rid in resp:
        by_bench[resp[rid]["bench"]].append(rid)
    for bench, ids in by_bench.items():
        kind = meta[ids[0]]["grade_kind"]
        n_missing = sum(1 for r in rows if r["bench"] == bench) - len(ids)
        scored: Dict[str, Any] = {"n": len(ids), "missing": n_missing}
        gens = [resp[i]["response"] for i in ids]
        if kind == "mcqa":
            res = [G.grade_mcqa(g, meta[i]["gold"], meta[i].get("n_options", 4))
                   for g, i in zip(gens, ids)]
            scored["accuracy"] = sum(r["correct"] for r in res) / len(res)
            scored["no_answer_frac"] = sum(r["pred"] is None for r in res) / len(res)
        elif kind == "pubmedqa":
            scored.update(G.grade_pubmedqa_batch(gens, [meta[i]["gold"] for i in ids]))
            scored.pop("preds", None)
        elif kind == "legalbench":
            per_task: Dict[str, List[int]] = defaultdict(list)
            for j, i in enumerate(ids):
                per_task[meta[i]["task"]].append(j)
            task_scores = []
            for task, js in per_task.items():
                t = G.grade_legalbench_task(task, [gens[j] for j in js],
                                            [meta[ids[j]]["gold"] for j in js])
                task_scores.append(t["score"])
            scored["balanced_acc_mean"] = sum(task_scores) / len(task_scores)
            scored["n_tasks"] = len(task_scores)
        elif kind == "fin_numeric":
            res = [G.grade_fin_numeric(g, meta[i]["gold"]) for g, i in zip(gens, ids)]
            scored["accuracy"] = sum(r["correct"] for r in res) / len(res)
        elif kind == "ihc":
            items = [{"response": g, "grader_code": meta[i]["grader_code"], "attack": meta[i].get("attack", ""),
                      "task_type": meta[i].get("task_type", "?")} for g, i in zip(gens, ids)]
            scored.update(G.grade_ihc_batch(items))
        elif kind == "tatqa":
            items = [{"response": g, "gold_answer": meta[i]["gold"],
                      "gold_scale": meta[i].get("gold_scale", ""),
                      "answer_type": meta[i].get("answer_type", "arithmetic"),
                      "uid": i} for g, i in zip(gens, ids)]
            scored.update(G.grade_tatqa_batch(items))
        trunc = sum(1 for i in ids if resp[i].get("finish") == "length")
        scored["truncated_frac"] = trunc / len(ids)
        metrics[bench] = scored
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--benchmarks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=1.0, help="1.0 = training-matched default; 0.95 with --top-k 20 = the Qwen thinking preset")
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--shard-rank", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--aggregate-only", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.aggregate_only:
        aggregate(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
