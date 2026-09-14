"""Evaluate one model on the MOPD benchmark set using all GPUs of one node.

Data-parallel: the parent shards (benchmark, row) pairs round-robin over --gpus,
spawns one worker process per GPU (each an offline vLLM engine, tp=1), waits,
merges the shards, grades, writes <out>/metrics.json + <out>/<bench>.gen.jsonl.

DEFAULT SAMPLING (user decision 2026-08-30, DEVIATES from the MOPD paper): ``--preset qwen-thinking``
= temperature 0.6, top_p 0.95, top_k 20 (Qwen3 thinking-mode recommendation), 16k tokens,
AIME avg@8, LCB pass@1 (n=1), IFEval/IFBench once.  The paper's protocol (T=1.0, no top-p/top-k,
AIME avg@32) is still available as ``--preset paper``.

Examples
  python -m mopd.eval.run_eval --model outputs/sft/qwen3-8b --out outputs/eval/sft-8b
  python -m mopd.eval.run_eval --model models/Qwen3-8B-Base --out outputs/eval_6bench/base-8b-probe \\
         --benchmarks aime24,ifeval --limit 30 --n 1 --prompt-mode raw   # base-model probe
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

from mopd.common.io import read_jsonl, write_json, write_jsonl
from mopd.eval.benchmarks import BENCHMARKS, DEFAULT_SET, DOMAIN_OF

PRESETS = {
    "paper": dict(temperature=1.0, top_p=1.0, top_k=-1, presence_penalty=0.0),
    "qwen-thinking": dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0),
    "qwen-nonthinking": dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5),
    "greedy": dict(temperature=0.0, top_p=1.0, top_k=-1, presence_penalty=0.0),
}


def parse_gpus(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--benchmarks", default=",".join(DEFAULT_SET))
    p.add_argument("--gpus", default="0-7")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--preset", default="qwen-thinking", choices=list(PRESETS),
                   help="default qwen-thinking (0.6/0.95/20; user decision, differs from the paper's T=1.0)")
    p.add_argument("--n", type=int, default=None, help="samples per prompt for ALL benches (default: per-bench)")
    p.add_argument("--n-math", type=int, default=None)
    p.add_argument("--n-code", type=int, default=None)
    p.add_argument("--n-if", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--limit", type=int, default=None, help="first K rows of every bench (smoke / probe)")
    p.add_argument("--prompt-mode", default="chat", choices=["chat", "raw"])
    p.add_argument("--thinking", default="true", help="enable_thinking flag for the chat template")
    p.add_argument("--math-style", default="qwen", choices=["qwen", "nemo"])
    p.add_argument("--lcb-start-date", default="", help="e.g. 2025-02-01 (Qwen 'v6 25.02-25.05' window)")
    p.add_argument("--lcb-end-date", default="")
    p.add_argument("--lcb-system", default="false",
                   help="LCB system message as a separate system turn (true) or folded into the user turn (false). "
                        "Default false: our SFT data has no system turns and the SFT'd model emitted <|im_end|> "
                        "immediately after a system-prompted LCB request (ck100 eval: 1055/1055 empty).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grade-workers", type=int, default=None)
    p.add_argument("--skip-generation", action="store_true", help="re-grade existing .gen.jsonl only")
    p.add_argument("--num-nodes", type=int, default=1, help="multi-node DP: total nodes (shards are global)")
    p.add_argument("--node-rank", type=int, default=0, help="multi-node DP: this node's rank")
    p.add_argument("--aggregate-only", action="store_true", help="multi-node: merge out shards + grade (run on one node)")
    p.add_argument("--chunk", type=int, default=32, help="prompts per generate() call; results are flushed per chunk (resume granularity)")
    # worker mode
    p.add_argument("--worker", action="store_true")
    p.add_argument("--shard", default=None)
    p.add_argument("--out-shard", default=None)
    return p


def _truthy(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y")


def _n_for(args, bench) -> int:
    if args.n is not None:
        return args.n
    kind = bench.kind
    v = {"math": args.n_math, "code": args.n_code, "if": args.n_if}[kind]
    return v if v is not None else bench.default_n


def load_rows(args, name: str) -> List[Dict[str, Any]]:
    b = BENCHMARKS[name]
    if name == "lcb_v6":
        return b.load(limit=args.limit, start_date=args.lcb_start_date, end_date=args.lcb_end_date)
    return b.load(limit=args.limit)


# ----------------------------------------------------------------------------- worker
def run_worker(args):
    """RESUMABLE: generation runs in chunks of --chunk prompts and every chunk is appended to
    out_shard immediately; on restart, (bench, idx) pairs already present in out_shard are skipped.
    A cancelled job therefore loses at most one chunk per GPU."""
    from mopd.eval.generate import VllmGenerator

    tasks = read_jsonl(args.shard)  # {bench, idx, messages, n}
    done = set()
    if os.path.exists(args.out_shard):
        done = {(r["bench"], r["idx"]) for r in read_jsonl(args.out_shard)}
    todo = [t for t in tasks if (t["bench"], t["idx"]) not in done]
    print(f"[eval-worker] shard={len(tasks)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return
    gen = VllmGenerator(args.model, args.tokenizer, tp=args.tp, max_model_len=args.max_model_len,
                        gpu_mem_util=args.gpu_mem_util, seed=args.seed)
    sp = PRESETS[args.preset]
    # group by n (one SamplingParams per group), then chunk each group
    by_n: Dict[int, List[Dict[str, Any]]] = {}
    for t in todo:
        by_n.setdefault(t["n"], []).append(t)
    n_written = 0
    for n, group in by_n.items():
        for s0 in range(0, len(group), args.chunk):
            chunk = group[s0:s0 + args.chunk]
            outs = gen.generate([t["messages"] for t in chunk], n=n, max_tokens=args.max_tokens,
                                seed=args.seed, mode=args.prompt_mode, thinking=_truthy(args.thinking), **sp)
            with open(args.out_shard, "a") as f:
                for t, o in zip(chunk, outs):
                    f.write(json.dumps({"bench": t["bench"], "idx": t["idx"], "samples": o}, ensure_ascii=False) + "\n")
            n_written += len(chunk)
            print(f"[eval-worker] progress {n_written}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- parent
def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.worker:
        return run_worker(args)

    os.makedirs(args.out, exist_ok=True)
    benches = [b for b in args.benchmarks.split(",") if b]
    for b in benches:
        assert b in BENCHMARKS, f"unknown benchmark {b}; known: {list(BENCHMARKS)}"
    rows_by_bench = {b: load_rows(args, b) for b in benches}
    msg_kw = dict(style=args.math_style, lcb_system=_truthy(args.lcb_system))

    shard_dir = os.path.join(args.out, "_shards")
    gpus = parse_gpus(args.gpus)
    assert len(gpus) % args.tp == 0
    groups = [gpus[i:i + args.tp] for i in range(0, len(gpus), args.tp)]
    n_shards = len(groups) * args.num_nodes
    if not args.skip_generation and not args.aggregate_only:
        tasks: List[Dict[str, Any]] = []
        for b in benches:
            bench = BENCHMARKS[b]
            n = _n_for(args, bench)
            for i, r in enumerate(rows_by_bench[b]):
                tasks.append({"bench": b, "idx": i, "messages": bench.messages(r, **msg_kw), "n": n})
        os.makedirs(shard_dir, exist_ok=True)
        # global round-robin sharding (identical on every node); node r owns shards r*G .. r*G+G-1
        shards = [[] for _ in range(n_shards)]
        for k, t in enumerate(tasks):
            shards[k % n_shards].append(t)
        mine = list(range(args.node_rank * len(groups), (args.node_rank + 1) * len(groups)))
        procs = []
        for gi in mine:
            g = groups[gi % len(groups)]
            sh = shards[gi]
            sp = os.path.join(shard_dir, f"shard_{gi}.jsonl")
            op = os.path.join(shard_dir, f"out_{gi}.jsonl")
            if not os.path.exists(sp):  # resume: keep the shard/out files of a previous attempt
                write_jsonl(sp, sh)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, g))
            env["VLLM_PORT"] = str(40000 + (g[0] % 8) * 128)  # distinct engine port per physical GPU
            cmd = [sys.executable, "-m", "mopd.eval.run_eval", "--worker", "--shard", sp, "--out-shard", op] + \
                  _passthrough(args)
            log = open(os.path.join(shard_dir, f"worker_{gi}.log"), "w")
            procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), op, log))
            print(f"[eval] node {args.node_rank} worker {gi} gpus={g} tasks={len(sh)}", flush=True)
        failed = 0
        for p, op, log in procs:
            p.wait()
            log.close()
            if p.returncode != 0 or not os.path.exists(op):
                failed += 1
                print(f"[eval] worker FAILED rc={p.returncode} ({op})", flush=True)
        if failed:
            raise SystemExit(f"{failed} worker(s) failed; see {shard_dir}/worker_*.log")
        if args.num_nodes > 1:
            print(f"[eval] node {args.node_rank} generation done; aggregate with --aggregate-only", flush=True)
            return
    if not args.skip_generation:
        merged: Dict[str, Dict[int, Any]] = {b: {} for b in benches}
        for gi in range(n_shards):
            op = os.path.join(shard_dir, f"out_{gi}.jsonl")
            assert os.path.exists(op), f"missing shard output {op}"
            for r in read_jsonl(op):
                merged[r["bench"]][r["idx"]] = r["samples"]  # duplicates (resume overlap) -> last wins
        for b in benches:
            rows = rows_by_bench[b]
            out = []
            for i, r in enumerate(rows):
                samples = merged[b].get(i)
                assert samples is not None, f"missing generation {b}[{i}]"
                out.append({"idx": i, "id": r.get("id") or r.get("key") or r.get("question_id"),
                            "messages": BENCHMARKS[b].messages(r, **msg_kw), "samples": samples})
            write_jsonl(os.path.join(args.out, f"{b}.gen.jsonl"), out)

    # ----- grade
    metrics: Dict[str, Any] = {"model": args.model, "preset": args.preset, "prompt_mode": args.prompt_mode,
                               "thinking": _truthy(args.thinking), "max_tokens": args.max_tokens,
                               "time": time.strftime("%Y-%m-%d %H:%M:%S"), "benchmarks": {}}
    for b in benches:
        rows = rows_by_bench[b]
        gens = read_jsonl(os.path.join(args.out, f"{b}.gen.jsonl"))
        assert len(gens) == len(rows)
        texts = [[s["text"] for s in g["samples"]] for g in gens]
        bench = BENCHMARKS[b]
        if bench.kind == "code":
            res = bench.score(rows, texts, num_workers=args.grade_workers)
        else:
            res = bench.score(rows, texts)
        per_row = res.pop("per_row", None)
        # generation stats
        ntok = [s["n_tokens"] for g in gens for s in g["samples"]]
        trunc = sum(1 for g in gens for s in g["samples"] if s["finish_reason"] == "length")
        finished_think = sum(1 for g in gens for s in g["samples"] if "</think>" in s["text"]) if _truthy(args.thinking) and args.prompt_mode == "chat" else None
        res.update({"avg_tokens": sum(ntok) / max(len(ntok), 1), "truncated_frac": trunc / max(len(ntok), 1),
                    "finished_thinking_frac": (finished_think / max(len(ntok), 1)) if finished_think is not None else None,
                    "domain": DOMAIN_OF[b]})
        metrics["benchmarks"][b] = res
        if per_row is not None:
            write_jsonl(os.path.join(args.out, f"{b}.scores.jsonl"),
                        [{"idx": i, "id": g["id"], "correct": c} for i, (g, c) in enumerate(zip(gens, per_row))])
        print(f"[eval] {b}: {res['score']:.2f} ({res['metric']}, n={res['n']}, avg_tokens={res['avg_tokens']:.0f}, "
              f"truncated={100*res['truncated_frac']:.1f}%)", flush=True)
    # domain averages
    dom: Dict[str, List[float]] = {}
    for b, r in metrics["benchmarks"].items():
        dom.setdefault(r["domain"], []).append(r["score"])
    metrics["domain_avg"] = {d: sum(v) / len(v) for d, v in dom.items()}
    write_json(os.path.join(args.out, "metrics.json"), metrics)
    print(json.dumps({k: (v if k != "benchmarks" else {b: r["score"] for b, r in v.items()}) for k, v in metrics.items()}, indent=2))


def _passthrough(args) -> List[str]:
    keys = ["model", "tokenizer", "out", "tp", "preset", "max_tokens", "max_model_len", "gpu_mem_util", "prompt_mode",
            "thinking", "seed", "chunk"]
    out: List[str] = []
    for k in keys:
        v = getattr(args, k)
        if v is not None:
            out += [f"--{k.replace('_', '-')}", str(v)]
    return out


if __name__ == "__main__":
    main()
