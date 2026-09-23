"""ONE job, ONE engine per GPU, every benchmark: in-domain + IF + OOD (math, code,
safety, GPQA).

    python -m mopd.eval.run_all --model M --out O --benchmarks medqa,...,truthfulqa_mc \
        --gpus 0-7 --num-nodes 2 --node-rank $SLURM_NODEID
    python -m mopd.eval.run_all --merge-only     --benchmarks ... --out O --model M
    python -m mopd.eval.run_all --aggregate-only --benchmarks ... --out O --model M

WHY: the legacy flow spent two Slurm allocations and two rounds of 16 vLLM start-ups per
checkpoint (``eval_domain.sbatch`` + ``eval_6bench.sbatch``), and adding the four phase-3
OOD sets would have made it four.  Here every GPU loads the student ONCE and serves all
benchmarks, each with its own sampling protocol (``mopd/eval/registry.py``).

Scheduling: tasks are dealt LPT (longest processing time first) — sorted by expected cost
(``n x max_tokens`` for generation, ~0 for log-prob scoring) and then round-robin over the
global shards, so the 8-sample 32k AIME prompts start immediately and every GPU finishes
at about the same time instead of one shard trailing with the long tail.

Two request kinds share the engine:
  gen  sample ``n`` completions                       (all benchmarks but the two below)
  lp   score fixed continuations via prompt_logprobs  (sycophancy, truthfulqa_mc)
lp requests carry no generation at all, so ~71k of them cost a few minutes of prefill.

Resume: every chunk is appended to the worker's out-shard and re-read on restart, so a
cancelled job loses at most one chunk per GPU (same contract as ``run_eval.py``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mopd.common.io import read_jsonl, write_json, write_jsonl
from mopd.eval.registry import FULL_SET, registry, spec


# ----------------------------------------------------------------------------- args
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--benchmarks", default=",".join(FULL_SET))
    p.add_argument("--gpus", default="0-7")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--limit", type=int, default=None, help="first K rows of every bench (smoke)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seqs-per-call", type=int, default=256,
                   help="target SEQUENCES (sum of n) per generate() call; one mixed call per "
                        "chunk keeps the batch full across benchmarks instead of draining at "
                        "every sampling-group boundary")
    p.add_argument("--chunk", type=int, default=0,
                   help="hard cap on prompts per generate() call (0 = only --seqs-per-call)")
    p.add_argument("--lp-chunk", type=int, default=512, help="log-prob requests per call")
    p.add_argument("--grade-workers", type=int, default=None)
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--merge-only", action="store_true", help="merge shards -> *.gen/*.lp.jsonl")
    p.add_argument("--aggregate-only", action="store_true", help="merge (if needed) + grade")
    p.add_argument("--prefix-caching", default="auto", choices=["auto", "on", "off"],
                   help="auto = off when the shard has log-prob tasks (vLLM cannot serve "
                        "prompt_logprobs from a cached prefix)")
    # per-bench overrides
    p.add_argument("--n", type=int, default=None, help="override samples/prompt for ALL gen benches")
    p.add_argument("--n-of", default="", help="bench=n,bench=n (e.g. gpqa=4,aime24=8)")
    p.add_argument("--max-tokens-of", default="", help="bench=tokens,...")
    # worker mode
    p.add_argument("--worker", action="store_true")
    p.add_argument("--shard", default=None)
    p.add_argument("--out-shard", default=None)
    return p


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


def _kv(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for part in s.split(","):
        if part.strip():
            k, v = part.split("=")
            out[k.strip()] = int(v)
    return out


def _kv_str(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in s.split(","):
        if part.strip():
            k, v = part.split("=")
            out[k.strip()] = v.strip()
    return out


def bench_list(args) -> List[str]:
    names = [b for b in args.benchmarks.split(",") if b]
    reg = registry()
    for b in names:
        if b not in reg:
            raise SystemExit("unknown benchmark %r; known: %s" % (b, sorted(reg)))
    return names


def resolved(args, name: str):
    """Spec with the CLI overrides applied (n / max_tokens)."""
    sp = spec(name)
    n = sp.n
    if sp.request == "gen":
        if args.n is not None:
            n = args.n
        n = _kv(args.n_of).get(name, n)
    mt = _kv(args.max_tokens_of).get(name, sp.max_tokens)
    return sp, n, mt


# ----------------------------------------------------------------------------- tasks
def load_rows(args, names: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for b in names:
        sp = spec(b)
        out[b] = sp.load(limit=args.limit)
        print("[run_all] %-14s %6d rows (%s/%s)" % (b, len(out[b]), sp.group, sp.request),
              flush=True)
    return out


def build_tasks(args, names: Sequence[str],
                rows: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for b in names:
        sp, n, mt = resolved(args, b)
        if sp.request == "gen":
            for i in range(len(rows[b])):
                tasks.append({"b": b, "i": i, "k": "gen", "w": float(n) * mt})
        else:
            for i, r in enumerate(rows[b]):
                for ci in range(len(sp.build(r))):
                    tasks.append({"b": b, "i": i, "k": "lp", "c": ci, "w": 256.0})
    # LPT: expensive first, then round-robin -> shards end together
    tasks.sort(key=lambda t: (-t["w"], t["b"], t["i"], t.get("c", 0)))
    return tasks


def shard_tasks(tasks: Sequence[Dict[str, Any]], n_shards: int) -> List[List[Dict[str, Any]]]:
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(n_shards)]
    load = [0.0] * n_shards
    for t in tasks:                      # greedy LPT: give each task to the lightest shard
        j = min(range(n_shards), key=lambda x: load[x])
        shards[j].append({k: v for k, v in t.items() if k != "w"})
        load[j] += t["w"]
    return shards


def task_key(t: Dict[str, Any]) -> Tuple:
    return (t["b"], t["i"], t["k"], t.get("c", -1))


# ----------------------------------------------------------------------------- worker
def run_worker(args) -> None:
    from mopd.eval.generate import VllmGenerator

    tasks = read_jsonl(args.shard)
    done = set()
    if os.path.exists(args.out_shard):
        done = {task_key(r) for r in read_jsonl(args.out_shard)}
    todo = [t for t in tasks if task_key(t) not in done]
    print("[worker] shard=%d done=%d todo=%d" % (len(tasks), len(done), len(todo)), flush=True)
    if not todo:
        return
    names = sorted({t["b"] for t in todo})
    rows = load_rows(args, names)
    has_lp = any(t["k"] == "lp" for t in todo)
    pc = {"auto": (False if has_lp else None), "on": True, "off": False}[args.prefix_caching]
    gen = VllmGenerator(args.model, args.tokenizer, tp=args.tp, max_model_len=args.max_model_len,
                        gpu_mem_util=args.gpu_mem_util, seed=args.seed, enable_prefix_caching=pc)

    def flush(recs: List[Dict[str, Any]]) -> None:
        with open(args.out_shard, "a") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- generation: ONE mixed engine call per chunk (per-prompt SamplingParams), so
    # benchmarks with different protocols share the same continuous batch. The task list is
    # already cost-sorted, so the long 32k x n=8 prompts enter first and the cheap ones fill
    # the batch behind them.
    gen_tasks = [t for t in todo if t["k"] == "gen"]
    items = []
    for t in gen_tasks:
        sp, n, mt = resolved(args, t["b"])
        items.append((t, {"messages": sp.build(rows[t["b"]][t["i"]]), "n": n, "max_tokens": mt,
                          "sampling": sp.sampling, "mode": sp.mode,
                          "thinking": sp.thinking, "seed": args.seed}))
    n_done = 0
    pos = 0
    while pos < len(items):
        seqs = 0
        end = pos
        while end < len(items):
            nxt = int(items[end][1]["n"])
            if end > pos and (seqs + nxt > args.seqs_per_call
                              or (args.chunk and end - pos >= args.chunk)):
                break
            seqs += nxt
            end += 1
        chunk = items[pos:end]
        pos = end
        outs = gen.generate_mixed([it for _, it in chunk])
        flush([{"b": t["b"], "i": t["i"], "k": "gen",
                "s": [{"t": o["text"], "nt": o["n_tokens"], "fr": o["finish_reason"]}
                      for o in out]}
               for (t, _), out in zip(chunk, outs)])
        n_done += len(chunk)
        print("[worker] gen %d/%d (%d seqs, %s)"
              % (n_done, len(items), seqs, ",".join(sorted({t["b"] for t, _ in chunk}))),
              flush=True)

    # ---- log-prob scoring
    lp = [t for t in todo if t["k"] == "lp"]
    for s0 in range(0, len(lp), args.lp_chunk):
        chunk = lp[s0:s0 + args.lp_chunk]
        specs = []
        for t in chunk:
            sp = spec(t["b"])
            specs.append(sp.build(rows[t["b"]][t["i"]])[t["c"]])
        res = gen.score_continuations(specs)
        flush([{"b": t["b"], "i": t["i"], "k": "lp", "c": t["c"],
                "lp": r["logprob"], "nt": r["n_tokens"]} for t, r in zip(chunk, res)])
        print("[worker] lp %d/%d" % (min(s0 + args.lp_chunk, len(lp)), len(lp)), flush=True)


# ----------------------------------------------------------------------------- merge
def merge(args, names: Sequence[str], rows: Dict[str, List[Dict[str, Any]]],
          n_shards: int) -> None:
    shard_dir = os.path.join(args.out, "_shards")
    gen: Dict[str, Dict[int, Any]] = {b: {} for b in names}
    lp: Dict[str, Dict[int, Dict[int, Any]]] = {b: {} for b in names}
    for gi in range(n_shards):
        op = os.path.join(shard_dir, "out_%d.jsonl" % gi)
        assert os.path.exists(op), "missing shard output %s" % op
        for r in read_jsonl(op):
            if r["k"] == "gen":
                gen[r["b"]][r["i"]] = r["s"]
            else:
                lp[r["b"]].setdefault(r["i"], {})[r["c"]] = {"logprob": r["lp"],
                                                             "n_tokens": r["nt"]}
    for b in names:
        sp = spec(b)
        rr = rows[b]
        if sp.request == "gen":
            out = []
            for i, r in enumerate(rr):
                s = gen[b].get(i)
                assert s is not None, "missing generation %s[%d]" % (b, i)
                out.append({"idx": i, "id": r.get("id") or r.get("key") or r.get("question_id"),
                            "samples": [{"text": x["t"], "n_tokens": x["nt"],
                                         "finish_reason": x["fr"]} for x in s]})
            write_jsonl(os.path.join(args.out, "%s.gen.jsonl" % b), out)
        else:
            out = []
            for i, r in enumerate(rr):
                got = lp[b].get(i) or {}
                want = len(sp.build(r))
                assert len(got) == want, "missing log-probs %s[%d]: %d/%d" % (b, i, len(got), want)
                out.append({"idx": i, "id": r.get("id"),
                            "choices": [{"ci": c, **got[c]} for c in sorted(got)]})
            write_jsonl(os.path.join(args.out, "%s.lp.jsonl" % b), out)
        print("[run_all] merged %s" % b, flush=True)


# ----------------------------------------------------------------------------- grade
def grade(args, names: Sequence[str], rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "model": args.model, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runner": "mopd.eval.run_all", "max_model_len": args.max_model_len,
        "limit": args.limit, "benchmarks": {}, "protocol": {},
    }
    recorded = {}
    pj = os.path.join(args.out, "protocol.json")
    if os.path.exists(pj):
        recorded = json.load(open(pj))
        metrics["protocol_source"] = "protocol.json (written at generation time)"
    else:
        metrics["protocol_source"] = ("registry at grade time -- this run predates "
                                      "protocol.json; check the generation diagnostics if the "
                                      "code changed between generation and grading")
    for b in names:
        sp, n, mt = resolved(args, b)
        metrics["protocol"][b] = recorded.get(b) or {
            "request": sp.request, "n": n, "max_tokens": mt,
            "thinking": sp.thinking, "prompt_mode": sp.mode, **sp.sampling}
        rr = rows[b]
        if sp.request == "gen":
            gens = read_jsonl(os.path.join(args.out, "%s.gen.jsonl" % b))
            assert len(gens) == len(rr), "%s: %d generations for %d rows" % (b, len(gens), len(rr))
            texts = [[s["text"] for s in g["samples"]] for g in gens]
            res = sp.score(rr, texts, args.out)
            ntok = [s["n_tokens"] for g in gens for s in g["samples"]]
            trunc = sum(1 for g in gens for s in g["samples"] if s["finish_reason"] == "length")
            fin = sum(1 for g in gens for s in g["samples"] if "</think>" in s["text"])
            res.update({"avg_tokens": sum(ntok) / max(len(ntok), 1),
                        "truncated_frac": trunc / max(len(ntok), 1),
                        "finished_thinking_frac": fin / max(len(ntok), 1) if sp.thinking else None})
        else:
            got = read_jsonl(os.path.join(args.out, "%s.lp.jsonl" % b))
            assert len(got) == len(rr), "%s: %d scored rows for %d" % (b, len(got), len(rr))
            lp = [{c["ci"]: {"logprob": c["logprob"], "n_tokens": c["n_tokens"]}
                   for c in g["choices"]} for g in got]
            res = sp.score(rr, lp, args.out)
        res.pop("per_row", None)
        metrics["benchmarks"][b] = res
        print("[run_all] %-14s %s = %s" % (b, res.get("metric"),
                                           None if res.get("score") is None
                                           else round(res["score"], 2)), flush=True)
    # group averages only where the members share a direction; "safety" mixes ASR (lower
    # better), a sycophancy rate (lower better) and MC2 (higher better), so it gets none.
    groups: Dict[str, List[float]] = {}
    drop = set()
    for b, r in metrics["benchmarks"].items():
        g = spec(b).group
        if r.get("score") is None:
            continue
        if not r.get("higher_is_better", True):
            drop.add(g)
        groups.setdefault(g, []).append(r["score"])
    metrics["group_avg"] = {g: sum(v) / len(v) for g, v in sorted(groups.items())
                            if g not in drop}
    metrics["scores"] = {b: r.get("score") for b, r in metrics["benchmarks"].items()}
    # a benchmark may report a second, permissive reading of the same samples (GPQA
    # general): keep it beside the headline so watchers and tables see both
    for b, r in metrics["benchmarks"].items():
        if r.get("score_general") is not None:
            metrics["scores"]["%s_general" % b] = r["score_general"]
    write_json(os.path.join(args.out, "metrics.json"), metrics)
    return metrics


# ----------------------------------------------------------------------------- parent
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        run_worker(args)
        return 0
    os.makedirs(args.out, exist_ok=True)
    names = bench_list(args)
    gpus = parse_gpus(args.gpus)
    assert len(gpus) % args.tp == 0, "gpus %s not divisible by tp %d" % (gpus, args.tp)
    groups = [gpus[i:i + args.tp] for i in range(0, len(gpus), args.tp)]
    n_shards = len(groups) * args.num_nodes
    rows = load_rows(args, names)

    if args.merge_only or args.aggregate_only:
        if args.merge_only or not all(
                os.path.exists(os.path.join(args.out, "%s.%s.jsonl" %
                                            (b, "gen" if spec(b).request == "gen" else "lp")))
                for b in names):
            merge(args, names, rows, n_shards)
        if args.merge_only:
            return 0
        m = grade(args, names, rows)
        print(json.dumps({"scores": m["scores"], "group_avg": m["group_avg"]}, indent=2))
        return 0

    tasks = build_tasks(args, names, rows)
    shards = shard_tasks(tasks, n_shards)
    # the protocol is recorded HERE, with the samples, not at grading time: generation and
    # grading are separate job steps and the registry can change in between (it did once --
    # see tmp/fix_protocol_record.py).
    if args.node_rank == 0:
        write_json(os.path.join(args.out, "protocol.json"),
                   {b: dict(request=spec(b).request, n=resolved(args, b)[1],
                            max_tokens=resolved(args, b)[2], thinking=spec(b).thinking,
                            prompt_mode=spec(b).mode, **spec(b).sampling) for b in names})
    shard_dir = os.path.join(args.out, "_shards")
    os.makedirs(shard_dir, exist_ok=True)
    mine = list(range(args.node_rank * len(groups), (args.node_rank + 1) * len(groups)))
    print("[run_all] %d tasks -> %d shards (%d local: %s)" % (len(tasks), n_shards, len(mine), mine),
          flush=True)
    procs = []
    repo = os.environ.get("MOPD_REPO") or os.getcwd()
    for gi in mine:
        g = groups[gi % len(groups)]
        sp_path = os.path.join(shard_dir, "shard_%d.jsonl" % gi)
        op = os.path.join(shard_dir, "out_%d.jsonl" % gi)
        if not os.path.exists(sp_path):        # resume: keep a previous attempt's shard
            write_jsonl(sp_path, shards[gi])
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, g))
        env["VLLM_PORT"] = str(40000 + (g[0] % 8) * 128)
        # per-worker compile caches: concurrent jobs corrupted the shared inductor cache
        c = os.path.join(repo, "tmp", "cache")
        jid = os.environ.get("SLURM_JOB_ID", "x")
        env["VLLM_CACHE_ROOT"] = os.path.join(c, "vllm_%s_%d" % (jid, gi))
        env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(c, "ind_%s_%d" % (jid, gi))
        env["TRITON_CACHE_DIR"] = os.path.join(c, "triton_%s_%d" % (jid, gi))
        env["XDG_CACHE_HOME"] = os.path.join(c, "xdg")
        cmd = [sys.executable, "-m", "mopd.eval.run_all", "--worker",
               "--shard", sp_path, "--out-shard", op] + _passthrough(args)
        log = open(os.path.join(shard_dir, "worker_%d.log" % gi), "a")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), op, log))
        print("[run_all] worker %d gpus=%s tasks=%d" % (gi, g, len(shards[gi])), flush=True)
    failed = 0
    for p, op, log in procs:
        p.wait()
        log.close()
        if p.returncode != 0 or not os.path.exists(op):
            failed += 1
            print("[run_all] worker FAILED rc=%s (%s)" % (p.returncode, op), flush=True)
    if failed:
        raise SystemExit("%d worker(s) failed; see %s/worker_*.log" % (failed, shard_dir))
    print("[run_all] node %d generation done" % args.node_rank, flush=True)
    return 0


def _passthrough(args) -> List[str]:
    keys = ["model", "tokenizer", "out", "benchmarks", "tp", "max_model_len", "gpu_mem_util",
            "seed", "chunk", "seqs_per_call", "lp_chunk", "limit", "n", "n_of", "max_tokens_of",
            "prefix_caching"]
    out: List[str] = []
    for k in keys:
        v = getattr(args, k)
        if v is None or v == "":
            continue
        out += ["--%s" % k.replace("_", "-"), str(v)]
    return out


if __name__ == "__main__":
    sys.exit(main())
