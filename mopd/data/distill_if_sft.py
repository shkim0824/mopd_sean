"""IF SFT data = IFBench-recipe prompts + responses distilled from a teacher and
kept only if the verifier passes ALL constraints (paper: "SFT prompts are
constructed following the method introduced by IFBench and distilled on
gpt-oss-120b").

GPU job (1 node, 8 GPU DP): teacher served offline with vLLM, k candidates per
prompt, first candidate that passes every IFEvalG constraint (strict, on the
de-thinked answer) is kept. Output rows match sft_{math,code}.jsonl:
  {id, domain:"if", messages:[{user},{assistant}], source, teacher}
The assistant text is normalised to the Qwen3 thinking format. gpt-oss emits
its reasoning in a separate channel -> it becomes the <think> block; a
non-reasoning teacher yields an empty <think> block (still valid).

  python -m mopd.data.distill_if_sft --teacher models/gpt-oss-120b --tp 2 --gpus 0-7 \\
      --prompts data/train/sft_if_prompts.jsonl --out data/train/sft_if.jsonl --k 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

from mopd.common.chat import normalize_assistant
from mopd.common.io import read_jsonl, write_json, write_jsonl
from mopd.eval.run_eval import parse_gpus
from mopd.graders.if_grader import ifevalg_reward


def _split_reasoning(text: str, reasoning: str | None) -> Dict[str, str]:
    if reasoning is None and "</think>" in text:
        pre, post = text.split("</think>", 1)
        reasoning, text = pre.split("<think>", 1)[-1], post
    return {"reasoning": (reasoning or "").strip(), "answer": text.strip()}


HARMONY = ("<|channel|>", "<|message|>", "<|end|>", "<|start|>", "<|return|>", "<|call|>", "<|constrain|>")


def parse_harmony(text: str) -> Dict[str, str]:
    """gpt-oss (harmony) raw output with special tokens kept:
    '<|channel|>analysis<|message|>REASONING<|end|><|start|>assistant<|channel|>final<|message|>ANSWER<|return|>'
    -> {reasoning, answer}. Falls back to the whole text as answer when no final channel is present."""
    reasoning, answer = "", text
    if "<|channel|>analysis<|message|>" in text:
        r = text.split("<|channel|>analysis<|message|>", 1)[1]
        reasoning = r.split("<|end|>", 1)[0] if "<|end|>" in r else r.split("<|channel|>final", 1)[0]
    if "<|channel|>final<|message|>" in text:
        answer = text.split("<|channel|>final<|message|>", 1)[1]
    elif reasoning:
        answer = ""  # never reached the final channel (truncated)
    for tok in HARMONY:
        answer = answer.replace(tok, "")
        reasoning = reasoning.replace(tok, "")
    return {"reasoning": reasoning.strip(), "answer": answer.strip()}


def is_gpt_oss(path: str) -> bool:
    return "gpt-oss" in os.path.basename(path.rstrip("/")).lower() or "gpt_oss" in path.lower()


def worker(args):
    """Generate for one shard, CHUNKED and INCREMENTAL: results are appended to out_shard after every
    chunk and prompts whose id is already in out_shard are skipped, so an engine crash (CUDA illegal
    address seen late in a 1.7 h run) loses at most one chunk and a relaunch resumes."""
    import json as _json
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = read_jsonl(args.shard)
    done = set()
    if os.path.exists(args.out_shard):
        done = {r["id"] for r in read_jsonl(args.out_shard)}
    rows = [r for r in rows if r["id"] not in done]
    print(f"[distill-worker] shard={len(done) + len(rows)} already_done={len(done)} todo={len(rows)}", flush=True)
    if not rows:
        return
    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    gptoss = is_gpt_oss(args.teacher)
    kw = {}
    if gptoss and args.reasoning_effort:  # gpt-oss chat template kwarg (low|medium|high)
        kw["reasoning_effort"] = args.reasoning_effort
    prompts_all = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                           add_generation_prompt=True, **kw) for r in rows]
    # drop prompts that cannot fit (some Tulu-SFT prompts are >80k tokens) -> vLLM would kill the engine
    budget = args.max_model_len - args.max_tokens
    keep = [i for i, p in enumerate(prompts_all) if len(tok(p, add_special_tokens=False)["input_ids"]) <= budget]
    print(f"[distill-worker] todo={len(rows)} fit={len(keep)} skipped_too_long={len(rows) - len(keep)} (budget {budget} tokens)", flush=True)
    rows = [rows[i] for i in keep]
    prompts = [prompts_all[i] for i in keep]
    llm = LLM(model=args.teacher, tensor_parallel_size=args.tp, dtype="auto", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9, trust_remote_code=True, disable_custom_all_reduce=(args.tp > 1),
              enforce_eager=(args.tp > 2))
    stop_ids = None
    if gptoss:
        stop_ids = [i for i in (tok.convert_tokens_to_ids(t) for t in ("<|return|>", "<|call|>")) if isinstance(i, int) and i >= 0]
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed,
                        skip_special_tokens=not gptoss, stop_token_ids=stop_ids)
    CH = args.chunk
    for s0 in range(0, len(rows), CH):
        outs = llm.generate(prompts[s0:s0 + CH], sp, use_tqdm=(s0 == 0))
        with open(args.out_shard, "a") as f:
            for r, o in zip(rows[s0:s0 + CH], outs):
                cands = []
                for c in o.outputs:
                    d = parse_harmony(c.text) if gptoss else _split_reasoning(c.text, None)
                    d["finish_reason"] = c.finish_reason
                    cands.append(d)
                f.write(_json.dumps({"id": r["id"], "cands": cands}, ensure_ascii=False) + "\n")
        print(f"[distill-worker] progress {min(s0 + CH, len(rows))}/{len(rows)}", flush=True)


def _launch(args, gi, g, sp, op, wd, attempt):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, g)), VLLM_PORT=str(40000 + (g[0] % 8) * 128))
    cmd = [sys.executable, "-m", "mopd.data.distill_if_sft", "--worker", "--shard", sp, "--out-shard", op,
           "--teacher", args.teacher, "--tp", str(args.tp), "--k", str(args.k), "--temperature", str(args.temperature),
           "--top-p", str(args.top_p), "--max-tokens", str(args.max_tokens), "--max-model-len", str(args.max_model_len),
           "--reasoning-effort", args.reasoning_effort or "", "--seed", str(args.seed + attempt), "--chunk", str(args.chunk)]
    log = open(os.path.join(wd, f"worker_{gi}.attempt{attempt}.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--prompts", default="data/train/sft_if_prompts.jsonl")
    ap.add_argument("--out", default="data/train/sft_if.jsonl")
    ap.add_argument("--gpus", default="0-7")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=256, help="prompts per generate() call; results flushed after each")
    ap.add_argument("--retries", type=int, default=3, help="engine-crash relaunches per shard")
    ap.add_argument("--num-nodes", type=int, default=1, help="multi-node: total nodes running this script")
    ap.add_argument("--node-rank", type=int, default=0, help="multi-node: this node's rank (shards are global)")
    ap.add_argument("--aggregate-only", action="store_true", help="skip generation; merge existing out_*.jsonl")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out-shard", default=None)
    args = ap.parse_args(argv)
    if args.worker:
        return worker(args)

    rows = read_jsonl(args.prompts, limit=args.limit)
    gpus = parse_gpus(args.gpus)
    groups = [gpus[i:i + args.tp] for i in range(0, len(gpus), args.tp)]
    per_node = len(groups)
    n_shards = per_node * args.num_nodes
    wd = args.out + ".shards"
    os.makedirs(wd, exist_ok=True)
    # global shard layout: shard g = rows[g::n_shards]; node r owns shards r*per_node .. r*per_node+per_node-1
    shards = []
    for gi in range(n_shards):
        sp = os.path.join(wd, f"shard_{gi}.jsonl")
        op = os.path.join(wd, f"out_{gi}.jsonl")
        if args.node_rank == 0 and not os.path.exists(sp):
            write_jsonl(sp, rows[gi::n_shards])
        shards.append((gi, groups[gi % per_node], sp, op))
    if not args.aggregate_only:
        mine = [sh for sh in shards if sh[0] // per_node == args.node_rank]
        # other nodes wait for rank 0 to write the shard files
        for _ in range(60):
            if all(os.path.exists(sp) for _, _, sp, _ in mine):
                break
            time.sleep(5)
        procs = {gi: _launch(args, gi, g, sp, op, wd, 0) for gi, g, sp, op in mine}
        attempts = {gi: 0 for gi, *_ in mine}
        todo = {gi: len(read_jsonl(sp)) for gi, g, sp, op in mine}
        by_gi = {gi: (g, sp, op) for gi, g, sp, op in mine}
        while procs:
            for gi in list(procs):
                p = procs[gi]
                if p.poll() is None:
                    continue
                g, sp, op = by_gi[gi]
                n_done = len(read_jsonl(op)) if os.path.exists(op) else 0
                if p.returncode == 0:
                    print(f"[distill] shard {gi} done ({n_done}/{todo[gi]})", flush=True)
                    del procs[gi]
                elif attempts[gi] < args.retries:
                    attempts[gi] += 1
                    print(f"[distill] shard {gi} worker crashed rc={p.returncode} at {n_done}/{todo[gi]} -> relaunch #{attempts[gi]}", flush=True)
                    procs[gi] = _launch(args, gi, g, sp, op, wd, attempts[gi])
                else:
                    print(f"[distill] shard {gi} FAILED after {args.retries} relaunches at {n_done}/{todo[gi]}", flush=True)
                    del procs[gi]
            time.sleep(20)
        if args.num_nodes > 1:
            print(f"[distill] node {args.node_rank} finished its {len(mine)} shards (aggregate with --aggregate-only)", flush=True)
            return
    outs = [op for _, _, _, op in shards if os.path.exists(op)]
    n_out = sum(len(read_jsonl(op)) for op in outs)
    print(f"[distill] generated for {n_out}/{len(rows)} prompts", flush=True)
    if n_out < 0.9 * len(rows):
        raise SystemExit(f"only {n_out}/{len(rows)} prompts generated; see {wd}/worker_*.log")
    procs_out = outs
    by_id = {r["id"]: r for r in rows}
    kept, stats = [], {"prompts": len(rows), "kept": 0, "no_pass": 0, "cand_pass_rate": 0.0}
    n_c = n_pass = 0
    for op in procs_out:
        for r in read_jsonl(op):
            row = by_id[r["id"]]
            best = None
            for c in r["cands"]:
                n_c += 1
                ok = ifevalg_reward(c["answer"], row["ground_truth"], all_or_nothing=True) == 1.0
                n_pass += ok
                if ok and best is None:
                    best = c
            if best is None:
                stats["no_pass"] += 1
                continue
            kept.append({"id": row["id"], "domain": "if",
                         "messages": [{"role": "user", "content": row["prompt"]},
                                      {"role": "assistant", "content": normalize_assistant(best["answer"], best["reasoning"])}],
                         "source": row["source"], "teacher": os.path.basename(args.teacher.rstrip("/"))})
    stats["kept"] = len(kept)
    stats["cand_pass_rate"] = n_pass / max(n_c, 1)
    write_jsonl(args.out, kept)
    write_json(args.out + ".stats.json", stats)
    print(stats)


if __name__ == "__main__":
    main()
