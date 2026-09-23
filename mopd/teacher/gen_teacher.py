"""Teacher rejection-sampling generator: k samples per prompt with offline vLLM,
DP workers over row shards (same shard/.done pattern as eval.run_domain_eval).

  python -m mopd.teacher.gen_teacher --model M --pool pool.jsonl --out O --k 4 \
      --shard-rank R --num-shards N --tp 2 [--max-tokens 16384] [--max-model-len 20480] \
      [--temperature 0.6] [--chunk 1024]

Pool rows: {"id", "input", "output", "meta"} (input already carries the eval-aligned
'Answer: <letter>' suffix). Output rows: {"id", "gold", "meta", "samples": [{"text",
"finish"}...]}. Resume-safe: a partial shardNNN.jsonl.tmp is continued, not redone.
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=20480)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--shard-rank", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--min-gen", type=int, default=2048)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rank = args.shard_rank

    rows = [json.loads(l) for l in open(args.pool)]
    for i, r in enumerate(rows):
        r.setdefault("id", str(i))
    mine = rows[rank::args.num_shards]
    out_f = os.path.join(args.out, f"shard{rank:03d}.jsonl")
    tmp = out_f + ".tmp"
    if os.path.exists(out_f + ".done"):
        print(f"[gen {rank}] already done"); return
    done = 0
    if os.path.exists(tmp):  # resume: count complete lines already written
        with open(tmp) as f:
            done = sum(1 for _ in f)
        print(f"[gen {rank}] resuming after {done}/{len(mine)} rows")
    todo = mine[done:]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              tensor_parallel_size=args.tp, gpu_memory_utilization=0.9,
              trust_remote_code=True, enable_prefix_caching=True)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95, top_k=20,
                        max_tokens=args.max_tokens, seed=args.seed + 1000 * rank)
    # pre-rendered pools (tau2: {prompt_text, ...} with system + history + tools already in the Qwen3 template) are used
    # verbatim; the MCQ pools ({input}) are rendered as a single user turn as before.
    prompts = [r["prompt_text"] if r.get("prompt_text") else
               tok.apply_chat_template([{"role": "user", "content": r["input"]}],
                                       add_generation_prompt=True, tokenize=False,
                                       enable_thinking=True) for r in todo]
    # guard: a prompt longer than the context minus a minimal generation budget would make
    # vLLM raise and kill the whole shard -> skip it (recorded) instead.
    limit = args.max_model_len - args.min_gen
    keep_idx, skipped = [], []
    for i, ptxt in enumerate(prompts):
        n = len(tok(ptxt, add_special_tokens=False)["input_ids"])
        (keep_idx if n <= limit else skipped).append(i)
    if skipped:
        with open(os.path.join(args.out, f"skipped_shard{rank:03d}.jsonl"), "a") as fsk:
            for i in skipped:
                fsk.write(json.dumps({"id": todo[i]["id"], "reason": "prompt_too_long"}) + "\n")
        print(f"[gen {rank}] skipped {len(skipped)} over-long prompts (> {limit} tok)", flush=True)
    todo = [todo[i] for i in keep_idx]
    prompts = [prompts[i] for i in keep_idx]
    with open(tmp, "a") as f:
        for s in range(0, len(prompts), args.chunk):
            outs = llm.generate(prompts[s:s + args.chunk], sp)
            for r, o in zip(todo[s:s + args.chunk], outs):
                f.write(json.dumps({"id": r["id"], "gold": r.get("output"), "meta": r.get("meta") or {k: r.get(k) for k in ("sub_domain", "source_dialog_id", "turn_index", "correct", "reward") if k in r},
                                    "samples": [{"text": c.text, "finish": c.finish_reason}
                                                for c in o.outputs]},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"[gen {rank}] {done + min(s + args.chunk, len(prompts))}/{len(mine)}", flush=True)
    os.replace(tmp, out_f)
    open(out_f + ".done", "w").write("ok")
    print(f"[gen {rank}] wrote {len(mine)} rows x k={args.k}")


if __name__ == "__main__":
    main()
