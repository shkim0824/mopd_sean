"""pass@8 sweep worker: one GPU, one shard. Generates n=8 @ T=1.0/top_p=1.0/32k budget
(matching RL rollout conditions) and grades with math_verify. Raw outputs saved gzipped
so a judge re-grade never needs regeneration."""
import argparse, gzip, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, default=64)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.problems)]
    mine = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
    done_path = f"{a.out_dir}/shard{a.shard:03d}.done"
    if os.path.exists(done_path):
        print(f"shard {a.shard}: already done"); return
    print(f"shard {a.shard}: {len(mine)} problems", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, tensor_parallel_size=1, max_model_len=32768,
              gpu_memory_utilization=0.90, enforce_eager=False, dtype="bfloat16")

    prompts, params = [], []
    for r in mine:
        text = tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                       tokenize=False, add_generation_prompt=True)
        n_in = len(tok(text).input_ids)
        prompts.append(text)
        params.append(SamplingParams(n=8, temperature=1.0, top_p=1.0,
                                     max_tokens=max(256, 32768 - n_in - 8), seed=r["uid"]))
    outs = llm.generate(prompts, params)

    from math_verify import parse, verify
    def graded(pred_text, gold):
        try:
            g = parse("\\boxed{" + str(gold) + "}") if "\\boxed" not in str(gold) else parse(str(gold))
            p = parse(pred_text)
            return bool(verify(g, p))
        except Exception:
            return False

    res_f = open(f"{a.out_dir}/shard{a.shard:03d}.jsonl", "w")
    raw_f = gzip.open(f"{a.out_dir}/shard{a.shard:03d}.raw.jsonl.gz", "wt")
    for r, o in zip(mine, outs):
        passes, lens = [], []
        for c in o.outputs:
            t = c.text
            lens.append(len(c.token_ids))
            passes.append(1 if graded(t, r["expected_answer"]) else 0)
            raw_f.write(json.dumps({"uid": r["uid"], "text": t}, ensure_ascii=False) + "\n")
        rec = {k: r[k] for k in ("uid", "dataset", "pass_rate_30b", "expected_answer")}
        rec["zero_tier_probe"] = r.get("zero_tier_probe", False)
        rec["n"] = len(passes); rec["passed"] = sum(passes)
        rec["gen_tokens_mean"] = sum(lens) // max(1, len(lens))
        res_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    res_f.close(); raw_f.close()
    open(done_path, "w").write("ok")
    print(f"shard {a.shard}: DONE", flush=True)

if __name__ == "__main__":
    main()
