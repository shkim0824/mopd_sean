"""Extractor/rejection canary: turn the existing Qwen3.6-35B MedQA eval outputs
(1,273 single-sample responses, known 94.6% accurate) into gen-shard format and run
reject_filter on them. Expected: sft_rows close to 0.946*1273 minus trunc/malformed,
'wrong_answer' about 5%, assistant text = '<think>...</think>\n\n...Answer: X'."""
import glob, json, os, sys
from mopd.eval.domains.benches import BENCHES

eval_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
meta = {r["id"]: r for r in BENCHES["medqa"]()}
with open(os.path.join(out_dir, "pool.jsonl"), "w") as fp, \
     open(os.path.join(out_dir, "shard000.jsonl"), "w") as fs:
    n = 0
    for f in sorted(glob.glob(os.path.join(eval_dir, "shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line); m = meta[r["id"]]
            fp.write(json.dumps({"id": r["id"], "input": m["prompt"], "output": m["gold"],
                                 "meta": {"source": "canary-medqa-test", "kind": "mcqa4"}}) + "\n")
            fs.write(json.dumps({"id": r["id"], "gold": m["gold"],
                                 "meta": {"source": "canary-medqa-test", "kind": "mcqa4"},
                                 "samples": [{"text": r["response"], "finish": r["finish"]}]}) + "\n")
            n += 1
print(f"canary rows: {n}")
