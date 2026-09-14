import pyarrow.parquet as pq, json, glob, sys
R="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean"
run=sys.argv[1]; steps=[int(s) for s in sys.argv[2].split(",")]
idx={}
for l in open(f"{R}/data/rl_pools/fin_train.jsonl"): idx[json.loads(l)["input"][:300]]="fin"
files=glob.glob(f"{R}/outputs/opd/{run}/completions/completions_*.parquet")
bystep={int(f.rsplit("_",1)[1].split(".")[0]):f for f in files}
for s in steps:
    f=bystep.get(s) or bystep[max(k for k in bystep if k<=s)]
    t=pq.read_table(f).to_pandas()
    pc=[c for c in t.columns if "prompt" in c][0]; cc=[c for c in t.columns if "completion" in c][0]
    for p,c in zip(t[pc],t[cc]):
        q=p.split("\n",1)[1] if p.startswith("user\n") else p
        if idx.get(q[:300])!="fin": continue
        if "</think>" in c: continue
        print(f"step {s} NOT-CLOSED chars={len(c)} head={c[:60]!r} tail={c[-100:]!r}")
