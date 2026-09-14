import pyarrow.parquet as pq, json, glob, os, collections, sys
R="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean"
pools={"law":"law_rl_v2_train.jsonl","fin":"fin_train.jsonl","if":"if_train.jsonl","med":"med_opd_train.jsonl"}
idx={}
for d,f in pools.items():
    for l in open(f"{R}/data/rl_pools/{f}"):
        idx[json.loads(l)["input"][:400]]=d
for run in sys.argv[1:]:
    files=sorted(glob.glob(f"{R}/outputs/opd/{run}/completions/completions_*.parquet"))
    print("==",run,"step files:",len(files))
    for f in files[:3]+files[-2:]:
        t=pq.read_table(f, columns=["step","prompt"]).to_pydict()
        c=collections.Counter(idx.get((p.split("\n",1)[1] if p.startswith("user\n") else p)[:400],"?") for p in t["prompt"])
        print(" step",t["step"][0],"n=",len(t["prompt"]),dict(c))
