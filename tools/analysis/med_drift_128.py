import pyarrow.parquet as pq, json, glob, sys, statistics as st
R="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean"
run=sys.argv[1]; steps=[int(s) for s in sys.argv[2].split(",")]
idx={}
for l in open(f"{R}/data/rl_pools/med_opd_train.jsonl"): idx[json.loads(l)["input"][:300]]="med"
files=sorted(glob.glob(f"{R}/outputs/opd/{run}/completions/completions_*.parquet"), key=lambda f:int(f.rsplit("_",1)[1].split(".")[0]))
bystep={int(f.rsplit("_",1)[1].split(".")[0]):f for f in files}
for s in steps:
    f=bystep.get(s) or bystep[max(k for k in bystep if k<=s)]
    t=pq.read_table(f).to_pandas()
    pc=[c for c in t.columns if "prompt" in c][0]; cc=[c for c in t.columns if "completion" in c][0]
    rows=[]
    for p,c in zip(t[pc],t[cc]):
        q=p.split("\n",1)[1] if p.startswith("user\n") else p
        if idx.get(q[:300])!="med": continue
        closed="</think>" in c; n=len(c)
        tail=c[-80:].replace("\n"," ")
        rows.append((n,closed,tail))
    rows.sort()
    L=[r[0] for r in rows]; nc=[r for r in rows if not r[1]]
    print(f"step {s}: n={len(rows)} chars p10/50/90/max = {L[int(.1*len(L))]}/{L[len(L)//2]}/{L[int(.9*len(L))]}/{L[-1]}  not-closed={len(nc)} their chars={[r[0] for r in nc]}")
    for r in nc[:3]: print("   NC tail:", repr(r[2]))
    for r in rows[-2:]: print("   longest tail:", r[0], r[1], repr(r[2]))
