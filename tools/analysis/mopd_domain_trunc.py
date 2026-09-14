import pyarrow.parquet as pq, json, glob, os, collections, sys, re
R="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean"
pools={"law":"law_rl_v2_train.jsonl","fin":"fin_train.jsonl","if":"if_train.jsonl","med":"med_opd_train.jsonl"}
idx={}
for d,f in pools.items():
    for l in open(f"{R}/data/rl_pools/{f}"):
        idx[json.loads(l)["input"][:300]]=d
def dom(p):
    q=p.split("\n",1)[1] if p.startswith("user\n") else p
    return idx.get(q[:300], "if?")   # unmatched short prompts are IF (templating), verified earlier
for run in sys.argv[1:]:
    files=sorted(glob.glob(f"{R}/outputs/opd/{run}/completions/completions_*.parquet"))
    steps=[int(re.search(r"_(\d+)\.parquet",f).group(1)) for f in files]
    pick=[s for s in steps if s in (25,50,75,100,125,150,175,200)] + steps[-1:]
    print("==",run,"latest step",steps[-1])
    print("step | domain: n / mean chars / closed </think> % / >50k chars (≈cap) %")
    for s in sorted(set(pick)):
        t=pq.read_table(files[steps.index(s)], columns=["prompt","completion"]).to_pydict()
        agg=collections.defaultdict(list)
        for p,c in zip(t["prompt"],t["completion"]): agg[dom(p).rstrip("?")].append(c)
        row=[]
        for d in ["law","fin","if","med"]:
            cs=agg.get(d,[]); 
            if not cs: row.append(f"{d}: -"); continue
            row.append(f"{d}: {len(cs)}/{sum(len(c) for c in cs)//len(cs)}/{100*sum('</think>' in c for c in cs)//len(cs)}%/{100*sum(len(c)>50000 for c in cs)//len(cs)}%")
        print(s, "|", " | ".join(row))
