"""Screen the held-out med candidates by the policy's pass@k (gen_teacher probe) and write the RL pool.
usage: PYTHONPATH=. python3 tmp/filter_med_v2_pool.py outputs/teacher_gen/probe-med-v2-c6_3ep-k8 data/rl_pools/med_v2_candidates.jsonl \
         --lo 1 --hi 7 [--max-short N] [--short-th 200] --out data/rl_pools/med_v2_train.jsonl
"""
import json, glob, sys, collections, argparse, random
from mopd.graders.domains.extract import extract_letter
p=argparse.ArgumentParser(); p.add_argument("gen"); p.add_argument("pool"); p.add_argument("--lo",type=int,default=1); p.add_argument("--hi",type=int,default=7)
p.add_argument("--max-short",type=int,default=None); p.add_argument("--short-th",type=int,default=200); p.add_argument("--out"); p.add_argument("--seed",type=int,default=0)
a=p.parse_args()
pool={}
for l in open(a.pool):
    r=json.loads(l); pool[r["id"]]=r
res={}; trunc=0; ns=0
for f in sorted(glob.glob(a.gen+"/shard*.jsonl")):
    for l in open(f):
        g=json.loads(l); gold=str(g["gold"]).strip().upper(); n=0
        for s in g["samples"]:
            ns+=1
            if s.get("finish")!="stop": trunc+=1; continue
            n+= (extract_letter(s["text"], n_options=4)==gold)
        res[g["id"]]=(n,len(g["samples"]))
print(f"probed {len(res)}/{len(pool)} prompts | trunc/sample {trunc/max(1,ns):.3f}")
def bucket(r): return "short" if r["meta"]["stem_len"]<a.short_th else "long"
stat=collections.defaultdict(collections.Counter); hist=collections.Counter()
for pid,(n,k) in res.items():
    r=pool[pid]; cat="all" if n==k else ("zero" if n==0 else "mixed"); hist[n]+=1
    stat[(r["meta"]["source"],bucket(r))][cat]+=1
print("#correct histogram:", dict(sorted(hist.items())))
for key,c in sorted(stat.items()):
    t=sum(c.values()); print(f"  {key[0]:14s} {key[1]:5s} n={t:5d} all={c['all']/t:.2f} mixed={c['mixed']/t:.2f} zero={c['zero']/t:.2f}")
keep=[pid for pid,(n,k) in res.items() if a.lo<=n<=a.hi]
rng=random.Random(a.seed); rng.shuffle(keep)
if a.max_short is not None:
    short=[x for x in keep if bucket(pool[x])=="short"][:a.max_short]; long_=[x for x in keep if bucket(pool[x])=="long"]
    keep=short+long_; rng.shuffle(keep)
out=[]
for pid in keep:
    r=pool[pid]; n,k=res[pid]
    out.append({"input":r["input"],"output":r["output"],"meta":{**r["meta"],"id":pid,"pass":n,"k":k}})
print(f"kept {len(out)} (pass in [{a.lo},{a.hi}])", collections.Counter((x["meta"]["source"],bucket(x)) for x in out), "answers", collections.Counter(x["output"] for x in out))
if a.out:
    with open(a.out,"w") as f:
        for x in out: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    print("wrote", a.out)
