"""Merge screened held-out pools into the final med RL v2 pool.
usage: python3 tmp/merge_med_v2_pool.py --tbqa data/rl_pools/med_v2_train_tbqa.jsonl --mm data/rl_pools/med_v2_train_mm.jsonl --short-frac 0.3 --out data/rl_pools/med_v2_train.jsonl
keeps ALL TextBookQA + ALL long-stem MedMCQA mixed prompts; caps short-stem m23k so that short <= short_frac of the total."""
import json, random, argparse, collections
p=argparse.ArgumentParser(); p.add_argument("--tbqa"); p.add_argument("--mm"); p.add_argument("--short-frac",type=float,default=0.3); p.add_argument("--short-th",type=int,default=200); p.add_argument("--out"); p.add_argument("--seed",type=int,default=0)
a=p.parse_args(); rng=random.Random(a.seed)
tb=[json.loads(l) for l in open(a.tbqa)] if a.tbqa else []
mm=[json.loads(l) for l in open(a.mm)]
long_=[r for r in mm if r["meta"]["stem_len"]>=a.short_th]; short=[r for r in mm if r["meta"]["stem_len"]<a.short_th]
base=tb+long_; n_short=int(a.short_frac/(1-a.short_frac)*len(base)); rng.shuffle(short); short=short[:n_short]
pool=base+short; rng.shuffle(pool)
print(f"tbqa {len(tb)} + long-mm {len(long_)} + short-mm {len(short)} (cap {n_short} of {sum(1 for r in mm if r['meta']['stem_len']<a.short_th)}) = {len(pool)}")
print("sources:", collections.Counter(r["meta"]["source"] for r in pool), "answers:", collections.Counter(r["output"] for r in pool))
ps=[r["meta"]["pass"] for r in pool]; print("pass histogram:", dict(sorted(collections.Counter(ps).items())))
with open(a.out,"w") as f:
    for r in pool: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("wrote", a.out)
