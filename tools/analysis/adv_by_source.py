import pyarrow.parquet as pq, json, glob, re, sys, collections, statistics
sys.path.insert(0, "/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean")
from mopd.graders.domains.extract import extract_letter
R="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean"
def norm(s): return re.sub(r"\s+"," ", s.strip().lower())[:300]
pool={}
for l in open(f"{R}/data/rl_pools/med_opd_train.jsonl"):
    r=json.loads(l); pool[norm(r["input"])]=(r["meta"]["source"], r["output"], {"mcqa4":4,"mcqa5":5}.get(r["meta"].get("kind"),4))
for run in sys.argv[1:]:
    fs=sorted(glob.glob(f"{R}/outputs/opd/{run}/completions/completions_*.parquet"), key=lambda f:int(f.rsplit("_",1)[1].split(".")[0]))
    agg=collections.defaultdict(lambda: collections.defaultdict(list))
    for f in fs:
        step=int(f.rsplit("_",1)[1].split(".")[0]); band="early(1-50)" if step<=50 else "mid(51-125)" if step<=125 else "late(126-200)"
        t=pq.read_table(f).to_pandas()
        for p,c,a in zip(t["prompt"], t["completion"], t["advantage"]):
            q=p.split("\n",1)[1] if p.startswith("user\n") else p
            info=pool.get(norm(q))
            if not info: continue
            src,gold,n=info
            ok=(extract_letter(c, n_options=n)==str(gold).strip().upper())
            agg[band][src].append((float(a), abs(float(a)), ok, "</think>" in c, len(c)))
    print(f"== {run}")
    for band in ["early(1-50)","mid(51-125)","late(126-200)"]:
        for src in sorted(agg[band]):
            v=agg[band][src]; n=len(v)
            print(f"  {band:14s} {src:14s} n={n:5d} mean_adv={statistics.mean(x[0] for x in v):+.3f} mean|adv|={statistics.mean(x[1] for x in v):.3f} student_acc={100*sum(x[2] for x in v)/n:.1f}% closed={100*sum(x[3] for x in v)/n:.0f}% chars_p50={sorted(x[4] for x in v)[n//2]}")
