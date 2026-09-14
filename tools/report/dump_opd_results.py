import json,glob,os,re
S="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean"
print("=== mopd_domains domain evals (tag: bench acc trunc n)")
for f in sorted(glob.glob(S+"/mopd_sean/outputs/eval_domain/*/metrics.json")):
    tag=os.path.basename(os.path.dirname(f))
    if not re.search(r"opd-|law|fin|casehold|finqa|tatqa|ot3|base", tag, re.I): continue
    try: m=json.load(open(f))
    except Exception as e: print(tag,"ERR",e); continue
    for b,r in m.items():
        if not isinstance(r,dict): continue
        acc=r.get("accuracy", r.get("f1", r.get("score")))
        acc=None if acc is None else round(acc*100,1)
        tr=round(r.get("truncated_frac",0)*100,1); n=r.get("n")
        print("%s: %s acc=%s trunc=%s n=%s" % (tag,b,acc,tr,n))
print("=== mopd full evals (OOD)")
for f in sorted(glob.glob(S+"/mopd_sean/outputs/eval_6bench/*/metrics.json")):
    tag=os.path.basename(os.path.dirname(f))
    if not re.search(r"full-|sftot3-4b-final-32k", tag): continue
    m=json.load(open(f)); b=m["benchmarks"]
    print(tag, " ".join("%s=%.1f" % (k,v["score"]) for k,v in b.items()), "| avg", " ".join("%s=%.1f" % (k,v) for k,v in m["domain_avg"].items()))
