import pyarrow.parquet as pq, os, hashlib
B="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean/outputs/opd"
def loops(run,step):
    f="%s/%s/completions/completions_%05d.parquet" % (B,run,step)
    if not os.path.exists(f): return None
    t=pq.read_table(f).to_pydict()
    return {hashlib.md5(p.encode()).hexdigest()[:8] for p,c in zip(t["prompt"],t["completion"]) if "</think>" not in c}
runs=["law-pg","law-pg-v2","law-top64-v2","law-top64","law-top16-v2"]
def jac(a,b):
    if not a or not b: return float("nan")
    return len(a&b)/max(1,len(a|b))
for s in (78,80,82,84,88):
    L={r:loops(r,s) for r in runs}
    line=", ".join("%s:%s/32" % (r, len(v) if v is not None else "-") for r,v in L.items())
    print("step %d: unfinished per run -> %s | Jaccard pg-v1 vs pg-v2 %.2f; pg-v1 vs top64-v2 %.2f" % (s, line, jac(L["law-pg"],L["law-pg-v2"]), jac(L["law-pg"],L["law-top64-v2"])))
