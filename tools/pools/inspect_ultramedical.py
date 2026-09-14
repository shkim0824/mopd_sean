import pyarrow.parquet as pq, glob, collections, json, re, random
D="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/domains/med_ultramedical/data"
fs=sorted(glob.glob(D+"/train-*.parquet")); t=pq.read_table(fs[0]); print("cols", t.column_names, "rows shard0", t.num_rows)
rows=[]
for f in fs:
    tt=pq.read_table(f); rows+=tt.to_pylist()
print("total rows", len(rows))
print("type:", collections.Counter(r.get("type") for r in rows).most_common(20))
def first_user(r):
    c=r.get("conversations") or []
    for m in c:
        if isinstance(m,dict) and m.get("from") in ("human","user"): return m.get("value","")
    return ""
opt=re.compile(r"(^|\n)\s*[A-E][\.\):]\s")
bytype=collections.defaultdict(list)
for r in rows: bytype[r.get("type")].append(r)
for ty,rs in bytype.items():
    qs=[first_user(r) for r in rs]; mcq=[q for q in qs if len(opt.findall(q))>=4]
    L=sorted(len(q) for q in mcq) if mcq else [0]
    ans=collections.Counter(str(r.get("answer"))[:3] for r in rs)
    print(f"{ty}: n={len(rs)} mcq-like={len(mcq)} len p50={L[len(L)//2]} p90={L[9*len(L)//10]} answers={ans.most_common(8)}")
random.seed(0)
for ty in ("TextBookQA","MedQA-Evol","MedQA","MedMCQA"):
    rs=bytype.get(ty) or []
    for r in random.sample(rs, min(2,len(rs))):
        print("--", ty, "| answer:", str(r.get("answer"))[:20], "| id:", r.get("id"), "\n", first_user(r)[:700].replace("\n"," | "))
