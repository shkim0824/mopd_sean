import pyarrow.parquet as pq, pyarrow.compute as pc, glob, collections, json, re, random, time
t0=time.time()
D="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/domains"
rows=[]
for f in sorted(glob.glob(D+"/med_ultramedical/data/train-*.parquet")):
    t=pq.read_table(f, columns=["id","type","conversations","answer"]); t=t.filter(pc.equal(t.column("type"),"Exam"))
    ids=t.column("id").to_pylist(); ans=t.column("answer").to_pylist(); conv=t.column("conversations").to_pylist()
    for i,a,c in zip(ids,ans,conv):
        q=""
        for m in (c or []):
            if isinstance(m,dict) and m.get("from") in ("human","user"): q=m.get("value",""); break
        rows.append((i,a,q))
print("exam rows", len(rows), f"{time.time()-t0:.0f}s", flush=True)
pref=collections.Counter(re.split(r"[,_\-\d]",str(i))[0] for i,_,_ in rows); print("id prefixes:", pref.most_common(12))
print("id samples:", [r[0] for r in rows[:3]], [r[0] for r in rows[-3:]])
opt=re.compile(r"\n\s*A[\.\):]\s")
def stem(q):
    m=opt.search(q); return (q[:m.start()] if m else q)
def norm(q): return re.sub(r"\W+"," ",q.lower()).strip()
mq=set(norm(json.loads(l)["question"]) for l in open(D+"/med_medqa/phrases_no_exclude_train.jsonl"))
mqt=set(norm(json.loads(l)["question"]) for l in open(D+"/med_medqa/phrases_no_exclude_test.jsonl"))
mm=set(norm(x) for x in pq.read_table(D+"/med_medmcqa/data/train-00000-of-00001.parquet",columns=["question"]).column("question").to_pylist())
by=collections.defaultdict(collections.Counter); lens=collections.defaultdict(list); new=[]
for i,a,q in rows:
    s=norm(stem(q)); p=re.split(r"[,_\-\d]",str(i))[0]
    k="medqa-train" if s in mq else ("medqa-TEST" if s in mqt else ("medmcqa-train" if s in mm else "new"))
    by[p][k]+=1; lens[(p,k)].append(len(stem(q)))
    if k=="new": new.append((i,a,q))
for p,c in by.items():
    print(p, dict(c), {k:(sorted(v)[len(v)//2]) for (pp,k),v in lens.items() if pp==p})
print("NEW (not in MedQA-train/test or MedMCQA-train):", len(new), f"{time.time()-t0:.0f}s")
random.seed(1)
for i,a,q in random.sample(new,3): print("--", i, "| ans", a, "\n", q[:600].replace("\n"," | "))
