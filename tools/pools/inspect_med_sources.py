import pyarrow.parquet as pq, json, glob, collections, random
D="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/domains"
def q(xs, ps=(.1,.25,.5,.75,.9,.95)):
    xs=sorted(xs); return {p:xs[min(len(xs)-1,int(p*len(xs)))] for p in ps}
t=pq.read_table(f"{D}/med_m23k/data/train-00000-of-00001.parquet").to_pylist()
print("m23k", len(t), list(t[0].keys()))
print(" source:", collections.Counter(r["source"] for r in t).most_common(20))
print(" metadata sample:", str(t[0]["metadata"])[:300]); print(" prompt sample:", t[0]["prompt"][:400].replace("\n"," | "))
print(" answer_letter:", collections.Counter(r["answer_letter"] for r in t).most_common(12))
bysrc=collections.defaultdict(list)
for r in t: bysrc[r["source"]].append(len(r["prompt"]))
for s,v in bysrc.items(): print(f"  {s}: n={len(v)} promptlen p50={q(v)[.5]} p90={q(v)[.9]}")
t=pq.read_table(f"{D}/med_medmcqa/data/train-00000-of-00001.parquet")
print("medmcqa-train", t.num_rows, t.column_names)
qs=t.column("question").to_pylist(); ct=t.column("choice_type").to_pylist(); cop=t.column("cop").to_pylist()
L=[len(x or "") for x in qs]; print(" stem len q:", q(L,(.1,.25,.5,.75,.9,.95,.99)))
print(" choice_type:", collections.Counter(ct).most_common(), " cop range:", min(cop), max(cop))
for th in (120,200,300,500): print(f"  single & stem>={th}:", sum(1 for l,c in zip(L,ct) if c=="single" and l>=th))
print(" subjects:", collections.Counter(t.column("subject_name").to_pylist()).most_common(22))
random.seed(0); idx=[i for i,(l,c) in enumerate(zip(L,ct)) if c=="single" and l>=300]
for i in random.sample(idx,3): print("  --", qs[i][:350].replace("\n"," "), "|| opa:", str(t.column("opa")[i])[:50], "| cop", cop[i])
for sp in ("train","validation","test"):
    tt=pq.read_table(f"{D}/med_medqa_bigbio/med_qa_en_4options_source/{sp}-00000-of-00001.parquet"); print("medqa4opt", sp, tt.num_rows, tt.column_names)
r=tt.slice(0,1).to_pylist()[0]; print(" ", {k:str(v)[:200] for k,v in r.items()})
j=json.load(open(f"{D}/med_r1_distill/medical_r1_distill_sft.json")); r=j[0] if isinstance(j,list) else j
print("r1_distill", len(j), {k:str(v)[:200] for k,v in r.items()})
tt=pq.read_table(sorted(glob.glob(f"{D}/med_ii_medical/data/train-*.parquet"))[0]); print("ii_medical shard0", tt.num_rows, tt.column_names)
r=tt.slice(0,1).to_pylist()[0]; print(" ", {k:str(v)[:200] for k,v in r.items()})
