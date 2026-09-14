# Held-out medical RL pool v2: prompts DISJOINT from the C6 SFT set (MedQA-train).
#  (1) m23k MedMCQA subset (m1 paper: difficulty-filtered, R1-verified)  (2) MedMCQA-train single-choice long stems
#  + MedQA 4-option dev (1,272) as RL validation, + MedQA test pool for a pass@k headroom probe (read-only diagnostic).
import pyarrow.parquet as pq, json, re, ast, glob, collections, random, sys, os
sys.path.insert(0, os.getcwd())
from mopd.eval.domains.benches import _mcqa_prompt, _letters
from mopd.data.domains.prep_rl_pools import build_eval_grams, contaminated
D="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/domains"
MIN_STEM=int(sys.argv[1]) if len(sys.argv)>1 else 200
def norm(q): return re.sub(r"\W+"," ",(q or "").lower()).strip()
# --- MedMCQA train index
t=pq.read_table(f"{D}/med_medmcqa/data/train-00000-of-00001.parquet")
cols=("question","opa","opb","opc","opd","cop","choice_type","subject_name","id")
mm=[{c:t.column(c)[i].as_py() for c in cols} for i in range(t.num_rows)]
mincop=min(int(r["cop"]) for r in mm); print("medmcqa cop min", mincop, "max", max(int(r["cop"]) for r in mm))
for r in mm: r["cop"]=int(r["cop"])-(1 if mincop==1 else 0)
mm_by_q={}
for r in mm: mm_by_q.setdefault(norm(r["question"]), r)
def mm_row(r, src, extra=None):
    opts=[str(r["opa"]),str(r["opb"]),str(r["opc"]),str(r["opd"])]
    if r["choice_type"]!="single" or not (0<=r["cop"]<4) or any(not o.strip() for o in opts): return None
    return {"input":_mcqa_prompt(str(r["question"]).strip(),opts),"output":_letters(4)[r["cop"]],
            "meta":{"source":src,"kind":"mcqa4","stem_len":len(str(r["question"])),"subject":r["subject_name"],"mm_id":r["id"],**(extra or {})}}
# --- (1) m23k MedMCQA
m=pq.read_table(f"{D}/med_m23k/data/train-00000-of-00001.parquet").to_pylist()
first=[r for r in m if r["source"]=="openlifescienceai/medmcqa"][0]
md=first["metadata"]; md=ast.literal_eval(md) if isinstance(md,str) else md
print("m23k metadata keys:", list(md.keys()) if isinstance(md,dict) else type(md))
cand=[]; seen=set(); stats=collections.Counter(); agree=[0,0]
for r in m:
    if r["source"]!="openlifescienceai/medmcqa": continue
    md=r["metadata"]; md=ast.literal_eval(md) if isinstance(md,str) else md
    q=md.get("question") if isinstance(md,dict) else None
    row=mm_by_q.get(norm(q)) if q else None
    if row is None: stats["m23k_unmatched"]+=1; continue
    x=mm_row(row,"m23k-MedMCQA",{"m23k":True})
    if x is None: stats["m23k_invalid"]+=1; continue
    agree[1]+=1; agree[0]+= (x["output"]==str(r["answer_letter"]).strip().upper())
    k=norm(row["question"])
    if k in seen: stats["dup"]+=1; continue
    seen.add(k); cand.append(x); stats["m23k_kept"]+=1
print("m23k letter agreement with MedMCQA gold:", agree, f"{agree[0]/max(1,agree[1]):.3f}")
# --- (2) MedMCQA-train long stems (not already taken)
for r in mm:
    if len(str(r["question"]))<MIN_STEM: continue
    k=norm(r["question"])
    if k in seen: stats["long_dup"]+=1; continue
    x=mm_row(r,"MedMCQA-long")
    if x is None: continue
    seen.add(k); cand.append(x); stats["long_kept"]+=1
print("before decon:", len(cand), dict(stats))
grams=build_eval_grams("med")
cand=[x for x in cand if not contaminated(x["input"],grams)]
print("after decon:", len(cand), collections.Counter(x["meta"]["source"] for x in cand), "answers", collections.Counter(x["output"] for x in cand))
random.Random(0).shuffle(cand)
for i,x in enumerate(cand): x["id"]=f"mv2-{i:05d}"
with open("data/rl_pools/med_v2_candidates.jsonl","w") as f:
    for x in cand: f.write(json.dumps(x,ensure_ascii=False)+"\n")
L=sorted(x["meta"]["stem_len"] for x in cand); print("stem_len p10/50/90:", L[len(L)//10], L[len(L)//2], L[9*len(L)//10])
# --- val: MedQA 4-option dev
tv=pq.read_table(f"{D}/med_medqa_bigbio/med_qa_en_4options_source/validation-00000-of-00001.parquet").to_pylist()
val=[]
for i,r in enumerate(tv):
    opts=r["options"]; opts=ast.literal_eval(opts) if isinstance(opts,str) else opts
    opts=[(o["key"],o["value"]) for o in opts] if isinstance(opts,list) else sorted(opts.items())
    val.append({"id":f"mqdev-{i:04d}","input":_mcqa_prompt(r["question"].strip(),[v for _,v in opts]),"output":str(r["answer_idx"]).strip().upper(),"meta":{"source":"MedQA-4opt-dev","kind":"mcqa4"}})
assert all(len(ast.literal_eval(r["options"]) if isinstance(r["options"],str) else r["options"])==4 for r in tv)
with open("data/rl_pools/med_v2_val.jsonl","w") as f:
    for x in val: f.write(json.dumps(x,ensure_ascii=False)+"\n")
print("val rows", len(val), collections.Counter(x["output"] for x in val)); print(val[0]["input"][:300].replace("\n"," | "))
# --- MedQA test pool for headroom probe (diagnostic only)
te=[json.loads(l) for l in open(f"{D}/med_medqa/phrases_no_exclude_test.jsonl")]
tp=[{"id":f"mqtest-{i:04d}","input":_mcqa_prompt(r["question"].strip(),[r["options"][k] for k in sorted(r["options"])]),"output":r["answer_idx"].strip().upper(),"meta":{"source":"MedQA-4opt-test","kind":"mcqa4"}} for i,r in enumerate(te)]
with open("data/distill_pools/medqa_test_probe.jsonl","w") as f:
    for x in tp: f.write(json.dumps(x,ensure_ascii=False)+"\n")
print("test probe rows", len(tp))
