# Held-out MedQA-style pool from UltraMedical TextBookQA (GPT-4 vignettes from textbooks; disjoint from MedQA/MedMCQA).
import pyarrow.parquet as pq, pyarrow.compute as pc, glob, collections, json, re, random, sys, os
sys.path.insert(0, os.getcwd())
from mopd.eval.domains.benches import _mcqa_prompt
from mopd.data.domains.prep_rl_pools import build_eval_grams, contaminated
D="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/domains"
N=int(sys.argv[1]) if len(sys.argv)>1 else 12000
rows=[]
for f in sorted(glob.glob(D+"/med_ultramedical/data/train-*.parquet")):
    t=pq.read_table(f, columns=["id","type","conversations","answer"]); t=t.filter(pc.equal(t.column("type"),"Exam"))
    for i,a,c in zip(t.column("id").to_pylist(), t.column("answer").to_pylist(), t.column("conversations").to_pylist()):
        if not str(i).startswith("TextBookQA"): continue
        q=""
        for m in (c or []):
            if isinstance(m,dict) and m.get("from") in ("human","user"): q=m.get("value",""); break
        rows.append((i,str(a).strip().upper(),q))
print("TextBookQA rows", len(rows))
optre=re.compile(r"^\s*\(?([A-E])[\.\):]\s*(.+?)\s*$")
cand=[]; bad=collections.Counter(); seen=set()
for i,a,q in rows:
    lines=q.strip().split("\n"); opts={}; stem_lines=[]
    for ln in lines:
        m=optre.match(ln)
        if m and (not opts or m.group(1)==chr(ord(max(opts))+1)): opts[m.group(1)]=m.group(2)
        elif not opts: stem_lines.append(ln)
        else: bad["trailing_text"]+=1; break
    stem="\n".join(stem_lines).strip()
    if sorted(opts)!=["A","B","C","D"]: bad["not4opts"]+=1; continue
    if a not in opts: bad["ans_not_in_opts"]+=1; continue
    if len(stem)<80: bad["short_stem"]+=1; continue
    k=re.sub(r"\W+"," ",stem.lower()).strip()
    if k in seen: bad["dup"]+=1; continue
    seen.add(k)
    cand.append({"id":None,"input":_mcqa_prompt(stem,[opts[x] for x in "ABCD"]),"output":a,"meta":{"source":"UM-TextBookQA","kind":"mcqa4","stem_len":len(stem),"um_id":i}})
print("parsed", len(cand), dict(bad), "answers", collections.Counter(x["output"] for x in cand))
random.Random(0).shuffle(cand); cand=cand[:int(N*1.1)]
grams=build_eval_grams("med"); cand=[x for x in cand if not contaminated(x["input"],grams)][:N]
for j,x in enumerate(cand): x["id"]=f"tbqa-{j:05d}"
with open("data/rl_pools/med_tbqa_candidates.jsonl","w") as f:
    for x in cand: f.write(json.dumps(x,ensure_ascii=False)+"\n")
L=sorted(x["meta"]["stem_len"] for x in cand); print("wrote", len(cand), "stem_len p10/50/90:", L[len(L)//10], L[len(L)//2], L[9*len(L)//10])
print(cand[0]["input"][:500].replace("\n"," | "), "->", cand[0]["output"])
