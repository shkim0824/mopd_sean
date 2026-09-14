# MedQA-Evol candidate pool from UltraMedical (type=Exam, ids of the MedQA-Evol subset): 4-option MCQ, answer letter,
# dedup, eval-decontaminated. Same prompt format as the other med pools (_mcqa_prompt).
import pyarrow.parquet as pq, pyarrow.compute as pc, glob, collections, json, re, random, sys, os
sys.path.insert(0, os.getcwd())
from mopd.eval.domains.benches import _mcqa_prompt
from mopd.data.domains.prep_rl_pools import build_eval_grams, contaminated
D="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/domains"
N=int(sys.argv[1]) if len(sys.argv)>1 else 20000
pref=collections.Counter(); rows=[]
for f in sorted(glob.glob(D+"/med_ultramedical/data/train-*.parquet")):
    t=pq.read_table(f, columns=["id","type","conversations","answer"]); t=t.filter(pc.equal(t.column("type"),"Exam"))
    for i,a,c in zip(t.column("id").to_pylist(), t.column("answer").to_pylist(), t.column("conversations").to_pylist()):
        p=re.split(r"[^A-Za-z]", str(i))[0]; pref[p]+=1
        if "evol" not in str(i).lower(): continue
        q=""
        for m in (c or []):
            if isinstance(m,dict) and m.get("from") in ("human","user"): q=m.get("value",""); break
        rows.append((i,str(a).strip().upper(),q))
print("Exam id prefixes:", pref.most_common(10)); print("MedQA-Evol rows", len(rows))
if not rows: sys.exit("no Evol rows found")
optre=re.compile(r"^\s*\(?([A-E])[\.\):]\s*(.+?)\s*$")
cand=[]; bad=collections.Counter(); seen=set(); nopt=collections.Counter()
for i,a,q in rows:
    lines=q.strip().split("\n"); opts={}; stem_lines=[]
    for ln in lines:
        m=optre.match(ln)
        if m and (not opts or m.group(1)==chr(ord(max(opts))+1)): opts[m.group(1)]=m.group(2)
        elif not opts: stem_lines.append(ln)
        else: bad["trailing_text"]+=1; break
    stem="\n".join(stem_lines).strip(); nopt[len(opts)]+=1
    if sorted(opts) not in (["A","B","C","D"],["A","B","C","D","E"]): bad["not4or5opts"]+=1; continue
    if a not in opts: bad["ans_not_in_opts"]+=1; continue
    if len(stem)<120: bad["short_stem"]+=1; continue
    k=re.sub(r"\W+"," ",stem.lower()).strip()
    if k in seen: bad["dup"]+=1; continue
    seen.add(k); letters="".join(sorted(opts))
    cand.append({"id":None,"input":_mcqa_prompt(stem,[opts[x] for x in letters]),"output":a,"meta":{"source":"UM-MedQA-Evol","kind":"mcqa%d"%len(opts),"stem_len":len(stem),"um_id":i}})
print("parsed", len(cand), dict(bad), "n_options", dict(nopt), "answers", collections.Counter(x["output"] for x in cand))
random.Random(0).shuffle(cand); cand=cand[:int(N*1.15)]
grams=build_eval_grams("med"); before=len(cand); cand=[x for x in cand if not contaminated(x["input"],grams)][:N]
print("decon dropped", before-len(cand))
for j,x in enumerate(cand): x["id"]=f"evol-{j:05d}"
with open("data/rl_pools/med_evol_candidates.jsonl","w") as f:
    for x in cand: f.write(json.dumps(x,ensure_ascii=False)+"\n")
L=sorted(x["meta"]["stem_len"] for x in cand); print("wrote", len(cand), "stem_len p10/50/90:", L[len(L)//10], L[len(L)//2], L[9*len(L)//10])
print(cand[0]["input"][:700].replace("\n"," | "), "->", cand[0]["output"])
