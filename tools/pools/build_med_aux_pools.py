# Auxiliary medical RL candidates for a portfolio pool:
#  (1) HeadQA-en train (5-option MCQ, hard exam questions) -> mcqa5 prompts (same format as the other pools)
#  (2) UltraMedical Literature (= PubMedQA abstracts, answer yes/no/maybe) -> our PubMedQA eval prompt format (kind=pubmedqa)
# Output: data/rl_pools/med_aux_candidates.jsonl (both, shuffled, eval-decontaminated)
import pyarrow.parquet as pq, pyarrow.compute as pc, glob, collections, json, re, random, sys, os
sys.path.insert(0, os.getcwd())
from mopd.eval.domains.benches import _mcqa_prompt
from mopd.data.domains.prep_rl_pools import build_eval_grams, contaminated
D="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/domains"
NPUB=int(sys.argv[1]) if len(sys.argv)>1 else 8000
grams=build_eval_grams("med"); cand=[]; bad=collections.Counter()
# HeadQA
items=json.load(open(D+"/med_headqa/train.json"))
for it in items:
    d=it["data"]; opts=d["Options"]; letters=sorted(opts)
    if letters!=list("ABCDE"[:len(letters)]) or len(letters)<4: bad["headqa_opts"]+=1; continue
    q=d["Question"].strip(); a=d["Correct Option"].strip().upper()
    if a not in opts or len(q)<15: bad["headqa_bad"]+=1; continue
    cand.append({"id":None,"input":_mcqa_prompt(q,[opts[x] for x in letters]),"output":a,"meta":{"source":"HeadQA-en","kind":"mcqa%d"%len(letters),"topic":it.get("topic_name"),"stem_len":len(q)}})
nh=len(cand); print("HeadQA parsed", nh, dict(bad))
# PubMedQA (UltraMedical Literature)
rows=[]
for f in sorted(glob.glob(D+"/med_ultramedical/data/train-*.parquet")):
    t=pq.read_table(f, columns=["id","type","conversations","answer"]); t=t.filter(pc.equal(t.column("type"),"Literature"))
    for i,a,c in zip(t.column("id").to_pylist(), t.column("answer").to_pylist(), t.column("conversations").to_pylist()):
        q=[m.get("value","") for m in (c or []) if m.get("from") in ("human","user")]
        if q: rows.append((i,str(a).strip().upper(),q[0]))
random.Random(1).shuffle(rows); rows=rows[:int(NPUB*1.3)]
optre=re.compile(r"\n([A-C])\. (yes|no|maybe)\s*$", re.M)
for i,a,q in rows:
    m=dict(re.findall(r"\n([A-C])\. (yes|no|maybe)", q))
    if sorted(m)!=["A","B","C"] or a not in m: bad["pub_opts"]+=1; continue
    body=q[:q.find("\nA. ")] if "\nA. " in q else q
    body=re.sub(r"\n[A-C]\. (yes|no|maybe)\s*","\n",body).strip()
    lines=[l for l in body.split("\n") if l.strip()]
    if len(lines)<2: bad["pub_short"]+=1; continue
    question=lines[-1].strip(); ctx="\n".join(l.replace("Context: ","",1) for l in lines[:-1]).strip()
    if not question.endswith("?") or len(ctx)<200: bad["pub_format"]+=1; continue
    prompt=(f"Abstract:\n{ctx}\n\nQuestion: {question}\n\nPlease reason step by step, then answer strictly with yes, no, or maybe on the last line in the form \"Answer: <yes/no/maybe>\".")
    cand.append({"id":None,"input":prompt,"output":m[a],"meta":{"source":"UM-PubMedQA","kind":"pubmedqa","um_id":i,"stem_len":len(ctx)}})
print("PubMedQA parsed", len(cand)-nh, dict(bad), "labels", collections.Counter(x["output"] for x in cand[nh:]))
before=len(cand); cand=[x for x in cand if not contaminated(x["input"],grams)]; print("decon dropped", before-len(cand))
pub=[x for x in cand if x["meta"]["kind"]=="pubmedqa"][:NPUB]; head=[x for x in cand if x["meta"]["kind"]!="pubmedqa"]
cand=head+pub; random.Random(0).shuffle(cand)
for j,x in enumerate(cand): x["id"]=f"aux-{j:05d}"
with open("data/rl_pools/med_aux_candidates.jsonl","w") as f:
    for x in cand: f.write(json.dumps(x,ensure_ascii=False)+"\n")
print("wrote", len(cand), collections.Counter(x["meta"]["source"] for x in cand))
print(pub[0]["input"][:400].replace("\n"," | "), "->", pub[0]["output"])
