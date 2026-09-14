import json,glob,collections,re,sys
from mopd.graders.domains.extract import extract_letter
pool={json.loads(l)["id"]:json.loads(l) for l in open("data/rl_pools/med_aux_candidates.jsonl")}
ynm=re.compile(r"answer\s*[:\-]?\s*\**\s*(yes|no|maybe)\b", re.I)
def grade(r,s):
    if s.get("finish")!="stop": return None
    t=s["text"]; t=t.split("</think>")[-1] if "</think>" in t else t
    if r["meta"]["kind"]=="pubmedqa":
        m=ynm.findall(t); return (m[-1].lower()==r["output"]) if m else False
    n=5 if r["meta"]["kind"]=="mcqa5" else 4
    return extract_letter(t, n_options=n)==r["output"]
stat=collections.defaultdict(collections.Counter); hist=collections.defaultdict(collections.Counter); tr=collections.Counter(); pred=collections.Counter()
for f in sorted(glob.glob("outputs/teacher_gen/probe-med-aux-c6_4ep-k8/shard*.jsonl")):
    for l in open(f):
        g=json.loads(l); r=pool[g["id"]]; src=r["meta"]["source"]; n=0
        for s in g["samples"]:
            v=grade(r,s); tr[src]+= (s.get("finish")!="stop"); n+= bool(v)
            if src=="UM-PubMedQA" and s.get("finish")=="stop":
                t=s["text"].split("</think>")[-1]; m=ynm.findall(t); pred[m[-1].lower() if m else "none"]+=1
        key=src+("/"+r["output"] if src=="UM-PubMedQA" else "")
        cat="all" if n==8 else ("zero" if n==0 else "mixed"); stat[key][cat]+=1; hist[key][n]+=1
for k,c in sorted(stat.items()):
    t=sum(c.values()); print("%-18s n=%5d all=%.2f mixed=%.2f zero=%.2f | hist %s" % (k,t,c["all"]/t,c["mixed"]/t,c["zero"]/t,dict(sorted(hist[k].items()))))
print("truncated samples by source:", dict(tr))
print("policy PubMedQA prediction distribution:", dict(pred))
