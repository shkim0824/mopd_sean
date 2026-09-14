"""v5 = portfolio pool (MedQA-Evol mixed + HeadQA-en mixed + TBQA mixed + MedMCQA-long mixed);
   v6 = on-distribution pool (MedQA 4-option dev split: 900 train prompts) + MedQA-Evol mixed; val = remaining 372 dev items."""
import json,glob,collections,random,re
from mopd.graders.domains.extract import extract_letter
def rd(p): return [json.loads(l) for l in open(p)]
evol=rd("data/rl_pools/med_v4_train.jsonl")
v3=rd("data/rl_pools/med_v3_train.jsonl"); src=collections.Counter(r["meta"].get("source") for r in v3); print("v3 sources:", dict(src))
other=[r for r in v3 if r["meta"].get("source") not in ("MedQA-train","medqa_train","MedQA","medqa") and "MedQA" not in str(r["meta"].get("source"))]
print("v3 non-MedQA-train rows:", len(other), collections.Counter(r["meta"].get("source") for r in other))
# HeadQA mixed from the aux probe
pool={r["id"]:r for r in rd("data/rl_pools/med_aux_candidates.jsonl")}; head=[]
for f in sorted(glob.glob("outputs/teacher_gen/probe-med-aux-c6_4ep-k8/shard*.jsonl")):
    for l in open(f):
        g=json.loads(l); r=pool[g["id"]]
        if r["meta"]["source"]!="HeadQA-en": continue
        n=sum(1 for s in g["samples"] if s.get("finish")=="stop" and extract_letter(s["text"], n_options=5)==r["output"])
        if 1<=n<=7: r=dict(r); r["meta"]=dict(r["meta"]); r["meta"]["policy_pass8"]=n; head.append(r)
print("HeadQA mixed:", len(head))
v5=evol+head+other; random.Random(0).shuffle(v5)
for j,r in enumerate(v5): r["id"]=f"v5-{j:05d}"
with open("data/rl_pools/med_v5_train.jsonl","w") as f:
    for r in v5: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("v5 pool:", len(v5), collections.Counter(r["meta"].get("source") for r in v5))
# v6: dev split
dev=rd("data/rl_pools/med_v2_val.jsonl"); random.Random(0).shuffle(dev); dtr,dval=dev[:900],dev[900:]
for r in dtr: r["meta"]=dict(r.get("meta") or {}); r["meta"]["source"]="MedQA-dev-train"
v6=dtr+[dict(r) for r in evol]; random.Random(1).shuffle(v6)
for j,r in enumerate(v6): r["id"]=f"v6-{j:05d}"
with open("data/rl_pools/med_v6_train.jsonl","w") as f:
    for r in v6: f.write(json.dumps(r,ensure_ascii=False)+"\n")
with open("data/rl_pools/med_v6_val.jsonl","w") as f:
    for r in dval: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("v6 pool:", len(v6), "val:", len(dval), "dev sample:", dtr[0]["input"][:120].replace("\n"," "), "->", dtr[0]["output"])
