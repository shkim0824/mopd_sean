"""Build the med v4 RL pool from the MedQA-Evol candidates:
   keep prompts where (a) the 35B teacher's answer == label (label verification) and (b) the policy's pass@8 is in [lo,hi].
usage: PYTHONPATH=. python3 tmp/build_med_v4_pool.py --probe outputs/teacher_gen/probe-med-evol-c6_4ep-k8 \
         --verify outputs/teacher_gen/verify-med-evol-35b-k1 --pool data/rl_pools/med_evol_candidates.jsonl --lo 1 --hi 7 --out data/rl_pools/med_v4_train.jsonl
"""
import json, glob, argparse, collections, random
from mopd.graders.domains.extract import extract_letter
p=argparse.ArgumentParser(); p.add_argument("--probe",required=True); p.add_argument("--verify",required=True); p.add_argument("--pool",required=True)
p.add_argument("--lo",type=int,default=1); p.add_argument("--hi",type=int,default=7); p.add_argument("--out",required=True); p.add_argument("--no-verify",action="store_true")
a=p.parse_args()
pool={}
for l in open(a.pool):
    r=json.loads(l); pool[r["id"]]=r
def read(d):
    res={}
    for f in sorted(glob.glob(d+"/shard*.jsonl")):
        for l in open(f):
            g=json.loads(l); res[g["id"]]=g
    return res
probe=read(a.probe); verify=read(a.verify) if not a.no_verify else {}
print(f"pool {len(pool)} | probed {len(probe)} | verified {len(verify)}")
hist=collections.Counter(); agree=collections.Counter(); keep=[]; nopt=lambda r: 5 if r["meta"]["kind"]=="mcqa5" else 4
for pid,r in pool.items():
    g=probe.get(pid)
    if not g: continue
    gold=str(g["gold"]).strip().upper(); n=0; k=0
    for s in g["samples"]:
        k+=1
        if s.get("finish")!="stop": continue
        n+= (extract_letter(s["text"], n_options=nopt(r))==gold)
    hist[n]+=1
    ok_label=True
    if verify:
        v=verify.get(pid)
        if not v: agree["unverified"]+=1; continue
        s=v["samples"][0]; t=extract_letter(s["text"], n_options=nopt(r)) if s.get("finish")=="stop" else None
        if t is None: agree["teacher_no_answer"]+=1; ok_label=False
        elif t==gold: agree["agree"]+=1
        else: agree["disagree"]+=1; ok_label=False
    if ok_label and a.lo<=n<=a.hi: keep.append((pid,n))
print("policy #correct/8 histogram:", dict(sorted(hist.items())))
print("teacher label agreement:", dict(agree))
random.Random(0).shuffle(keep)
with open(a.out,"w") as f:
    for pid,n in keep:
        r=dict(pool[pid]); r["meta"]=dict(r["meta"]); r["meta"]["policy_pass8"]=n; f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"wrote {len(keep)} -> {a.out}; pass8 distribution among kept:", dict(sorted(collections.Counter(n for _,n in keep).items())))
