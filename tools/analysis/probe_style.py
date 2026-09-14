# Per-source response-length / think-style stats of the policy's probe generations (chars, ~4 chars/token)
import json, glob, sys, collections, statistics as st
def stats(gen_dir, pool, label, bucket=None):
    meta={}
    for l in open(pool):
        r=json.loads(l); meta[r["id"]]=r["meta"]
    L=collections.defaultdict(list); think=collections.defaultdict(list); n=0
    for f in sorted(glob.glob(gen_dir+"/shard*.jsonl")):
        for l in open(f):
            g=json.loads(l); m=meta.get(g["id"],{}); src=m.get("source","?")
            if bucket: src=src+("/short" if m.get("stem_len",999)<200 else "/long")
            for s in g["samples"]:
                t=s["text"]; L[src].append(len(t)); i=t.find("</think>"); think[src].append(i if i>=0 else len(t))
            n+=1
    for src in sorted(L):
        v=sorted(L[src]); w=sorted(think[src]); q=lambda a,p: a[min(len(a)-1,int(p*len(a)))]
        print(f"{label:12s} {src:24s} n={len(v)//8:5d}  resp chars p50={q(v,.5):6d} p90={q(v,.9):6d}  think p50={q(w,.5):6d}  (~tok p50={q(v,.5)//4})")
stats("outputs/teacher_gen/probe-medqa-test-c6_3ep-k8","data/distill_pools/medqa_test_probe.jsonl","MedQA-test")
stats("outputs/teacher_gen/probe-med-tbqa-c6_3ep-k8","data/rl_pools/med_tbqa_candidates.jsonl","TBQA")
stats("outputs/teacher_gen/probe-med-v2-c6_3ep-k8","data/rl_pools/med_v2_candidates.jsonl","MedMCQA",bucket=True)
stats("outputs/teacher_gen/probe-med-c6_3ep-k8","data/distill_pools/medqa45_pool.jsonl","MedQA-train")
