import json,glob,os,re,collections
R="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean/outputs/eval_domain"
rows=collections.defaultdict(dict)
for f in glob.glob(R+"/rl4-med-v*/metrics.json"):
    tag=os.path.basename(os.path.dirname(f))
    if "partial" in tag: continue
    m=json.load(open(f)); r=m.get("medqa")
    if not r or r.get("missing",0)>0: continue
    mm=re.match(r"rl4-med-(v\w+)-ck(\d+)-medqa(-T1)?$",tag)
    if not mm: continue
    rows[mm.group(1)][(int(mm.group(2)),"T1" if mm.group(3) else "T06")]=round(r["accuracy"]*100,1)
for v in sorted(rows):
    cks=sorted({k[0] for k in rows[v]})
    print(v, " | ".join("ck%d: %s/%s" % (c, rows[v].get((c,"T06"),"-"), rows[v].get((c,"T1"),"-")) for c in cks))
