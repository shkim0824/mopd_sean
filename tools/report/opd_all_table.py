#!/usr/bin/env python3
"""Single-domain OPD tables (law / fin / IF / med) with base + teacher rows and s̃ (base->teacher fraction)."""
import json, os
S="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean"
DE=S+"/mopd_sean/outputs/eval_domain"; FE=S+"/mopd_sean/outputs/eval_6bench"
BASE={"medqa":69.8,"casehold":63.2,"finqa":58.3,"ifeval":51.0,"ifbench":27.7}; TEACH={"medqa":80.7,"casehold":78.3,"finqa":74.6,"ifeval":79.1,"ifbench":58.7}
def dom(tag):
    f=f"{DE}/{tag}/metrics.json"
    if not os.path.exists(f): return {}
    m=json.load(open(f)); o={}
    for b,r in m.items():
        if isinstance(r,dict) and "accuracy" in r: o[b]=(round(r["accuracy"]*100,1), round(r.get("truncated_frac",0)*100,1))
    return o
def full(tag):
    f=f"{FE}/{tag}/metrics.json"
    if not os.path.exists(f): return {}
    m=json.load(open(f)); b=m["benchmarks"]; o={k:(round(v["score"],1),None) for k,v in b.items()}
    o["math"]=(round(m["domain_avg"]["math"],1),None); o["code"]=(round(m["domain_avg"]["code"],1),None); return o
def cell(d,k):
    if k not in d: return "–"
    v,tr=d[k]; s=f"{v:.1f}"
    if tr is not None and tr>=5: s+=f" (trunc {tr:.0f}%)"
    return s
def sn(d,k): return (d[k][0]-BASE[k])/(TEACH[k]-BASE[k])
def merged(*ds):
    o={}
    for d in ds: o.update(d)
    return o
base=merged(dom("base-ot3-alldom-T1"), full("sftot3-4b-final-32k"))
def table(title, bench, teacher_row, runs, ck_tag, final_tag, full_tag, extra=None):
    print(f"\n## {title}\n")
    print(f"| model | {bench} T1.0 (ck200) | {bench} T0.6 (final) | IFEval | IFBench | Math (AIME24-26) | Code (LCB v6) | s̃ {bench} |"); print("|---|---|---|---|---|---|---|---|")
    def row(name, t1, t06, fl):
        d=merged(fl); s=f"{100*((t1[bench][0]-BASE[bench])/(TEACH[bench]-BASE[bench])):.0f}%" if bench in t1 else "–"
        print(f"| {name} | {cell(t1,bench)} | {cell(t06,bench)} | {cell(d,'ifeval')} | {cell(d,'ifbench')} | {cell(d,'math')} | {cell(d,'code')} | {s} |")
    row("base OT3", base, dom("base-ot3-alldom-T06"), base)
    row(*teacher_row)
    for run,label in runs:
        row(label, dom(ck_tag(run)), dom(final_tag(run)), full(full_tag(run)))
table("LAW OPD (teacher rl2-law-distill-ck200 = law SFT->RL)", "casehold",
      ("teacher-law", dom("rl2-law-distill-ck200-casehold-T1"), dom("rl2-law-distill-ck200-casehold"), full("full-law-rl-ck200-32k")),
      [("law-top64-v2","law top64 v2"),("law-pg-v2","law pg v2"),("law-pg-sftw75","law pg from SFT warm-up ck75 (branch)"),("law-pg","law pg v1"),("law-top64","law top64 v1 (collapsed)"),("law-top16","law top16 v1 (collapsed)"),("law-top16-v2","law top16 v2 (collapsed)")],
      lambda r:f"opd-{r}-ck200-casehold-T1", lambda r:f"opd-{r}-final-casehold", lambda r:f"full-opd-{r}-final-32k")
table("FIN OPD (teacher rl-fin-ck200)", "finqa",
      ("teacher-fin", dom("rl-fin-ck200-T1-32k"), dom("rl-fin-ck200"), full("full-fin-rl-ck200-32k")),
      [("fin-top64","fin top64"),("fin-top16","fin top16"),("fin-pg","fin pg")],
      lambda r:f"opd-{r}-ck200-finqa-T1", lambda r:f"opd-{r}-final-finqa", lambda r:f"full-opd-{r}-final-32k")
print("\n## IF OPD (teacher rl-if-ck200; in-domain = IFEval/IFBench from the 6-bench, 32k T0.6)\n")
print("| model | IFEval | IFBench | Math (AIME24-26) | Code (LCB v6) | s̃ IF |"); print("|---|---|---|---|---|---|")
def ifrow(name,d):
    s=f"{100*(sn(d,'ifeval')+sn(d,'ifbench'))/2:.0f}%" if "ifeval" in d else "–"
    print(f"| {name} | {cell(d,'ifeval')} | {cell(d,'ifbench')} | {cell(d,'math')} | {cell(d,'code')} | {s} |")
ifrow("base OT3", base); ifrow("teacher-if", full("rl-if-ck200-32k-fixedrope"))
for r,l in [("if-top64","if top64"),("if-top16","if top16"),("if-pg","if pg")]: ifrow(l, full(f"full-opd-if-{r}-final-32k"))
