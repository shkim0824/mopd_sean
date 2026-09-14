#!/usr/bin/env python3
"""Consolidated MOPD / medical-OPD tables (base + teacher rows + teacher-normalized retention).
Reads mopd_domains/outputs/eval/<tag>/metrics.json (domain benches, T1.0 unless '-final-' T0.6)
and mopd/outputs/eval/<tag>/metrics.json (6-bench OOD, 32k, T0.6)."""
import json, os, sys, glob
S = "/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean"
if not os.path.isdir(S): S = "/data/ib-a100-cluster-a-pri-lmalign_942/personal/sean"
DE = S + "/mopd_sean/outputs/eval_domain"; FE = S + "/mopd_sean/outputs/eval_6bench"
T_IN = {"med": 80.7, "law": 78.3, "fin": 74.6, "ifeval": 79.1, "ifbench": 58.7}   # teacher in-domain refs (T1.0 / 6-bench)
def dom(tag):
    f = f"{DE}/{tag}/metrics.json"
    if not os.path.exists(f): return {}
    m = json.load(open(f)); out = {}
    for b, r in m.items():
        if not isinstance(r, dict): continue
        acc = r.get("macro_f1") if b == "pubmedqa" else r.get("accuracy")
        if acc is None: continue
        out[b] = (round(acc * 100, 1), round(r.get("truncated_frac", 0) * 100, 1))
    return out
def full(tag):
    f = f"{FE}/{tag}/metrics.json"
    if not os.path.exists(f): return {}
    m = json.load(open(f)); b = m["benchmarks"]
    out = {k: (round(v["score"], 1), None) for k, v in b.items()}
    out["math"] = (round(m["domain_avg"]["math"], 1), None); out["code"] = (round(m["domain_avg"]["code"], 1), None)
    return out
def cell(d, k):
    if k not in d: return "–"
    v, tr = d[k]; s = f"{v:.1f}"
    if tr is not None and tr >= 5: s += f" (trunc {tr:.0f}%)"
    return s
BASE = {"medqa": 69.8, "casehold": 63.2, "finqa": 58.3, "ifeval": 51.0, "ifbench": 27.7}   # base student OT3 (T1.0 / 6-bench)
TEACH = {"medqa": 80.7, "casehold": 78.3, "finqa": 74.6, "ifeval": 79.1, "ifbench": 58.7}
def norm(d, k):  # s~_d = (s_d - s_d^s) / (s_d^t - s_d^s)
    return (d[k][0] - BASE[k]) / (TEACH[k] - BASE[k])
def retention(d):
    """mean over domains of s~_d; IF domain = mean of IFEval and IFBench s~"""
    parts = []
    for k in ("medqa", "casehold", "finqa"):
        if k in d: parts.append(norm(d, k))
    if "ifeval" in d and "ifbench" in d: parts.append((norm(d, "ifeval") + norm(d, "ifbench")) / 2)
    if not parts: return "–"
    return f"{100*sum(parts)/len(parts):.0f}%" + ("" if len(parts) == 4 else f" ({len(parts)}dom)")
COLS = ["medqa", "casehold", "finqa", "ifeval", "ifbench", "math", "code"]
HDR = "| model | MedQA | CaseHOLD | FinQA | IFEval | IFBench | Math (AIME24-26) | Code (LCB v6) | s̃ (base→teacher 진행률, domain 평균) |"
SEP = "|---|---|---|---|---|---|---|---|---|"
def row(name, d): return f"| {name} | " + " | ".join(cell(d, c) for c in COLS) + f" | {retention(d)} |"
def merged(*ds):
    o = {}
    for d in ds: o.update(d)
    return o
# ---- reference rows
base = merged(dom("base-ot3-alldom-T1"), full("sftot3-4b-final-32k"))
teachers = {
 "teacher-law (rl2-law-distill-ck200)": merged(dom("rl2-law-distill-ck200-casehold-T1"), dom("teacher-law-dom3-T1"), full("full-law-rl-ck200-32k")),
 "teacher-fin (rl-fin-ck200)": merged(dom("rl-fin-ck200-T1-32k"), dom("teacher-fin-dom3-T1"), full("full-fin-rl-ck200-32k")),
 "teacher-if (rl-if-ck200)": merged(dom("teacher-if-dom3-T1"), full("rl-if-ck200-32k-fixedrope")),
 "teacher-med (SFT C6-4ep)": merged(dom("sftd-med-c6_4ep-medqa-T1"), dom("teacher-med-dom3-T1"), full("full-med-c6_4ep-32k")),
 "teacher-law-RLonly (rl-law-ck200, no SFT)": merged(dom("rl-law-ck200-T1-26k"), dom("teacher-rllaw-dom3-T1"), full("full-law-rlonly-ck200-32k")),
}
RUNS = [("mopd-3dom-rllaw-96", "3-dom PG 96/upd: RL-ONLY law teacher + fin + if"), ("mopd-4dom-128-merged", "4-dom PG, 128/upd, init = MERGED student"), ("mopd-4dom-128-merged-sftp", "EVOL-FREE: 4-dom PG 128/upd, MERGED init"), ("mopd-4dom-128-sftp", "EVOL-FREE: 4-dom PG 128/upd, OT3 init"), ("mopd-medlaw-64-sftp", "EVOL-FREE: 2-dom med+law PG 64/upd"), ("mopd-4dom-32", "4-dom PG, 32/upd (8/dom)"), ("mopd-4dom-128", "4-dom PG, 128/upd (32/dom)"),
        ("mopd-4dom-32-top64", "4-dom top-64, 32/upd"), ("mopd-4dom-128-top64", "4-dom top-64, 128/upd"),
        ("mopd-finif-64", "2-dom fin+if PG, 64/upd"), ("mopd-medlaw-64", "2-dom med+law PG, 64/upd")]
only = sys.argv[1:]  # optional run filter
print("## MOPD (student Qwen3-4B-OT3; domain benches T1.0 (final = T0.6), OOD 32k T0.6)\n")
print(HDR); print(SEP); print(row("base OT3", base))
for n, d in teachers.items(): print(row(n, d))
print(row("merged-4teachers-uniform (init, no training)", merged(dom("merged-4t-uniform-dom3-T1"), full("full-merged-4t-uniform-32k"))))
for run, desc in RUNS:
    if only and run not in only: continue
    print(f"| **{run}** ({desc}) | | | | | | | | |")
    for ck in [25, 50, 75, 100, 125, 150, 175, 200]:
        d = merged(dom(f"opd-{run}-ck{ck}-dom3-T1"), full(f"full-opd-{run}-ck{ck}-32k"))
        if d: print(row(f"{run} ck{ck}", d))
    for ck in [25, 50, 75, 100, 125, 150, 175, 200]:   # extra T0.6 ckpt evals (clean replacement finals)
        d = dom(f"opd-{run}-ck{ck}-dom3-T06")
        if d: print(row(f"{run} ck{ck} (T0.6)", d))
    d = merged(dom(f"opd-{run}-final-dom3"), full(f"full-opd-{run}-final-32k"))
    if d: print(row(f"{run} final = ck200 (T0.6)", d))
# ---- medical OPD table
print("\n## Medical OPD from the SFT teacher (MedQA T1.0 per ckpt; final = T0.6; OOD 32k)\n")
print("| model | MedQA | MedXpertQA | PubMedQA F1 | IFEval | IFBench | Math | Code | s̃ MedQA (base→teacher 진행률) |"); print("|---|---|---|---|---|---|---|---|---|")
def mrow(name, d):
    r = f"{100*norm(d,'medqa'):.0f}%" if "medqa" in d else "–"
    return f"| {name} | " + " | ".join(cell(d, c) for c in ["medqa", "medxpertqa", "pubmedqa", "ifeval", "ifbench", "math", "code"]) + f" | {r} |"
print(mrow("base OT3", merged(dom("base-ot3-alldom-T1"), full("sftot3-4b-final-32k"))))
print(mrow("teacher-med (SFT C6-4ep)", merged(dom("sftd-med-c6_4ep-medqa-T1"), dom("sftd-med-c6_4ep-xpert-pubmed"), full("full-med-c6_4ep-32k"))))
MED_RUNS = [("opd-med-top64", "opd-med-top64"), ("opd-med-top16", "opd-med-top16"), ("opd-med-pg", "opd-med-pg"),
            ("opd-opd_med_pg-sftp", "opd-med-pg EVOL-FREE (pool = SFT teacher prompts only)"),
            ("opd-med-pg-sftw150", "branch v1: SFT warm-up ck150 (6.4k x 1 pass) -> OPD pg"),
            ("opd-med-pg-sftp-sftw150", "branch v2 EVOL-FREE: warm-up v2 ck150 -> OPD pg")]
for pre, label in MED_RUNS:
    print(f"| **{label}** | | | | | | | | |")
    for ck in [25, 50, 75, 100, 125, 150, 175, 200]:
        d = dom(f"{pre}-ck{ck}-medqa-T1")
        if d: print(mrow(f"{label.split(' ')[0]} ck{ck}", d))
    d = merged(dom(f"{pre}-final-medqa"), dom(f"{pre}-final-xpert-pubmed"), full(f"full-{pre}-final-32k"))
    if d: print(mrow(f"{label.split(' ')[0]} final (T0.6)", d))
