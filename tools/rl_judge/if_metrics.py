import json,glob,os,re
S="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean/outputs/eval_6bench/"
def ck(t):
    m=re.search(r"ck(\d+)",t); return int(m.group(1)) if m else 0
tags=["sftot3-4b-final-32k"]+sorted([os.path.basename(p) for p in glob.glob(S+"rl-if-*")],key=ck)
for t in tags:
    p=f"{S}{t}/metrics.json"
    if not os.path.exists(p): print(t,"(no metrics)"); continue
    m=json.load(open(p)); b=m["benchmarks"]; ie=b.get("ifeval",{}); ib=b.get("ifbench",{})
    a=[b[x]["score"] for x in ["aime24","aime25","aime26"] if x in b]
    print(f"{t:30s} preset={m.get('preset')}")
    print(f"   IFEval  prompt strict {ie.get('prompt_level_strict_acc',0):6.2f} loose {ie.get('prompt_level_loose_acc',0):6.2f} | inst strict {ie.get('inst_level_strict_acc',0):6.2f} loose {ie.get('inst_level_loose_acc',0):6.2f} | tok {ie.get('avg_tokens',0):6.0f} trunc {ie.get('truncated_frac',0):.3f} finished_think {ie.get('finished_thinking_frac',0):.3f}")
    print(f"   IFBench prompt strict {ib.get('prompt_level_strict_acc',0):6.2f} loose {ib.get('prompt_level_loose_acc',0):6.2f} | inst strict {ib.get('inst_level_strict_acc',0):6.2f} loose {ib.get('inst_level_loose_acc',0):6.2f} | tok {ib.get('avg_tokens',0):6.0f} trunc {ib.get('truncated_frac',0):.3f} finished_think {ib.get('finished_thinking_frac',0):.3f}")
    print(f"   if-domain {m['domain_avg'].get('if',0):.2f} | aime24/25/26 {' '.join(f'{x:.2f}' for x in a)} math {m['domain_avg'].get('math',0):.2f} | lcb {b.get('lcb_v6',{}).get('score',0):.2f} (no_code {b.get('lcb_v6',{}).get('no_code_block')}) code {m['domain_avg'].get('code',0):.2f}")
