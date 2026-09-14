#!/usr/bin/env python3
# Diagnose GRPO pool difficulty from NeMo-RL rollout dumps.
# Per prompt-group (G=8): count successes (reward>0.5); classify all-correct / all-wrong / mixed.
import json, os, sys, glob, re
from collections import defaultdict

domain = sys.argv[1]
base = f"/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean/logs/nemo_{domain}"

# merge ALL exp_* dirs: map step -> file, prefer the higher exp_NNN for duplicate steps
allf = {}
for exp in sorted(glob.glob(os.path.join(base, "exp_*"))):
    for f in glob.glob(os.path.join(exp, "train_data_step*.jsonl")):
        st = int(re.search(r"step(\d+)\.jsonl", f).group(1))
        allf[st] = f  # later (higher exp_) overwrites
print(f"[{domain}] merged {len(allf)} steps across exp dirs, range {min(allf)}..{max(allf)}")
steps_avail = sorted(allf)
# sample ~8 steps evenly
if len(steps_avail) > 8:
    idxs = [round(i*(len(steps_avail)-1)/7) for i in range(8)]
    sample = [steps_avail[i] for i in sorted(set(idxs))]
else:
    sample = steps_avail

print(f"{'step':>5} {'meanR':>7} {'succ%':>7} {'all8%':>7} {'zero%':>7} {'mixed%':>7} {'genlen':>7}")
agg = defaultdict(list)
for st in sample:
    G = 8
    rows = []
    genlens = []
    with open(allf[st]) as fh:
        for line in fh:
            d = json.loads(line)
            r = d["rewards"][0] if isinstance(d["rewards"], list) else d["rewards"]
            il = d["input_lengths"][0]
            rows.append(r)
            try:
                tl = len(d["token_ids"][0]); genlens.append(tl-il)
            except Exception: pass
    # rollouts are laid out prompt-major: consecutive blocks of G gens = one prompt
    groups = [rows[i:i+G] for i in range(0, len(rows), G)]
    n = len(groups); all8=zero=mixed=0; rs=[]
    for rr in groups:
        succ = sum(1 for x in rr if x > 0.5); rs += rr
        if succ == len(rr): all8 += 1
        elif succ == 0: zero += 1
        else: mixed += 1
    meanR = sum(rs)/len(rs) if rs else 0
    succpct = 100*sum(1 for x in rs if x>0.5)/len(rs) if rs else 0
    gl = sorted(genlens); glp90 = gl[int(0.9*len(gl))] if gl else 0
    print(f"{st:>5} {meanR:>7.3f} {succpct:>6.1f} {100*all8/n:>6.1f} {100*zero/n:>6.1f} {100*mixed/n:>6.1f} {glp90:>7}")
    agg['all8'].append(100*all8/n); agg['zero'].append(100*zero/n); agg['mixed'].append(100*mixed/n); agg['meanR'].append(meanR)
def m(k): return sum(agg[k])/len(agg[k])
print(f"[{domain}] AVG  meanR={m('meanR'):.3f}  all-correct(easy)={m('all8'):.1f}%  all-wrong(hard)={m('zero'):.1f}%  MIXED(learnable)={m('mixed'):.1f}%")
