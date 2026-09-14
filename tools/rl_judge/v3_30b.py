import json,glob,re,collections,statistics
S='/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean'; D=S+'/mopd_sean/data/rl3dom/'
v3=[json.loads(l) for l in open(D+'nano_v3_train.jsonl')]
sweep=[json.loads(l) for l in open(D+'nano_sweep_problems.jsonl')]
def nk(s): return re.sub(r'\s+',' ',s).strip()[:300]
q2uid={}
for r in sweep: q2uid.setdefault(nk(r['question']),str(r['uid']))
res={}
for d in ['nano_sweep','nano_sweep_r2']:
    for f in glob.glob(f'{S}/mopd_sean/tmp/{d}/shard*.jsonl'):
        if '.raw' in f: continue
        for l in open(f): o=json.loads(l); res[str(o['uid'])]=o
v3u={q2uid.get(nk(r['input'])) for r in v3}; v3u.discard(None)
def p30(o):
    try: return float(o['pass_rate_30b'])
    except: return None
def p4(o): return float(o['passed'])/float(o['n'])
groups={'v3 train (0<p4b<=0.8)':[o for u,o in res.items() if u in v3u],
        'excluded easy (p4b>0.8)':[o for u,o in res.items() if u not in v3u and p4(o)>0.8 and str(o.get('zero_tier_probe'))=='False'],
        'excluded zero (p4b=0)':[o for u,o in res.items() if u not in v3u and p4(o)==0 and str(o.get('zero_tier_probe'))=='False']}
for name,g in groups.items():
    x=[p30(o) for o in g if p30(o) is not None]
    c=collections.Counter(round(v*8) for v in x)
    print(f'{name:28s} n={len(g):5d} mean pass_30B={statistics.mean(x):.3f}  hist(k/8 for 30B): '+' '.join(f'{k}:{100*c[k]/len(x):.0f}%' for k in sorted(c)))
# within v3: by 4B pass bin, the 30B pass
print('\nv3 by 4B pass bin -> 30B mean pass:')
for lo,hi in [(0,0.13),(0.13,0.26),(0.26,0.51),(0.51,0.76),(0.76,0.81)]:
    x=[p30(o) for o in groups['v3 train (0<p4b<=0.8)'] if lo<p4(o)<=hi and p30(o) is not None]
    print(f'   4B pass ({lo:.2f},{hi:.2f}]: n={len(x):5d} 30B mean {statistics.mean(x):.3f}; share 30B<=0.25: {100*sum(1 for v in x if v<=0.25)/len(x):.0f}%')
# hard-for-both: 4B<=0.25 and 30B<=0.25
both=[o for o in groups['v3 train (0<p4b<=0.8)'] if p4(o)<=0.25 and p30(o) is not None and p30(o)<=0.25]
print(f'\nhard for both (4B<=2/8 & 30B<=2/8): {len(both)} of v3')
