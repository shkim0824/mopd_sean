import json,glob,re,collections,statistics,os
S='/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean'; D=S+'/mopd_sean/data/rl3dom/'
sweep={str(r['uid']):r for r in (json.loads(l) for l in open(D+'nano_sweep_problems.jsonl'))}
res={}
for d in ['nano_sweep','nano_sweep_r2']:
    for f in glob.glob(f'{S}/mopd_sean/tmp/{d}/shard*.jsonl'):
        if '.raw' in f: continue
        for l in open(f): o=json.loads(l); res[str(o['uid'])]=o
def p4(o): return float(o['passed'])/float(o['n'])
def p30(o):
    try: return float(o['pass_rate_30b'])
    except: return None
def tier(p):
    if p==0: return '0/8'
    if p<=0.25: return '1-2/8'
    if p<=0.625: return '3-5/8'
    if p<=0.8: return '6/8'
    return '7-8/8'
if False: print('=== sweep main tier by source x 4B tier (n, mean 30B pass, share 30B>=3/8) ===')
tab=collections.defaultdict(list)
for u,o in res.items():
    if str(o.get('zero_tier_probe'))!='False': continue
    src='dapo' if 'dapo' in o['dataset'] else 'skywork'
    tab[(src,tier(p4(o)))].append(p30(o))
for src in ['dapo','skywork']:
    tot=sum(len(v) for (s,t),v in tab.items() if s==src)
    for t in ['0/8','1-2/8','3-5/8','6/8','7-8/8']:
        v=[x for x in tab[(src,t)] if x is not None]; n=len(tab[(src,t)])
        print(f'  {src:8s} {t:7s} n={n:5d} ({100*n/tot:4.1f}%)  30B mean {statistics.mean(v) if v else float("nan"):.2f}  30B>=3/8 {100*sum(1 for x in v if x>=0.375)/max(1,len(v)):3.0f}%  30B>=4/8 {100*sum(1 for x in v if x>=0.5)/max(1,len(v)):3.0f}%')
# zero-tier probe result recap
zp=[o for o in res.values() if str(o.get('zero_tier_probe'))=='True']
print(f'zero-tier probe (Nano-30B pass=0 rows): n={len(zp)}, 4B 0/8: {sum(1 for o in zp if p4(o)==0)}, 4B 1-2/8: {sum(1 for o in zp if 0<p4(o)<=0.25)}, >0.25: {sum(1 for o in zp if p4(o)>0.25)}')
# answer-type of the 0/8 tier (is it integer/verifiable?)
def isint(s): return re.fullmatch(r'\s*-?\d+\s*',str(s)) is not None
z=[o for o in res.values() if str(o.get('zero_tier_probe'))=='False' and p4(o)==0]
print(f'0/8 tier: n={len(z)}, integer gold {100*sum(1 for o in z if isint(o["expected_answer"]))/len(z):.0f}%, mean gen tokens {statistics.mean(float(o["gen_tokens_mean"]) for o in z):.0f}')
# contamination vs AIME eval prompts
def norm(s): return re.sub(r'[^a-z0-9]','',s.lower())
aime=[]
for b in ['aime24','aime25','aime26']:
    f=f'{S}/mopd_sean/outputs/eval_6bench/sftot3-4b-final-32k/{b}.gen.jsonl'
    for l in open(f):
        o=json.loads(l); m=o['messages']
        q=m[0]['content'] if isinstance(m,list) else None
        if q: aime.append((b,o.get('id',''),q))
print('aime prompts loaded:',len(aime))
def grams(s,n=12):
    t=norm(s); return {t[i:i+n] for i in range(0,max(1,len(t)-n),3)}
pool_grams={}
for u,r in sweep.items(): pool_grams[u]=grams(r['question'])
hits=[]
for b,i,q in aime:
    g=grams(q)
    if not g: continue
    best=(0,None)
    for u,pg in pool_grams.items():
        ov=len(g&pg)/len(g)
        if ov>best[0]: best=(ov,u)
    if best[0]>=0.5: hits.append((b,i,round(best[0],2),best[1],res.get(best[1],{}).get('dataset','?'),sweep[best[1]]['question'][:120].replace('\n',' ')))
print('AIME prompts with >=50% 12-gram overlap in the sweep pool:',len(hits))
for h in hits[:30]: print('  ',h)
