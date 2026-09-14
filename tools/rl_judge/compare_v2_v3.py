import json, re, random, collections, statistics, glob, os
S='/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean'
D=S+'/mopd_sean/data/rl3dom/'
def load(p): return [json.loads(l) for l in open(p)]
v2=load(D+'math_train.jsonl'); v3=load(D+'nano_v3_train.jsonl'); band=load(D+'math_train_band.jsonl'); sweep=load(D+'nano_sweep_problems.jsonl')
print('sweep keys:', list(sweep[0].keys()))
print('rows v2',len(v2),'v3',len(v3),'v2band',len(band),'sweep',len(sweep))

def strip_cmds(s): return re.sub(r'\\[a-zA-Z]+','',s)
def norm_ans(a):
    a=str(a).strip().strip('$').strip()
    a=re.sub(r'\\[dt]frac',r'\\frac',a)
    for t in ['\\left','\\right','\\,','\\!','\;','\\ ',' ','\\displaystyle']: a=a.replace(t,'')
    a=re.sub(r'\^\{?\\circ\}?','',a)
    m=re.fullmatch(r'\\boxed\{(.*)\}',a)
    if m: a=m.group(1)
    a=re.sub(r'\\text\{([^}]*)\}',r'\1',a)
    return a.rstrip('.')
def atype(a):
    n=norm_ans(a)
    if re.fullmatch(r'-?\d+',n): return 'integer'
    if re.fullmatch(r'-?\d*\.\d+',n): return 'decimal'
    if re.fullmatch(r'-?\\frac\{-?\d+\}\{\d+\}|-?\d+/\d+',n): return 'fraction'
    letters=re.findall(r'[a-zA-Z]',strip_cmds(n))
    if re.search(r'[a-zA-Z]{4,}',strip_cmds(n)): return 'words/text'
    if ',' in n or '\\{' in n or ';' in n: return 'multi/set/tuple'
    if letters: return 'symbolic(vars)'
    return 'closed-numeric(sqrt/pi/pow)'
def ans_stats(rows,name):
    c=collections.Counter(atype(r['output']) for r in rows)
    N=len(rows)
    print(f'\n[{name}] answer types (N={N}):')
    for k,v in c.most_common(): print(f'   {k:28s} {v:6d} {100*v/N:5.1f}%')
    ints=[int(norm_ans(r['output'])) for r in rows if atype(r['output'])=='integer']
    if ints:
        inrange=sum(1 for x in ints if 0<=x<=999)
        print(f'   integers in [0,999]: {inrange}/{len(ints)} ({100*inrange/len(ints):.1f}%) ; median |int| = {statistics.median(abs(x) for x in ints)}')
    alen=[len(norm_ans(r['output'])) for r in rows]
    plen=[len(r['input']) for r in rows]
    pw=[len(r['input'].split()) for r in rows]
    print(f'   answer len chars: median {statistics.median(alen)}, mean {statistics.mean(alen):.1f}')
    print(f'   prompt len chars: median {statistics.median(plen)}, mean {statistics.mean(plen):.0f}, p90 {sorted(plen)[int(0.9*N)]} ; words median {statistics.median(pw)}')
    kw={'prove/show that':r'\b(prove|show that)\b','find all/determine all':r'\b(find all|determine all)\b','maximum/minimum':r'\b(max|min)(imum|imal|imize)?\b','probability':r'\bprobabilit','integer(s) word':r'\binteger','function(al)':r'\bfunction','how many':r'\bhow many\b','sequence':r'\bsequence','triangle/circle (geom)':r'\b(triangle|circle|quadrilateral|tangent)\b','polynomial':r'\bpolynomial','remainder/mod':r'\b(remainder|modulo|mod)\b','physics/chem units':r'\b(cm|kg|joule|velocity|molar|mol\b|gram|meter|newton|volt)'}
    print('   prompt keyword shares:')
    for k,p in kw.items():
        n=sum(1 for r in rows if re.search(p,r['input'],re.I)); print(f'      {k:24s} {100*n/N:5.1f}%')
ans_stats(v2,'v2 Nemotron-RL-Math-v2 train (all)')
ans_stats(band,'v2 band 0.2-0.8 (532, in-run)')
ans_stats(v3,'v3 nano 0<p<=0.8 train (all)')
# v3 by source
for ds in sorted(set(r['dataset'] for r in v3)):
    sub=[r for r in v3 if r['dataset']==ds]; ans_stats(sub,f'v3 source={ds}')
# v3 pass histogram
ps=[float(r['pass8_4b']) for r in v3]
print('\n[v3] pass8 histogram:'); c=collections.Counter(ps)
for k in sorted(c): print(f'   {k:.3f}: {c[k]:5d} ({100*c[k]/len(ps):.1f}%)')
print(f'   mean {statistics.mean(ps):.3f}')
for ds in sorted(set(r['dataset'] for r in v3)):
    sub=[float(r['pass8_4b']) for r in v3 if r['dataset']==ds]; print(f'   {ds}: n={len(sub)} mean {statistics.mean(sub):.3f}')
print('\n[v3] pass by answer type:')
bt=collections.defaultdict(list)
for r in v3: bt[atype(r['output'])].append(float(r['pass8_4b']))
for k,v in sorted(bt.items(),key=lambda x:-len(x[1])): print(f'   {k:28s} n={len(v):5d} mean pass {statistics.mean(v):.3f}  share<=0.25: {100*sum(1 for x in v if x<=0.25)/len(v):.0f}%')
# whole sweep (all tiers) type vs pass, using sweep shards
uid2row={}
for i,r in enumerate(sweep): uid2row[str(r.get('uid',i))]=r
res={}
for d in ['nano_sweep','nano_sweep_r2']:
    for f in glob.glob(f'{S}/mopd_sean/tmp/{d}/shard*.jsonl'):
        if '.raw' in f: continue
        for l in open(f):
            o=json.loads(l); res[str(o['uid'])]=o
print('\nsweep results loaded:',len(res))
# map v3 rows to sweep uid by input text
key2uid={}
for i,r in enumerate(sweep):
    k=re.sub(r'\s+',' ',r.get('input',r.get('problem','')) or '').strip()[:300]; key2uid.setdefault(k,str(r.get('uid',i)))
hit=0; gt=[]; 
for r in v3:
    k=re.sub(r'\s+',' ',r['input']).strip()[:300]; u=key2uid.get(k)
    if u and u in res:
        hit+=1; o=res[u]; r['_uid']=u; r['_gen']=float(o.get('gen_tokens_mean',0)); r['_p30']=o.get('pass_rate_30b')
print(f'v3 rows matched to sweep: {hit}/{len(v3)}')
print('\n[v3] pass bin vs mean gen tokens (32k budget):')
bins=[(0,0.13),(0.13,0.26),(0.26,0.51),(0.51,0.76),(0.76,0.81)]
for lo,hi in bins:
    sub=[r for r in v3 if '_gen' in r and lo<float(r['pass8_4b'])<=hi]
    if sub: g=[r['_gen'] for r in sub]; print(f'   pass in ({lo:.2f},{hi:.2f}]: n={len(sub):5d} gen_tokens mean {statistics.mean(g):6.0f} median {statistics.median(g):6.0f}  share>=24k: {100*sum(1 for x in g if x>=24000)/len(g):.0f}%')
# sweep-wide: all tiers (incl. >0.8 and 0) gen tokens
allg=[(float(o.get('gen_tokens_mean',0)), float(o['passed'])/float(o['n'])) for o in res.values() if o.get('zero_tier_probe','False')=='False']
print('\n[sweep all main-tier] pass bin vs gen tokens:')
for lo,hi in [(-0.01,0.0),(0,0.13),(0.13,0.26),(0.26,0.51),(0.51,0.76),(0.76,0.99),(0.99,1.0)]:
    g=[x for x,p in allg if lo<p<=hi]
    if g: print(f'   pass in ({lo:.2f},{hi:.2f}]: n={len(g):5d} gen_tokens mean {statistics.mean(g):6.0f}')
# overlap v2 vs v3
def k(s): return re.sub(r'[^a-z0-9]','',s.lower())[:120]
v3k=set(k(r['input']) for r in v3); swk=set(k(r.get('input','')) for r in sweep)
ov=sum(1 for r in v2 if k(r['input']) in v3k); ov2=sum(1 for r in v2 if k(r['input']) in swk)
print(f'\noverlap: v2 prompts also in v3 train: {ov}/{len(v2)} ; v2 prompts in full sweep pool: {ov2}/{len(v2)}')
# samples
random.seed(0)
def show(r,extra=''):
    t=re.sub(r'\s+',' ',r['input']).strip()
    print(f'--- {extra} | gold: {r["output"][:80]!r}'); print('   '+t[:650]+('...' if len(t)>650 else ''))
print('\n\n############ v2 random samples (all) ############')
for r in random.sample(v2,6): show(r,'v2')
print('\n############ v2 band(0.2-0.8) samples ############')
for r in random.sample(band,4): show(r,'v2band')
print('\n############ v3 samples by pass ############')
for lo,hi in [(0,0.13),(0.26,0.51),(0.62,0.76)]:
    sub=[r for r in v3 if lo<float(r['pass8_4b'])<=hi]
    for r in random.sample(sub,4): show(r,f'v3 pass={r["pass8_4b"]} src={r["dataset"].split("_")[-1] if "dapo" not in r["dataset"] else "dapo"} gen={r.get("_gen","?")}')
print('\n############ v3 full-text 2 examples (format check) ############')
for r in v3[:2]: print(repr(r['input'][:900])); print('gold:',repr(r['output']))
print('\n############ v2 full-text 1 example (format check) ############')
print(repr(v2[0]['input'][:900])); print('gold:',repr(v2[0]['output']))
