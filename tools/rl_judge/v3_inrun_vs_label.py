import json,glob,re,collections,statistics,os,sys
S='/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean'; D=S+'/mopd_sean/data/rl3dom/'
v3=[json.loads(l) for l in open(D+'nano_v3_train.jsonl')]
sweep=[json.loads(l) for l in open(D+'nano_sweep_problems.jsonl')]
def nk(s): return re.sub(r'\s+',' ',s).strip()[:300]
def nk2(s): return re.sub(r'[^a-z0-9]','',s.lower())[:150]
q2uid={}
for r in sweep: q2uid.setdefault(nk(r['question']),str(r['uid'])); q2uid.setdefault('#'+nk2(r['question']),str(r['uid']))
key2row={}
for r in v3:
    u=q2uid.get(nk(r['input'])) or q2uid.get('#'+nk2(r['input'])); r['_uid']=u
    key2row[nk(r['input'])]=r; key2row['#'+nk2(r['input'])]=r
lc=json.load(open(S+'/mopd_sean/tmp/v3_labelcheck.json'))
res={}
for d in ['nano_sweep','nano_sweep_r2']:
    for f in glob.glob(f'{S}/mopd_sean/tmp/{d}/shard*.jsonl'):
        if '.raw' in f: continue
        for l in open(f): o=json.loads(l); res[str(o['uid'])]=o
def isint(s): return s is not None and re.fullmatch(r'-?\d+',s) is not None
def cat(g,m):
    if re.fullmatch(r'[A-E]',m): return 'MC letter boxed'
    if re.fullmatch(r'(yes|no|Yes|No|true|false|True|False)',m): return 'yes/no answer'
    if isint(g) and not re.fullmatch(r'-?[\d.]+',m): return 'gold int, model non-numeric'
    if isint(g) and isint(m) and abs(int(g)-int(m))==1: return 'off-by-one integer'
    if isint(g) and isint(m): return 'integer vs integer (other)'
    return 'expression mismatch (other)'
def category(r):
    u=r.get('_uid'); o=lc.get(u) if u else None
    if o is None: return 'unmapped'
    if o['pass']<=0.25 and o['majc']>=5 and o['maj']!=o['gold']: return 'HARD+mismatch: '+cat(o['gold'],o['maj'])
    if o['pass']<=0.25: return 'HARD, no consistent mismatch'
    return 'mid/easy (pass>0.25)'
# --- stream rollout logs
files=sorted(glob.glob(S+'/mopd_sean/logs/nemo_math_v3/exp_001/train_data_step*.jsonl'),key=lambda f:int(re.search(r'step(\d+)',f).group(1)))
print('rollout files:',len(files),'steps',re.search(r'step(\d+)',files[0]).group(1),'..',re.search(r'step(\d+)',files[-1]).group(1),flush=True)
def strip_prompt(p):
    p=p.strip()
    if p.startswith('<|im_start|>user'): p=p[len('<|im_start|>user'):].lstrip('\n')
    i=p.find('<|im_end|>'); 
    if i>=0: p=p[:i]
    return p.strip()
stats=collections.defaultdict(lambda:[0,0,0,0])  # n, correct, judge_used, judge_granted (judge not logged -> 0)
dec=json.JSONDecoder(); unmatched=0; nlines=0
for fi,f in enumerate(files):
    with open(f,buffering=1024*1024*16) as fh:
        for line in fh:
            nlines+=1
            q=line.find('"',line.find('[['))
            try: prompt,_=dec.raw_decode(line,q)
            except Exception: continue
            j=line.find('"rewards": [')
            if j<0: continue
            j2=line.find(']',j)
            try: rw=float(line[j+12:j2])
            except Exception: continue
            sp=strip_prompt(prompt)
            r=key2row.get(nk(sp)) or key2row.get('#'+nk2(sp))
            if r is None: unmatched+=1; continue
            st=stats[id(r)]; st[0]+=1; st[1]+= 1 if rw>0.5 else 0
            r.setdefault('_st',st)
    if (fi+1)%10==0: print(f'  {fi+1}/{len(files)} files, prompts {len(stats)}',flush=True)
print(f'lines {nlines}, unmatched {unmatched}, prompts seen {len(stats)}',flush=True)
seen=[r for r in v3 if '_st' in r]
tot_n=sum(r['_st'][0] for r in seen); tot_c=sum(r['_st'][1] for r in seen); tot_j=sum(r['_st'][2] for r in seen); tot_jg=sum(r['_st'][3] for r in seen)
print(f'in-run samples {tot_n}: reward mean {tot_c/tot_n:.3f}; judge used on {100*tot_j/tot_n:.1f}% of samples, judge-granted {tot_jg} ({100*tot_jg/max(1,tot_c):.1f}% of all correct)')
print('\n[calibration] sweep label bin -> in-run reward (prompts seen so far):')
for pb in [0.125,0.25,0.375,0.5,0.625,0.75]:
    g=[r for r in seen if abs(float(r['pass8_4b'])-pb)<1e-6]
    if g:
        n=sum(r['_st'][0] for r in g); c=sum(r['_st'][1] for r in g); jg=sum(r['_st'][3] for r in g)
        print(f'   label {pb:.3f}: prompts {len(g):4d}, in-run reward {c/n:.3f} (judge-granted share of correct {100*jg/max(1,c):.0f}%)')
print('\n[by category] sweep pass vs in-run reward:')
bycat=collections.defaultdict(list)
for r in seen: bycat[category(r)].append(r)
for c_,g in sorted(bycat.items(),key=lambda x:-len(x[1])):
    n=sum(r['_st'][0] for r in g); c=sum(r['_st'][1] for r in g); jg=sum(r['_st'][3] for r in g)
    sp=statistics.mean(float(r['pass8_4b']) for r in g)
    p30=[float(res[r['_uid']]['pass_rate_30b']) for r in g if r.get('_uid') in res and res[r['_uid']].get('pass_rate_30b') not in (None,'None','')]
    print(f'   {c_:45s} prompts {len(g):4d}  sweep pass {sp:.3f}  in-run reward {c/n:.3f}  judge-granted/correct {100*jg/max(1,c):3.0f}%  30B pass {statistics.mean(p30) if p30 else float("nan"):.2f}')
# all v3 (not only seen): 30B pass by category
print('\n[all v3] 30B pass distribution by category:')
allcat=collections.defaultdict(list)
for r in v3: allcat[category(r)].append(r)
for c_,g in sorted(allcat.items(),key=lambda x:-len(x[1])):
    p30=[float(res[r['_uid']]['pass_rate_30b']) for r in g if r.get('_uid') in res and res[r['_uid']].get('pass_rate_30b') not in (None,'None','')]
    if p30:
        hi=sum(1 for x in p30 if x>=0.5); lo=sum(1 for x in p30 if x<=0.125)
        print(f'   {c_:45s} n={len(g):4d}  30B mean {statistics.mean(p30):.2f}  30B>=0.5: {100*hi/len(p30):3.0f}%  30B<=1/8: {100*lo/len(p30):3.0f}%')
