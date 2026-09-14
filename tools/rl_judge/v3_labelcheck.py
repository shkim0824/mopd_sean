import json,gzip,glob,re,collections,statistics,sys,time
S='/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean'; D=S+'/mopd_sean/data/rl3dom/'
v3=[json.loads(l) for l in open(D+'nano_v3_train.jsonl')]
sweep=[json.loads(l) for l in open(D+'nano_sweep_problems.jsonl')]
def nk(s): return re.sub(r'\s+',' ',s).strip()[:300]
def nk2(s): return re.sub(r'[^a-z0-9]','',s.lower())[:150]
q2uid={}; 
for r in sweep: q2uid.setdefault(nk(r['question']),str(r['uid'])); q2uid.setdefault('#'+nk2(r['question']),str(r['uid']))
uid2v3={}
for r in v3:
    u=q2uid.get(nk(r['input'])) or q2uid.get('#'+nk2(r['input']))
    if u: uid2v3[u]=r
print('mapped',len(uid2v3),'/',len(v3),flush=True)
res={}
for d in ['nano_sweep','nano_sweep_r2']:
    for f in glob.glob(f'{S}/mopd_sean/tmp/{d}/shard*.jsonl'):
        if '.raw' in f: continue
        for l in open(f):
            o=json.loads(l); res[str(o['uid'])]=o
# gen tokens vs pass (v3 + all main tier)
print('\n[v3] pass bin vs mean gen tokens:')
for lo,hi in [(0,0.13),(0.13,0.26),(0.26,0.51),(0.51,0.76),(0.76,0.81)]:
    g=[float(res[u]['gen_tokens_mean']) for u,r in uid2v3.items() if u in res and lo<float(r['pass8_4b'])<=hi]
    if g: print(f'   ({lo:.2f},{hi:.2f}]: n={len(g):5d} gen_tokens mean {statistics.mean(g):6.0f} median {statistics.median(g):6.0f} share>=24k {100*sum(1 for x in g if x>=24000)/len(g):.0f}%')
print('[sweep main tier] pass bin vs gen tokens:')
allg=[(float(o['gen_tokens_mean']), float(o['passed'])/float(o['n'])) for o in res.values() if str(o.get('zero_tier_probe','False'))=='False']
for lo,hi in [(-0.01,0.0),(0,0.13),(0.13,0.26),(0.26,0.51),(0.51,0.76),(0.76,0.99),(0.99,1.0)]:
    g=[x for x,p in allg if lo<p<=hi]
    if g: print(f'   ({lo:.2f},{hi:.2f}]: n={len(g):5d} gen_tokens mean {statistics.mean(g):6.0f}')
# flags without raw
def flags(q):
    return {'chinese':bool(re.search(r'[\u4e00-\u9fff]',q)),
            'multipart':bool(re.search(r'\(1\)[\s\S]*\(2\)|\b[Aa]\)[\s\S]*\b[Bb]\)|\(i\)[\s\S]*\(ii\)|\(a\)[\s\S]*\(b\)',q)),
            'mchoice':bool(re.search(r'\(A\)[\s\S]*\(B\)[\s\S]*\(C\)|\bA\)[\s\S]*\bB\)[\s\S]*\bC\)|\nA\.[\s\S]*\nB\.[\s\S]*\nC\.|\(A\)\s*\$',q)),
            'figure':bool(re.search(r'\b(figure|diagram|shown|graph paper|picture|as shown)\b',q,re.I)),
            'proof':bool(re.search(r'\b(prove|show that)\b',q,re.I))}
print('\n[v3] pass histogram by source:')
for ds in sorted(set(r['dataset'] for r in v3)):
    c=collections.Counter(float(r['pass8_4b']) for r in v3 if r['dataset']==ds); n=sum(c.values())
    print('   ',ds.split('_')[-1] if 'dapo' not in ds else 'dapo', ' '.join(f'{k:.3f}:{100*c[k]/n:.0f}%' for k in sorted(c)))
print('\n[v3] prompt flags (share, mean pass):')
fl=collections.defaultdict(list)
for r in v3:
    f=flags(r['input'])
    for k,v in f.items():
        if v: fl[k].append(float(r['pass8_4b']))
for k,v in fl.items(): print(f'   {k:10s} n={len(v):5d} ({100*len(v)/len(v3):4.1f}%) mean pass {statistics.mean(v):.3f}')
# raw pass
def last_boxed(t):
    i=t.rfind('\\boxed')
    if i<0: return None
    j=t.find('{',i)
    if j<0 or j-i>8: return None
    depth=0;k=j
    while k<len(t):
        c=t[k]
        if c=='{': depth+=1
        elif c=='}':
            depth-=1
            if depth==0: return t[j+1:k]
        k+=1
    return None
def norm(a):
    if a is None: return None
    a=a.strip().strip('$').strip()
    a=re.sub(r'\\[dt]frac',r'\\frac',a)
    for t in ['\\left','\\right','\\,','\\!','\;','\\ ',' ','\\displaystyle','\\mathrm','\\textbf','\\%']: a=a.replace(t,'')
    a=re.sub(r'\^\{?\\circ\}?','',a); a=re.sub(r'\\text\{([^}]*)\}',r'\1',a)
    a=re.sub(r'^[a-zA-Z]\s*=\s*','',a)  # 'k = n-1' -> 'n-1'
    a=a.rstrip('.').strip('()') if re.fullmatch(r'\(-?[\d\\][^,]*\)',a) else a.rstrip('.')
    return a
want=set(uid2v3)
per=collections.defaultdict(lambda:{'ans':[],'nobox':0,'len':[],'think_closed':0})
t0=time.time(); nfile=0
for d in ['nano_sweep','nano_sweep_r2']:
    for f in sorted(glob.glob(f'{S}/mopd_sean/tmp/{d}/shard*.raw.jsonl.gz')):
        nfile+=1
        try:
          with gzip.open(f,'rt') as fh:
            for l in fh:
                try: o=json.loads(l)
                except: continue
                u=str(o['uid'])
                if u not in want: continue
                t=o['text']; b=last_boxed(t); p=per[u]
                p['len'].append(len(t)); p['think_closed']+= ('</think>' in t)
                if b is None: p['nobox']+=1
                else: p['ans'].append(norm(b))
        except (EOFError,OSError) as e: print('  truncated file',f,e,flush=True)
        print(f'  {nfile} files, {len(per)} uids, {time.time()-t0:.0f}s',flush=True)
out={}
summ=collections.defaultdict(lambda:collections.Counter())
examples=[]
for u,p in per.items():
    r=uid2v3[u]; pr=float(r['pass8_4b']); g=norm(r['output'])
    c=collections.Counter(a for a in p['ans'] if a)
    maj,mc=(c.most_common(1)[0] if c else (None,0))
    n=len(p['ans'])+p['nobox']
    b='<=0.25' if pr<=0.25 else ('0.375-0.5' if pr<=0.5 else '0.625-0.75')
    s=summ[b]; s['problems']+=1; s['samples']+=n; s['nobox']+=p['nobox']; s['unclosed_think']+= n-p['think_closed']
    s['wrong_boxed']+= sum(1 for a in p['ans'] if a!=g)
    if mc>=5 and maj!=g: s['consistent_mismatch(maj>=5!=gold)']+=1; examples.append((pr,r['dataset'].split('_')[-1],g,maj,mc,p['nobox'],re.sub(r'\s+',' ',r['input'])[:230]))
    if mc>=5 and maj==g and pr<=0.25: s['maj==gold_but_lowpass(grader?)']+=1
    if p['nobox']>=6: s['>=6/8 no boxed (budget)']+=1
    out[u]={'pass':pr,'gold':g,'maj':maj,'majc':mc,'nobox':p['nobox'],'ans':p['ans'],'lens':p['len']}
print('\n[v3 raw check] per pass bin:')
for b in ['<=0.25','0.375-0.5','0.625-0.75']:
    s=summ[b]; N=s['problems']; ns=s['samples']
    print(f'  bin {b}: problems {N}, samples {ns}; no-boxed {100*s["nobox"]/ns:.1f}% of samples, unclosed-think {100*s["unclosed_think"]/ns:.1f}%, wrong-boxed {100*s["wrong_boxed"]/ns:.1f}%; '
          f'problems with >=6/8 no-boxed: {s[">=6/8 no boxed (budget)"]} ({100*s[">=6/8 no boxed (budget)"]/N:.1f}%); consistent-mismatch: {s["consistent_mismatch(maj>=5!=gold)"]} ({100*s["consistent_mismatch(maj>=5!=gold)"]/N:.1f}%); maj==gold-but-lowpass: {s["maj==gold_but_lowpass(grader?)"]}')
print('\n[examples: majority answer (>=5/8) != gold]  (pass, src, gold, majority, count, nobox, question)')
examples.sort()
for e in examples[:60]: print('  ',e)
json.dump(out,open(f'{S}/mopd_sean/tmp/v3_labelcheck.json','w'))
print('DONE')
