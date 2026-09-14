import json,glob,re,collections,statistics
S='/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean'; D=S+'/mopd_sean/data/rl3dom/'
v3=[json.loads(l) for l in open(D+'nano_v3_train.jsonl')]
def nk(s): return re.sub(r'\s+',' ',s).strip()[:300]
def nk2(s): return re.sub(r'[^a-z0-9]','',s.lower())[:150]
key2p={}
for r in v3: key2p[nk(r['input'])]=float(r['pass8_4b']); key2p['#'+nk2(r['input'])]=float(r['pass8_4b'])
def strip_prompt(p):
    p=p.strip()
    if p.startswith('<|im_start|>user'): p=p[len('<|im_start|>user'):].lstrip('\n')
    i=p.find('<|im_end|>')
    if i>=0: p=p[:i]
    return p.strip()
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
            if depth==0: return re.sub(r'\s+','',t[j+1:k])
        k+=1
    return None
files=sorted(glob.glob(S+'/mopd_sean/logs/nemo_math_v3/exp_001/train_data_step*.jsonl'),key=lambda f:int(re.search(r'step(\d+)',f).group(1)))
dec=json.JSONDecoder()
groups=collections.defaultdict(lambda:{'r':[],'a':[]})  # (step,promptkey) -> rewards, answers
for f in files:
    step=int(re.search(r'step(\d+)',f).group(1))
    with open(f,buffering=1024*1024*16) as fh:
        for line in fh:
            q=line.find('"',line.find('[['))
            try: prompt,end=dec.raw_decode(line,q)
            except Exception: continue
            # response string follows: , "resp"
            q2=line.find('"',end+1)
            try: resp,_=dec.raw_decode(line,q2)
            except Exception: resp=''
            j=line.find('"rewards": [')
            if j<0: continue
            try: rw=float(line[j+12:line.find(']',j)])
            except Exception: continue
            sp=strip_prompt(prompt); p=key2p.get(nk(sp)); 
            if p is None: p=key2p.get('#'+nk2(sp))
            if p is None: continue
            g=groups[(step,nk(sp))]; g['r'].append(rw); g['a'].append(last_boxed(resp)); g['p']=p
print('groups:',len(groups))
wins=[(1,40),(41,80),(81,110)]
print('window | label bin | groups | all-zero% | all-one% | mixed% | distinct answers/group | no-box/sample% | mean reward')
for lo,hi in wins:
    for name,cond in [('0.125',lambda p:abs(p-0.125)<1e-6),('0.25',lambda p:abs(p-0.25)<1e-6),('0.375-0.5',lambda p:0.3<p<0.55),('0.625-0.75',lambda p:p>0.6)]:
        gs=[g for (s,k),g in groups.items() if lo<=s<=hi and cond(g['p']) and len(g['r'])>=8]
        if not gs: continue
        az=sum(1 for g in gs if sum(g['r'])==0)/len(gs); ao=sum(1 for g in gs if all(x>0.5 for x in g['r']))/len(gs)
        da=statistics.mean(len(set(a for a in g['a'] if a)) for g in gs)
        nb=sum(1 for g in gs for a in g['a'] if a is None)/sum(len(g['a']) for g in gs)
        mr=statistics.mean(x for g in gs for x in g['r'])
        print(f'{lo:3d}-{hi:3d} | {name:9s} | {len(gs):5d} | {100*az:5.1f} | {100*ao:5.1f} | {100*(1-az-ao):5.1f} | {da:4.2f} | {100*nb:4.1f} | {mr:.3f}')
print('DONE')
