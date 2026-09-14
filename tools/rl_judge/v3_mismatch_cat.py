import json,re,collections
S='/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean'; D=S+'/mopd_sean/data/rl3dom/'
v3=[json.loads(l) for l in open(D+'nano_v3_train.jsonl')]
sweep=[json.loads(l) for l in open(D+'nano_sweep_problems.jsonl')]
def nk(s): return re.sub(r'\s+',' ',s).strip()[:300]
def nk2(s): return re.sub(r'[^a-z0-9]','',s.lower())[:150]
q2uid={}
for r in sweep: q2uid.setdefault(nk(r['question']),str(r['uid'])); q2uid.setdefault('#'+nk2(r['question']),str(r['uid']))
uid2v3={}
for r in v3:
    u=q2uid.get(nk(r['input'])) or q2uid.get('#'+nk2(r['input']))
    if u: uid2v3[u]=r
lc=json.load(open(S+'/mopd_sean/tmp/v3_labelcheck.json'))
def isint(s): return s is not None and re.fullmatch(r'-?\d+',s) is not None
def cat(g,m,q):
    if re.fullmatch(r'[A-E]',m): return 'MC letter boxed'
    if re.fullmatch(r'(yes|no|Yes|No|true|false|True|False)',m): return 'yes/no answer'
    if isint(g) and not re.fullmatch(r'-?[\d.]+',m): return 'gold int, model non-numeric (untransformed form?)'
    if isint(g) and isint(m) and abs(int(g)-int(m))==1: return 'off-by-one integer'
    if isint(g) and isint(m): return 'integer vs integer (other)'
    return 'other'
tot=collections.Counter(); bysrc=collections.Counter(); hard=collections.Counter(); ex=collections.defaultdict(list)
transform=re.compile(r'原始|请给出|m\s*\+\s*n|标准答案|答案的格式|格式为|provide the (sum|value)|give the (sum|value)|format of|in the form',re.I)
for u,o in lc.items():
    r=uid2v3[u]; src='dapo' if 'dapo' in r['dataset'] else 'skywork'
    if o['pass']<=0.25: hard[src]+=1
    if o['majc']>=5 and o['maj']!=o['gold'] and o['pass']<=0.25:
        c=cat(o['gold'],o['maj'],r['input']); tot[c]+=1; bysrc[(src,c)]+=1
        fl=('CN ' if re.search(r'[一-鿿]',r['input']) else '')+('XFORM ' if transform.search(r['input']) else '')
        ex[(src,c)].append((o['pass'],o['gold'],o['maj'],o['majc'],fl,re.sub(r'\s+',' ',r['input'])[:260]))
print('hard-bin problems by source:',dict(hard))
print('\nconsistent-mismatch categories (hard bin, maj>=5/8 != gold):')
for c,n in tot.most_common(): print(f'   {c:50s} {n:4d}   dapo {bysrc[("dapo",c)]:4d} / skywork {bysrc[("skywork",c)]:4d}')
nmm=sum(tot.values()); print(f'   total {nmm} = {100*nmm/sum(hard.values()):.0f}% of hard bin; dapo {sum(v for (s,c),v in bysrc.items() if s=="dapo")} ({100*sum(v for (s,c),v in bysrc.items() if s=="dapo")/hard["dapo"]:.0f}% of dapo hard), skywork {sum(v for (s,c),v in bysrc.items() if s=="skywork")} ({100*sum(v for (s,c),v in bysrc.items() if s=="skywork")/hard["skywork"]:.0f}% of skywork hard)')
# transform / CN shares among mismatches vs among hard bin
def share(pred,pool):
    return sum(1 for x in pool if pred(x))/max(1,len(pool))
hardrows=[uid2v3[u] for u,o in lc.items() if o['pass']<=0.25]
mmrows=[uid2v3[u] for u,o in lc.items() if o['pass']<=0.25 and o['majc']>=5 and o['maj']!=o['gold']]
for name,pred in [('chinese',lambda r: re.search(r'[一-鿿]',r['input']) is not None),('transform-instruction',lambda r: transform.search(r['input']) is not None)]:
    print(f'   {name}: share in hard bin {100*share(pred,hardrows):.0f}%, share in consistent-mismatch {100*share(pred,mmrows):.0f}%')
import random; random.seed(1)
for key in [('skywork','integer vs integer (other)'),('skywork','other'),('skywork','gold int, model non-numeric (untransformed form?)'),('dapo','integer vs integer (other)'),('dapo','gold int, model non-numeric (untransformed form?)')]:
    L=ex.get(key,[]); print(f'\n### {key} (n={len(L)}) examples:')
    for e in random.sample(L,min(8,len(L))): print('   ',e)
