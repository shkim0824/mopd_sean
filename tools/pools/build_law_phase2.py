import json,glob,re,random,collections,os
import pyarrow.parquet as pq
from mopd.data.domains.prep_rl_pools import build_eval_grams, contaminated
random.seed(42)
D='/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/domains/law_lawma'
SUF='\n\nPlease reason step by step, then give your final answer on the last line in the form "Answer: <letter>".'
# --- schema ---
t=pq.read_table(sorted(glob.glob(D+'/lawma-instructions/data/train-*.parquet'))[0])
print('lawma-instructions columns:',t.column_names, 'rows/shard',t.num_rows)
row=t.slice(0,1).to_pylist()[0]
for k,v in row.items(): print('  ',k,'=',repr(str(v)[:300]))
# --- exclusion set: opinions in caselawqa-8k test/val ---
excl=set()
for f in glob.glob(D+'/caselawqa-8k/*/*.parquet'):
    tb=pq.read_table(f); cols=tb.column_names
    for r in tb.to_pylist():
        op=(r.get('opinion') or '').strip(); excl.add(op[:300])
print('caselawqa-8k opinions to exclude:',len(excl))
# --- collect candidates ---
eg=build_eval_grams('law')
LETTER=re.compile(r'^\s*\(?([A-J])\)?\s*$')
bytask=collections.defaultdict(list); n_all=0; drop=collections.Counter(); percase=collections.Counter()
for f in sorted(glob.glob(D+'/lawma-instructions/data/train-*.parquet')):
    for r in pq.read_table(f).to_pylist():
        n_all+=1
        ins=r.get('instruction') or ''; out=r.get('output') or ''; task=r.get('task') or '?'
        m=LETTER.match(out)
        if not m: drop['non_letter_answer']+=1; continue
        if len(ins)>22000: drop['too_long']+=1; continue
        body=ins.split('\n\n',1)[1].strip() if '\n\n' in ins else ins
        if body[:300] in excl: drop['caselawqa_excl']+=1; continue
        opkey=body[:300]
        if percase[opkey]>=2: drop['opinion_cap']+=1; continue
        percase[opkey]+=1
        bytask[task].append((ins, m.group(1)))
print('rows scanned',n_all,'drop',dict(drop),'tasks with letter answers',len(bytask))
sel=[]
for task,items in bytask.items():
    random.shuffle(items)
    # balance answers within task: cap per letter so no letter exceeds 50%
    cnt=collections.Counter(); take=[]
    for ins,a in items:
        if len(take)>=200: break
        if cnt[a]>=100: continue
        take.append((ins,a)); cnt[a]+=1
    for ins,a in take: sel.append({'input':ins.rstrip()+SUF,'output':a,'meta':{'source':'lawma','task':task,'kind':'mcqa_lawma'}})
print('after task cap (<=200/task, <=100/letter):',len(sel))
kept=[r for r in sel if not contaminated(r['input'],eg)]
tc=sorted(collections.Counter(r['meta']['task'] for r in kept).values()); print('after eval decon:',len(kept),'| tasks',len(tc),'| per-task min/p50/max',tc[0] if tc else 0, tc[len(tc)//2] if tc else 0, tc[-1] if tc else 0)
# --- HolySaint MBE ---
mbe=json.load(open('/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/domains/law_mbe/raw_dataset.json'))
bar=set(json.loads(l)['input'][:120] for l in open('data/rl_pools/law_train.jsonl') if json.loads(l)['meta']['source']=='barexam_qa')
mrows=[]; seenq=set()
for it in mbe:
    q=(it.get('question') or '').strip(); opts=it.get('options') or {}; ans=(it.get('correct_answer') or '').strip().upper()[-1:]
    if not q or len(opts)<4 or ans not in 'ABCD' or q in seenq: continue
    seenq.add(q)
    txt=q+'\n\n'+'\n'.join(f'{k[-1].upper()}. {v}' for k,v in sorted(opts.items()))
    if txt[:120] in bar: continue
    mrows.append({'input':txt+SUF,'output':ans,'meta':{'source':'mbe_holysaint','kind':'mcqa4'}})
mrows=[r for r in mrows if not contaminated(r['input'],eg)]
print('MBE rows',len(mrows))
allr=kept+mrows; random.shuffle(allr)
for i,r in enumerate(allr): r['id']=f'law2-{i:06d}'
os.makedirs('data/rl_pools',exist_ok=True)
with open('data/rl_pools/law_phase2_candidates.jsonl','w') as f:
    for r in allr: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('PHASE2 CANDIDATES:',len(allr), dict(collections.Counter(r['meta']['source'] for r in allr)))
L=sorted(len(r['input']) for r in allr); print('prompt chars p50',L[len(L)//2],'p90',L[int(.9*len(L))],'max',L[-1])
