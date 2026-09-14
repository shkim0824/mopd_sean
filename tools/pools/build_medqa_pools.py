import json,ast,re,glob,collections
import pyarrow.parquet as pq
D='/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/domains'
SUF='\n\nPlease reason step by step, then give your final answer on the last line in the form "Answer: <letter>".'
keep=set(json.load(open('data/distill_pools/medqa_train_keep_idx_relaxed.json')))
rows4=[json.loads(l) for l in open(D+'/med_medqa/phrases_no_exclude_train.jsonl')]
print('4opt rows',len(rows4),'keys',list(rows4[0].keys())[:8])
def fmt(q,opts): return q.strip()+'\n\n'+'\n'.join(f'{k}. {v}' for k,v in opts)
p1=[]
for i,r in enumerate(rows4):
    if i not in keep: continue
    opts=r['options']; opts=sorted(opts.items()) if isinstance(opts,dict) else opts
    p1.append({'id':f'mq4-{i:05d}','input':fmt(r['question'],opts)+SUF,'output':r['answer_idx'].strip().upper(),'meta':{'source':'MedQA-4opt','kind':'mcqa4','qid':i}})
# 5-option from bigbio source (same order as GBaker? match by question text)
t=pq.read_table(glob.glob(D+'/med_medqa_bigbio/med_qa_en_source/train-*.parquet')[0]).to_pylist()
q2i={rows4[i]['question'].strip()[:200]:i for i in range(len(rows4))}
p2=[];miss=0
for r in t:
    i=q2i.get(r['question'].strip()[:200]);
    if i is None or i not in keep: miss+=1; continue
    opts=ast.literal_eval(r['options']) if isinstance(r['options'],str) else r['options']
    opts=[(o['key'],o['value']) for o in opts]
    p2.append({'id':f'mq5-{i:05d}','input':fmt(r['question'],opts)+SUF,'output':r['answer_idx'].strip().upper(),'meta':{'source':'MedQA-5opt','kind':'mcqa5','qid':i}})
print('P1 4opt',len(p1),'| P2 5opt',len(p2),'(unmatched/excluded',miss,')')
for name,p in [('medqa4_pool',p1),('medqa5_pool',p2)]:
    with open(f'data/distill_pools/{name}.jsonl','w') as f:
        for r in p: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('answer dist 4opt',collections.Counter(r['output'] for r in p1),'5opt',collections.Counter(r['output'] for r in p2))
# II-Medical MCQA scan (shard 0)
tb=pq.read_table(D+'/med_ii_medical/data/train-00000-of-00041.parquet',columns=['problem'])
pr=tb.column('problem').to_pylist(); opt=re.compile(r'(^|\n)\s*[A-E][\.\)]\s')
n_mcq=sum(1 for x in pr if x and len(opt.findall(x))>=3); vign=sum(1 for x in pr if x and len(opt.findall(x))>=3 and len(x)>500)
print(f'II-Medical shard0: {len(pr)} rows | MCQ-like {n_mcq} ({n_mcq/len(pr):.1%}) | vignette-length MCQ {vign}')
