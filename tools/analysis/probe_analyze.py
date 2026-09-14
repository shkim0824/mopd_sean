"""pass@k analysis of a gen_teacher probe: per-prompt #correct of k -> zero / mixed / all distribution.
usage: python3 tmp/probe_analyze.py outputs/teacher_gen/probe-law-ck104-k8 [--write data/rl_pools/law_rl_v2_train.jsonl --pool data/rl_pools/law_train.jsonl]
"""
import json,glob,sys,collections,argparse
from mopd.graders.domains.extract import extract_letter, extract_yesno_maybe
p=argparse.ArgumentParser(); p.add_argument('gen'); p.add_argument('--write'); p.add_argument('--pool'); a=p.parse_args()
NOPT={'mcqa4':4,'mcqa5':5}
hist=collections.Counter(); bysrc=collections.defaultdict(collections.Counter); rows=[]
for f in sorted(glob.glob(a.gen+'/shard*.jsonl')):
    for l in open(f):
        g=json.loads(l); kind=g['meta'].get('kind','mcqa4'); gold=str(g['gold']).strip().upper(); src=g['meta'].get('source','?')
        n=0; k=len(g['samples']); trunc=0
        for s in g['samples']:
            if s.get('finish')!='stop': trunc+=1; continue
            t=s['text']
            pred=(extract_yesno_maybe(t,allow_maybe=False) or '').upper() if kind=='yesno' else extract_letter(t,n_options=NOPT.get(kind,4))
            n+= (pred==gold)
        cat='all' if n==k else ('zero' if n==0 else 'mixed')
        hist[cat]+=1; bysrc[src][cat]+=1; rows.append((g['id'],src,kind,n,k,trunc))
tot=sum(hist.values()); print(f'prompts={tot} k={rows[0][4] if rows else 0} | all-correct {hist["all"]/tot:.3f} | mixed {hist["mixed"]/tot:.3f} | all-wrong {hist["zero"]/tot:.3f} | mean pass {sum(r[3]/r[4] for r in rows)/tot:.3f} | trunc/sample {sum(r[5] for r in rows)/sum(r[4] for r in rows):.3f}')
for src,c in bysrc.items():
    t=sum(c.values()); print(f'  {src:16s} n={t:4d} all={c["all"]/t:.2f} mixed={c["mixed"]/t:.2f} zero={c["zero"]/t:.2f}')
pc=collections.Counter(r[3] for r in rows); print('  #correct histogram:',dict(sorted(pc.items())))
