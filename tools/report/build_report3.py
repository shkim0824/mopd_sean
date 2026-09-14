import json
d=json.load(open('all_evals.json')); dom=d['dom']; full=d['full']
BASE={'medqa':69.8,'casehold':63.2,'finqa':58.3,'ifeval':51.0,'ifbench':27.7}
TEACH={'medqa':80.7,'casehold':78.3,'finqa':74.6,'ifeval':79.1,'ifbench':58.7}
RLONLY_LAW=69.8
def acc(tag,b):
    r=dom.get(tag,{}).get(b)
    if not r or 'accuracy' not in r: return None
    return (r['accuracy'], r.get('truncated_frac',0.0))
def fb(tag,b):
    r=full.get(tag); return None if not r or b not in r else r[b]
def math_of(tag):
    r=full.get(tag); return None if not r else sum(r[b] for b in ('aime24','aime25','aime26'))/3
def cell(v):
    if v is None: return '–'
    if isinstance(v,tuple):
        a,t=v; s=f'{a:.1f}'
        if t>=5: s+=f' †{t:.0f}%'
        return s
    return f'{v:.1f}'
def sn(v,b,teach=None):
    if v is None: return None
    a=v[0] if isinstance(v,tuple) else v; t=teach if teach is not None else TEACH[b]
    return (a-BASE[b])/(t-BASE[b])
def pct(x):
    if x is None: return '–'
    v=round(100*x); v=0 if v==0 else v; return f'{v:d}%'
def mean(xs):
    xs=[x for x in xs if x is not None]; return None if not xs else sum(xs)/len(xs)
def if_sn(ie,ib): return mean([sn(ie,'ifeval'), sn(ib,'ifbench')])
BT='base-ot3-alldom-T1'; BF='sftot3-4b-final-32k'
def ood(ft): return [cell(fb(ft,'ifeval')), cell(fb(ft,'ifbench')), cell(math_of(ft)), cell(fb(ft,'lcb_v6'))]

# ---------- reference ----------
def ref_row(name, medqa, casehold, finqa, ft, note=''):
    ie=fb(ft,'ifeval'); ib=fb(ft,'ifbench')
    s4=mean([sn(medqa,'medqa'), sn(casehold,'casehold'), sn(finqa,'finqa'), if_sn(ie,ib)])
    return [name, cell(medqa), cell(casehold), cell(finqa)] + ood(ft) + [pct(s4), note]
REF_HDR=['model','MedQA','CaseHOLD','FinQA','IFEval','IFBench','Math (AIME24–26)','Code (LCB v6)','s̃ 4-dom','note']
REF=[
 ref_row('base OT3 (Qwen3-4B-OT3)', acc(BT,'medqa'), acc(BT,'casehold'), acc(BT,'finqa'), BF, 'student init'),
 ref_row('teacher law (SFT→RL)', acc('teacher-law-dom3-T1','medqa'), acc('rl2-law-distill-ck200-casehold-T1','casehold'), acc('teacher-law-dom3-T1','finqa'), 'full-law-rl-ck200-32k', '35B-trace SFT → GRPO'),
 ref_row('teacher law RL-only', acc('teacher-rllaw-dom3-T1','medqa'), acc('rl-law-ck200-T1-26k','casehold'), acc('teacher-rllaw-dom3-T1','finqa'), 'full-law-rlonly-ck200-32k', 'OT3 → GRPO only (no SFT)'),
 ref_row('teacher fin', acc('teacher-fin-dom3-T1','medqa'), acc('teacher-fin-dom3-T1','casehold'), acc('rl-fin-ck200-T1-32k','finqa'), 'full-fin-rl-ck200-32k', 'OT3 → GRPO'),
 ref_row('teacher IF', acc('teacher-if-dom3-T1','medqa'), acc('teacher-if-dom3-T1','casehold'), acc('teacher-if-dom3-T1','finqa'), 'rl-if-ck200-32k-fixedrope', 'OT3 → GRPO'),
 ref_row('teacher med (SFT C6-4ep)', acc('sftd-med-c6_4ep-medqa-T1','medqa'), acc('teacher-med-dom3-T1','casehold'), acc('teacher-med-dom3-T1','finqa'), 'full-med-c6_4ep-32k', '35B-trace SFT, 4 epochs'),
 ref_row('merged-4teachers (untrained init)', acc('merged-4t-uniform-dom3-T1','medqa'), acc('merged-4t-uniform-dom3-T1','casehold'), acc('merged-4t-uniform-dom3-T1','finqa'), 'full-merged-4t-uniform-32k', 'fp32 average of the 4 teachers'),
]
# ---------- PG summary ----------
PG_HDR=['model','teacher','MedQA','CaseHOLD','FinQA','IFEval','IFBench','Math','Code','s̃ (own domain)','loop episode (steps)']
def pgrow(name, teacher, bench, tag_in, ft, note, teach=None):
    v=acc(tag_in,bench) if tag_in else None
    cells={'medqa':'–','casehold':'–','finqa':'–'}
    if bench in cells: cells[bench]=cell(v)
    ie=fb(ft,'ifeval'); ib=fb(ft,'ifbench')
    s=if_sn(ie,ib) if bench=='if' else sn(v,bench,teach)
    return [name, teacher, cells['medqa'], cells['casehold'], cells['finqa']] + ood(ft) + [pct(s), note]
r0=REF[0]
PG=[
 ['base OT3','–',r0[1],r0[2],r0[3],r0[4],r0[5],r0[6],r0[7],'0%',''],
 ['teacher IF','–',REF[4][1],REF[4][2],REF[4][3],REF[4][4],REF[4][5],REF[4][6],REF[4][7],'100% (IF)',''],
 pgrow('IF PG','IF','if',None,'full-opd-if-if-pg-final-32k','없음'),
 ['teacher fin','–',REF[3][1],REF[3][2],REF[3][3],REF[3][4],REF[3][5],REF[3][6],REF[3][7],'100% (fin)',''],
 pgrow('fin PG','fin','finqa','opd-fin-pg-ck200-finqa-T1','full-opd-fin-pg-final-32k','없음'),
 ['teacher law (SFT→RL)','–',REF[1][1],REF[1][2],REF[1][3],REF[1][4],REF[1][5],REF[1][6],REF[1][7],'100% (law)',''],
 pgrow('law PG','law SFT→RL','casehold','opd-law-pg-v2-ck200-casehold-T1','full-opd-law-pg-v2-final-32k','80–94, 회복'),
 ['teacher med (SFT C6-4ep)','–',REF[5][1],REF[5][2],REF[5][3],REF[5][4],REF[5][5],REF[5][6],REF[5][7],'100% (med)',''],
 pgrow('med PG','med SFT','medqa','opd-opd_med_pg-sftp-ck200-medqa-T1','full-opd-opd_med_pg-sftp-final-32k','85–150, 회복'),
 ['teacher law RL-only','–',REF[2][1],REF[2][2],REF[2][3],REF[2][4],REF[2][5],REF[2][6],REF[2][7],'100% (law, 69.8 기준)',''],
 pgrow('law PG (RL-only teacher)','law RL-only','casehold','opd-opd_law_pg-rlteacher-ck200-casehold-T1','full-opd-opd_law_pg-rlteacher-final-32k','30–40, 회복 (word salad)', teach=RLONLY_LAW),
]
# ---------- per-domain ----------
def one_row(name, bench, tag_in, ft, note, teach=None):
    v=acc(tag_in,bench); return [name, cell(v)] + ood(ft) + [pct(sn(v,bench,teach)), note]
LAW_HDR=['model','CaseHOLD (ck200, T1.0)','IFEval','IFBench','Math','Code','s̃ CaseHOLD','loop episode (steps) · note']
LAW=[
 ['base OT3', cell(acc(BT,'casehold'))]+ood(BF)+['0%',''],
 ['teacher law (SFT→RL)', cell(acc('rl2-law-distill-ck200-casehold-T1','casehold'))]+ood('full-law-rl-ck200-32k')+['100%','아래 8행의 teacher'],
 one_row('law top-64','casehold','opd-law-top64-v2-ck200-casehold-T1','full-opd-law-top64-v2-final-32k','75–98 회복 · best law OPD'),
 one_row('law PG','casehold','opd-law-pg-v2-ck200-casehold-T1','full-opd-law-pg-v2-final-32k','80–94 회복'),
 one_row('law PG (v1: truncated 샘플 loss 포함, 저장 50)','casehold','opd-law-pg-ck200-casehold-T1','full-opd-law-pg-final-32k','78–99 회복 (마스킹 없이도)'),
 one_row('law top-64 (v1)','casehold','opd-law-top64-ck200-casehold-T1','full-opd-law-top64-final-32k','72부터 미회복'),
 one_row('law top-16 (v1)','casehold','opd-law-top16-ck200-casehold-T1','full-opd-law-top16-final-32k','69부터 미회복'),
 one_row('law top-16','casehold','opd-law-top16-v2-ck200-casehold-T1','full-opd-law-top16-v2-final-32k','미회복 (전 샘플 마스킹 → loss 0)'),
 one_row('law PG from SFT warm-up ck75 (branch)','casehold','opd-law-pg-sftw75-ck200-casehold-T1','full-opd-law-pg-sftw75-final-32k','없음 · OOD는 warm-up에서 상속'),
 ['teacher law RL-only', cell(acc('rl-law-ck200-T1-26k','casehold'))]+ood('full-law-rlonly-ck200-32k')+['100% (자기 기준)','아래 행의 teacher'],
 one_row('law PG (RL-only teacher)','casehold','opd-opd_law_pg-rlteacher-ck200-casehold-T1','full-opd-opd_law_pg-rlteacher-final-32k','30–40 회복 (word salad) · s̃는 69.8 기준', teach=RLONLY_LAW),
]
FIN_HDR=['model','FinQA (ck200, T1.0)','IFEval','IFBench','Math','Code','s̃ FinQA','note']
FIN=[
 ['base OT3', cell(acc(BT,'finqa'))]+ood(BF)+['0%',''],
 ['teacher fin', cell(acc('rl-fin-ck200-T1-32k','finqa'))]+ood('full-fin-rl-ck200-32k')+['100%',''],
 one_row('fin top-64','finqa','opd-fin-top64-ck200-finqa-T1','full-opd-fin-top64-final-32k','loop 없음'),
 one_row('fin top-16','finqa','opd-fin-top16-ck200-finqa-T1','full-opd-fin-top16-final-32k','loop 없음'),
 one_row('fin PG','finqa','opd-fin-pg-ck200-finqa-T1','full-opd-fin-pg-final-32k','loop 없음'),
]
IF_HDR=['model','IFEval','IFBench','Math','Code','s̃ IF','note']
def ifrow(name,tag,note): ie=fb(tag,'ifeval'); ib=fb(tag,'ifbench'); return [name, cell(ie), cell(ib), cell(math_of(tag)), cell(fb(tag,'lcb_v6')), pct(if_sn(ie,ib)), note]
IF=[ifrow('base OT3',BF,''), ifrow('teacher IF','rl-if-ck200-32k-fixedrope',''), ifrow('IF top-64','full-opd-if-if-top64-final-32k','loop 없음'), ifrow('IF top-16','full-opd-if-if-top16-final-32k','loop 없음'), ifrow('IF PG','full-opd-if-if-pg-final-32k','loop 없음')]
MED_HDR=['model','MedQA (ck200, T1.0)','IFEval','IFBench','Math','Code','s̃ MedQA','loop episode · note']
MED=[
 ['base OT3', cell(acc(BT,'medqa'))]+ood(BF)+['0%',''],
 ['teacher med (SFT C6-4ep)', cell(acc('sftd-med-c6_4ep-medqa-T1','medqa'))]+ood('full-med-c6_4ep-32k')+['100%',''],
 one_row('med PG','medqa','opd-opd_med_pg-sftp-ck200-medqa-T1','full-opd-opd_med_pg-sftp-final-32k','85–150 회복'),
 one_row('med PG from SFT warm-up ck150 (branch)','medqa','opd-med-pg-sftp-sftw150-ck200-medqa-T1','full-opd-med-pg-sftp-sftw150-final-32k','없음 · init = 6.4k×1 warm-up ck150'),
]
MED_OLD=[
 one_row('med top-64 (old pool)','medqa','opd-med-top64-ck200-medqa-T1','full-opd-med-top64-final-32k','early-EOS drift → code 붕괴'),
 one_row('med top-16 (old pool)','medqa','opd-med-top16-ck200-medqa-T1','full-opd-med-top16-final-32k','평탄'),
 one_row('med PG (old pool)','medqa','opd-med-pg-ck200-medqa-T1','full-opd-med-pg-final-32k','85–150 회복 (175)'),
 one_row('med PG from old-pool warm-up ck150 (branch v1)','medqa','opd-med-pg-sftw150-ck200-medqa-T1','full-opd-med-pg-sftw150-final-32k','없음'),
]
# ---------- warm-ups ----------
WU_REC_HDR=['run','teacher · data source','rows (traces)','epochs · row-passes','recipe','compute','result (T1.0)','follow-up']
WU_REC=[
 ['law warm-up','law SFT→RL teacher, k=1 T1.0 on 9,971 law OPD-pool prompts → reject filter (letter-exact, finished, single </think>) 7,950 accepted (80%) → 6,400 sampled','6,400 (1 per prompt)','1 · 6.4k','padding, max_len 20480, lr 1e-5 cosine, 32 seqs per step × 200 steps, save 25','1 node, 14 min','CaseHOLD 70.8 → 77.1 (ck25 → ck200), 포화 ck75','ck75 → law PG branch'],
 ['med warm-up','med SFT teacher, k=1 on the 8,384 SFT-teacher prompts → 7,893 accepted → 6,400','6,400','1 · 6.4k','same','1 node, 22 min','MedQA 62.5 → 70.0 (base 수준)','ck150 → med PG branch'],
 ['scaled-up B','med SFT teacher, k=4 → keep ≤3 distinct correct traces per prompt','24,401 (8,320 prompts)','4 · 98k','teacher recipe: flatten packing 8192, lr 1e-5 cosine, save each epoch','2 nodes, 1h15m','MedQA 76.4 → 77.2 → 78.1 → 78.7','epoch 4 → med PG branch (진행 중)'],
 ['mixed law+med warm-up','law_sft6400에서 3,200행(= law warm-up ck100 노출량) + scaled-up B 데이터 24,401행 ×2(= B 2 epoch 노출량), 섞어서 1 epoch','52,002','1 · 52k','teacher recipe','2 nodes, 40 min','MedQA 76.7 · CaseHOLD 73.8 · FinQA 51.9','4-dom MOPD 128 (대기 중)'],
]
WU_HDR=['model','bench','in-domain (last ckpt, T1.0)','IFEval','IFBench','Math','Code','s̃','note']
def wurow(name, run, bench, lastck, note):
    v=acc(f'sftw-{run}-ck{lastck}-{bench}-T1',bench); ft=f'full-sftw-{run}-final-32k'
    return [name, bench, cell(v)] + ood(ft) + [pct(sn(v,bench)), note]
def wu_one(name, bench, tag_in, ft, note, teach=None):
    v=acc(tag_in,bench); return [name, bench, cell(v)] + ood(ft) + [pct(sn(v,bench,teach)), note]
def latest_ck(pre, bench):
    for c in [200,175,150,125,100,75,50,25]:
        if f'{pre}-ck{c}-{bench}-T1' in dom: return c
    return None
def branch_row(name, pre, bench, note):
    c=latest_ck(pre,bench)
    if c is None: return [name, bench, '–','–','–','–','–','–', note+' · 아직 checkpoint 없음']
    v=acc(f'{pre}-ck{c}-{bench}-T1',bench); ft=f'full-{pre}-final-32k'
    return [f'{name} (ck{c})', bench, cell(v)] + ood(ft) + [pct(sn(v,bench)), note]
WU=[
 ['base OT3 (law bench)','casehold', cell(acc(BT,'casehold'))]+ood(BF)+['0%',''],
 ['teacher law (SFT→RL)','casehold', cell(acc('rl2-law-distill-ck200-casehold-T1','casehold'))]+ood('full-law-rl-ck200-32k')+['100%',''],
 wurow('law warm-up ck200 (6.4k × 1 pass)','sft_warmup_law','casehold',200,'ck75 이후 포화'),
 wu_one('law PG branch from warm-up ck75 (ck200)','casehold','opd-law-pg-sftw75-ck200-casehold-T1','full-opd-law-pg-sftw75-final-32k','OPD가 OOD를 복구하지 못함'),
 ['base OT3 (med bench)','medqa', cell(acc(BT,'medqa'))]+ood(BF)+['0%',''],
 ['teacher med (SFT C6-4ep)','medqa', cell(acc('sftd-med-c6_4ep-medqa-T1','medqa'))]+ood('full-med-c6_4ep-32k')+['100%',''],
 wurow('med warm-up ck200 (6.4k × 1)','sft_warmup_med_sftp','medqa',200,'base 수준'),
 wu_one('med PG branch from warm-up ck150 (ck200)','medqa','opd-med-pg-sftp-sftw150-ck200-medqa-T1','full-opd-med-pg-sftp-sftw150-final-32k','loop 없음'),
 wurow('scaled-up B epoch 4 (24.4k × 4)','sft_warmup_med_c6k3_4ep','medqa',2164,'best medical model'),
 branch_row('med PG branch from scaled-up B epoch 4','opd-med-pg-sftwB2164','medqa','진행 중'),
]
WU_APP=[
 ['base OT3 (med bench)','medqa', cell(acc(BT,'medqa'))]+ood(BF)+['0%',''],
 ['teacher med (SFT C6-4ep)','medqa', cell(acc('sftd-med-c6_4ep-medqa-T1','medqa'))]+ood('full-med-c6_4ep-32k')+['100%',''],
 wurow('med warm-up v1 ck200 (old pool, 6.4k × 1)','sft_warmup_med','medqa',200,'base 수준'),
 wu_one('med PG branch from old-pool warm-up ck150 (ck200)','medqa','opd-med-pg-sftw150-ck200-medqa-T1','full-opd-med-pg-sftw150-final-32k','loop 없음'),
 wurow('35B-trace control ck200 (같은 6,400 prompts × 1)','sft_warmup_med_35b','medqa',200,'style-shift dip'),
 wurow('scaled-up A epoch 4 (7.9k × 4)','sft_warmup_med_c6k1_4ep','medqa',748,'epoch 3에서 포화'),
 branch_row('med PG branch from scaled-up A epoch 3','opd-med-pg-sftwA561','medqa','진행 중'),
]
# ---------- MOPD ----------
MOPD_RUNS=[
 ('mopd-finif-64','2-dom fin+IF PG, 64 per update','OT3','fin IF','없음',200,False),
 ('mopd-medlaw-64-sftp','2-dom med+law PG, 64 per update','OT3','med law','75–100 회복',200,False),
 ('mopd-4dom-128-sftp','4-dom PG, 128 per update (32 per domain)','OT3','med law fin IF','med loop ck150–200, 일부 완화',200,False),
 ('mopd-4dom-128-merged-sftp','4-dom PG 128, merged-teacher init','merged','med law fin IF','없음',200,False),
 ('mopd-3dom-rllaw-96','3-dom PG 96 (32 per domain), RL-only law + fin + IF','OT3','law(RL-only) fin IF','law word-salad 45–60, ck125 회복',200,True),
 ('mopd-4dom-128-mixsft','4-dom PG 128, mixed law+med warm-up init · 진행 중','mixed SFT','med law fin IF','진행 중 (최신 ckpt 표시)',None,False),
]
MOPD_HDR=['run','init','domains','ckpt','MedQA','CaseHOLD','FinQA','IFEval','IFBench','Math','Code','s̃ 4-dom','s̃ own domains','loop episode · note']
def mopd_row(run,label,init,doms,note,ck,rlonly):
    t=f'opd-{run}-ck{ck}-dom3-T1'; ft=f'full-opd-{run}-ck{ck}-32k'
    m=acc(t,'medqa'); c=acc(t,'casehold'); f=acc(t,'finqa'); ie=fb(ft,'ifeval'); ib=fb(ft,'ifbench')
    parts={'med':sn(m,'medqa'),'law':sn(c,'casehold',RLONLY_LAW if rlonly else None),'fin':sn(f,'finqa'),'if':if_sn(ie,ib)}
    s4=mean(list(parts.values())) if ft in full else None; own=mean([parts['med'] if 'med' in doms else None, parts['law'] if 'law' in doms else None, parts['fin'] if 'fin' in doms else None, parts['if'] if 'IF' in doms else None]) if (ft in full or 'IF' not in doms) else None
    return [label, init, doms, f'ck{ck}', cell(m), cell(c), cell(f), cell(ie), cell(ib), cell(math_of(ft)), cell(fb(ft,'lcb_v6')), pct(s4), pct(own), note]
MOPD=[]
MOPD.append(['base OT3','–','–','–',r0[1],r0[2],r0[3],r0[4],r0[5],r0[6],r0[7],'0%','0%',''])
for r,lab in [(1,'teacher law (SFT→RL)'),(2,'teacher law RL-only'),(3,'teacher fin'),(4,'teacher IF'),(5,'teacher med'),(6,'merged-4teachers init (untrained)')]:
    x=REF[r]; MOPD.append([lab,'–','–','–',x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],'–',x[9]])
MIX=ref_row('mixed law+med SFT warm-up ckpt (untrained init)', acc('sftw-mix-lawmed-final-dom3-T1','medqa'), acc('sftw-mix-lawmed-final-dom3-T1','casehold'), acc('sftw-mix-lawmed-final-dom3-T1','finqa'), 'full-sftw-mix-lawmed-final-32k', '3,200 law + 24,401×2 med rows, 1 epoch')
MOPD.append([MIX[0],'–','–','–',MIX[1],MIX[2],MIX[3],MIX[4],MIX[5],MIX[6],MIX[7],MIX[8] if 'full-sftw-mix-lawmed-final-32k' in full else '–','–',MIX[9]])
def latest_dom3(run):
    for c in [200,175,150,125,100,75,50,25]:
        if f'opd-{run}-ck{c}-dom3-T1' in dom: return c
    return None
MOPD_RUNS=[(run,label,init,doms,note,(ck if ck else latest_dom3(run)),rl) for run,label,init,doms,note,ck,rl in MOPD_RUNS]
MOPD_RUNS=[r for r in MOPD_RUNS if r[5]]
for run,label,init,doms,note,ck,rl in MOPD_RUNS: MOPD.append(mopd_row(run,label,init,doms,note,ck,rl))
BEST=[]
for run,label,init,doms,note,ck,rl in MOPD_RUNS:
    best=None
    for c in [50,100,150,200]:
        if f'opd-{run}-ck{c}-dom3-T1' not in dom or f'full-opd-{run}-ck{c}-32k' not in full: continue
        row=mopd_row(run,label,init,doms,'',c,rl); s=float(row[11].rstrip('%'))
        if best is None or s>best[0]: best=(s,row)
    if best and best[1][3]!=f'ck{ck}': r=best[1]; r[13]='ck200보다 좋은 checkpoint'; BEST.append(r)
CURVE_HDR=['ckpt','MedQA','CaseHOLD','FinQA','IFEval','IFBench','Math','Code','s̃ 4-dom','s̃ own']
CURVES={}
for run,label,init,doms,note,ck,rl in MOPD_RUNS:
    rows=[]
    for c in [25,50,75,100,125,150,175,200]:
        if f'opd-{run}-ck{c}-dom3-T1' in dom: rows.append(mopd_row(run,f'ck{c}',init,doms,'',c,rl)[3:13])
    CURVES[label]=rows
def single_curve(pre, bench, teach=None):
    rows=[]
    for c in [25,50,75,100,125,150,175,200]:
        v=acc(f'{pre}-ck{c}-{bench}-T1',bench)
        if v: rows.append([f'ck{c}', cell(v), pct(sn(v,bench,teach))])
    return rows
SC={}
for pre,bench,lab,teach in [('opd-law-pg-v2','casehold','law PG',None),('opd-law-top64-v2','casehold','law top-64',None),('opd-law-pg','casehold','law PG (v1)',None),('opd-law-pg-sftw75','casehold','law PG branch (warm-up ck75)',None),('opd-opd_law_pg-rlteacher','casehold','law PG, RL-only teacher (s̃ 69.8 기준)',RLONLY_LAW),
                            ('opd-fin-pg','finqa','fin PG',None),('opd-fin-top64','finqa','fin top-64',None),('opd-fin-top16','finqa','fin top-16',None),
                            ('opd-opd_med_pg-sftp','medqa','med PG',None),('opd-med-pg-sftp-sftw150','medqa','med PG branch (warm-up v2 ck150)',None),('opd-med-pg','medqa','med PG (old pool)',None),('opd-med-top64','medqa','med top-64 (old pool)',None),('opd-med-top16','medqa','med top-16 (old pool)',None),
                            ('sftw-sft_warmup_law','casehold','law warm-up SFT',None),('sftw-sft_warmup_med','medqa','med warm-up v1',None),('sftw-sft_warmup_med_sftp','medqa','med warm-up v2',None),('sftw-sft_warmup_med_35b','medqa','35B-trace control',None),('opd-med-pg-sftwB2164','medqa','med PG branch from scaled-up B (진행 중)',None),('opd-med-pg-sftwA561','medqa','med PG branch from scaled-up A (진행 중)',None)]:
    SC[lab]=(bench,single_curve(pre,bench,teach))
SC['scaled-up A (epochs 1–4)']=('medqa',[[f'epoch {i+1} (ck{c})', cell(acc(f'sftw-sft_warmup_med_c6k1_4ep-ck{c}-medqa-T1','medqa')), pct(sn(acc(f'sftw-sft_warmup_med_c6k1_4ep-ck{c}-medqa-T1','medqa'),'medqa'))] for i,c in enumerate([187,374,561,748])])
SC['scaled-up B (epochs 1–4)']=('medqa',[[f'epoch {i+1} (ck{c})', cell(acc(f'sftw-sft_warmup_med_c6k3_4ep-ck{c}-medqa-T1','medqa')), pct(sn(acc(f'sftw-sft_warmup_med_c6k3_4ep-ck{c}-medqa-T1','medqa'),'medqa'))] for i,c in enumerate([541,1082,1623,2164])])
# IF curves from 6-bench per ckpt
IFC={}
for v in ['pg','top64','top16']:
    rows=[]
    for c in [25,50,75,100,125,150,175,200]:
        t=f'full-opd-if-if-{v}-ck{c}-if'
        if t in full: rows.append([f'ck{c}', cell(fb(t,'ifeval')), cell(fb(t,'ifbench')), pct(if_sn(fb(t,'ifeval'),fb(t,'ifbench')))])
    IFC[f'IF {v}']=rows
json.dump({'REF':(REF_HDR,REF),'PG':(PG_HDR,PG),'LAW':(LAW_HDR,LAW),'FIN':(FIN_HDR,FIN),'IF':(IF_HDR,IF),'MED':(MED_HDR,MED),'MED_OLD':(MED_HDR,MED_OLD),'WU_REC':(WU_REC_HDR,WU_REC),'WU':(WU_HDR,WU),'WU_APP':(WU_HDR,WU_APP),'MOPD':(MOPD_HDR,MOPD),'BEST':(MOPD_HDR,BEST),'CURVES':CURVES,'CURVE_HDR':CURVE_HDR,'SC':SC,'IFC':IFC}, open('report_tables3.json','w'), ensure_ascii=False)
def md(hdr, rows):
    out=['| '+' | '.join(hdr)+' |', '|'+'|'.join(['---']*len(hdr))+'|']
    for r in rows: out.append('| '+' | '.join(str(x) for x in r)+' |')
    return '\n'.join(out)
for k in ['PG','REF','LAW','FIN','IF','MED','MED_OLD','WU','MOPD','BEST']:
    h,r=json.load(open('report_tables3.json'))[k]; print(f'\n### {k}\n'+md(h,r))
