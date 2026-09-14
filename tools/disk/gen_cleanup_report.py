import json, html, re, sys
S="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean"
AFTER=sys.argv[1] if len(sys.argv)>1 else 'dust_d4.json'; BEFORE=sys.argv[2] if len(sys.argv)>2 else None
def load(p):
    d=json.load(open(p)); idx={}
    def size(n):
        try: return int(str(n.get('size','0')).rstrip('B').strip())
        except: return 0
    def walk(n):
        n['_s']=size(n); n['_r']=n['name'].replace(S+'/','').replace(S,'.'); idx[n['_r']]=n
        for c in n.get('children',[]): walk(c)
    walk(d); return d, idx
A, ai = load(AFTER)
B, bi = (load(BEFORE) if BEFORE else (None, {}))
def G(b): return b/1e9
def fmt(b):
    if b>=1e12: return f"{b/1e12:.2f} TB"
    if b>=1e9: return f"{b/1e9:.1f} GB"
    if b>=1e6: return f"{b/1e6:.0f} MB"
    return f"{b/1e3:.0f} KB"
def kids(path, idx=ai): 
    n=idx.get(path); return sorted(n.get('children',[]), key=lambda c:-c['_s']) if n else []
def sz(path, idx=ai): return idx[path]['_s'] if path in idx else 0
CK={'fin-pg':4,'fin-top16':4,'fin-top64':4,'if-pg':8,'if-top16':8,'if-top64':8,'law-pg':4,'law-pg-sftw75':4,'law-pg-v2':8,'law-top16':4,'law-top16-v2':8,'law-top64':4,'law-top64-v2':8,'med-pg':8,'med-pg-sftp-sftw150':8,'med-pg-sftw150':8,'med-pg-sftwA561':6,'med-pg-sftwB2164':7,'med-top16':8,'med-top64':8,'mopd-3dom-rllaw-96':8,'mopd-4dom-128':8,'mopd-4dom-128-merged':8,'mopd-4dom-128-merged-sftp':8,'mopd-4dom-128-merged-w4411':0,'mopd-4dom-128-mixsft':2,'mopd-4dom-128-sftp':8,'mopd-4dom-128-top64':8,'mopd-4dom-32':8,'mopd-4dom-32-top64':8,'mopd-finif-64':8,'mopd-medlaw-64':8,'mopd-medlaw-64-sftp':8,'opd_law_pg-rlteacher':8,'opd_med_pg-sftp':8,'qwen3-1p7b-18k':44,'qwen3-4b':44}
def nck(path):
    n=path.split('/')[-1]
    if n in CK: return CK[n]
    k=sum(1 for c in kids(path) if re.match(r'(checkpoint-\d+|step_\d+)$', c['_r'].split('/')[-1]))
    return k if k else '–'
# ---------- T1 top level before/after
T1=[]
tops=sorted(set([c['_r'] for c in A.get('children',[])] + ([c['_r'] for c in B.get('children',[])] if B else [])))
for t in tops:
    T1.append((t, sz(t,bi) if B else None, sz(t)))
T1.sort(key=lambda r:-(r[2]))
# ---------- T3 keep-family checkpoint table
symlinked = {  # results step dirs referenced by teacher/eval symlink dirs (readlink, 2026-09-14)
 'mopd_domains/outputs/rl/grpo_law_distill_4b': 'step_50/100/150/200 (rl2-law-distill-ck*; ck200 = law teacher)',
 'mopd_domains/outputs/rl/grpo_law_4b': 'step_100/150/190/200 (rl-law-ck*; ck200 = RL-only law teacher)',
 'mopd_domains/outputs/rl/grpo_fin_4b': 'step_100/150/200 (rl-fin-ck*; ck200 = fin teacher)',
 'mopd_domains/outputs/rl/grpo_med_4b': 'step_100/150/200 (rl-med-ck*, dead end)',
 'mopd_domains/outputs/rl/grpo_law_phase2_4b': 'step_50/100 (rl3-law-phase2, rejected)',
 'mopd_domains/outputs/rl/grpo_med_v2_4b': 'step_25..150 (rl4-med-v2, dead end)', 'mopd_domains/outputs/rl/grpo_med_v3_4b': 'step_25..150 (dead end)',
 'mopd_domains/outputs/rl/grpo_med_v4_4b': 'step_25..200 (dead end)', 'mopd_domains/outputs/rl/grpo_med_v4a_4b': 'step_25 (dead end)', 'mopd_domains/outputs/rl/grpo_med_v5_4b': 'step_25..200 (dead end)', 'mopd_domains/outputs/rl/grpo_med_v6_4b': 'step_25..200 (dead end)',
 'mopd_rl/outputs/rl/grpo_if_4b': 'step_110/200 (rl-if-ck110, rl-if-ck200 = IF teacher; mopd_rl/models/rl-if-ck200 is file-level symlinks into step_200)',
 'mopd_rl/outputs/rl/grpo_code_4b': 'step_50/100/150 (rl-code-ck*; file-level symlinks)',
 'mopd_rl/outputs/rl/grpo_math_v3_4b': 'step_70/120/180 (rl-mathv3-ck*; file-level symlinks)',
}
def role(path):
    n=path.split('/')[-1]
    if path.startswith('mopd_domains/outputs/opd/'): return ('OPD/MOPD run', '유지 (완료 run은 가중치만 80 GB; 진행 중 run은 완료 후 strip)')
    if path.startswith('mopd_domains/outputs/sft/sft_distill_med_c6_4ep'): return ('med teacher (SFT C6-4ep)', 'checkpoint-3368 유지, optimizer state 제거 가능')
    if path.startswith('mopd_domains/outputs/sft/sft_distill_law_2ep'): return ('law SFT (RL teacher의 init)', 'checkpoint-104 유지, 나머지 제거 가능')
    if path.startswith('mopd_domains/results/sft_distill') or path.startswith('mopd_domains/results/sft_med') or path.startswith('mopd_domains/results/smoke'): return ('SFT teacher 후보 (탈락)', 'past 이동 + 마지막 ckpt만 유지')
    if path.startswith('mopd_domains/results/sft_warmup'): return ('SFT warm-up', 'OPD init으로 쓴 ckpt만 유지 (law ck75, med_sftp ck150, c6k1 ck561, c6k3 ck2164, mix ck1110)')
    if path.startswith('mopd_domains/results/grpo_'): return ('GRPO teacher run (NeMo-RL)', 'symlink된 step의 consolidated 가중치만 유지, 나머지 step·DCP shard 제거')
    if path.startswith('mopd_rl/results/'): return ('GRPO teacher run (NeMo-RL, math/code/IF)', '참조되는 step의 consolidated 가중치만 유지 (IF teacher = step_200!), 나머지 step·DCP shard 제거')
    if path.startswith('mopd_rl/models/'): return ('teacher / student 사본', '유지')
    if path.startswith('mopd/outputs/sft_ot3/qwen3-1p7b'): return ('1.7B-OT3 (유지 대상)', '마지막 ckpt 가중치만 유지 (optimizer·중간 ckpt 제거 → 약 4 GB)')
    if path.startswith('mopd/outputs/sft_ot3/qwen3-4b'): return ('4B-OT3 SFT run', '최종본은 models/Qwen3-4B-OT3 → 중간 ckpt 제거 가능')
    if path.startswith('mopd_domains/models/merged'): return ('merged init', '유지')
    if path.startswith('mopd_domains/models/'): return ('teacher symlink dir (16 MB)', '유지')
    return ('', '')
T3=[]
for base in ['mopd_domains/outputs/opd','mopd_domains/results','mopd_rl/results','mopd_rl/models','mopd/outputs/sft_ot3','mopd_domains/models']:
    for c in kids(base):
        r=c['_r']
        if c['_s'] < 1e9: continue
        ro,pr=role(r); T3.append((r, c['_s'], nck(r), ro, symlinked.get(r,''), pr))
# ---------- T4 past
T4=[(c['_r'], c['_s']) for c in kids('past')] + [(g['_r'], g['_s']) for c in kids('past') for g in kids(c['_r'])]
# ---------- T5 recommendations (deletable caches/logs)
T5=[('mopd_domains/tmp/cache', sz('mopd_domains/tmp/cache'), '평가·학습 job별 torch inductor 컴파일 캐시(ind_<job>_<rank>); 끝난 job의 것은 재생성 가능한 임시 파일 → 삭제 권고 (진행 중 job 것 제외)'),
    ('mopd_domains/logs', sz('mopd_domains/logs'), 'NeMo-RL rollout 덤프·tensorboard (nemo_med_v2..v6, nemo_fin/law) → 분석 끝난 run은 삭제 또는 past'),
    ('mopd_rl/logs', sz('mopd_rl/logs'), 'NeMo-RL rollout 덤프 (nemo_code/math/if) → 동일'),
    ('mtm', sz('mtm'), '별도 프로젝트 (MTM): outputs 3.6 TB + models 1.4 TB. 이번 정리 범위 밖이라 손대지 않음 → 결정 필요')]
# ---------- tree text
def tree(path, depth, maxd, minG=1.0, out=None, prefix=''):
    out=out if out is not None else []
    n=ai.get(path); 
    if not n: return out
    for c in kids(path):
        if G(c['_s'])<minG: continue
        out.append(f"{prefix}{fmt(c['_s']):>9}  {c['_r'].split('/')[-1]}")
        if depth<maxd: tree(c['_r'], depth+1, maxd, minG, out, prefix+'    ')
    return out
lines=[f"{fmt(A['_s']):>9}  {S} (after)"]+tree('.',1,2,minG=1.0)
open('cleanup_tree.txt','w').write('\n'.join(lines))
def md(hdr, rows):
    o=['| '+' | '.join(hdr)+' |','|'+'|'.join(['---']*len(hdr))+'|']
    for r in rows: o.append('| '+' | '.join(str(x) for x in r)+' |')
    return '\n'.join(o)
mdout=[]
mdout.append('### 최상위 (before → after)\n'+md(['dir','before','after'],[(t, fmt(b) if b is not None else '–', fmt(a)) for t,b,a in T1]))
mdout.append('\n### 4B-OT3 계열 checkpoint (측정만; 이동 여부는 사용자 결정)\n'+md(['path','size','ckpt dirs','role','참조되는 step','제안'],[(p,fmt(s),n,ro,sl,pr) for p,s,n,ro,sl,pr in T3]))
mdout.append('\n### past/ (정리 후)\n'+md(['path','size'],[(p,fmt(s)) for p,s in T4]))
mdout.append('\n### 권고 (승인 필요)\n'+md(['path','size','내용'],[(p,fmt(s),w) for p,s,w in T5]))
open('cleanup_tables.md','w').write('\n'.join(mdout))
json.dump({'T1':T1,'T3':T3,'T4':T4,'T5':T5,'total_after':A['_s'],'total_before':(B['_s'] if B else None)}, open('cleanup_tables.json','w'), ensure_ascii=False)
print('\n'.join(lines[:45])); print('...'); print(mdout[0][:1500])
