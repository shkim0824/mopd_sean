import json, sys, re
S="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean"
AFTER=sys.argv[1] if len(sys.argv)>1 else 'dust_d4_after.json'
P=json.load(open('cleanable_params.json')) if len(sys.argv)>2 else {}
# measured or assumed layout ratios (bytes kept per unit when pruned)
NEMO_KEEP_PER_STEP = P.get('nemo_consolidated_bytes', 8.1e9)   # consolidated HF bf16 weights per kept step
SFT_WEIGHTS_4B     = P.get('sft_weights_bytes_4b', 8.1e9)       # bf16 weights of one 4B checkpoint
SFT_WEIGHTS_1P7B   = P.get('sft_weights_bytes_1p7b', 3.5e9)
d=json.load(open(AFTER)); idx={}
def size(n):
    try: return int(str(n.get('size','0')).rstrip('B').strip())
    except: return 0
def walk(n):
    n['_s']=size(n); n['_r']=n['name'].replace(S+'/','').replace(S,'.'); idx[n['_r']]=n
    for c in n.get('children',[]): walk(c)
walk(d)
def sz(p): return idx[p]['_s'] if p in idx else 0
def kids(p): return sorted(idx[p].get('children',[]), key=lambda c:-c['_s']) if p in idx else []
def nsteps(p): return sum(1 for c in kids(p) if re.match(r'(checkpoint-\d+|step_\d+)$', c['_r'].split('/')[-1]))
rows=[]  # (category, path, current, keep, action, approval)
def add(cat, path, cur, keep, action, approval): rows.append((cat, path, cur, max(0,keep), action, approval))
# 1. compile caches
add('임시 캐시', 'mopd_domains/tmp/cache', sz('mopd_domains/tmp/cache'), 0, '6,523개 job 캐시 폴더 전부 끝난 job의 것(큐에 없음) → 전량 삭제 가능', '삭제 승인')
# 2. NeMo-RL GRPO teacher runs (mopd_domains + mopd_rl)
REF={'grpo_law_distill_4b':4,'grpo_law_4b':4,'grpo_fin_4b':3,'grpo_med_4b':3,'grpo_law_phase2_4b':2,'grpo_med_v2_4b':6,'grpo_med_v3_4b':6,'grpo_med_v4_4b':8,'grpo_med_v4a_4b':1,'grpo_med_v5_4b':8,'grpo_med_v6_4b':8,'grpo_if_4b':2,'grpo_code_4b':3,'grpo_math_v3_4b':3}
TEACHER={'grpo_law_distill_4b','grpo_law_4b','grpo_fin_4b','grpo_if_4b'}
for base in ['mopd_domains/results','mopd_rl/results']:
    for c in kids(base):
        n=c['_r'].split('/')[-1]
        if not n.startswith('grpo_'): continue
        keep_steps = 1 if n in TEACHER else 0   # keep only the step the teacher symlink points to (final); dead ends: keep last consolidated only
        keep = (keep_steps if keep_steps else 1)*NEMO_KEEP_PER_STEP
        act = ('teacher step(ck200)의 consolidated 가중치만 유지, 나머지 step + DCP shard + optimizer 제거' if n in TEACHER else 'dead end: 마지막 step consolidated만 남기고 제거 (또는 past)')
        add('GRPO teacher run (NeMo-RL)', c['_r'], c['_s'], keep, act, 'ckpt 결정')
# 3. SFT candidates / teachers / warm-ups (trl checkpoints with optimizer states)
for c in kids('mopd_domains/results'):
    n=c['_r'].split('/')[-1]
    if n.startswith('grpo_'): continue
    k=nsteps(c['_r']) or 1
    if n=='sft_distill_med_c6_4ep': add('SFT teacher', c['_r'], c['_s'], SFT_WEIGHTS_4B, 'checkpoint-3368 가중치만 유지 (optimizer state·다른 epoch 제거)', 'ckpt 결정')
    elif n=='sft_distill_law_2ep': add('SFT teacher', c['_r'], c['_s'], SFT_WEIGHTS_4B, 'checkpoint-104 가중치만 유지', 'ckpt 결정')
    elif n.startswith('sft_warmup'):
        used={'sft_warmup_law':1,'sft_warmup_med_sftp':1,'sft_warmup_med_c6k1_4ep':1,'sft_warmup_med_c6k3_4ep':1,'sft_warmup_mix_lawmed':1}.get(n,0)
        add('SFT warm-up', c['_r'], c['_s'], (used or 1)*SFT_WEIGHTS_4B, 'OPD init으로 쓴 ckpt 가중치만 유지' if used else '마지막 ckpt 가중치만 유지 (또는 past)', 'ckpt 결정')
    else: add('SFT teacher 후보 (탈락)', c['_r'], c['_s'], SFT_WEIGHTS_4B, '마지막 ckpt 가중치만 남기고 past로', 'ckpt 결정')
# 4. sft_ot3
add('OT3 SFT run', 'mopd/outputs/sft_ot3/qwen3-1p7b-18k', sz('mopd/outputs/sft_ot3/qwen3-1p7b-18k'), SFT_WEIGHTS_1P7B, '1.7B-OT3: 마지막 ckpt 가중치만 유지 (44 ckpt + optimizer 제거)', 'ckpt 결정')
add('OT3 SFT run', 'mopd/outputs/sft_ot3/qwen3-4b', sz('mopd/outputs/sft_ot3/qwen3-4b'), 0, '최종본은 models/Qwen3-4B-OT3 → run 폴더의 ckpt 전부 제거 가능 (로그만 유지)', 'ckpt 결정')
# 5. logs
add('NeMo-RL 로그', 'mopd_domains/logs', sz('mopd_domains/logs'), 0.02*sz('mopd_domains/logs'), 'rollout 덤프·tensorboard 삭제 또는 past (요약 로그만 유지)', '삭제 승인')
add('NeMo-RL 로그', 'mopd_rl/logs', sz('mopd_rl/logs'), 0.02*sz('mopd_rl/logs'), '동일', '삭제 승인')
# 6. OPD runs: running ones will be stripped; finished: optional intermediate-ckpt pruning (rule says keep all)
run_sz=sum(c['_s'] for c in kids('mopd_domains/outputs/opd'))
running=sum(c['_s'] for c in kids('mopd_domains/outputs/opd') if c['_s']>1.0e11)
add('OPD/MOPD run (진행 중)', 'mopd_domains/outputs/opd/* (진행 중 3개)', running, 3*8.0e10, '완료 후 optimizer state strip (자동)', '없음 (자동)')
add('OPD/MOPD run (완료)', 'mopd_domains/outputs/opd/* (완료 35개)', run_sz-running, run_sz-running, '규칙상 모든 ckpt 유지 (선택: 완료 run의 중간 ckpt 제거 시 약 2.2 TB 추가)', '규칙 유지')
# 7. past
add('past/ (보관)', 'past/models', sz('past/models'), 0, 'HF에서 재다운로드 가능한 base model 25개 → 삭제 시 전량 회수', '삭제 승인')
add('past/ (보관)', 'past/code (conductor·PoC·nt 등)', sz('past/code'), 0.05*sz('past/code'), '코드만 남기고 outputs/data 삭제 시', '삭제 승인')
add('past/ (보관)', 'past/data', sz('past/data'), 0, 'conductor 시대 데이터 (images 652 GB 등)', '삭제 승인')
add('past/ (보관)', 'past/venvs', sz('past/venvs'), 0, '이동으로 이미 못 쓰는 venv', '삭제 승인')
json.dump({'rows':rows,'params':{'nemo_keep':NEMO_KEEP_PER_STEP,'sft4b':SFT_WEIGHTS_4B,'sft1p7b':SFT_WEIGHTS_1P7B}}, open('cleanable.json','w'), ensure_ascii=False)
def fmt(b):
    if b>=1e12: return f"{b/1e12:.2f} TB"
    if b>=1e9: return f"{b/1e9:.1f} GB"
    return f"{b/1e6:.0f} MB"
tot=sum(r[2] for r in rows); keep=sum(r[3] for r in rows)
print('| 분류 | 경로 | 현재 | 유지 후 | 회수 | 조치 | 승인 |'); print('|---|---|---|---|---|---|---|')
agg={}
for cat,p,cur,k,act,ap in rows:
    a=agg.setdefault(cat,[0,0]); a[0]+=cur; a[1]+=k
for cat,(cur,k) in sorted(agg.items(), key=lambda x:-(x[1][0]-x[1][1])):
    print(f"| {cat} | (합계) | {fmt(cur)} | {fmt(k)} | {fmt(cur-k)} | | |")
print(f"\nTOTAL current {fmt(tot)} -> keep {fmt(keep)} -> freeable {fmt(tot-keep)}")
