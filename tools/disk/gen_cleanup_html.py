import json, html, sys
S="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean"
AFTER=sys.argv[1] if len(sys.argv)>1 else 'dust_d4.json'; BEFORE=sys.argv[2] if len(sys.argv)>2 else None
T=json.load(open('cleanup_tables.json')); C=json.load(open('cleanable.json'))
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
def fmt(b):
    if b is None: return '–'
    if b>=1e12: return f"{b/1e12:.2f} TB"
    if b>=1e9: return f"{b/1e9:.1f} GB"
    if b>=1e6: return f"{b/1e6:.0f} MB"
    return f"{b/1e3:.0f} KB"
def esc(x): return html.escape(str(x))
def kids(path): 
    n=ai.get(path); return sorted(n.get('children',[]), key=lambda c:-c['_s']) if n else []
KEEP={'mopd','mopd_domains','mopd_rl','mopd_sean','models','data','images','.venv','.hf_cache','.uv_cache','.nltk_data','localbin','tmp_enroot'}
def cls_top(name):
    if name=='past': return 'past'
    if name=='mtm': return 'other'
    if name in KEEP: return 'keep'
    return ''
def bar(b, maxb): 
    w=max(0.4, 100*b/maxb) if maxb else 0
    return f'<span class="bar"><i style="width:{w:.1f}%"></i></span>'
def tree_html(path, depth, maxd, minb=1e9):
    ch=[c for c in kids(path) if c['_s']>=minb]
    if not ch: return ''
    maxb=max(c['_s'] for c in ch)
    out=['<ul class="tree">']
    for c in ch:
        name=c['_r'].split('/')[-1]; sub=tree_html(c['_r'], depth+1, maxd, minb) if depth<maxd else ''
        label=f'<span class="sz">{fmt(c["_s"])}</span>{bar(c["_s"],maxb)}<span class="nm">{esc(name)}</span>'
        if sub: out.append(f'<li><details{" open" if depth<=1 else ""}><summary>{label}</summary>{sub}</details></li>')
        else: out.append(f'<li><div class="leaf">{label}</div></li>')
    out.append('</ul>'); return '\n'.join(out)

# ---- cleanable rows -> per-path freeable map
free={r[1]:(r[2]-r[3]) for r in C['rows']}
def freeable_under(path):
    tot=0
    for p,f in free.items():
        pp=p.split(' ')[0]
        if pp==path or pp.startswith(path+'/'): tot+=f
    return tot
def fmt2(b): return fmt(b)
cl_rows=sorted(C['rows'], key=lambda r:-(r[2]-r[3]))
agg={}
for cat,pth,cur,k,act,ap in C['rows']:
    a=agg.setdefault(cat,[0,0,ap]); a[0]+=cur; a[1]+=k
cat_rows=sorted(agg.items(), key=lambda x:-(x[1][0]-x[1][1]))
mxf=max(cur-k for cat,(cur,k,ap) in cat_rows) if cat_rows else 1
cl_html='<div class="tw"><table><thead><tr><th>분류</th><th>현재</th><th>유지 후</th><th>회수 가능</th><th class="barcell">회수 (bar)</th><th>승인</th></tr></thead><tbody>'+''.join(f'<tr><th scope="row">{esc(cat)}</th><td>{fmt(cur)}</td><td>{fmt(k)}</td><td><b>{fmt(cur-k)}</b></td><td class="barcell">{bar(cur-k,mxf)}</td><td class="txt">{esc(ap)}</td></tr>' for cat,(cur,k,ap) in cat_rows)+'</tbody></table></div>'
tot_cur=sum(r[2] for r in C['rows']); tot_keep=sum(r[3] for r in C['rows'])
det_html='<details><summary>경로별 상세 (한 행 = 하나의 폴더)</summary><div class="tw"><table><thead><tr><th>경로</th><th>분류</th><th>현재</th><th>유지 후</th><th>회수</th><th>조치</th><th>승인</th></tr></thead><tbody>'+''.join(f'<tr><th scope="row">{esc(pth)}</th><td class="txt">{esc(cat)}</td><td>{fmt(cur)}</td><td>{fmt(k)}</td><td>{fmt(cur-k)}</td><td class="txt">{esc(act)}</td><td class="txt">{esc(ap)}</td></tr>' for cat,pth,cur,k,act,ap in cl_rows)+'</tbody></table></div></details>'
# ---- folder graph depth 2-3 (excluding mtm) with freeable overlay
def graph_rows(path, depth, maxd, minb=2e9):
    out=[]
    for c in kids(path):
        if c['_s']<minb or c['_r'].split('/')[0]=='mtm' and depth>1: continue
        out.append((depth, c['_r'], c['_s'], freeable_under(c['_r'])))
        if depth<maxd: out+=graph_rows(c['_r'], depth+1, maxd, minb)
    return out
G=[r for r in graph_rows('.',1,3) if not r[1].startswith('mtm/')]
mxg=max(r[2] for r in G) if G else 1
def gbar(total, freeb, mx):
    w=max(0.3,100*total/mx); fw=(100*freeb/total) if total else 0
    return f'<span class="bar gbar" style="width:{w:.1f}%"><i style="width:{fw:.1f}%"></i></span>'
graph_html='<div class="graph">'+''.join(f'<div class="grow d{d}"><span class="gname">{esc(pth.split("/")[-1]) if d>1 else esc(pth)}</span><span class="gsz">{fmt(sz)}</span>{gbar(sz,fr,mxg)}<span class="gfree">{("−"+fmt(fr)) if fr>0 else ""}</span></div>' for d,pth,sz,fr in G)+'</div>'
# ---- mopd_sean tree
sean_tree=tree_html('mopd_sean',1,3,minb=1e6)

total_after=T['total_after']; total_before=T.get('total_before')
# T1 rows with bars
mx=max(a for _,_,a in T['T1'])
t1=''.join(f'<tr class="{cls_top(t)}"><th scope="row">{esc(t)}</th><td>{fmt(b)}</td><td>{fmt(a)}</td><td class="barcell">{bar(a,mx)}</td><td>{ {"keep":"유지 (4B-OT3 계열 / 공용 환경)","past":"보관 (archive)","other":"범위 밖 (MTM)","":"이동됨 → past"}[cls_top(t)] if a>0 else "이동됨 → past"}</td></tr>' for t,b,a in T['T1'])
t3=''.join(f'<tr><th scope="row">{esc(p)}</th><td>{fmt(s)}</td><td>{n}</td><td class="txt">{esc(ro)}</td><td class="txt">{esc(sl)}</td><td class="txt">{esc(pr)}</td></tr>' for p,s,n,ro,sl,pr in T['T3'])
t4=''.join(f'<tr><th scope="row">{esc(p)}</th><td>{fmt(s)}</td></tr>' for p,s in T['T4'])
t5=''.join(f'<tr><th scope="row">{esc(p)}</th><td>{fmt(s)}</td><td class="txt">{esc(w)}</td></tr>' for p,s,w in T['T5'])
sean=''.join(f'<tr><th scope="row">{esc(c["_r"].split("/")[-1])}</th><td>{fmt(c["_s"])}</td><td class="txt">{esc({"mopd":"공통 패키지(mopd/distill = MOPDTrainer), OT3 SFT, 6-bench 평가","mopd_domains":"도메인 OPD/MOPD 드라이버·런처·설정, graders, teacher 궤적 생성, NeMo-RL teacher 설정, third_party 벤치마크 데이터(1.3 GB), 분석 스크립트","mopd_rl":"fin·IF·math·code RL teacher (NeMo-RL) 설정·env·grader vendoring","README.md":"레이아웃·teacher 계보·run 위치·명령","INVENTORY.md":"이 페이지의 표"}.get(c["_r"].split("/")[-1],""))}</td></tr>' for c in kids('mopd_sean'))
page=f"""<title>MOPD 저장소 정리 지도</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--ground:#f6f5f1;--paper:#fff;--ink:#1f2933;--muted:#5b6672;--rule:#d9dcd6;--accent:#0f6b63;--accent-ink:#0a4f49;--warn:#b0561a;--keep:#e6efe9;--past:#efe9dc;--other:#ece9f2;--band:#e6efe9;--barbg:#e4e7e1}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--ground:#151a1f;--paper:#1d242b;--ink:#e6e9ec;--muted:#9aa5b1;--rule:#334049;--accent:#5fd3c5;--accent-ink:#8fe6dc;--warn:#f0a35a;--keep:#1f3a35;--past:#3a3122;--other:#2c2740;--band:#1f3a35;--barbg:#2a333b}}}}
:root[data-theme="dark"]{{--ground:#151a1f;--paper:#1d242b;--ink:#e6e9ec;--muted:#9aa5b1;--rule:#334049;--accent:#5fd3c5;--accent-ink:#8fe6dc;--warn:#f0a35a;--keep:#1f3a35;--past:#3a3122;--other:#2c2740;--band:#1f3a35;--barbg:#2a333b}}
body{{background:var(--ground);color:var(--ink);font-family:"IBM Plex Sans","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;font-size:14px;line-height:1.55;margin:0}}
main{{max-width:1280px;margin:0 auto;padding:32px 28px 64px}} header{{border-bottom:1px solid var(--rule);padding-bottom:18px;margin-bottom:26px}}
h1{{font-size:26px;font-weight:600;margin:0 0 6px;text-wrap:balance}} .sub{{color:var(--muted);margin:0;max-width:90ch}}
.eyebrow{{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin:0 0 4px;font-weight:600}}
h2{{font-size:19px;font-weight:600;margin:0 0 8px}} section{{margin:32px 0}} .lede{{max-width:85ch;margin:0 0 12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:16px 0 0}}
.card{{background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:14px 16px}} .card .k{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 4px}} .card .v{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:24px;font-weight:500;color:var(--accent-ink);margin:0}} .card p{{margin:6px 0 0;color:var(--muted)}}
.tw{{overflow-x:auto;background:var(--paper);border:1px solid var(--rule);border-radius:6px;margin:10px 0 6px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{padding:7px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule);font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px}}
thead th{{background:var(--band);color:var(--accent-ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:12px;font-weight:600;position:sticky;top:0}}
tbody th{{text-align:left;font-family:"IBM Plex Sans","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;font-weight:500;white-space:normal;min-width:200px}}
td.txt{{text-align:left;font-family:"IBM Plex Sans","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;color:var(--muted);white-space:normal;min-width:180px;max-width:420px}}
td.barcell{{width:220px}} .bar{{display:inline-block;width:200px;height:9px;background:var(--barbg);border-radius:2px;vertical-align:middle;margin:0 8px}} .bar i{{display:block;height:100%;background:var(--accent);border-radius:2px}}
tr.keep th,tr.keep td{{background:var(--keep)}} tr.past th,tr.past td{{background:var(--past)}} tr.other th,tr.other td{{background:var(--other)}}
ul.tree{{list-style:none;padding-left:18px;margin:2px 0}} ul.tree>li{{margin:2px 0}} .tree .sz{{display:inline-block;width:82px;text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;color:var(--accent-ink)}} .tree .nm{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px}} .tree summary{{cursor:pointer;list-style:none;padding:2px 0}} .tree summary::before{{content:"▸ ";color:var(--muted)}} .tree details[open]>summary::before{{content:"▾ "}} .tree .leaf{{padding:2px 0 2px 1.1em}}
.treebox{{background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:10px 14px}}
.graph{{background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:10px 14px;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px}} .grow{{display:grid;grid-template-columns:260px 90px 1fr 110px;align-items:center;gap:8px;padding:2px 0}} .grow.d2 .gname{{padding-left:18px;color:var(--ink)}} .grow.d3 .gname{{padding-left:36px;color:var(--muted)}} .grow.d1 .gname{{font-weight:600}} .gsz{{text-align:right;color:var(--accent-ink)}} .gbar{{display:inline-block;height:10px;background:var(--barbg);border-radius:2px;margin:0}} .gbar i{{display:block;height:100%;background:var(--warn);border-radius:2px}} .gfree{{color:var(--warn);text-align:right}}
.note{{color:var(--muted);font-size:12.5px;margin:4px 0 0;max-width:95ch}} .legend span{{display:inline-block;padding:2px 8px;border-radius:3px;margin-right:8px;font-size:12px}}
</style>
<main>
<header><p class="eyebrow">$S = {esc(S)} · dust -d 4 · 2026-09-14</p><h1>MOPD 저장소 정리 지도</h1>
<p class="sub">4B-OT3 OPD/MOPD 실험과 1.7B-OT3만 남기고 나머지(더 큰 teacher, 다른 student, non-think, conductor 시대 자산)는 past/로 옮긴 뒤의 상태. 코드 사본은 mopd_sean/에 통합했고, 4B-OT3 계열 checkpoint는 측정만 했다.</p>
<div class="cards">
<div class="card"><p class="k">공유 폴더 전체</p><p class="v">{fmt(total_before) if total_before else '–'} → {fmt(total_after)}</p><p>before → after (past/ 포함, 삭제 없음)</p></div>
<div class="card"><p class="k">past/ (보관)</p><p class="v">{fmt(ai['past']['_s']) if 'past' in ai else '–'}</p><p>중간 ckpt·optimizer state 제거 후</p></div>
<div class="card"><p class="k">mopd_sean (코드 통합본)</p><p class="v">{fmt(ai['mopd_sean']['_s']) if 'mopd_sean' in ai else '–'}</p><p>mopd + mopd_domains + mopd_rl 코드 사본</p></div>
<div class="card"><p class="k">4B-OT3 계열 checkpoint</p><p class="v">{fmt(sum(s for _,s,_,_,_,_ in T['T3']))}</p><p>측정만 (이동 여부는 사용자 결정)</p></div>
</div></header>
<section><p class="eyebrow">1</p><h2>최상위 폴더 (before → after)</h2><p class="legend"><span style="background:var(--keep)">유지</span><span style="background:var(--past)">past (보관)</span><span style="background:var(--other)">범위 밖 (MTM)</span></p>
<div class="tw"><table><thead><tr><th>dir</th><th>before</th><th>after</th><th>after (bar)</th><th>분류</th></tr></thead><tbody>{t1}</tbody></table></div></section>
<section><p class="eyebrow">2</p><h2>남은 것: 폴더 트리 (1 GB 이상만, 접이식)</h2><div class="treebox">{tree_html('.',1,3)}</div></section>
<section><p class="eyebrow">3</p><h2>mopd_sean 구성</h2><div class="tw"><table><thead><tr><th>folder</th><th>size</th><th>내용</th></tr></thead><tbody>{sean}</tbody></table></div><p class="note">원본 세 repo의 코드만 rsync로 복사한 뒤 절대 경로를 mopd_sean으로 치환(125개 코드 파일)하고, outputs/results/models/data/logs는 원본을 가리키는 symlink로 연결. import 검사와 eval smoke까지 통과(README.md 참고).</p><h3>mopd_sean 트리 (1 MB 이상)</h3><div class="treebox">{sean_tree}</div></section>
<section><p class="eyebrow">4</p><h2>4B-OT3 계열 checkpoint (측정만)</h2><p class="lede">한 행 = 하나의 run 폴더. "참조되는 step"은 teacher/eval용 symlink 폴더(mopd_domains/models, mopd_rl/models)가 가리키는 step. 제안은 용량 관리 관점의 초안이며 실행하지 않았다.</p>
<div class="tw"><table><thead><tr><th>path</th><th>size</th><th>ckpt dirs</th><th>role</th><th>참조되는 step</th><th>제안</th></tr></thead><tbody>{t3}</tbody></table></div></section>
<section><p class="eyebrow">5</p><h2>past/ 내용 (정리 후)</h2><div class="tw"><table><thead><tr><th>path</th><th>size</th></tr></thead><tbody>{t4}</tbody></table></div><p class="note">manifest: $S/past/CLEANUP_2026-09-14.md (이동 목록 + pruning 전후 용량)</p></section>
<section><p class="eyebrow">6</p><h2>정리 가능 용량 (mtm 제외; 측정한 checkpoint 구조 기준)</h2><p class="lede">NeMo-RL step = optimizer 32.2 GB + DCP shard 16.1 GB + consolidated 16.1 GB(fp32); trl SFT checkpoint = optimizer 56.3 GB + 가중치 8.8 GB; 1.7B checkpoint = 24.1 + 4.1 GB; 완료 OPD run = 9 × 8.84 GB. "유지 후"는 teacher가 가리키는 step의 consolidated, 각 run의 마지막(또는 사용한) checkpoint 가중치만 남길 때의 크기.</p><div class="cards"><div class="card"><p class="k">대상 합계</p><p class="v">{fmt(tot_cur)}</p><p>mopd_domains·mopd_rl·mopd·past 안의 정리 후보</p></div><div class="card"><p class="k">유지 후</p><p class="v">{fmt(tot_keep)}</p><p>teacher·최종 가중치·완료 run</p></div><div class="card"><p class="k">회수 가능</p><p class="v">{fmt(tot_cur-tot_keep)}</p><p>승인 종류별 내역은 표 참고</p></div></div>{cl_html}{det_html}<h3>폴더 용량 그래프 (depth 2–3, 2 GB 이상; 주황 = 회수 가능분)</h3>{graph_html}</section>
<section><p class="eyebrow">6b</p><h2>optimizer state만 삭제할 때 (사용자 선호안; 모든 checkpoint의 가중치 유지)</h2><p class="lede">trl/DeepSpeed checkpoint의 <code>global_step*</code>·optimizer.pt·scheduler.pt·rng_state·latest·zero_to_fp32.py, NeMo-RL step의 <code>policy/optimizer</code>는 <b>같은 run을 중단 지점에서 이어갈 때만</b> 필요하다. 평가와 새 학습(SFT·OPD/MOPD·NeMo-RL)은 HF 가중치(config + model*.safetensors + tokenizer)만 읽으므로 영향이 없다 — 이 연구의 OPD branch는 모두 optimizer 없이 저장된(save_only_model) warm-up checkpoint에서 시작했고, teacher 폴더도 consolidated 가중치만 가리킨다. NeMo DCP shard는 같은 가중치의 분산 포맷 중복본(해당 NeMo-RL run 재개 전용)이며 HF consolidated 사본이 step마다 남아 있다. <b>아직 아무것도 삭제하지 않았다.</b></p><div class="tw"><table><thead><tr><th>그룹</th><th>현재</th><th>optimizer state (resume 전용)</th><th>NeMo DCP shard (가중치 중복본, resume 전용)</th><th>optimizer만 삭제 후</th><th>optimizer + DCP 삭제 후</th></tr></thead><tbody><tr><th scope="row">mopd_domains/results</th><td>11.17 TB</td><td>5.94 TB</td><td>2.04 TB</td><td>5.23 TB</td><td>3.19 TB</td></tr><tr><th scope="row">mopd_rl/results</th><td>2.71 TB</td><td>1.19 TB</td><td>595.4 GB</td><td>1.51 TB</td><td>918.2 GB</td></tr><tr><th scope="row">mopd/outputs/sft_ot3</th><td>1.64 TB</td><td>1.06 TB</td><td>0</td><td>581.3 GB</td><td>581.3 GB</td></tr><tr><th scope="row">mopd_domains/outputs/opd</th><td>2.82 TB</td><td>450.5 GB</td><td>0</td><td>2.37 TB</td><td>2.37 TB</td></tr><tr><th scope="row">합계</th><td>18.35 TB</td><td>8.65 TB</td><td>2.64 TB</td><td>9.70 TB</td><td>7.06 TB</td></tr></tbody></table></div><details><summary>run별 상세 (optimizer 또는 DCP가 1 GB 이상인 run)</summary><div class="tw"><table><thead><tr><th>run</th><th>ckpt 수</th><th>현재</th><th>optimizer</th><th>DCP</th><th>optimizer 삭제 후</th></tr></thead><tbody><tr><th scope="row">mopd/outputs/sft_ot3/qwen3-1p7b-18k</th><td>44</td><td>1.24 TB</td><td>1.06 TB</td><td>0</td><td>183.6 GB</td></tr><tr><th scope="row">mopd_rl/outputs/rl/grpo_math_v3_4b</th><td>20</td><td>1.29 TB</td><td>645.2 GB</td><td>321.9 GB</td><td>644.1 GB</td></tr><tr><th scope="row">mopd_domains/outputs/rl/grpo_law_distill_4b</th><td>20</td><td>1.29 TB</td><td>644.5 GB</td><td>321.8 GB</td><td>644.0 GB</td></tr><tr><th scope="row">mopd_domains/outputs/rl/grpo_fin_4b</th><td>20</td><td>1.29 TB</td><td>644.5 GB</td><td>321.8 GB</td><td>644.0 GB</td></tr><tr><th scope="row">mopd_domains/outputs/rl/grpo_med_4b</th><td>20</td><td>1.29 TB</td><td>644.5 GB</td><td>321.8 GB</td><td>644.0 GB</td></tr><tr><th scope="row">mopd_domains/outputs/rl/grpo_law_4b</th><td>20</td><td>1.29 TB</td><td>644.5 GB</td><td>321.8 GB</td><td>644.0 GB</td></tr><tr><th scope="row">mopd_rl/outputs/rl/grpo_code_4b</th><td>17</td><td>1.10 TB</td><td>547.8 GB</td><td>273.6 GB</td><td>547.4 GB</td></tr><tr><th scope="row">mopd_domains/outputs/rl/grpo_law_phase2_4b</th><td>10</td><td>644.2 GB</td><td>322.3 GB</td><td>160.9 GB</td><td>322.0 GB</td></tr><tr><th scope="row">mopd_domains/outputs/opd/med-pg-sftwA561</th><td>6</td><td>391.1 GB</td><td>337.9 GB</td><td>0</td><td>53.2 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c6_4ep</th><td>4</td><td>270.9 GB</td><td>225.3 GB</td><td>0</td><td>45.7 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c4_3ep</th><td>3</td><td>205.3 GB</td><td>168.9 GB</td><td>0</td><td>36.4 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c3_3ep</th><td>3</td><td>204.9 GB</td><td>168.9 GB</td><td>0</td><td>36.0 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c7_3ep</th><td>3</td><td>206.2 GB</td><td>168.9 GB</td><td>0</td><td>37.2 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c6_3ep</th><td>3</td><td>205.8 GB</td><td>168.9 GB</td><td>0</td><td>36.8 GB</td></tr><tr><th scope="row">mopd_domains/results/sft_medo1_endist</th><td>3</td><td>205.0 GB</td><td>168.9 GB</td><td>0</td><td>36.1 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c1_3ep</th><td>3</td><td>204.6 GB</td><td>168.9 GB</td><td>0</td><td>35.7 GB</td></tr><tr><th scope="row">mopd_domains/results/sft_medo1_enmix</th><td>3</td><td>204.5 GB</td><td>168.9 GB</td><td>0</td><td>35.6 GB</td></tr><tr><th scope="row">mopd_domains/results/sft_medo1_en</th><td>3</td><td>204.5 GB</td><td>168.9 GB</td><td>0</td><td>35.5 GB</td></tr><tr><th scope="row">mopd_domains/outputs/rl/grpo_med_v6_4b</th><td>8</td><td>257.6 GB</td><td>0</td><td>128.7 GB</td><td>257.6 GB</td></tr><tr><th scope="row">mopd_domains/outputs/rl/grpo_med_v5_4b</th><td>8</td><td>257.6 GB</td><td>0</td><td>128.7 GB</td><td>257.6 GB</td></tr><tr><th scope="row">mopd_domains/outputs/rl/grpo_med_v4_4b</th><td>8</td><td>257.6 GB</td><td>0</td><td>128.7 GB</td><td>257.6 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_2ep</th><td>2</td><td>141.9 GB</td><td>112.6 GB</td><td>0</td><td>29.3 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_law_2ep</th><td>2</td><td>141.0 GB</td><td>112.6 GB</td><td>0</td><td>28.4 GB</td></tr><tr><th scope="row">mopd_domains/results/sft_med_4b</th><td>2</td><td>140.2 GB</td><td>112.6 GB</td><td>0</td><td>27.6 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_v2a</th><td>2</td><td>139.4 GB</td><td>112.6 GB</td><td>0</td><td>26.7 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c6</th><td>2</td><td>140.6 GB</td><td>112.6 GB</td><td>0</td><td>28.0 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c4</th><td>2</td><td>140.2 GB</td><td>112.6 GB</td><td>0</td><td>27.5 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_v2b</th><td>2</td><td>139.8 GB</td><td>112.6 GB</td><td>0</td><td>27.2 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c3</th><td>2</td><td>139.8 GB</td><td>112.6 GB</td><td>0</td><td>27.1 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c1</th><td>2</td><td>139.5 GB</td><td>112.6 GB</td><td>0</td><td>26.8 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c2</th><td>2</td><td>139.9 GB</td><td>112.6 GB</td><td>0</td><td>27.2 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c1_lr5</th><td>2</td><td>139.5 GB</td><td>112.6 GB</td><td>0</td><td>26.8 GB</td></tr><tr><th scope="row">mopd_domains/outputs/opd/mopd-4dom-128-mixsft</th><td>2</td><td>130.7 GB</td><td>112.6 GB</td><td>0</td><td>18.0 GB</td></tr><tr><th scope="row">mopd_domains/outputs/rl/grpo_med_v2_4b</th><td>6</td><td>193.2 GB</td><td>0</td><td>96.6 GB</td><td>193.2 GB</td></tr><tr><th scope="row">mopd_domains/outputs/rl/grpo_med_v3_4b</th><td>6</td><td>193.2 GB</td><td>0</td><td>96.6 GB</td><td>193.2 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_1ep</th><td>1</td><td>76.7 GB</td><td>56.3 GB</td><td>0</td><td>20.4 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_law_1ep</th><td>1</td><td>75.9 GB</td><td>56.3 GB</td><td>0</td><td>19.6 GB</td></tr><tr><th scope="row">mopd_domains/outputs/sft/sft_distill_med_c5</th><td>1</td><td>74.5 GB</td><td>56.3 GB</td><td>0</td><td>18.2 GB</td></tr><tr><th scope="row">mopd_domains/results/smoke_sft_med</th><td>1</td><td>75.0 GB</td><td>56.3 GB</td><td>0</td><td>18.7 GB</td></tr><tr><th scope="row">mopd_domains/outputs/rl/grpo_med_v4a_4b</th><td>1</td><td>32.2 GB</td><td>0</td><td>16.1 GB</td><td>32.2 GB</td></tr></tbody></table></div></details></section>
<section><p class="eyebrow">7</p><h2>권고 (승인이 필요한 항목)</h2><div class="tw"><table><thead><tr><th>path</th><th>size</th><th>내용</th></tr></thead><tbody>{t5}</tbody></table></div></section>
</main>"""
open('cleanup_map.html','w').write(page); print('html', len(page))
