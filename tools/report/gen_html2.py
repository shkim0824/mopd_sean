import json, html
T=json.load(open('report_tables2.json'))
def esc(x): return html.escape(str(x))
def table(hdr, rows, best_col=None):
    best=None
    if best_col is not None:
        vals=[]
        for r in rows:
            name=str(r[0]); ref=name.startswith('base') or name.startswith('teacher') or 'untrained' in name
            try: vals.append(None if ref else float(str(r[best_col]).rstrip('%')))
            except: vals.append(None)
        cand=[v for v in vals if v is not None]
        if cand: best=max(cand)
    out=[f'<div class="tw"><table><thead><tr>'+''.join(f'<th>{esc(h)}</th>' for h in hdr)+'</tr></thead><tbody>']
    for r in rows:
        name=str(r[0]); ref=name.startswith('base') or name.startswith('teacher') or 'untrained' in name
        cells=[]
        for i,c in enumerate(r):
            s=esc(c)
            if '†' in s: a,b=s.split('†',1); s=f'{a}<span class="tr">†{b}</span>'
            if best_col is not None and i==best_col and best is not None and not ref:
                try:
                    if float(str(c).rstrip('%'))==best: s=f'<span class="best">{s}</span>'
                except: pass
            cells.append(f'<td>{s}</td>' if i else f'<th scope="row">{s}</th>')
        out.append(f'<tr class="{"ref" if ref else ""}">'+''.join(cells)+'</tr>')
    out.append('</tbody></table></div>'); return '\n'.join(out)
def sec(id_, eyebrow, title, lede, body):
    return f'<section id="{id_}"><p class="eyebrow">{eyebrow}</p><h2>{title}</h2>{"<p class=lede>"+lede+"</p>" if lede else ""}{body}</section>'
def details(label, inner): return f'<details><summary>{esc(label)}</summary>{inner}</details>'
curves=''.join(details(lab, table(T['CURVE_HDR'], rows)) for lab,rows in T['CURVES'].items() if rows)
bench_name={'casehold':'CaseHOLD T1.0','medqa':'MedQA T1.0','finqa':'FinQA T1.0'}
sc=''.join(details(lab, table(['ckpt', bench_name[b], 's̃'], rows)) for lab,(b,rows) in T['SC'].items() if rows)
ifc=''.join(details(lab, table(['ckpt','IFEval','IFBench','s̃ IF'], rows)) for lab,rows in T['IFC'].items() if rows)
notation="""
<ul class="notation">
<li><b>모든 in-domain 점수는 T=1.0</b>(MedQA 1,273 · CaseHOLD 5,314 · FinQA, 26,624 토큰 상한)이며 checkpoint = ck200(진행 중인 run은 표시된 ckpt). OOD 6-bench(Math = AIME24–26 평균, Code = LiveCodeBench v6, IFEval, IFBench)는 OT3 표준 평가 설정(32k, Qwen thinking 샘플링 preset)이다.</li>
<li><b>s̃</b> = 도메인별 (student − base)/(teacher − base)의 도메인 평균. base OT3 = 0%, teacher = 100%. 4-dom = med·law·fin·IF 평균(IF = IFEval·IFBench 평균), own domains = 학습한 도메인만. RL-only law teacher를 쓴 run은 그 teacher의 CaseHOLD 69.8을 100%로 둔다.</li>
<li>정확도는 공식 extraction 기준, <span class="tr">†N%</span>는 잘림 비율 5% 이상 주석. 모든 표에 base·teacher 행을 둔다. 한 행 = 한 모델(checkpoint).</li>
<li>의료 prompt pool은 med SFT teacher의 학습 프롬프트 8,384개(MedQA-train 4·5지선다)이며 MedQA-Evol은 제외한 것이 기본이다. "old pool"이라고 표시한 행만 MedQA-Evol 17,888개를 포함한 초기 pool이다.</li>
<li>학생은 Qwen3-4B-OT3(또는 명시된 init). OPD = PG(Â = clip(log π_T − log π_θ, ±5)) 또는 teacher top-k 정규화(Eq. 5), 32 prompts × 200 step, 완료 상한 16,384 토큰, lr 1e-6, 저장 25 step. v2 = truncated 샘플을 loss에서 제외(mask_truncated_completions) + 저장 25 step, 그 외 동일.</li>
</ul>"""
findings="""
<ol class="findings">
<li><b>같은 계보(OT3→GRPO) teacher는 어떤 변형으로도 전이된다.</b> fin·IF는 top-64·top-16·PG 모두 ck75 전후에 teacher 수준(FinQA 75–76, IFEval 76–78)에 도달하고 math·code 손실이 없다.</li>
<li><b>분포가 다른 law teacher(35B-trace SFT→RL)는 전이되지만 step 65–100의 loop blow-up을 거친다.</b> PG는 마스킹 유무(v1·v2)와 무관하게 15–20 step 안에 회복하고, top-64는 마스킹(v2)이 있을 때만, top-16은 어느 쪽도 회복하지 못한다. 최종 CaseHOLD 74–75(s̃ 72–81%), OOD는 teacher(math 38, code 27)보다 훨씬 많이 보존(math 50–51, code 40).</li>
<li><b>의료 SFT teacher → OT3 단일 OPD는 전이가 거의 없다</b>(MedQA 72.5, s̃ 25%; teacher 80.7). 6.4k warm-up에서 분기해도 73.4(33%)에 그친다.</li>
<li><b>MOPD의 병목은 의료이고, 4 teacher uniform merge를 init으로 쓰면 가장 좋다.</b> OT3 init 4-dom PG 128 = s̃ 59%(med s̃ 음수, med loop episode), merged init = <b>81%</b>(MedQA 76.6 · CaseHOLD 74.8 · FinQA 74.0 · IFEval 77.8 · IFBench 54.0 · math 61.2 · code 53.0). merge가 망가뜨린 code(10.2)는 ck50까지 복구된다.</li>
<li><b>Off-policy SFT warm-up은 in-domain이 가장 빠르지만 teacher의 OOD 손상을 그대로 상속한다.</b> law는 6.4k 궤적 1 pass로 77(s̃ 91%)이지만 math 40·code 15가 되고 OPD branch도 복구하지 못한다. 의료의 "0 이득"은 데이터 예산 효과: 6.4k×1 pass는 base, 7.9k×4 epoch는 75.0, 24.4k×4 epoch는 78.7(teacher 80.7).</li>
<li><b>RL-only law teacher 가설(SFT가 OPD를 막는다)</b>: SFT를 뺀 teacher로도 blow-up은 사라지지 않고 step 30에 더 일찍 온다(반복이 아닌 영·중 word salad, 길이 overshoot가 기제). 학생은 ck75에 teacher 수준(69.2 vs 69.8)에 수렴하고 OOD도 teacher와 같아져(math 56.7 · code 51.8 · IFEval 43.8) math·code 손실은 사라지지만 절대 CaseHOLD 상한이 5점 낮다. 3-dom MOPD에서도 law 도메인만 같은 퇴화를 보였고 ck125부터 회복 중이다.</li>
</ol>"""
status="""
<ul class="status">
<li><b>진행 중</b> — 3-dom PG 96(RL-only law + fin + IF): step 약 190 / 200. 완료 시 ck200 T1.0 dom3 + 6-bench가 자동 평가된다. 이후 T0.6 final 평가는 파이프라인에서 제거했다.</li>
<li><b>완료</b> — 단일 OPD: law 9(v1 3, v2 3, warm-up branch 1, RL-only 1, 그 외), fin 3, IF 3, med 4 + old pool 4; MOPD 4(+ old-pool 6은 표에서 제외); warm-up 6.</li>
<li><b>운영</b> — 41–66 밖 노드의 외부 취소 반복(9/13 21:55: 4-dom eval, RL-only law OPD step 40 → ck25 자동 재개; 재개 후 같은 step 30에 blow-up 재현 = state-driven). 모든 checkpoint는 optimizer state만 제거하고 보존.</li>
</ul>"""
page=f"""<title>OPD·MOPD 실험 종합</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--ground:#f6f5f1;--paper:#ffffff;--ink:#1f2933;--muted:#5b6672;--rule:#d9dcd6;--accent:#0f6b63;--accent-ink:#0a4f49;--warn:#b0561a;--best:#0f6b63;--ref:#eef1ec;--band:#e6efe9;}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--ground:#151a1f;--paper:#1d242b;--ink:#e6e9ec;--muted:#9aa5b1;--rule:#334049;--accent:#5fd3c5;--accent-ink:#8fe6dc;--warn:#f0a35a;--best:#5fd3c5;--ref:#222b32;--band:#1f3a35;}}}}
:root[data-theme="dark"]{{--ground:#151a1f;--paper:#1d242b;--ink:#e6e9ec;--muted:#9aa5b1;--rule:#334049;--accent:#5fd3c5;--accent-ink:#8fe6dc;--warn:#f0a35a;--best:#5fd3c5;--ref:#222b32;--band:#1f3a35;}}
body{{background:var(--ground);color:var(--ink);font-family:"IBM Plex Sans","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;font-size:14px;line-height:1.55;margin:0}}
main{{max-width:1380px;margin:0 auto;padding:32px 28px 64px}}
header{{border-bottom:1px solid var(--rule);padding-bottom:20px;margin-bottom:28px}}
h1{{font-size:28px;font-weight:600;margin:0 0 6px;letter-spacing:-0.01em;text-wrap:balance}}
.sub{{color:var(--muted);margin:0;max-width:90ch}}
.eyebrow{{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin:0 0 4px;font-weight:600}}
h2{{font-size:20px;font-weight:600;margin:0 0 8px;text-wrap:balance}}
h3{{font-size:15px;font-weight:600;margin:22px 0 6px}}
section{{margin:34px 0}}
.lede{{max-width:80ch;margin:0 0 14px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;margin:18px 0 0}}
.card{{background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:14px 16px}}
.card .k{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 4px}}
.card .v{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:26px;font-weight:500;color:var(--accent-ink);margin:0}}
.card p{{margin:6px 0 0;color:var(--muted)}}
.tw{{overflow-x:auto;background:var(--paper);border:1px solid var(--rule);border-radius:6px;margin:10px 0 6px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule);font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px}}
thead th{{background:var(--band);color:var(--accent-ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:12px;font-weight:600;position:sticky;top:0}}
tbody th{{text-align:left;font-family:"IBM Plex Sans","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;font-weight:500;position:sticky;left:0;background:var(--paper);white-space:normal;min-width:180px;max-width:360px}}
thead th:first-child{{text-align:left;left:0;z-index:2}}
td:last-child{{text-align:left;font-family:"IBM Plex Sans","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;color:var(--muted);white-space:normal;min-width:180px}}
table.wide td{{white-space:normal}}
tr.ref th,tr.ref td{{background:var(--ref);color:var(--muted)}} tr.ref th{{color:var(--ink)}}
.tr{{color:var(--warn);font-size:11px;margin-left:2px}} .best{{color:var(--best);font-weight:600}}
details{{margin:8px 0;border:1px solid var(--rule);border-radius:6px;background:var(--paper);padding:2px 10px}}
summary{{cursor:pointer;padding:8px 2px;font-weight:500}} details .tw{{border:none;margin:0 0 8px}}
.findings li,.status li,.notation li{{margin:8px 0;max-width:95ch}} .findings{{padding-left:22px}} .status,.notation{{padding-left:20px}}
.note{{color:var(--muted);font-size:12.5px;margin:4px 0 0;max-width:95ch}}
</style>
<main>
<header>
<p class="eyebrow">Qwen3-4B-OT3 student · 4 domain teachers · 2026-09-09 → 09-14 · in-domain T=1.0</p>
<h1>OPD·MOPD 실험 종합</h1>
<p class="sub">단일 도메인 on-policy distillation, multi-teacher MOPD, off-policy SFT warm-up의 전체 결과. 표는 클러스터의 metrics.json에서 직접 생성했고 한 행은 하나의 모델(checkpoint)이다. 모든 in-domain 점수는 T=1.0.</p>
<div class="cards">
<div class="card"><p class="k">최고 MOPD (s̃ 4-dom)</p><p class="v">81%</p><p>4-dom PG 128, merged-teacher init, ck200 — MedQA 76.6 · CaseHOLD 74.8 · FinQA 74.0 · IFEval 77.8 · math 61.2 · code 53.0</p></div>
<div class="card"><p class="k">단일 PG OPD, 같은 계보 teacher</p><p class="v">104% · 89%</p><p>fin PG (FinQA 75.2) · IF PG (IFEval 75.6, IFBench 55.7), OOD 손실 없음</p></div>
<div class="card"><p class="k">단일 PG OPD, 분포가 다른 teacher</p><p class="v">72% · 25%</p><p>law PG v2 (CaseHOLD 74.1, loop 후 회복) · med PG (MedQA 72.5)</p></div>
<div class="card"><p class="k">최고 의료 (SFT warm-up B)</p><p class="v">78.7</p><p>MedQA T1.0, 24.4k med-teacher traces × 4 epoch (teacher 80.7)</p></div>
</div>
</header>
{sec('notation','표기','읽는 법','',notation)}
{sec('pg','요약','단일 도메인 PG OPD 한눈에 보기 (law는 v2)', '각 도메인의 PG run(32 prompts × 200 step, ck200)과 그 teacher. 단일 도메인 run은 자기 도메인 벤치만 측정했다(다른 in-domain 칸은 –). s̃는 자기 도메인 기준.', table(*T['PG'], best_col=9))}
{sec('ref','기준 모델','base · teacher · merged init', 'teacher는 각자 도메인 밖에서는 base보다 나쁘다(s̃ 4-dom 음수 또는 한 자릿수).', table(*T['REF']))}
{sec('law','단일 도메인 OPD · LAW','CaseHOLD (teacher = 35B-trace SFT → GRPO law, 그리고 RL-only law)', '모든 SFT→RL-teacher run이 step 65–100에 loop blow-up을 거친다. v2 = truncated 샘플 loss 제외 + 저장 25 step(상한 16k 동일). 마지막 두 행은 SFT 단계가 없는 RL-only teacher를 쓴 가설 검증.', table(*T['LAW'], best_col=6))}
{sec('fin','단일 도메인 OPD · FIN','FinQA (teacher = OT3 → GRPO fin)', '같은 계보의 teacher: 세 변형 모두 teacher를 넘고 OOD 손실이 없다.', table(*T['FIN'], best_col=6))}
{sec('if','단일 도메인 OPD · IF','IFEval · IFBench (teacher = OT3 → GRPO IF)', 'IFBench 표준오차 약 2.9라 변형 간 차이는 구분되지 않는다.', table(*T['IF'], best_col=5) + '<h3>checkpoint 곡선 (IFEval · IFBench, 6-bench 32k)</h3>' + ifc)}
{sec('med','단일 도메인 OPD · MED','MedQA (teacher = 35B-trace SFT C6-4ep)', 'SFT teacher의 지식은 on-policy distillation으로 거의 옮겨지지 않는다. 아래 두 번째 표는 MedQA-Evol이 포함된 초기 pool로 돌린 변형 비교(참고).', table(*T['MED'], best_col=6) + '<h3>참고: old pool (MedQA-Evol 17,888 포함)</h3>' + table(*T['MED_OLD'], best_col=6))}
{sec('wu','Off-policy SFT warm-up','teacher 궤적으로 SFT → OPD 분기', 'teacher가 OPD pool 프롬프트에 생성한 궤적을 reject filter(정답 letter 일치, 완결, 단일 </think>)로 거른 뒤 SFT. law는 6.4k 궤적 1 pass면 충분(추론 과제)하지만 OOD를 teacher처럼 잃는다. 의료는 25k row-pass 이상 필요(지식 암기)하고, 노출량이 늘수록 in-domain은 teacher에 가까워지며 OOD도 teacher 프로파일이 된다.', '<h3>실험 목록 (recipe)</h3>' + table(*T['WU_REC']).replace('<table>','<table class="wide">') + '<h3>결과 (한 행 = 한 모델)</h3>' + table(*T['WU'], best_col=7))}
{sec('mopd','MOPD','multi-teacher on-policy distillation, ck200 (T1.0 in-domain + 6-bench OOD)', '학생 OT3(또는 merged init) ← 여러 teacher 동시 distillation, 도메인 균형 배치(K prompts per domain), PG, 200 step. 의료 pool은 SFT-teacher 프롬프트(기본). 마지막 행은 진행 중인 RL-only-law 3-domain run(ck150).', table(*T['MOPD'], best_col=11) + ('<h3>ck200이 아닌 checkpoint가 더 좋은 run</h3>' + table(*T['BEST'], best_col=11) if T['BEST'][1] else '') + '<h3>checkpoint 곡선 (한 행 = 한 checkpoint)</h3>' + curves)}
{sec('scurves','checkpoint 곡선 · 단일 OPD와 warm-up','in-domain T1.0', '', sc)}
{sec('findings','발견','지금까지의 결론', '', findings)}
{sec('status','상태','진행 중 · 완료 · 운영', '', status)}
</main>
"""
open('mopd_report.html','w').write(page); print('html', len(page))
