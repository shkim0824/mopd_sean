# tools/watch — the tracking loop (학습 추적 → 결과 수신 → 후속 제출 → 재제출)

> **범용 워처는 `$S/tools/watch/` 로 옮겼다** (모든 프로젝트의 학습·평가에 사용: `wait_events.sh`,
> `poll_results.sh`, `poll_remote.sh` — `REPO=`/`EVAL_GLOBS=` 로 레포를 지정). 이 디렉터리에는 MOPD 전용
> 워처만 남긴다(`wait_opd_done.sh`, `watch_opd_resub.sh`, `mopd_finals.sh`, `poll_opd_many_cpu.sh`,
> `watch_dom.sh`, `watch_full.sh`). 아래 1절의 클러스터 쪽 자동화는 MOPD 레포 기준 설명이다.

이 디렉터리는 **랩톱 쪽 알림 도구**이고, 실제 자동화는 **클러스터 로그인 노드에서 detached로 도는 루프**들이다.
랩톱/VPN/세션이 끊겨도 클러스터 쪽은 계속 굴러가고, 다시 붙으면 워처만 재무장하면 복구된다.

## 1. 클러스터 쪽 (내가 없어도 굴러가는 부분)

| 구성요소 | 역할 |
|---|---|
| `scripts/guard_jobs.sh` (node manager v3.5, 60 s 루프) | pending job 재배치(T1/T2/T3) + 범위 내 외부 GPU job 취소 + **레지스트리 기반 재제출** |
| `tmp/guard_registry.tsv` | `jobid <TAB> run <TAB> done-marker <TAB> 재제출 커맨드 <TAB> tries` — job이 done-marker 없이 끝나면 guard가 ≤3회 재제출. 학습은 최신 ckpt resume, eval은 완료 shard 재사용 |
| `scripts/auto_eval.sh` / `sft_auto_eval.sh` / `rl_auto_eval.sh` | 학습이 체크포인트를 저장할 때마다 해당 ckpt 평가를 제출(+레지스트리 등록). `rl_auto_eval.sh`는 "다음 step"을 기다리므로 **마지막 ckpt는 수동 변환/제출 필요** |
| `tmp/phase2/*_chain.sh` (`law_warmup_chain.sh`, `mix_chain.sh`, `start_chains.sh`) | 단계 전이: trace 생성 → reject filter → SFT → ckpt 평가 → 포화 규칙 → 분기 OPD 제출. idempotent (로그인 노드 재시작 후 그냥 다시 실행) |
| `tmp/phase2/finals_watch.sh <run>` | 학습 종료(`opd_done.json` + `final/`) 감지 → optimizer state strip → final 6-bench 제출 + 등록 |
| `scripts/rl_ihc_loop.sh` | 다중 라운드 루프(공격 합성 → GRPO → HF 뷰 변환 → 교사 고정), `SYNTH_TRIES`/`GRPO_TRIES`로 외부 kill 재시도 |

## 2. 랩톱 쪽 (세션을 깨우는 부분) — 전부 **1회 발사 후 재무장** 방식

| 도구 | 트리거 |
|---|---|
| `wait_events.sh` | 한 번의 ssh로 여러 신호를 동시에 감시: `WATCH_JOBS` 중 큐에서 사라진 job, `WATCH_METRICS`의 `metrics.json` 등장, `DRIVERS` 로그의 패턴 수 변화, guard 로그의 `RESUBMIT/GIVE UP/BACKOFF/RANGE:` 줄 수 변화. ssh 3연속 실패 시 exit 2 (Kerberos 만료 → kinit 후 재무장) |
| `poll_tree_results.sh` + `poll_tree_remote.sh` | cpu-instance 경유(=Kerberos 티켓이 만료돼도 동작)로 5분마다 결과 스냅샷을 떠서 STATE 파일과 비교 → **새 결과만** 출력하고 종료 |
| `wait_opd_done.sh` / `watch_dom.sh` / `watch_full.sh` | 특정 run/job 하나를 기다려 결과(도메인 정확도·truncation / 6-bench)를 출력 |
| `watch_opd_resub.sh` | job 종료 시 `final/`이 없으면 자동 재제출(최대 3회) |
| `poll_opd_many_cpu.sh` / `mopd_finals.sh` | 여러 OPD run의 auto-eval 로그·결과 동시 감시 / 종료된 run의 strip + 최종 평가 |

## 3. 알림 처리 규칙 (이 순서를 지킬 때만 루프가 닫힌다)

1. 알림 도착 → **출력 파일을 읽는다**(워처는 무엇이 바뀌었는지만 출력한다).
2. **queue-exit ≠ 성공**: `sacct State` + done-marker를 둘 다 확인한다.
3. 결과를 해석해서 `PIPELINE_STATE.md`(+ 해당 메모리)에 한 줄 기록하고, 요청이 없어도 사용자에게 보고한다.
4. 필요한 후속 작업을 제출한다(다음 ckpt 평가, finals, 다음 단계) — 제출할 때마다 레지스트리에 등록한다.
5. **워처를 다시 무장한다.** 워처는 설계상 첫 이벤트에서 종료하므로, 재무장하지 않으면 추적이 조용히 끊긴다.
