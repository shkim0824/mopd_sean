#!/bin/bash
# Login-node NODE MANAGER + guard (run detached: setsid nohup bash scripts/guard_jobs.sh >> logs/guard_jobs.log 2>&1 &). Every 60 s
# it re-places my PENDING jobs according to Sean's rules (2026-09-14/15) and keeps killed training runs alive:
#   tier 1  n227-236  first priority: the whole range should run my jobs. Pending jobs of mine are pulled in as soon as nodes
#           there are idle or held by foreign jobs (pin via ExcNodeList, rename k-sean-* -> sean-*); foreign jobs on the range
#           are cancelled while a pinned job of mine is pending; starts are logged.
#   tier 2  n041-066  second priority: pending jobs are pinned there (k-sean-*) when it has idle room; NO job is ever cancelled
#           there; a job that still waits GRACE seconds after being pinned there is RELEASED to tier 3 (and may be pulled back
#           into tier 1/2 later; COOLDOWN before re-pinning to tier 2).
#   tier 3  rest of the cluster, no pin — but never n033-036 (NEVER_EXCL) and, operationally, not the kill zone (KILL_ZONE).
#   B       registry tmp/guard_registry.tsv (jobid, run, done-marker, resubmit cmd, tries<=MAX_TRIES): a registered training job
#           that left the queue without its done-marker is resubmitted (NeMo / trainer resume from the latest checkpoint).
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"; cd "${REPO}"
# tier-1 policy (user, 2026-09-17): every pending job of mine is pulled into n227-236 when the range can be made
# free, and foreign jobs holding range nodes are cancelled regardless of their size. Do not narrow this on my own.
REG="${REG:-tmp/guard_registry.tsv}"; MAX_TRIES="${MAX_TRIES:-3}"; RESUBMIT_DELAY="${RESUBMIT_DELAY:-600}"; declare -A delay_logged=() cpu_skip_logged=() CANCEL_CAP="${CANCEL_CAP:-100}"; GRACE="${GRACE:-300}"; COOLDOWN="${COOLDOWN:-900}"
CANCEL_WIN="${CANCEL_WIN:-600}"; CANCEL_WIN_MAX="${CANCEL_WIN_MAX:-6}"; backoff_until=0; cancel_times=()
BOTH_EXCL="ib-a100-cluster-a-n[001-040,067-226,237-258]"      # legacy: pinned to both of my ranges (treated as tier 2)
T3_EXCL="$(excl_union "${NEVER_EXCL}" "${KILL_ZONE}")"
ncancel=0
log() { echo "$(date '+%m-%d %H:%M') $*"; }
touch "${REG}"
declare -A tracked pinned1_at pinned2_at released_at confirm_miss
range_nodes=" $(scontrol show hostnames "${MINE_RANGE}" 2>/dev/null | tr '\n' ' ') "
tier_of() { case "$1" in "${MINE_EXCL}") echo 1;; "${RANGE_EXCL}"|"${BOTH_EXCL}") echo 2;; *) echo 3;; esac; }
covers() { [[ " $(scontrol show hostnames "$1" 2>/dev/null | tr '\n' ' ') " == *" $2 "* ]]; }   # covers <excl-list> <host>
log "node manager v3.14 (v3.13 + no cancel cap + CPU-only foreign jobs never cancelled) started (repo ${REPO}; tier1 ${MINE_RANGE}, tier2 ${RANGE_41_66}, tier3 excludes ${T3_EXCL}; grace ${GRACE}s cooldown ${COOLDOWN}s)"
while true; do
  now=$(date +%s)
  if [[ "${NODE_MANAGER:-1}" == 1 ]]; then
  # ---- inventory: my pending jobs (oldest first)
  P=()
  while read -r jid jname; do
    [[ -z "${jid}" ]] && continue
    info=$(scontrol show job "${jid}" 2>/dev/null) || continue
    reason=$(grep -oE "Reason=[^ ]*" <<<"${info}" | head -1 | cut -d= -f2)
    [[ "${reason}" == *Held* || "${reason}" == *Dependency* ]] && continue
    # v3.11 (Sean 2026-09-17 22:00, "33-34 job은 sean-으로, 이번만"): a job I pinned BY HAND with -w (ReqNodeList set) is left alone —
    # no tier pull, no T2 pin, no T3 exclusion fix (any ExcNodeList edit would contradict its ReqNodeList and strand it).
    req=$(grep -oE "ReqNodeList=[^ ]*" <<<"${info}" | head -1 | cut -d= -f2); [[ -n "${req}" && "${req}" != "(null)" ]] && continue
    n=$(grep -oE "NumNodes=[0-9]+" <<<"${info}" | head -1 | cut -d= -f2)
    excl=$(grep -oE "ExcNodeList=[^ ]*" <<<"${info}" | head -1 | cut -d= -f2); [[ "${excl}" == "(null)" ]] && excl=""
    sub=$(grep -oE "SubmitTime=[^ ]*" <<<"${info}" | head -1 | cut -d= -f2); age=$(( now - $(date -d "${sub}" +%s 2>/dev/null || echo "${now}") ))
    P+=("${jid}|${jname}|${n}|${excl}|${age}")
  done < <(squeue -h -t PD -o "%i %j" -S V 2>/dev/null | awk '$2 ~ /^(k-)?sean-/')
  # ---- capacities (GPU occupancy, not node state). Sean 2026-09-15: CPU-only jobs do not compete with my GPU jobs ->
  #      they are never counted as occupying a range node and never cancelled. A range node is FREE when no GPU job runs
  #      on it; FOREIGN-HELD when a foreign GPU job runs on it and none of my GPU jobs does.
  declare -A node_mine=() node_foreign=()
  while IFS='|' read -r jid jname jnodes; do
    [[ -z "${jid}" ]] && continue
    tres=$(scontrol show job "${jid}" 2>/dev/null | grep -oE "TRES=[^ ]*" | head -1)
    [[ "${tres}" == *gres/gpu* ]] || continue                      # CPU-only: ignore
    mine=0; [[ "${jname}" =~ ^(k-)?sean- ]] && mine=1
    for h in $(scontrol show hostnames "${jnodes}" 2>/dev/null); do
      [[ "${range_nodes}" == *" ${h} "* ]] || continue
      if (( mine == 1 )); then node_mine[${h}]=1; else node_foreign[${h}]="${node_foreign[${h}]:-} ${jid}"; fi
    done
  done < <(squeue -h -t R -w "${MINE_RANGE}" -o "%i|%j|%N" 2>/dev/null)
  idle1=0; foreign1=0; foreign_jobs=""
  for h in ${range_nodes}; do
    if [[ -n "${node_mine[${h}]:-}" ]]; then continue; fi
    if [[ -n "${node_foreign[${h}]:-}" ]]; then foreign1=$((foreign1 + 1)); foreign_jobs="${foreign_jobs} ${node_foreign[${h}]}"; else idle1=$((idle1 + 1)); fi
  done
  foreign_jobs=$(echo "${foreign_jobs}" | tr ' ' '\n' | grep . | sort -u | tr '\n' ' ')
  idle2=$(sinfo -h -n "${RANGE_41_66}" -t idle -o "%D" 2>/dev/null | head -1); idle2=${idle2:-0}
  demand1=0; demand2=0
  for e in "${P[@]}"; do IFS='|' read -r jid jname n excl age <<<"${e}"; t=$(tier_of "${excl}"); (( t == 1 )) && demand1=$((demand1 + n)); (( t == 2 )) && demand2=$((demand2 + n)); done
  cap1=$(( idle1 + foreign1 - demand1 )); cap2=$(( idle2 - demand2 ))
  t1_pending=""
  # ---- pass over pending jobs, oldest first
  for i in "${!P[@]}"; do
    IFS='|' read -r jid jname n excl age <<<"${P[$i]}"; t=$(tier_of "${excl}")
    # tier 1 pull
    if (( t != 1 )) && (( n <= cap1 )); then
      if scontrol update JobId="${jid}" ExcNodeList="${MINE_EXCL}" 2>/dev/null; then
        newname="${jname}"; [[ "${jname}" == k-sean-* ]] && { newname="sean-${jname#k-sean-}"; scontrol update JobId="${jid}" JobName="${newname}" 2>/dev/null || true; }
        log "T1 PIN: job ${jid} (${newname}, ${n} nodes) -> ${MINE_RANGE} [idle ${idle1}, foreign-held ${foreign1}, demand ${demand1}]"
        cap1=$((cap1 - n)); (( t == 2 )) && demand2=$((demand2 - n)); t=1; tracked[${jid}]="T1"; pinned1_at[${jid}]=${now}
      fi
    fi
    if (( t == 1 )); then
      # tier-1 pinned but the range is full of MY OWN jobs (nothing idle, nothing foreign to cancel): after GRACE, release the
      # job to tier 3 so it does not sit for hours; it is pulled back as soon as range capacity appears.
      since1=${pinned1_at[${jid}]:-$(( now - age ))}
      if (( idle1 + foreign1 == 0 )) && (( now - since1 > GRACE )); then
        if scontrol update JobId="${jid}" ExcNodeList="${T3_EXCL}" 2>/dev/null; then
          newname="${jname}"; [[ "${jname}" == sean-* ]] && { newname="k-sean-${jname#sean-}"; scontrol update JobId="${jid}" JobName="${newname}" 2>/dev/null || true; }
          log "T1 RELEASE: job ${jid} (${newname}, ${n} nodes) waited $(( (now - since1) / 60 )) min; ${MINE_RANGE} fully held by my own jobs -> whole cluster (excl ${T3_EXCL})"
          unset "pinned1_at[${jid}]"; tracked[${jid}]="T3"; demand1=$((demand1 - n)); t=3
        fi
      fi
      if (( t == 1 )); then t1_pending="${t1_pending} ${jid}"; tracked[${jid}]="${tracked[${jid}]:-T1}"; continue; fi
    fi
    # tier 2 pin (no cancels here, ever)
    if (( t == 3 )) && (( n <= cap2 )) && (( now - ${released_at[${jid}]:-0} > COOLDOWN )); then
      if scontrol update JobId="${jid}" ExcNodeList="${RANGE_EXCL}" 2>/dev/null; then
        log "T2 PIN: job ${jid} (${jname}, ${n} nodes) -> ${RANGE_41_66} [idle ${idle2}, demand ${demand2}]"
        cap2=$((cap2 - n)); pinned2_at[${jid}]=${now}; t=2; tracked[${jid}]="T2"; continue
      fi
    fi
    # tier 2 release after the grace period (pinned at submission: measured from submit time)
    if (( t == 2 )); then
      since=${pinned2_at[${jid}]:-$(( now - age ))}
      if (( now - since > GRACE )); then
        if scontrol update JobId="${jid}" ExcNodeList="${T3_EXCL}" 2>/dev/null; then
          log "T3 RELEASE: job ${jid} (${jname}, ${n} nodes) waited $(( (now - since) / 60 )) min on ${RANGE_41_66} [idle ${idle2}] -> whole cluster (excl ${T3_EXCL})"
          released_at[${jid}]=${now}; unset "pinned2_at[${jid}]"; tracked[${jid}]="T3"; cap2=$((cap2 + n))
        fi
      fi
      continue
    fi
    # tier 3 hygiene: never n033-036 (+ kill zone)
    if (( t == 3 )) && ! covers "${excl}" "ib-a100-cluster-a-n033"; then
      fixed=$(excl_union "${excl}" "${T3_EXCL}")
      scontrol update JobId="${jid}" ExcNodeList="${fixed}" 2>/dev/null && log "T3 FIX: job ${jid} (${jname}) exclusion ${excl:-none} -> ${fixed}"
    fi
  done
  # ---- tier 1 cancels: while a pinned job of mine pends, clear the foreign jobs off the range — running ones AND pending
  #      ones the scheduler has PLANNED onto range nodes (Sean 2026-09-14: "plan한 job이 있다면 그것도 종료"; a planned foreign
  #      job holds the idle nodes in state "planned" and pushes my pinned job's start out by days)
  # cancel rate limit (2026-09-15 17:45 incident: freed range nodes were re-taken by fresh 1-node foreign jobs every minute
  # while my pinned evals never started -> a cancel per minute with no effect). More than CANCEL_WIN_MAX cancels in
  # CANCEL_WIN seconds => BACKOFF: no cancels for CANCEL_WIN and the tier-1 pins are released to tier 3.
  recent=(); for t in "${cancel_times[@]}"; do (( now - t < CANCEL_WIN )) && recent+=("$t"); done; cancel_times=("${recent[@]}")
  if (( now < backoff_until )); then t1_cancel_ok=0; else t1_cancel_ok=1; fi
  if (( t1_cancel_ok == 1 )) && (( ${#cancel_times[@]} >= CANCEL_WIN_MAX )) && [[ -n "${t1_pending}" ]]; then
    backoff_until=$(( now + CANCEL_WIN )); t1_cancel_ok=0
    log "BACKOFF: ${#cancel_times[@]} cancels in the last $(( CANCEL_WIN / 60 )) min without my pinned jobs starting -> no cancels for $(( CANCEL_WIN / 60 )) min; releasing tier-1 pins:${t1_pending}"
    for rj in ${t1_pending}; do
      if scontrol update JobId="${rj}" ExcNodeList="${T3_EXCL}" 2>/dev/null; then
        rn=$(squeue -h -j "${rj}" -o %j 2>/dev/null); [[ "${rn}" == sean-* ]] && scontrol update JobId="${rj}" JobName="k-sean-${rn#sean-}" 2>/dev/null || true
        unset "pinned1_at[${rj}]"; tracked[${rj}]="T3"; fi
    done
    t1_pending=""
  fi
  if [[ -n "${t1_pending}" ]] && (( t1_cancel_ok == 1 )); then
    for fj in ${foreign_jobs}; do
      if ! scontrol show job "${fj}" 2>/dev/null | grep -oE "TRES=[^ ]*" | head -1 | grep -q "gres/gpu"; then
        [[ -z "${cpu_skip_logged[${fj}]:-}" ]] && { cpu_skip_logged[${fj}]=1; log "CPU-ONLY: not cancelling foreign job ${fj} ($(squeue -h -j "${fj}" -o "%j user=%u %N" 2>/dev/null))"; }
        continue
      fi
      info=$(squeue -h -j "${fj}" -o "%j user=%u %M %D nodes %N" 2>/dev/null)
      if scancel "${fj}" 2>/dev/null; then ncancel=$((ncancel + 1)); cancel_times+=("${now}"); log "RANGE: cancelled foreign job ${fj} (${info}) — my pinned pending:${t1_pending}"; fi
    done
    while read -r pj pname psched; do
      [[ -z "${pj}" ]] && continue; [[ "${pname}" =~ ^(k-)?sean- ]] && continue
      hit=""; for h in $(scontrol show hostnames "${psched}" 2>/dev/null); do [[ "${range_nodes}" == *" ${h} "* ]] && hit="${hit} ${h}"; done
      [[ -z "${hit}" ]] && continue
      scontrol show job "${pj}" 2>/dev/null | grep -oE "TRES=[^ ]*" | head -1 | grep -q "gres/gpu" || continue   # CPU-only planned jobs: ignore
      if scancel "${pj}" 2>/dev/null; then ncancel=$((ncancel + 1)); cancel_times+=("${now}"); log "PLANNED: cancelled foreign pending job ${pj} (${pname}) planned onto${hit} — my pinned pending:${t1_pending}"; fi
    done < <(scontrol show jobs 2>/dev/null | awk '/^JobId=/{id=$1; sub("JobId=","",id); name=$2; sub("JobName=","",name)} /JobState=/{st=$1} /SchedNodeList=/{for(i=1;i<=NF;i++) if($i ~ /^SchedNodeList=/){s=$i; sub("SchedNodeList=","",s)}} /^$/{ if (st ~ /PENDING/ && s != "" && s !~ /null/) print id, name, s; s=""; st="" }')
  fi
  # ---- start tracking
  for jid in "${!tracked[@]}"; do
    st=$(squeue -h -j "${jid}" -o "%T|%N" 2>/dev/null)
    case "${st}" in
      PENDING*) ;;
      RUNNING*|COMPLETING*) log "START (${tracked[${jid}]}): my job ${jid} running on ${st#*|}"; unset "tracked[${jid}]" ;;
      "") log "GONE (${tracked[${jid}]}): my job ${jid} left the queue ($(sacct -j "${jid}" -X -o State -n 2>/dev/null | head -1 | tr -d ' '))"; unset "tracked[${jid}]" ;;
    esac
  done
  else
    (( $(date +%s) % 3600 < 60 )) && log "node manager DISABLED (NODE_MANAGER=0): no tier pulls, no cancellations; registry guard only"
  fi
  # ---- B. resubmit guard
  if [[ -s "${REG}" ]]; then
    tmp="${REG}.new"; : > "${tmp}"
    while IFS=$'\t' read -r jid run marker cmd tries banked; do
      [[ -z "${jid}" ]] && continue; tries="${tries:-0}"; banked="${banked:-0}"
      # results already on disk for this run (mopd.eval.run_all banks every chunk); used below to
      # tell a partial success (TIMEOUT with progress) from a job that is simply broken
      now_banked=$(cat "$(dirname "${marker}")"/_shards/out_*.jsonl 2>/dev/null | wc -l | tr -d " ")
      now_banked="${now_banked:-0}"
      if [[ "${jid}" == done-* || "${jid}" == stalled-* || "${jid}" == dup-* ]] || squeue -h -j "${jid}" -o %T 2>/dev/null | grep -q .; then printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${jid}" "${run}" "${marker}" "${cmd}" "${tries}" "${banked}" >> "${tmp}"; continue; fi
      st=$(sacct -j "${jid}" -X -o State -n 2>/dev/null | head -1 | tr -d ' ')
      ls -a "$(dirname "${marker}")" >/dev/null 2>&1   # readdir: refresh the NFS attribute cache
      if [[ -e "${marker}" ]]; then log "DONE ${run} (job ${jid} ${st}; ${marker} present)"; printf '%s\t%s\t%s\t%s\t%s\t%s\n' "done-${jid}" "${run}" "${marker}" "${cmd}" "${tries}" "${now_banked}" >> "${tmp}"; unset 'confirm_miss[${jid}]'; continue; fi
      # a marker written in the same second the job left the queue can be invisible here for a
      # cycle (NFS), and resubmitting then duplicates a COMPLETED run -> require two sightings
      if [[ -z "${confirm_miss[${jid}]:-}" ]]; then confirm_miss[${jid}]=1; log "MISS? ${run} (job ${jid} ${st}; ${marker} not visible yet -- confirming next cycle)"; printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${jid}" "${run}" "${marker}" "${cmd}" "${tries}" "${banked}" >> "${tmp}"; continue; fi
            # a resubmission that made progress is a RESUME, not a retry: only a run that banked
      # nothing new since its last attempt counts against MAX_TRIES
      # one live job per run: if another registry row for the SAME marker still has a job in the
      # queue (a chain or a hand resubmission already took over), park this row instead of
      # submitting a second writer into the same output dir
      live_other=$(awk -F'\t' -v m="${marker}" -v me="${jid}" '$3==m && $1!=me && $1 !~ /^(done|stalled|dup)-/ {print $1}' "${REG}" | while read -r oj; do squeue -h -j "${oj}" -o %T 2>/dev/null | grep -q . && echo "${oj}"; done | head -1)
      if [[ -n "${live_other}" ]]; then log "DUP ${run}: job ${jid} ${st} but job ${live_other} is already running this run -> parked"; printf '%s\t%s\t%s\t%s\t%s\t%s\n' "dup-${jid}" "${run}" "${marker}" "${cmd}" "${tries}" "${now_banked}" >> "${tmp}"; continue; fi
      # v3.12 (Sean 2026-09-17 22:40): a job CANCELLED by someone else is usually a node owner clearing nodes for a big job. Excluding
      # nodes would be too broad (and the zone moves), so just wait RESUBMIT_DELAY before resubmitting; by then the owner's job
      # normally holds those nodes and the resubmission lands elsewhere.
      if [[ "${st}" == CANCELLED* ]]; then
        end_ts=$(date -d "$(sacct -j "${jid}" -X -o End -n 2>/dev/null | head -1 | tr -d ' ')" +%s 2>/dev/null || echo 0)
        if (( now - end_ts < RESUBMIT_DELAY )); then
          [[ -z "${delay_logged[${jid}]:-}" ]] && { delay_logged[${jid}]=1; log "WAIT ${run}: job ${jid} ${st} (cancelled by someone) -> resubmit after $(( RESUBMIT_DELAY / 60 )) min (about $(date -d @$((end_ts + RESUBMIT_DELAY)) +%H:%M))"; }
          printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${jid}" "${run}" "${marker}" "${cmd}" "${tries}" "${banked}" >> "${tmp}"; continue
        fi
      fi
      next_tries="${tries}"; kind="RESUBMIT"
      if (( now_banked > banked )); then kind="RESUME"; else next_tries=$(( tries + 1 )); fi
      [[ "${st}" == CANCELLED* ]] && next_tries="${tries}"   # v3.13: killed by someone else -> not the run's fault, no retry consumed
      # A run that used its ENTIRE walltime and still banked nothing new is not flaky: a single
      # chunk does not fit the limit, so the identical command banks nothing again. Park it
      # instead of spending MAX_TRIES x walltime x nodes rediscovering that. A run that died
      # early with no progress keeps the old retry path.
      if (( now_banked <= banked )) && [[ "${st}" == TIMEOUT* ]]; then
        log "STALLED ${run}: job ${jid} ${st} used its full walltime and banked nothing new (${now_banked}); an identical resubmission cannot help -- needs a smaller chunk or a longer limit"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "stalled-${jid}" "${run}" "${marker}" "${cmd}" "${tries}" "${now_banked}" >> "${tmp}"; continue
      fi
      if [[ "${kind}" == "RESUBMIT" ]] && (( tries >= MAX_TRIES )); then log "GIVE UP ${run}: job ${jid} ${st}, ${tries} resubmissions with no new results (banked ${now_banked})"; printf '%s\t%s\t%s\t%s\t%s\t%s\n' "done-${jid}" "${run}" "${marker}" "${cmd}" "${tries}" "${now_banked}" >> "${tmp}"; continue; fi
      new=$(bash -c "source scripts/_common.sh; ${cmd}" 2>&1 | grep -oE "[0-9]{6,}" | tail -1)
      if [[ -n "${new}" ]]; then log "${kind} ${run}: job ${jid} ended ${st} without ${marker} -> new job ${new} (try ${next_tries}/${MAX_TRIES}, banked ${banked} -> ${now_banked})"; printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${new}" "${run}" "${marker}" "${cmd}" "${next_tries}" "${now_banked}" >> "${tmp}"
      else log "RESUBMIT FAILED ${run} (job ${jid} ${st}); will retry next cycle"; printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${jid}" "${run}" "${marker}" "${cmd}" "$((tries + 1))" "${now_banked}" >> "${tmp}"; fi
    done < "${REG}"
    mv "${tmp}" "${REG}"
  fi
  sleep 60
done
