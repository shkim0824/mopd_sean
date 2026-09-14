#!/bin/bash
# Login-node guard (run detached: setsid nohup bash scripts/guard_jobs.sh > logs/guard_jobs.log 2>&1 &). Every 60 s:
#  A. RANGE GUARD (user rule 2026-09-14): while any of MY jobs pinned to the assigned range n227-236 is still PENDING,
#     every foreign job RUNNING on those nodes is cancelled (a job submitted into a momentarily idle range is otherwise
#     overtaken by higher-priority jobs). Cancellations are logged; a hard cap avoids an endless fight.
#  B. RESUBMIT GUARD: registry tmp/guard_registry.tsv (jobid <TAB> run <TAB> done-marker <TAB> resubmit command). A
#     registered job that left the queue without its done-marker is resubmitted with the stored command (train.resume=auto /
#     NeMo checkpoint resume), at most MAX_TRIES times; the registry line is updated with the new job id.
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"; cd "${REPO}"
REG="${REG:-tmp/guard_registry.tsv}"; MAX_TRIES="${MAX_TRIES:-3}"; CANCEL_CAP="${CANCEL_CAP:-40}"; ncancel=0
log() { echo "$(date '+%m-%d %H:%M') $*"; }
touch "${REG}"
log "guard started (repo ${REPO}, registry ${REG})"
while true; do
  # ---- A. range guard
  pinned=""
  for j in $(squeue -h -t PD -o "%i %j" 2>/dev/null | awk '$2 ~ /^sean-/ {print $1}'); do
    scontrol show job "$j" 2>/dev/null | grep -qF "ExcNodeList=${MINE_EXCL}" && pinned="${pinned} ${j}"
  done
  if [[ -n "${pinned}" ]]; then
    for j in $(squeue -h -t R,CG -o "%i %j" -w "${MINE_RANGE}" 2>/dev/null | grep -v " sean-\| k-sean-" | awk '{print $1}' | sort -u); do
      if (( ncancel >= CANCEL_CAP )); then log "cancel cap ${CANCEL_CAP} reached; not cancelling ${j}"; continue; fi
      info=$(squeue -h -j "$j" -o "%j user=%u %M %D nodes %N" 2>/dev/null)
      if scancel "$j" 2>/dev/null; then ncancel=$((ncancel + 1)); log "RANGE: cancelled foreign job ${j} (${info}) — my pinned pending:${pinned}"; fi
    done
  fi
  # ---- B. resubmit guard
  if [[ -s "${REG}" ]]; then
    tmp="${REG}.new"; : > "${tmp}"
    while IFS=$'\t' read -r jid run marker cmd tries; do
      [[ -z "${jid}" ]] && continue; tries="${tries:-0}"
      if [[ "${jid}" == done-* ]] || squeue -h -j "${jid}" -o %T 2>/dev/null | grep -q .; then printf '%s\t%s\t%s\t%s\t%s\n' "${jid}" "${run}" "${marker}" "${cmd}" "${tries}" >> "${tmp}"; continue; fi
      st=$(sacct -j "${jid}" -X -o State -n 2>/dev/null | head -1 | tr -d ' ')
      if [[ -e "${marker}" ]]; then log "DONE ${run} (job ${jid} ${st}; ${marker} present)"; printf '%s\t%s\t%s\t%s\t%s\n' "done-${jid}" "${run}" "${marker}" "${cmd}" "${tries}" >> "${tmp}"; continue; fi
      if (( tries >= MAX_TRIES )); then log "GIVE UP ${run}: job ${jid} ${st}, ${tries} resubmissions used"; printf '%s\t%s\t%s\t%s\t%s\n' "done-${jid}" "${run}" "${marker}" "${cmd}" "${tries}" >> "${tmp}"; continue; fi
      new=$(bash -c "source scripts/_common.sh; ${cmd}" 2>&1 | grep -oE "[0-9]{6,}" | tail -1)
      if [[ -n "${new}" ]]; then log "RESUBMIT ${run}: job ${jid} ended ${st} without ${marker} -> new job ${new} (try $((tries + 1)))"; printf '%s\t%s\t%s\t%s\t%s\n' "${new}" "${run}" "${marker}" "${cmd}" "$((tries + 1))" >> "${tmp}"
      else log "RESUBMIT FAILED ${run} (job ${jid} ${st}); will retry next cycle"; printf '%s\t%s\t%s\t%s\t%s\n' "${jid}" "${run}" "${marker}" "${cmd}" "$((tries + 1))" >> "${tmp}"; fi
    done < "${REG}"
    mv "${tmp}" "${REG}"
  fi
  sleep 60
done
