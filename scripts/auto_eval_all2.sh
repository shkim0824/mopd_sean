#!/bin/bash
# auto_eval_all2.sh -- same as scripts/auto_eval_all.sh, with three fixes (2026-09-22):
#   1. REGISTERS every submission in tmp/guard_registry.tsv, so a job cancelled from outside is
#      resubmitted by guard (auto_eval_all.sh does not, and five 13-bench runs were lost that way).
#   2. skips a checkpoint whose metrics.json already exists or whose job is already queued, so the
#      driver can be restarted safely at any time.
#   3. walltime 6h instead of 4h (the warm-up checkpoints generate long and were hitting the limit).
# A separate file on purpose: scripts/auto_eval_all.sh is being read by running drivers.
# Must live in scripts/ -- _common.sh derives REPO from ${_here}/..
#
#   auto_eval_all2.sh <run> full 50 100 150 200
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"
RUN="$1"; MODE="$2"; shift 2; STEPS="$*"; D="${REPO}/outputs/opd/${RUN}"; cd "${REPO}"

FULL="medqa,casehold,finqa,ifeval,ifbench,aime24,aime25,aime26,lcb_v6,gpqa,harmbench,sycophancy,truthfulqa_mc"
OOD="gpqa,harmbench,sycophancy,truthfulqa_mc"
DOM3="medqa,casehold,finqa"
case "${MODE}" in
  full) BENCH="${FULL}"; PRE=all ;;
  ood)  BENCH="${OOD}";  PRE=ood ;;
  dom3) BENCH="${DOM3}"; PRE=dom3 ;;
  *)    echo "bad mode ${MODE} (full|ood|dom3)"; exit 2 ;;
esac
reg() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" 0 0 >> tmp/guard_registry.tsv; }

for N in ${STEPS}; do
  CK="${D}/checkpoint-${N}"
  while true; do
    if [[ -f "${CK}/trainer_state.json" && -f "${CK}/config.json" && ( -f "${CK}/model.safetensors.index.json" || -f "${CK}/model.safetensors" ) ]]; then sleep 120; break; fi
    if [[ -f "${D}/opd_done.json" && ! -d "${CK}" ]]; then echo "$(date '+%m-%d %H:%M') run finished without checkpoint-${N}; skipping"; continue 2; fi
    sleep 300
  done
  TAG="${PRE}-opd-${RUN}-ck${N}"
  if [[ -f "outputs/eval_all/${TAG}/metrics.json" ]]; then echo "$(date '+%m-%d %H:%M') skip ${TAG} (done)"; continue; fi
  if squeue -h -u deploy -o "%j" 2>/dev/null | grep -qE "evalall-${TAG}$"; then echo "$(date '+%m-%d %H:%M') skip ${TAG} (queued)"; continue; fi
  placement 2
  ENVV="MODEL=${CK} TAG=${TAG} BENCHMARKS=${BENCH}"
  J=$(env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy env ${ENVV} \
      sbatch --parsable -N 2 ${PLACE_OPT} -J "${JP}-evalall-${TAG}" --cpus-per-task=32 --mem=600G -t 6:00:00 scripts/eval_all.sbatch)
  if [[ "${J}" =~ ^[0-9]+$ ]]; then
    reg "${J}" "${TAG}" "outputs/eval_all/${TAG}/metrics.json" \
      "placement 2; ${ENVV} sbatch --parsable -N 2 \${PLACE_OPT} -J \${JP}-evalall-${TAG} --cpus-per-task=32 --mem=600G -t 6:00:00 scripts/eval_all.sbatch"
    echo "$(date '+%m-%d %H:%M') ${TAG} -> ${J} (${PLACE_TIER}, registered)"
  else
    echo "$(date '+%m-%d %H:%M') FAILED ${TAG}: ${J}"
  fi
done
echo "$(date '+%m-%d %H:%M') AUTO-EVAL-ALL2 DONE for ${RUN} steps: ${STEPS}"
