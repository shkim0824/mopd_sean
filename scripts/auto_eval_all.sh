#!/bin/bash
# Evaluate the checkpoints of an OPD/MOPD run as they appear, ONE job per checkpoint
# (login-node loop; run under setsid nohup). Successor of auto_eval.sh: instead of a
# domain job + a 6-bench job it submits scripts/eval_all.sbatch once, so each GPU loads
# the student a single time and serves in-domain + IF + OOD (math, code, safety, GPQA).
#
#   auto_eval_all.sh <run> full 25 50 75 ... 200   -> all 13 benchmarks per ckpt
#   auto_eval_all.sh <run> ood  200                -> safety + GPQA only (already-evaluated ckpts)
#   auto_eval_all.sh <run> dom3 25 ... 200         -> the 3 in-domain benches only
#   run = outputs/opd/<run>;  tags: all-opd-<run>-ck<N> | ood-opd-<run>-ck<N> | dom3-opd-<run>-ck<N>
#
# auto_eval.sh is left in place for the runs already chained to it (editing a script a
# running bash is still reading corrupts it).
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
  tau2) BENCH="tau2"; PRE=tau2 ;;       # official tau2-bench telecom via scripts/eval_tau2.sbatch (own job)
  *)    echo "bad mode ${MODE} (full|ood|dom3)"; exit 2 ;;
esac

for N in ${STEPS}; do
  CK="${D}/checkpoint-${N}"
  while true; do   # a ckpt is complete when trainer_state.json + weights index exist (+ grace)
    if [[ -f "${CK}/trainer_state.json" && -f "${CK}/config.json" && ( -f "${CK}/model.safetensors.index.json" || -f "${CK}/model.safetensors" ) ]]; then sleep 120; break; fi
    if [[ -f "${D}/opd_done.json" && ! -d "${CK}" ]]; then echo "run finished without checkpoint-${N}; skipping"; continue 2; fi
    sleep 300
  done
  TAG="${PRE}-opd-${RUN}-ck${N}"
  placement 2
  MODEL="${CK}" TAG="${TAG}" BENCHMARKS="${BENCH}" \
    sbatch -N 2 ${PLACE_OPT} -J "${JP}-evalall-${TAG}" --cpus-per-task=32 --mem=600G -t 4:00:00 \
    scripts/eval_all.sbatch
  echo "$(date '+%H:%M') submitted ${MODE} eval for ${RUN} step ${N} (tier ${PLACE_TIER})"
done
echo "AUTO-EVAL-ALL DONE for ${RUN} steps: ${STEPS}"
