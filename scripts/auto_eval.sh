#!/bin/bash
# Evaluate the checkpoints of an OPD/MOPD run as they appear (login-node loop; run under setsid nohup):
#   auto_eval.sh <run> dom3 25 50 ... 200           -> MedQA+CaseHOLD+FinQA (T1.0) per ckpt; 6-bench every 50 steps (MOPD default)
#   auto_eval.sh <run> bench:medqa 25 50 ... 200    -> one domain bench per ckpt (single-domain OPD: medqa | casehold | finqa | ...)
#   auto_eval.sh <run> if 25 50 ... 200             -> IFEval+IFBench per ckpt via the 6-bench script (IF OPD)
#   run = outputs/opd/<run>; tags: opd-<run>-ck<N>-<dom3|bench>-T1, full-opd-<run>-ck<N>-32k, full-opd-<run>-ck<N>-if
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"
RUN="$1"; MODE="$2"; shift 2; STEPS="$*"; D="${REPO}/outputs/opd/${RUN}"; cd "${REPO}"
for N in ${STEPS}; do
  CK="${D}/checkpoint-${N}"
  while true; do   # a ckpt is complete when trainer_state.json + weights index exist (+ grace period)
    if [[ -f "${CK}/trainer_state.json" && -f "${CK}/config.json" && ( -f "${CK}/model.safetensors.index.json" || -f "${CK}/model.safetensors" ) ]]; then sleep 120; break; fi
    if [[ -f "${D}/opd_done.json" && ! -d "${CK}" ]]; then echo "run finished without checkpoint-${N}; skipping"; continue 2; fi
    sleep 300
  done
  case "${MODE}" in
    dom3)
      TAG="opd-${RUN}-ck${N}-dom3-T1"
      MODEL="${CK}" TAG="${TAG}" BENCHMARKS=medqa,casehold,finqa MAX_TOKENS=26624 TEMP=1.0 sbatch -N 2 $(prefer 2) -J "${JP}-domeval-${TAG}" scripts/eval_domain.sbatch
      if (( N % 50 == 0 )); then FT="full-opd-${RUN}-ck${N}-32k"; MODEL="${CK}" TAG="${FT}" MAX_TOKENS=32768 MAX_MODEL_LEN=40960 sbatch -N 2 $(prefer 2) -J "${JP}-eval6-${FT}" --cpus-per-task=32 --mem=600G -t 12:00:00 scripts/eval_6bench.sbatch; fi ;;
    bench:*)
      B="${MODE#bench:}"; TAG="opd-${RUN}-ck${N}-${B}-T1"
      MODEL="${CK}" TAG="${TAG}" BENCHMARKS="${B}" MAX_TOKENS=26624 TEMP=1.0 sbatch -N 2 $(prefer 2) -J "${JP}-domeval-${TAG}" scripts/eval_domain.sbatch ;;
    if)
      TAG="full-opd-if-${RUN}-ck${N}-if"
      MODEL="${CK}" TAG="${TAG}" BENCHMARKS=ifeval,ifbench MAX_TOKENS=32768 MAX_MODEL_LEN=40960 sbatch -N 2 $(prefer 2) -J "${JP}-eval6-${TAG}" --cpus-per-task=32 --mem=600G -t 12:00:00 scripts/eval_6bench.sbatch ;;
    *) echo "bad mode ${MODE}"; exit 2 ;;
  esac
  echo "$(date '+%H:%M') submitted ${MODE} eval(s) for ${RUN} step ${N}"
done
echo "AUTO-EVAL DONE for ${RUN} steps: ${STEPS}"
