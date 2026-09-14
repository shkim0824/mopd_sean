#!/bin/bash
# Evaluate SFT checkpoints (outputs/sft/<run>/checkpoint-N) as they appear, at T=1.0 (login-node loop):
#   sft_auto_eval.sh <run> <bench> 25 50 75 ... 200      -> tag sftw-<run>-ck<N>-<bench>-T1 per step
#   sft_auto_eval.sh <run> <bench> epochs <n_ckpts>       -> every checkpoint-N as it appears (save_strategy=epoch)
# After the LAST checkpoint: 6-bench (tag full-sftw-<run>-final-32k) and, for medqa, MedXpertQA+PubMedQA (T1.0).
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"; cd "${REPO}"
RUN="$1"; B="$2"; shift 2; D="${REPO}/outputs/sft/${RUN}"
submit_ck() { local CK="$1" N="$2"; local TAG="sftw-${RUN}-ck${N}-${B}-T1"
  MODEL="${CK}" TAG="${TAG}" BENCHMARKS="${B}" MAX_TOKENS=26624 TEMP=1.0 sbatch -N 2 $(prefer 2) -J "${JP}-domeval-${TAG}" scripts/eval_domain.sbatch
  echo "$(date '+%H:%M') submitted ${B} T1.0 eval for ${RUN} checkpoint-${N}"; }
finals() { local CK="$1"
  MODEL="${CK}" TAG="full-sftw-${RUN}-final-32k" MAX_TOKENS=32768 MAX_MODEL_LEN=40960 sbatch -N 2 $(prefer 2) -J "${JP}-eval6-full-sftw-${RUN}-final-32k" --cpus-per-task=32 --mem=600G -t 12:00:00 scripts/eval_6bench.sbatch
  [[ "${B}" == medqa ]] && MODEL="${CK}" TAG="sftw-${RUN}-final-xpert-pubmed-T1" BENCHMARKS=medxpertqa,pubmedqa MAX_TOKENS=26624 TEMP=1.0 sbatch -N 2 $(prefer 2) -J "${JP}-domeval-sftw-${RUN}-final-xpert-pubmed-T1" scripts/eval_domain.sbatch
  echo "$(date '+%H:%M') submitted finals for ${RUN} on ${CK}"; }
if [[ "$1" == "epochs" ]]; then
  NEXP="$2"; declare -A done; n=0; last=""
  while (( n < NEXP )); do
    for CK in $(ls -d "${D}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
      N=${CK##*-}; [[ -n "${done[$N]:-}" ]] && continue
      [[ -f "${CK}/config.json" && -f "${CK}/model.safetensors.index.json" ]] || continue
      sleep 120; submit_ck "${CK}" "${N}"; done[$N]=1; n=$((n+1)); last="${CK}"
    done
    (( n < NEXP )) && sleep 300
  done
  finals "${last}"
else
  STEPS="$*"; last=""
  for N in ${STEPS}; do
    CK="${D}/checkpoint-${N}"
    while true; do
      if [[ -f "${CK}/config.json" && -f "${CK}/model.safetensors.index.json" ]]; then sleep 120; break; fi
      if [[ -f "${D}/mopd_sft_done.json" && ! -d "${CK}" ]]; then echo "run finished without checkpoint-${N}; skipping"; continue 2; fi
      sleep 180
    done
    submit_ck "${CK}" "${N}"; last="${CK}"
  done
  [[ -n "${last}" ]] && finals "${last}"
fi
echo "AUTO-EVAL DONE for ${RUN}"
