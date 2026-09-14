#!/bin/bash
# Evaluate NeMo-RL steps as they appear: rl_auto_eval.sh <tagprefix> <rl_run> <ref_hf> <bench> <steps...>
#   rl_run = outputs/rl/<run> (step_N/policy/weights/model/consolidated), ref_hf = init model (old-schema config/tokenizer)
#   builds models/rl_steps/<tagprefix>-ck<N> (scripts/mk_eval_model.sh) and submits the domain bench at T=1.0;
#   bench "if" = IFEval + IFBench via scripts/eval_6bench.sbatch (tag <tagprefix>-ck<N>-if).
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"; cd "${REPO}"
PFX="$1"; RL="$2"; REF="$3"; BENCH="$4"; shift 4; STEPS="$*"; mkdir -p models/rl_steps
for N in ${STEPS}; do
  while true; do
    if [[ -f "${RL}/step_${N}/policy/weights/model/consolidated/model.safetensors.index.json" ]]; then
      nxt=$((N + ${NEXT:-10})); if [[ "${NEXT:-10}" == "0" || -d "${RL}/step_${nxt}" ]] || ! squeue -h -o %j | grep -q "${JP}-rl-"; then sleep 120; break; fi
    fi
    sleep 300
  done
  M="models/rl_steps/${PFX}-ck${N}"
  bash scripts/mk_eval_model.sh "${RL}/step_${N}" "${M}" "${REF}" || { echo "convert failed step ${N}"; continue; }
  placement 2
  if [[ "${BENCH}" == "if" ]]; then   # instruction following: IFEval + IFBench through the 6-bench runner (Qwen thinking preset, 32k)
    TAG="${PFX}-ck${N}-if"; MODEL="${M}" TAG="${TAG}" BENCHMARKS="ifeval,ifbench" MAX_TOKENS=32768 MAX_MODEL_LEN=40960 sbatch -N 2 ${PLACE_OPT} -J "${JP}-eval6-${TAG}" --cpus-per-task=32 --mem=600G -t 6:00:00 scripts/eval_6bench.sbatch
  else
    TAG="${PFX}-ck${N}-${BENCH}-T1"; MODEL="${M}" TAG="${TAG}" BENCHMARKS="${BENCH}" MAX_TOKENS=26624 TEMP=1.0 sbatch -N 2 ${PLACE_OPT} -J "${JP}-domeval-${TAG}" scripts/eval_domain.sbatch
  fi
  echo "$(date '+%H:%M') submitted eval for ${PFX} step ${N}"
done
echo "AUTO-EVAL DONE for ${PFX} steps: ${STEPS}"
