#!/bin/bash
# IH-Challenge (safety) teacher = the online alternating loop of lena's recipe (RECIPES.md §2.2) on our stack:
#   round r:  synthesise attacks vs the CURRENT defender (self-play attacker, budget 3)  -> data/rl_pools/ihc_<size>_r<r>.jsonl
#             GRPO for STEPS steps initialised from the current defender (fresh optimizer)   -> outputs/rl/<name>/r<r>/step_<STEPS>
#             HF view of the new weights (old-schema config)                                -> models/rl_steps/<name>-r<r>-step<STEPS>
#             becomes the next round's defender
# ROUNDS x STEPS = 200 steps (the domain-teacher budget). Login-node driver; run detached:
#   SIZE=4b BASE=models/Qwen3-4B-OT3 setsid nohup bash scripts/rl_ihc_loop.sh > logs/ihc_loop_4b.log 2>&1 &
#   SIZE=1p7b BASE=models/Qwen3-1.7B-OT3 ...
# Resume: rounds whose step_<STEPS> exists are skipped; a round whose synth file exists skips the synthesis.
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"; cd "${REPO}"
SIZE="${SIZE:-4b}"; BASE="${BASE:-models/Qwen3-4B-OT3}"; ROUNDS="${ROUNDS:-5}"; STEPS="${STEPS:-40}"; NODES="${NODES:-2}"
NAME="${NAME:-grpo_ihc_${SIZE}}"; CONFIG="${CONFIG:-configs/rl/grpo_ihc.yaml}"; BUDGET="${BUDGET:-3}"
SK="${SK:-data/domains/ih-challenge}"; VAL="${VAL:-${SK}/heldout_attacks_ot3-4b.jsonl}"
[[ -f "${VAL}" ]] || { echo "ABORT: validation/benchmark attacks missing: ${VAL} (run the held-out synth first)"; exit 1; }
wait_job() {  # wait_job <jobid> -> 0 when it left the queue with COMPLETED
  while squeue -h -j "$1" -o %T | grep -q .; do sleep 60; done
  local st; st=$(sacct -j "$1" -X -o State -n | head -1 | tr -d " "); echo "$(date '+%H:%M') job $1 -> ${st}"; [[ "${st}" == COMPLETED ]]
}
defender="$(realpath "${BASE}")"
echo "$(date '+%H:%M') IHC loop: size=${SIZE} base=${defender} rounds=${ROUNDS} x steps=${STEPS} nodes=${NODES} name=${NAME}"
for ((r = 0; r < ROUNDS; r++)); do
  SYN="data/rl_pools/ihc_${SIZE}_r${r}.jsonl"; CK="outputs/rl/${NAME}/r${r}"; HF="models/rl_steps/${NAME}-r${r}-step${STEPS}"
  if [[ -f "${HF}/config.json" ]]; then echo "round ${r}: done already (${HF})"; defender="$(realpath "${HF}")"; continue; fi
  # ---- 1. attack synthesis vs the current defender
  if [[ ! -s "${SYN}" ]]; then
    J=$(SKELETONS="${SK}/skeletons_train_r${r}.jsonl" DEFENDER="${defender}" OUT="${SYN}" BUDGET="${BUDGET}" \
        sbatch --parsable -J "${JP}-ihc-synth-${SIZE}-r${r}" $(place 1) scripts/ihc_synth.sbatch)
    echo "$(date '+%H:%M') round ${r}: synth job ${J} (defender ${defender})"; wait_job "${J}" || { echo "ABORT: synth failed"; exit 2; }
  fi
  N=$(wc -l < "${SYN}"); echo "$(date '+%H:%M') round ${r}: ${N} conflict prompts in ${SYN}"; [[ "${N}" -ge 100 ]] || { echo "ABORT: too few rows"; exit 2; }
  # ---- 2. GRPO round (fresh optimizer, init = current defender)
  if [[ ! -d "${CK}/step_${STEPS}" ]]; then
    J=$(CONFIG="${CONFIG}" NODES="${NODES}" RUN_NAME="${NAME}_r${r}" \
        OVERRIDES="policy.model_name=${defender} data.train.data_path=$(realpath "${SYN}") data.validation.data_path=$(realpath "${VAL}") checkpointing.checkpoint_dir=${REPO}/${CK} logger.log_dir=${REPO}/logs/nemo/${NAME}_r${r} grpo.max_num_steps=${STEPS}" \
        bash scripts/rl_nemo.sh | grep -oE "[0-9]{6,}" | tail -1)
    echo "$(date '+%H:%M') round ${r}: GRPO job ${J} -> ${CK}"; wait_job "${J}" || echo "WARNING: GRPO job ${J} did not end COMPLETED (checking for step_${STEPS} anyway)"
    [[ -d "${CK}/step_${STEPS}/policy/weights/model/consolidated" ]] || { echo "ABORT: ${CK}/step_${STEPS} missing"; exit 3; }
  fi
  # ---- 3. HF view for the next round's vLLM / NeMo init (old-schema config from the base)
  bash scripts/mk_eval_model.sh "${CK}/step_${STEPS}" "${HF}" "${BASE}" && defender="$(realpath "${HF}")"
  echo "$(date '+%H:%M') round ${r} done -> defender ${defender}"
done
ln -sfn "$(realpath "${defender}")" "models/teacher-safety-${SIZE}" && echo "$(date '+%H:%M') IHC loop ALL DONE -> models/teacher-safety-${SIZE} -> ${defender}"
