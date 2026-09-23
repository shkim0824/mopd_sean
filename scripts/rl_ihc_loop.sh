#!/bin/bash
# IH-Challenge (safety) teacher = the online alternating loop of lena's recipe (RECIPES.md §2.2) on our stack:
#   round r:  synthesise attacks vs the CURRENT defender (frozen Qwen3-4B attacker, budget 3)  -> data/rl_pools/ihc_<size>_r<r>.jsonl
#             GRPO for STEPS steps initialised from the current defender (fresh optimizer)   -> outputs/rl/<name>/r<r>/step_<STEPS>
#             HF view of the new weights (old-schema config)                                -> models/rl_steps/<name>-r<r>-step<STEPS>
#             becomes the next round's defender
# ROUNDS x STEPS = 200 steps (the domain-teacher budget). Login-node driver; run detached:
#   SIZE=4b BASE=models/Qwen3-4B-OT3 setsid nohup bash scripts/rl_ihc_loop.sh > logs/ihc_loop_4b.log 2>&1 &
#   SIZE=1p7b BASE=models/Qwen3-1.7B-OT3 ...
# Resume: rounds whose step_<STEPS> exists are skipped; a round whose synth file exists skips the synthesis.
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"; cd "${REPO}"
SIZE="${SIZE:-4b}"; BASE="${BASE:-models/Qwen3-4B-OT3}"; ROUNDS="${ROUNDS:-5}"; STEPS="${STEPS:-40}"; NODES="${NODES:-4}"; ATTACKER="${ATTACKER:-models/Qwen3-4B}"
NAME="${NAME:-grpo_ihc_${SIZE}}"; CONFIG="${CONFIG:-configs/rl/grpo_ihc.yaml}"; BUDGET="${BUDGET:-3}"
SK="${SK:-data/domains/ih-challenge}"; VAL="${VAL:-${SK}/heldout_attacks_ot3-4b.jsonl}"
WHOLE_EXC="${WHOLE_EXC:-ib-a100-cluster-a-n[033-036,089-128]}"   # whole-cluster tier: skip the nodes whose owner cancels k-sean- jobs on sight
# exclusion for the current tier: PLACE_OPT's list, or WHOLE_EXC when placement picked the whole cluster
tier_exc() { if [[ "${PLACE_TIER}" == whole-cluster ]]; then echo "${WHOLE_EXC}"; else echo "${PLACE_OPT#--exclude=}"; fi; }
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
    # a killed synth job (k-sean- jobs may be cancelled by node owners) is resubmitted up to SYNTH_TRIES times
    stry=0
    while [[ ! -s "${SYN}" ]]; do
      (( stry < ${SYNTH_TRIES:-3} )) || { echo "ABORT: synth failed ${stry} times (${SYN} missing)"; exit 2; }
      stry=$((stry + 1)); placement 1
      J=$(squeue -h -o "%i %j" 2>/dev/null | awk -v n="-ihc-synth-${SIZE}-r${r}$" '$2 ~ n {print $1}' | head -1)   # attach to a queued synth job
      [[ -n "${J}" ]] && echo "$(date '+%H:%M') round ${r}: synth job ${J} already queued (attached)" || \
      J=$(SKELETONS="${SK}/skeletons_train_r${r}.jsonl" DEFENDER="${defender}" ATTACKER="${ATTACKER}" OUT="${SYN}" BUDGET="${BUDGET}" \
          sbatch --parsable -J "${JP}-ihc-synth-${SIZE}-r${r}" -t 1:30:00 --exclude="$(tier_exc)" scripts/ihc_synth.sbatch)   # short limit: backfill-friendly
      echo "$(date '+%H:%M') round ${r}: synth job ${J} (submission ${stry}; defender ${defender})"
      wait_job "${J}" || echo "WARNING: synth job ${J} did not end COMPLETED (checking ${SYN})"
    done
  fi
  N=$(wc -l < "${SYN}"); echo "$(date '+%H:%M') round ${r}: ${N} conflict prompts in ${SYN}"; [[ "${N}" -ge 100 ]] || { echo "ABORT: too few rows"; exit 2; }
  # ---- 2. GRPO round (fresh optimizer, init = current defender)
  # A killed job (k-sean- jobs may be cancelled by node owners) is resubmitted up to GRPO_TRIES times; NeMo-RL resumes from
  # the round's latest checkpoint (save_period 20), so the round continues instead of restarting.
  try=0
  while [[ ! -d "${CK}/step_${STEPS}/policy/weights/model/consolidated" ]]; do
    (( try < ${GRPO_TRIES:-3} )) || { echo "ABORT: ${CK}/step_${STEPS} missing after ${try} submissions"; exit 3; }
    try=$((try + 1)); placement "${NODES}"
    J=$(squeue -h -o "%i %j" 2>/dev/null | awk -v n="-rl-${NAME}_r${r}$" '$2 ~ n {print $1}' | head -1)   # attach to a queued/running GRPO job
    [[ -n "${J}" ]] && echo "$(date '+%H:%M') round ${r}: GRPO job ${J} already in the queue (attached)" || \
    J=$(JOB_PREFIX="${JP}" EXC="$(tier_exc)" CONFIG="${CONFIG}" NODES="${NODES}" RUN_NAME="${NAME}_r${r}" \
        OVERRIDES="policy.model_name=${defender} data.train.data_path=$(realpath "${SYN}") data.validation.data_path=$(realpath "${VAL}") checkpointing.checkpoint_dir=${REPO}/${CK} logger.log_dir=${REPO}/logs/nemo/${NAME}_r${r} grpo.max_num_steps=${STEPS}" \
        bash scripts/rl_nemo.sh | grep -oE "[0-9]{6,}" | tail -1)
    echo "$(date '+%H:%M') round ${r}: GRPO job ${J} (submission ${try}) -> ${CK}"; wait_job "${J}" || echo "WARNING: GRPO job ${J} did not end COMPLETED (checking for step_${STEPS})"
  done
  # ---- 3. HF view for the next round's vLLM / NeMo init (old-schema config from the base)
  bash scripts/mk_eval_model.sh "${CK}/step_${STEPS}" "${HF}" "${BASE}" && defender="$(realpath "${HF}")"
  echo "$(date '+%H:%M') round ${r} done -> defender ${defender}"
done
ln -sfn "$(realpath "${defender}")" "models/teacher-safety-${SIZE}" && echo "$(date '+%H:%M') IHC loop ALL DONE -> models/teacher-safety-${SIZE} -> ${defender}"
