#!/bin/bash
# strip_optimizer.sh <run_dir...>: delete resume-only state, keep every checkpoint's model weights.
#   trl/DeepSpeed ckpts: global_step*/ optimizer.pt scheduler.pt rng_state_*.pth latest zero_to_fp32.py
#   NeMo-RL steps:      policy/optimizer + DCP shard-*.safetensors/.metadata (only when policy/weights/model/consolidated exists)
# Weights-only checkpoints still load for evaluation and as the init of NEW SFT/OPD/RL runs; only resuming the same run needs the deleted files.
# NEVER run on a directory whose job is still running (it may need to resume).
set -u
for run in "$@"; do
  [[ -d "${run}" ]] || { echo "skip (not a dir): ${run}"; continue; }
  name="$(basename "${run}")"
  if squeue -h -o %j | grep -q -- "opd-${name}$\|sft-${name}$\|rl-${name}$"; then echo "SKIP running: ${run}"; continue; fi
  before=$(dust -d 0 -o b -P -b "${run}" 2>/dev/null | awk '{print $1}' | tr -d B)
  for ck in "${run}"/checkpoint-*; do [[ -d "${ck}" ]] || continue; rm -rf "${ck}"/global_step* "${ck}"/optimizer.pt "${ck}"/scheduler.pt "${ck}"/rng_state_*.pth "${ck}"/latest "${ck}"/zero_to_fp32.py; done
  for st in "${run}"/step_*; do [[ -d "${st}" ]] || continue; rm -rf "${st}"/policy/optimizer; [[ -d "${st}"/policy/weights/model/consolidated ]] && rm -f "${st}"/policy/weights/model/shard-*.safetensors "${st}"/policy/weights/model/.metadata; done
  after=$(dust -d 0 -o b -P -b "${run}" 2>/dev/null | awk '{print $1}' | tr -d B)
  awk -v b="${before:-0}" -v a="${after:-0}" -v r="${run}" 'BEGIN{printf "stripped %s: %.1f GB -> %.1f GB\n", r, b/1e9, a/1e9}'
done
