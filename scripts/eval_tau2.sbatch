#!/bin/bash
# tau2-bench evaluation of one model (official harness; telecom base split by default).
# N nodes = N shards: every node serves the model with vLLM (8 GPUs, data-parallel) and runs the
# harness on every N-th task id; node 0 then merges the shards and scores with tau2's own metrics.
#   MODEL=<path> TAG=tau2-<name> [USER_LLM=gpt-4.1-2025-04-14] [TRIALS=1] [CONC=16] [NUM_TASKS=] \
#     sbatch -N 2 -J k-sean-tau2-<name> --cpus-per-task=32 --mem=600G -t 8:00:00 scripts/eval_tau2.sbatch
# Writes outputs/eval_all/<TAG>/metrics.json: scores.tau2_telecom = pass^1 x 100 on the whole split,
# *_clean = minus the tasks present in the AReaL SFT data, *_passK for K>1 when TRIALS>1.
# Resumable: every shard runs with --auto-resume.
# Reasoning parser default = vLLM's stock qwen3 (2026-09-17 16:20): the strict plugin rejected '<think>' + tool call turns
# without '</think>' that the stock parser (and hence any external evaluator) executes; pass REASONING_PARSER=qwen3_strict to keep the old behaviour.
#SBATCH --exclude=ib-a100-cluster-a-n[033-036]
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 8:00:00
#SBATCH --output=logs/tau2-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs
# Snapshot the inner scripts for this job: bash reads a script incrementally, so editing tmp/phase3/*.sh while a
# job is inside the harness makes the job die with a syntax error when it resumes reading (576675, 2026-09-17).
SNAP="outputs/eval_all/${TAG:?}/_scripts.${SLURM_JOB_ID}"; mkdir -p "${SNAP}"
cp -f tmp/phase3/tau2_run_inner.sh tmp/phase3/tau2_merge_inner.sh "${SNAP}/"
ENVS="MODEL='${MODEL:?}' TAG='${TAG:?}' DOMAIN='${DOMAIN:-telecom}' SPLIT='${SPLIT:-base}' TRIALS='${TRIALS:-1}' CONC='${CONC:-16}' \
 USER_LLM='${USER_LLM:-gpt-4.1-2025-04-14}' USER_ARGS='${USER_ARGS:-}' AGENT_TEMP='${AGENT_TEMP:-1.0}' AGENT_TOP_P='${AGENT_TOP_P:-1.0}' AGENT_TOP_K='${AGENT_TOP_K:--1}' \
 AGENT_MAXTOK='${AGENT_MAXTOK:-}' DP='${DP:-8}' MAX_MODEL_LEN='${MAX_MODEL_LEN:-32768}' ROPE_SCALING='${ROPE_SCALING:-}' REASONING_PARSER='${REASONING_PARSER:-qwen3}' VLLM_DTYPE='${VLLM_DTYPE:-bfloat16}' AGENT_IMPL='${AGENT_IMPL:-llm_agent}' AREAL_PROMPT_FILE='${AREAL_PROMPT_FILE:-}' MAX_RETRIES='${MAX_RETRIES:-1}' NUM_TASKS='${NUM_TASKS:-}' TASK_IDS='${TASK_IDS:-}' PORT='${PORT:-8123}'"
# --wait=0: without it srun kills the remaining shards WaitTime seconds after the FIRST shard exits (576709 lost shard 1 at 55/57 tasks)
srun --wait=0 --nodes="${SLURM_JOB_NUM_NODES}" --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --export=ALL \
  --container-image="${IMG}" --container-mounts="${MOUNT}" --container-workdir="${REPO}" \
  bash -lc "${PRELUDE} cd '${REPO}'; export HF_HOME='${HF_HOME_DIR}' HF_HUB_OFFLINE=1; ${ENVS} bash ${SNAP}/tau2_run_inner.sh"
srun --nodes=1 --ntasks=1 --export=ALL --container-image="${IMG}" --container-mounts="${MOUNT}" --container-workdir="${REPO}" \
  bash -lc "${PRELUDE} cd '${REPO}'; ${ENVS} bash ${SNAP}/tau2_merge_inner.sh"
