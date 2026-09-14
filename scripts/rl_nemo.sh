#!/bin/bash
# Submit a NeMo-RL v0.7.0 GRPO teacher run (ray cluster over N nodes via scripts/nemo/ray.sub, NVIDIA launcher).
#   CONFIG=configs/rl/grpo_law_distill.yaml NODES=4 bash scripts/rl_nemo.sh            # domain teacher (exact-match env)
#   CONFIG=configs/rl/grpo_if.yaml NODES=4 bash scripts/rl_nemo.sh                     # IF / code (local verifiers)
#   CONFIG=configs/rl/grpo_math_v3.yaml NODES=8 JUDGE=1 bash scripts/rl_nemo.sh        # + 1 judge node (LLM judge, +1 node)
#   SMOKE=1 -> 1 node, 2 steps, tiny batches, output outputs/rl/smoke_<name>
#   OVERRIDES="key=value ..." extra hydra overrides; RUN_NAME=<x> names the job/log (default = config basename)
# Env registration + processors: mopd/rl/nemo/run_grpo.py (PYTHONPATH = repo root on every ray worker).
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"
cd "${REPO}"; mkdir -p logs
CONFIG="$(realpath "${CONFIG:?set CONFIG=configs/rl/<x>.yaml}")"; NAME="${RUN_NAME:-$(basename "${CONFIG}" .yaml)}"; NODES="${NODES:-4}"
OVERRIDES="${OVERRIDES:-}"   # extra hydra key=value overrides appended to the command (e.g. policy.model_name=... data.train.data_path=...)
EXC="${EXC:-$(prefer "${NODES}" | sed 's/--exclude=//')}"
export CONTAINER="${NEMO_IMG}" MOUNTS="/mnt/datafs:/mnt/datafs,/tmp:/tmp" GPUS_PER_NODE=8 CPUS_PER_WORKER=32 NEMO_RL_VENV_DIR=/opt/ray_venvs \
  TORCH_CUDA_ARCH_LIST=8.0 HF_HOME="${HF_HOME_DIR}" HF_HUB_OFFLINE=1 PYTHONPATH="${REPO}" BASE_LOG_DIR="${REPO}/logs/nemo"
mkdir -p "${REPO}/logs/nemo"
BASECMD="cd /opt/nemo-rl && export TORCH_CUDA_ARCH_LIST=8.0 HF_HOME=${HF_HOME_DIR} HF_HUB_OFFLINE=1 PYTHONPATH=${REPO} NEMO_RL_VENV_DIR=/opt/ray_venvs && unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && /opt/nemo_rl_venv/bin/python ${REPO}/mopd/rl/nemo/run_grpo.py --config ${CONFIG}"
JUDGE_EXTRA=""; NJ=0
if [[ "${JUDGE:-0}" == "1" ]]; then
  # LLM judge on an extra node (math_with_judge env); the judge URL file is re-read by the env every step
  export MOPD_RL_JUDGE_NODE=1 MOPD_RL_JUDGE_IMAGE="${IMG}" MOPD_RL_JUDGE_MODEL_PATH="${JUDGE_MODEL:-${S}/past/models/Qwen3-30B-A3B}" \
    MOPD_RL_JUDGE_URL_FILE="${REPO}/tmp/judge_url" MOPD_RL_VENV_ACTIVATE="${VENV}"
  mkdir -p "${REPO}/tmp"
  JUDGE_EXTRA="for i in \$(seq 1 240); do [ -f ${REPO}/tmp/judge_url ] && curl -fsS \$(cat ${REPO}/tmp/judge_url)/models > /dev/null 2>&1 && break; sleep 10; done && curl -fsS \$(cat ${REPO}/tmp/judge_url)/models > /dev/null && "
  NJ=1
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
  export COMMAND="cd /opt/nemo-rl && ${JUDGE_EXTRA}${BASECMD#cd /opt/nemo-rl && } cluster.num_nodes=1 grpo.num_prompts_per_step=4 grpo.num_generations_per_prompt=4 policy.train_global_batch_size=16 policy.max_total_sequence_length=4096 grpo.max_num_steps=2 ${OVERRIDES} grpo.val_at_start=false grpo.val_period=1000 checkpointing.checkpoint_dir=${REPO}/outputs/rl/smoke_${NAME} checkpointing.save_period=1 logger.log_dir=${REPO}/logs/nemo/smoke_${NAME}"
  sbatch -N $((1 + NJ)) -J "${JP}-rl-smoke-${NAME}" --gres=gpu:8 --cpus-per-task=32 --mem=600G -t 2:00:00 --exclude="${EXC}" --output="${REPO}/logs/rl-smoke-${NAME}-%j.log" scripts/nemo/ray.sub
else
  export COMMAND="cd /opt/nemo-rl && ${JUDGE_EXTRA}${BASECMD#cd /opt/nemo-rl && } cluster.num_nodes=${NODES} ${OVERRIDES}"
  sbatch -N $((NODES + NJ)) -J "${JP}-rl-${NAME}" --gres=gpu:8 --cpus-per-task=32 --mem=600G -t 48:00:00 --exclude="${EXC}" --output="${REPO}/logs/rl-${NAME}-%j.log" scripts/nemo/ray.sub
fi
