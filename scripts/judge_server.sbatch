#!/bin/bash
# Standing LLM-judge server (math_with_judge rewards; Qwen3-30B-A3B validated 2026-09-04: 99.84% agreement), vLLM tp8 on
# one node; publishes its URL to tmp/judge_url (the RL env re-reads it every step). Alternative to JUDGE=1 in rl_nemo.sh.
#   JUDGE_MODEL=$S/past/models/Qwen3-30B-A3B sbatch -J k-sean-judge --cpus-per-task=32 --mem=600G -t 96:00:00 $(prefer 1) scripts/judge_server.sbatch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 96:00:00
#SBATCH --output=logs/judge-server-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs tmp
JUDGE_MODEL="${JUDGE_MODEL:-${S}/past/models/Qwen3-30B-A3B}"; PORT=${PORT:-8231}
srun --nodes=1 --ntasks=1 --container-image="${IMG}" --container-mounts="${MOUNT}" --container-workdir="${REPO}" bash -lc "
  ${FWD_ENV} ${PRELUDE} export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -m vllm.entrypoints.openai.api_server --model ${JUDGE_MODEL} --served-model-name judge \
    --tensor-parallel-size 8 --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --disable-custom-all-reduce --enforce-eager --port ${PORT} > ${REPO}/tmp/judge_server_\$SLURM_JOB_ID.log 2>&1 &
  SRV=\$!
  for i in \$(seq 1 240); do curl -fsS http://localhost:${PORT}/v1/models > /dev/null 2>&1 && break; sleep 10; done
  curl -fsS http://localhost:${PORT}/v1/models > /dev/null || { echo JUDGE_SERVER_FAILED; exit 3; }
  echo \"http://\$(hostname):${PORT}/v1\" > ${REPO}/tmp/judge_url; echo \"JUDGE_UP \$(cat ${REPO}/tmp/judge_url)\"
  while kill -0 \$SRV 2>/dev/null; do sleep 60; done; echo JUDGE_SERVER_EXITED; exit 1
"
