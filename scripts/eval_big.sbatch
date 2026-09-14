#!/bin/bash
# Domain-benchmark eval of a BIG / MoE model (e.g. the Qwen3.6-35B-A3B trace teacher, now under $S/past/models) inside the
# NeMo-RL container, whose vLLM handles the MoE architectures: one node = 8/TP data-parallel workers (TP=2 default,
# TP=8 -> one worker), then one aggregate step. Used to pick / validate a trace teacher before scripts/gen_teacher.sbatch.
#   MODEL=$S/past/models/Qwen3.6-35B-A3B TAG=big-35b-dom3-T1 BENCHMARKS=medqa,casehold,finqa TP=2 \
#     sbatch -J k-sean-evalbig-<tag> $(prefer 1) scripts/eval_big.sbatch            (LIMIT=48 for a smoke; T=1.0 default)
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 12:00:00
#SBATCH --output=logs/evalbig-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs tmp/cache
PY="${PY:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python}"
MODEL="$(realpath "${MODEL:?set MODEL}")"; TAG="${TAG:?set TAG}"; BENCHMARKS="${BENCHMARKS:?set BENCHMARKS}"
TP="${TP:-2}"; NSH=$((8 / TP)); OUT="${OUT:-${REPO}/outputs/eval_domain/${TAG}}"; mkdir -p "${OUT}"
LIMIT_ARG=""; [[ -n "${LIMIT:-}" ]] && LIMIT_ARG="--limit ${LIMIT}"
COMMON="--model '${MODEL}' --benchmarks '${BENCHMARKS}' --out '${OUT}' ${LIMIT_ARG} --max-tokens ${MAX_TOKENS:-31232} \
  --max-model-len ${MAX_MODEL_LEN:-32768} --temperature ${TEMP:-1.0} --tp ${TP} --num-shards ${NSH}"
echo "[evalbig] model=${MODEL} tp=${TP} shards=${NSH} benchmarks=${BENCHMARKS} T=${TEMP:-1.0} start=$(date +%s)"
srun --nodes=1 --ntasks=1 --export=ALL --container-image="${NEMO_IMG}" --container-mounts="${MOUNT}" --container-workdir="${REPO}" bash -lc "
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONPATH='${REPO}' DOMAINS_DATA='${REPO}/data/domains' DOMAINS_TP='${REPO}/third_party' HF_HOME='${HF_HOME_DIR}' HF_HUB_OFFLINE=1
export TORCH_CUDA_ARCH_LIST=8.0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_HOST_IP=127.0.0.1
cd '${REPO}'; pids=()
for ((i=0;i<${NSH};i++)); do
  lo=\$((i*${TP})); gpus=\$lo; for ((k=1;k<${TP};k++)); do gpus=\"\$gpus,\$((lo+k))\"; done
  export CUDA_VISIBLE_DEVICES=\$gpus VLLM_PORT=\$((41000 + i*1000))
  export VLLM_CACHE_ROOT=${REPO}/tmp/cache/vllm_${SLURM_JOB_ID}_\$i TORCHINDUCTOR_CACHE_DIR=${REPO}/tmp/cache/ind_${SLURM_JOB_ID}_\$i TRITON_CACHE_DIR=${REPO}/tmp/cache/triton_${SLURM_JOB_ID}_\$i XDG_CACHE_HOME=${REPO}/tmp/cache/xdg
  ${PY} -m mopd.eval.domains.run_domain_eval ${COMMON} --shard-rank \$i > '${OUT}'/worker_\$i.log 2>&1 &
  pids+=(\$!); echo \"[evalbig] shard \$i -> GPUs \$gpus\"
done
rc=0; for p in \"\${pids[@]}\"; do wait \$p || rc=1; done
echo \"[evalbig] shards done rc=\$rc\"
${PY} -m mopd.eval.domains.run_domain_eval --aggregate-only --benchmarks '${BENCHMARKS}' --out '${OUT}' ${LIMIT_ARG} > '${OUT}'/aggregate.log 2>&1 || rc=1
exit \$rc"
rc=$?
[[ -f "${OUT}/metrics.json" ]] && python3 -c "import json;m=json.load(open('${OUT}/metrics.json'));print({b:{k:round(100*v,1) for k,v in r.items() if isinstance(v,float)} for b,r in m.items()})"
echo "[evalbig] finished rc=${rc} end=$(date +%s)"; exit ${rc}
