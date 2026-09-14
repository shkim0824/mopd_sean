#!/bin/bash
# Teacher trajectory generation (rejection sampling input) on N nodes: every node runs 8/TP independent vLLM workers
# (nemo-rl container python: its vLLM handles the MoE teachers; TP=2 x DP=4 was 1.44x faster than TP8 on the 35B teacher).
# Node p owns global shards [p*NSH_NODE, (p+1)*NSH_NODE). Resume-safe: resubmit with the same TAG.
#   MODEL=models/teacher-med POOL=data/rl_pools/med_sftp_train.jsonl TAG=gen-med-teacher-k4 K=4 TP=1 TEMP=1.0 \
#     sbatch -N 2 -J k-sean-gen-<tag> $(prefer 2) scripts/gen_teacher.sbatch      -> outputs/teacher_gen/<TAG>
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 24:00:00
#SBATCH --output=logs/gen-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs tmp/cache
PY=${PY:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python}
MODEL="$(realpath "${MODEL:?}")"; POOL="$(realpath "${POOL:?}")"; TAG="${TAG:?}"; K="${K:?}"
TP=${TP:-2}; NSH_NODE=$((8 / TP)); NODES=$SLURM_JOB_NUM_NODES; TOTAL=${TOTAL:-$((NODES * NSH_NODE))}
SHARDS=${SHARDS:-}    # e.g. SHARDS="4,5,6,7" with -N 1 and TOTAL=<original total>: this node resumes just those ranks
OUT="${REPO}/outputs/teacher_gen/${TAG}"; CACHE="${REPO}/tmp/cache"; mkdir -p "${OUT}"
COMMON="--model ${MODEL} --pool ${POOL} --out ${OUT} --k ${K} --tp ${TP} --num-shards ${TOTAL} --max-tokens ${MAX_TOKENS:-16384} --max-model-len ${MAX_MODEL_LEN:-20480} --temperature ${TEMP:-0.6} --chunk ${CHUNK:-1024}"
echo "[gen] nodes=${NODES} tp=${TP} shards/node=${NSH_NODE} total_shards=${TOTAL} k=${K} pool=${POOL} start=$(date +%s)"
srun --kill-on-bad-exit=0 --ntasks=${NODES} --ntasks-per-node=1 --container-image="${NEMO_IMG}" \
  --container-mounts=/mnt/datafs:/mnt/datafs --container-workdir="${REPO}" bash -lc '
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    export PYTHONPATH='"${REPO}"' HF_HUB_OFFLINE=1 TORCH_CUDA_ARCH_LIST=8.0
    export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_HOST_IP=127.0.0.1
    export HF_HOME='"${CACHE}"'/hf XDG_CACHE_HOME='"${CACHE}"'/xdg
    node=${SLURM_PROCID:-0}; TP='"${TP}"'; NSH='"${NSH_NODE}"'; SHARDS='"${SHARDS}"'
    echo "[gen] node=$node host=$(hostname) start=$(date +%s) shards_override=${SHARDS:-none}"
    pids=(); ranks=(); rm -f '"${OUT}"'/failed_shard* 2>/dev/null
    IFS=, read -r -a OVR <<< "$SHARDS"
    for ((i=0;i<NSH;i++)); do
      if [ -n "$SHARDS" ]; then [ $i -lt ${#OVR[@]} ] || break; rank=${OVR[$i]}; else rank=$((node*NSH+i)); fi
      lo=$((i*TP)); gpus=$lo
      for ((k=1;k<TP;k++)); do gpus="$gpus,$((lo+k))"; done
      export CUDA_VISIBLE_DEVICES=$gpus
      export VLLM_CACHE_ROOT='"${CACHE}"'/vllm_'"${SLURM_JOB_ID}"'_$rank TORCHINDUCTOR_CACHE_DIR='"${CACHE}"'/ind_'"${SLURM_JOB_ID}"'_$rank TRITON_CACHE_DIR='"${CACHE}"'/triton_'"${SLURM_JOB_ID}"'_$rank
      '"${PY}"' -m mopd.teacher.gen_teacher '"${COMMON}"' --shard-rank $rank > '"${REPO}"'/logs/gen-'"${TAG}"'-shard$rank.log 2>&1 &
      pids+=($!); ranks+=($rank); echo "[gen] node=$node shard=$rank gpus=$gpus pid=${pids[-1]}"
    done
    rc=0; OUTD='"${OUT}"'; TOTAL='"${TOTAL}"'
    for ((i=0;i<${#pids[@]};i++)); do
      if ! wait ${pids[$i]}; then rc=1; touch $OUTD/failed_shard${ranks[$i]}; echo "[gen] node=$node shard ${ranks[$i]} FAILED"; fi
    done
    echo "[gen] node=$node all shards done rc=$rc end=$(date +%s)"
    t0=$(date +%s)
    while [ $(( $(ls $OUTD/*.done 2>/dev/null | wc -l) + $(ls $OUTD/failed_shard* 2>/dev/null | wc -l) )) -lt $TOTAL ]; do
      [ $(( $(date +%s) - t0 )) -gt 21600 ] && { echo "[gen] node=$node barrier TIMEOUT"; break; }; sleep 30
    done
    echo "[gen] node=$node barrier passed end=$(date +%s)"; exit $rc
  '
rc=$?; echo "[gen] job done rc=${rc} end=$(date +%s)"; ls "${OUT}"/*.done 2>/dev/null | wc -l | xargs echo "[gen] done shards:"; exit ${rc}
