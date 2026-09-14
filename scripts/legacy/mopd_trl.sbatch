#!/bin/bash
# LEGACY (paper recipe driver mopd.distill.train_mopd = math/code/if with task-reward mixing; the domain OPD/MOPD
# experiments use mopd.distill.train_opd -> scripts/opd.sbatch). 2 nodes: node0 = student (8 ranks, colocate vLLM),
# node1 = teacher prefill servers (3+ nodes: more teacher GPUs; placement spreads replicas, student stays on node0).
#   TEACHERS="math:outputs/rl/trl_math/final,code:outputs/rl/trl_code/final,if:outputs/rl/trl_if/final" \
#   WEIGHTS="math=0.35,code=0.30,if=0.35" CONFIG=configs/legacy_trl/mopd.yaml \
#   OVERRIDES="model.student=models/Qwen3-4B-OT3 train.output=outputs/opd/trl_mopd_pg" \
#   sbatch -N 2 -J k-sean-mopd-trl --cpus-per-task=32 --mem=600G -t 48:00:00 $(prefer 2) scripts/legacy/mopd_trl.sbatch
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 48:00:00
#SBATCH --output=logs/mopd-trl-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs
CONFIG="${CONFIG:-configs/legacy_trl/mopd.yaml}"; OVERRIDES="${OVERRIDES:-}"
TEACHERS="${TEACHERS:?set TEACHERS=domain:path[:tp],...}"; WEIGHTS="${WEIGHTS:-math=0.35,code=0.30,if=0.35}"
TEACHER_MAX_LEN="${TEACHER_MAX_LEN:-32768}"; TEACHER_GPU_UTIL="${TEACHER_GPU_UTIL:-0.90}"; TOPK="${TOPK:-64}"
read -r OUT BS PD < <(python3 scripts/cfgval.py "${CONFIG}" train.output train.batch_size train.per_device_bs | tr '\n' ' ')
for ov in ${OVERRIDES}; do
  case "${ov}" in train.output=*) OUT="${ov#train.output=}";; train.batch_size=*) BS="${ov#train.batch_size=}";; train.per_device_bs=*) PD="${ov#train.per_device_bs=}";; esac
done
mkdir -p "${OUT}/teacher_logs"
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
NODES_CSV="$(IFS=,; echo "${NODES[*]}")"

# ---- placement (stdlib python on the batch host) ---------------------------------------------
PLACE=(python3 -m mopd.distill.placement --nodes "${NODES_CSV}" --teachers "${TEACHERS}" --weights "${WEIGHTS}")
export PYTHONPATH="${REPO}"
"${PLACE[@]}" --emit json > "${OUT}/teacher_endpoints.json"
"${PLACE[@]}" --emit table | tee "${OUT}/placement.txt"
eval "$("${PLACE[@]}" --emit env)"
WORLD="${STUDENT_WORLD}"; GA=$(( BS / (PD * WORLD) ))
(( GA * PD * WORLD == BS )) || { echo "batch_size=${BS} not divisible by per_device_bs*world=$((PD*WORLD))"; exit 2; }

# ---- teacher servers (background steps; distinct HTTP + VLLM_PORT; 20 s same-node stagger) --------
PIDS=(); URLS=(); declare -A SEEN
for i in $(seq 0 $((N_SERVERS - 1))); do
  m="SERVER_${i}_MODEL"; n="SERVER_${i}_NODE"; g="SERVER_${i}_GPUS"; p="SERVER_${i}_PORT"; v="SERVER_${i}_VLLM_PORT"; t="SERVER_${i}_TP"; nm="SERVER_${i}_NAME"; u="SERVER_${i}_URL"
  tpf=""; [[ "${!t}" -ge 2 ]] && tpf="--disable-custom-all-reduce"
  cmd="VLLM_PORT=${!v} ${VLLM_ATTN:+VLLM_ATTENTION_BACKEND=${VLLM_ATTN} }python -m vllm.entrypoints.openai.api_server \
    --model '$(realpath "${!m}")' --served-model-name '${!nm}' --host 0.0.0.0 --port ${!p} \
    --tensor-parallel-size ${!t} --max-model-len ${TEACHER_MAX_LEN} --gpu-memory-utilization ${TEACHER_GPU_UTIL} \
    --dtype bfloat16 --max-logprobs ${TOPK} --logprobs-mode raw_logprobs --enable-prefix-caching \
    --max-num-batched-tokens ${TEACHER_MAX_BATCHED:-16384} --trust-remote-code ${tpf}"
  seen=${SEEN[${!n}]:-0}; delay=$(( seen * 20 )); SEEN[${!n}]=$(( seen + 1 ))
  ( (( delay > 0 )) && sleep "${delay}"; run_step "${!n}" "${!g}" "${OUT}/teacher_logs/server_${i}.log" "${cmd}" ) &
  PIDS+=("$!"); URLS+=("${!u}")
done
cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
echo "[mopd] waiting for ${N_SERVERS} teacher servers: ${URLS[*]}"
if ! wait_http "${URLS[@]}"; then
  alive=0; for p in "${PIDS[@]}"; do kill -0 "${p}" 2>/dev/null && alive=$((alive+1)); done
  echo "[mopd] teacher servers not ready (alive steps: ${alive}); see ${OUT}/teacher_logs"; exit 4
fi
echo "[mopd] teachers ready"

# ---- student trainer (node0, 8 ranks) -----------------------------------------------------------
CFG="$(accel_cfg "${GA}" "${OUT}")"
OV="--override teachers.endpoints_file=${OUT}/teacher_endpoints.json --override teachers.top_k=${TOPK}"
for ov in ${OVERRIDES}; do OV="${OV} --override ${ov}"; done
echo "[mopd] student node=${STUDENT_NODE} world=${WORLD} batch=${BS} grad_accum=${GA}"
run_step "${STUDENT_NODE}" "${STUDENT_GPUS}" "${OUT}/train.log" \
  "accelerate launch --config_file '${CFG}' --num_machines 1 --num_processes ${WORLD} --main_process_port ${ACCEL_PORT:-29500} \
   -m mopd.distill.train_mopd --config '${CONFIG}' ${OV}"
rc=$?; echo "[mopd] exit ${rc}"; exit ${rc}
