#!/bin/bash
# On-policy distillation launcher: single-domain OPD and multi-teacher MOPD are the SAME job, differing only in the
# config (data.paths = one or several domain pools) and TEACHERS (one entry per domain).
#   nodes[0:STUDENT_NODES] = student (accelerate multi-machine, 8 ranks/node, colocate vLLM), nodes[STUDENT_NODES:] =
#   teacher vLLM prefill servers (placement balanced by WEIGHTS).
#   STUDENT_NODES=2 TEACHERS="law:models/teacher-law,fin:models/teacher-fin,if:models/teacher-if,med:models/teacher-med" \
#   WEIGHTS="law=1.0,fin=1.0,if=1.0,med=1.0" TOPK=64 CONFIG=configs/mopd/4dom_pg128.yaml \
#   sbatch -N 4 -J k-sean-opd-<run> --cpus-per-task=32 --mem=600G -t 48:00:00 $(prefer 4) scripts/opd.sbatch
#   Single domain: STUDENT_NODES=1 TEACHERS="law:models/teacher-law" WEIGHTS="law=1.0" CONFIG=configs/opd/law_pg.yaml sbatch -N 2 ...
#   Resume: resubmit the same line (train.resume=auto picks the last complete checkpoint).
#SBATCH --exclude=ib-a100-cluster-a-n[033-036]   # user rule 2026-09-15 (CLI --exclude overrides; launchers pass a superset)
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 48:00:00
#SBATCH --output=logs/opd-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs
CONFIG="${CONFIG:?set CONFIG=configs/opd/<x>.yaml or configs/mopd/<x>.yaml}"; OVERRIDES="${OVERRIDES:-}"
TEACHERS="${TEACHERS:?set TEACHERS=domain:path[:tp],...}"; WEIGHTS="${WEIGHTS:-}"
TEACHER_MAX_LEN="${TEACHER_MAX_LEN:-32768}"; TEACHER_GPU_UTIL="${TEACHER_GPU_UTIL:-0.55}"; TEACHER_MAX_BATCHED="${TEACHER_MAX_BATCHED:-4096}"; TOPK="${TOPK:-64}"
STUDENT_NODES="${STUDENT_NODES:-1}"
read -r OUT BS PD < <(python3 scripts/cfgval.py "${CONFIG}" train.output train.batch_size train.per_device_bs | tr '\n' ' ')
python3 scripts/rope_guard.py "$(python3 scripts/cfgval.py "${CONFIG}" model.student)" $(echo "${TEACHERS}" | tr "," "\n" | cut -d: -f2 | tr "\n" " ") || exit 6   # transformers-5 rope schema guard (2026-09-17)
for ov in ${OVERRIDES}; do
  case "${ov}" in train.output=*) OUT="${ov#train.output=}";; train.batch_size=*) BS="${ov#train.batch_size=}";; train.per_device_bs=*) PD="${ov#train.per_device_bs=}";; esac
done
mkdir -p "${OUT}/teacher_logs"
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
SN=("${NODES[@]:0:${STUDENT_NODES}}"); TN=("${NODES[@]:${STUDENT_NODES}}")
[[ ${#TN[@]} -ge 1 ]] || { echo "need >= 1 teacher node (STUDENT_NODES=${STUDENT_NODES}, nodes=${#NODES[@]})"; exit 2; }
PLACE_NODES="${SN[0]},$(IFS=,; echo "${TN[*]}")"       # placement: first entry = student head (ignored for GPUs), rest = teacher pool
export PYTHONPATH="${REPO}"
PLACE=(python3 -m mopd.distill.placement --nodes "${PLACE_NODES}" --teachers "${TEACHERS}" --weights "${WEIGHTS}")
"${PLACE[@]}" --emit json > "${OUT}/teacher_endpoints.json"
"${PLACE[@]}" --emit table | tee "${OUT}/placement.txt"
eval "$("${PLACE[@]}" --emit env)"
WORLD=$(( 8 * STUDENT_NODES )); GA=$(( BS / (PD * WORLD) ))
(( GA * PD * WORLD == BS )) || { echo "batch_size=${BS} not divisible by per_device_bs*world=$((PD*WORLD))"; exit 2; }
echo "[opd] config=${CONFIG} student nodes: ${SN[*]} (world=${WORLD}) | teacher nodes: ${TN[*]} | batch=${BS} grad_accum=${GA}"
# ---- teacher servers (vLLM prompt_logprobs; TEACHER_GPU_UTIL<=0.55 + max-num-batched 4096 avoid the log_softmax OOM)
PIDS=(); URLS=(); declare -A SEEN
for i in $(seq 0 $((N_SERVERS - 1))); do
  m="SERVER_${i}_MODEL"; n="SERVER_${i}_NODE"; g="SERVER_${i}_GPUS"; p="SERVER_${i}_PORT"; v="SERVER_${i}_VLLM_PORT"; t="SERVER_${i}_TP"; nm="SERVER_${i}_NAME"; u="SERVER_${i}_URL"
  tpf=""; [[ "${!t}" -ge 2 ]] && tpf="--disable-custom-all-reduce"
  cmd="VLLM_PORT=${!v} ${VLLM_ATTN:+VLLM_ATTENTION_BACKEND=${VLLM_ATTN} }python -m vllm.entrypoints.openai.api_server \
    --model '$(realpath "${!m}")' --served-model-name '${!nm}' --host 0.0.0.0 --port ${!p} \
    --tensor-parallel-size ${!t} --max-model-len ${TEACHER_MAX_LEN} --gpu-memory-utilization ${TEACHER_GPU_UTIL} \
    --dtype bfloat16 --max-logprobs ${TOPK} --logprobs-mode raw_logprobs --enable-prefix-caching \
    --max-num-batched-tokens ${TEACHER_MAX_BATCHED} --trust-remote-code ${tpf}"
  seen=${SEEN[${!n}]:-0}; delay=$(( seen * 20 )); SEEN[${!n}]=$(( seen + 1 ))
  ( (( delay > 0 )) && sleep "${delay}"; run_step "${!n}" "${!g}" "${OUT}/teacher_logs/server_${i}.log" "${cmd}" ) &
  PIDS+=("$!"); URLS+=("${!u}")
done
cleanup() { kill "${PIDS[@]}" "${SPIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
echo "[opd] waiting for ${N_SERVERS} teacher servers"; wait_http "${URLS[@]}" || { echo "[opd] teacher servers not ready; see ${OUT}/teacher_logs"; exit 4; }
echo "[opd] teachers ready"
# ---- student: one accelerate launch per student node (machine_rank r), rendezvous on the head node
CFG="$(accel_cfg "${GA}" "${OUT}")"
OV="--override teachers.endpoints_file=${OUT}/teacher_endpoints.json --override teachers.top_k=${TOPK}"
for ov in ${OVERRIDES}; do OV="${OV} --override ${ov}"; done
HEAD="${SN[0]}"; HEAD_IP="$(getent hosts "${HEAD}" | awk '{print $1}' | head -1)"; HEAD_IP="${HEAD_IP:-${HEAD}}"; PORT="${ACCEL_PORT:-29500}"
SPIDS=()
for r in $(seq 0 $((STUDENT_NODES - 1))); do
  run_step "${SN[$r]}" "0,1,2,3,4,5,6,7" "${OUT}/train$([[ $r -gt 0 ]] && echo "_rank${r}").log" \
    "accelerate launch --config_file '${CFG}' --num_machines ${STUDENT_NODES} --machine_rank ${r} \
     --main_process_ip ${HEAD_IP} --main_process_port ${PORT} --num_processes ${WORLD} \
     -m mopd.distill.train_opd --config '${CONFIG}' ${OV}" &
  SPIDS+=("$!")
done
# fail-fast: the first student step to exit decides; kill the others so a dead rank cannot leave a zombie job
wait -n "${SPIDS[@]}"; rc=$?; sleep 30; kill "${SPIDS[@]}" 2>/dev/null; wait "${SPIDS[@]}" 2>/dev/null
echo "[opd] student exit ${rc}"; grep -h "\[opd\] step .* wall" "${OUT}/train.log" 2>/dev/null | tail -5
exit ${rc}
