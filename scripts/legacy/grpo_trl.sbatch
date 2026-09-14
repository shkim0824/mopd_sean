#!/bin/bash
# LEGACY (paper recipe, trl 0.29 GRPO expert; the 4B-OT3 domain teachers were trained with NeMo-RL -> scripts/rl_nemo.sh).
# Per-domain RL expert on N nodes (8 GPU each), colocate vLLM per rank, ZeRO-2; configs/legacy_trl/rl_*.yaml
# (mopd.rl.train_grpo + mopd.rl.rewards; needs the legacy data/train/rl_*.jsonl built by mopd.data.prep_train).
#   CONFIG=configs/legacy_trl/rl_math.yaml OVERRIDES="model.init=models/Qwen3-4B-OT3 rl.output=outputs/rl/trl_math" \
#   sbatch -N 2 -J k-sean-rl-trl-math --cpus-per-task=32 --mem=600G -t 60:00:00 $(prefer 2) scripts/legacy/grpo_trl.sbatch
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 60:00:00
#SBATCH --output=logs/grpo-trl-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs
NUM_NODES="${SLURM_NNODES:?}"; WORLD=$(( NUM_NODES * 8 ))
CONFIG="${CONFIG:-configs/legacy_trl/rl_math.yaml}"
OVERRIDES="${OVERRIDES:-}"
read -r OUT PPS NG PD < <(python3 scripts/cfgval.py "${CONFIG}" rl.output rl.prompts_per_step rl.num_generations rl.per_device_bs | tr '\n' ' ')
for ov in ${OVERRIDES}; do
  case "${ov}" in rl.output=*) OUT="${ov#rl.output=}";; rl.prompts_per_step=*) PPS="${ov#rl.prompts_per_step=}";;
    rl.num_generations=*) NG="${ov#rl.num_generations=}";; rl.per_device_bs=*) PD="${ov#rl.per_device_bs=}";; esac
done
GEN=$(( PPS * NG )); GA=$(( GEN / (PD * WORLD) ))
(( GA * PD * WORLD == GEN )) || { echo "prompts_per_step*num_generations=${GEN} not divisible by per_device_bs*world=$((PD*WORLD))"; exit 2; }
mkdir -p "${OUT}"
CFG="$(accel_cfg "${GA}" "${OUT}")"
HEAD_IP="$(head_ip)"; PORT="${ACCEL_PORT:-29500}"
# vLLM sleep mode uses a CuMem memory pool that rejects PYTORCH_ALLOC_CONF=expandable_segments -> unset for sleep runs (fork study)
ALLOC_FIX=""; if [[ "${OVERRIDES:-}" == *"vllm_sleep=true"* || "${VLLM_SLEEP:-}" == "1" ]]; then ALLOC_FIX="export PYTORCH_ALLOC_CONF=; export PYTORCH_CUDA_ALLOC_CONF=;"; fi
OV=""; for ov in ${OVERRIDES}; do OV="${OV} --override ${ov}"; done
echo "[grpo] nodes=${NUM_NODES} world=${WORLD} gen_batch=${GEN} grad_accum=${GA} head=${HEAD_IP} out=${OUT}"
LAUNCH="accelerate launch --config_file '${CFG}' --num_machines ${NUM_NODES} --num_processes ${WORLD} \
 --machine_rank \$SLURM_NODEID --main_process_ip ${HEAD_IP} --main_process_port ${PORT} \
 -m mopd.rl.train_grpo --config '${CONFIG}' ${OV}"
ln -sfn "train.${SLURM_JOB_ID}.log" "${OUT}/train.log"   # keep every attempt's log (resumes overwrite otherwise)
srun --nodes="${NUM_NODES}" --ntasks-per-node=1 --wait=0 --export=ALL \
  --container-image="${IMG}" --container-mounts="${MOUNT}" --container-workdir="${REPO}" \
  bash -lc "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; ${FWD_ENV} ${ALLOC_FIX} ${PRELUDE} cd '${REPO}'; ${LAUNCH}" > "${OUT}/train.${SLURM_JOB_ID}.log" 2>&1
rc=$?; echo "[grpo] exit ${rc}"; exit ${rc}
