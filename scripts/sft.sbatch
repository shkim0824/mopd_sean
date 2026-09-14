#!/bin/bash
# SFT on N nodes (8 GPU each, DeepSpeed ZeRO-2 data-parallel via accelerate). One script for every SFT stage:
#   OT3 general SFT (configs/sft/ot3_*.yaml), teacher SFT distillation (configs/sft/distill_*.yaml),
#   off-policy warm-ups (configs/sft/warmup_*.yaml). Resume: resubmit the same line (train.resume=auto).
#   cd $REPO && CONFIG=configs/sft/distill_med_c6_4ep.yaml sbatch -N 2 -J k-sean-sft-med-c6 $(prefer 2) scripts/sft.sbatch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 48:00:00
#SBATCH --output=logs/sft-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs
CONFIG="${CONFIG:?set CONFIG=configs/sft/<x>.yaml}"
OVERRIDES="${OVERRIDES:-}"
OUT="${OUT:-$(python3 scripts/cfgval.py "${CONFIG}" train.output)}"
for ov in ${OVERRIDES}; do [[ "${ov}" == train.output=* ]] && OUT="${ov#train.output=}"; done
mkdir -p "${OUT}"
GA="${GRAD_ACCUM:-$(python3 scripts/cfgval.py "${CONFIG}" train.grad_accum)}"
for ov in ${OVERRIDES}; do [[ "${ov}" == train.grad_accum=* ]] && GA="${ov#train.grad_accum=}"; done
CFG="$(accel_cfg "${GA}" "${OUT}")"
OV=""; for ov in ${OVERRIDES}; do OV="${OV} --override ${ov}"; done
NUM_NODES="${SLURM_NNODES:?}"; WORLD=$(( NUM_NODES * 8 )); HEAD_IP="$(head_ip)"
echo "[sft] repo=${REPO} config=${CONFIG} out=${OUT} nodes=${NUM_NODES} grad_accum=${GA} overrides=${OVERRIDES}"
LAUNCH="accelerate launch --config_file '${CFG}' --num_machines ${NUM_NODES} --num_processes ${WORLD} \
 --machine_rank \$SLURM_NODEID --main_process_ip ${HEAD_IP} --main_process_port ${ACCEL_PORT:-29591} \
 -m mopd.sft.train_sft --config '${CONFIG}' ${OV}"
ln -sfn "train.${SLURM_JOB_ID}.log" "${OUT}/train.log"   # keep every attempt's log (resumes overwrite otherwise)
srun --nodes="${NUM_NODES}" --ntasks-per-node=1 --wait=0 --export=ALL --container-image="${IMG}" --container-mounts="${MOUNT}" \
  --container-workdir="${REPO}" bash -lc "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; ${FWD_ENV} ${PRELUDE} cd '${REPO}'; ${LAUNCH}" > "${OUT}/train.${SLURM_JOB_ID}.log" 2>&1
rc=$?; echo "[sft] exit ${rc}"; exit ${rc}
