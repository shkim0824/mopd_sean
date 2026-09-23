#!/bin/bash
# Convert a NeMo-RL step (DCP weights) to an HF checkpoint dir, then restore the OLD-schema config/tokenizer of the init
# model (the transformers-5 config written by the converter loses rope fields for the eval stack -> 0 scores on long
# contexts). Only needed when a step has no policy/weights/model/consolidated; otherwise use scripts/mk_eval_model.sh.
#   STEP=outputs/rl/grpo_math_v3/step_200 OUT=models/rl_steps/rl-mathv3-ck200 REF=models/Qwen3-4B-OT3 sbatch -J k-sean-convert scripts/convert_nemo_step.sbatch
#SBATCH --exclude=ib-a100-cluster-a-n[033-036]   # user rule 2026-09-15 (CLI --exclude overrides; launchers pass a superset)
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 04:00:00
#SBATCH --output=logs/convert-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; set -e
STEP="$(realpath "${STEP:?}")"; OUT="$(realpath -m "${OUT:?}")"; REF="$(realpath "${REF:?}")"; mkdir -p "${OUT}"
srun --ntasks=1 --container-image="${NEMO_IMG}" --container-mounts=/mnt/datafs:/mnt/datafs,/tmp:/tmp --container-workdir=/opt/nemo-rl bash -c "
  /opt/nemo_rl_venv/bin/python examples/converters/convert_dcp_to_hf.py --config ${STEP}/config.yaml --dcp-ckpt-path ${STEP}/policy/weights --hf-ckpt-path ${OUT}"
for f in config.json generation_config.json tokenizer_config.json tokenizer.json vocab.json merges.txt special_tokens_map.json added_tokens.json chat_template.jinja; do [[ -f "${REF}/$f" ]] && cp -f "${REF}/$f" "${OUT}/$f"; done
ls "${OUT}"; echo CONVERT_OK
