#!/bin/bash
# IH-Challenge attack synthesis for one training round (or for the held-out benchmark): 8 data-parallel vLLM shards on
# one node (training container + mopd venv), then a merge step. Attacker = defender = the current policy (recipe v1
# self-play) unless ATTACKER is given.
#   SKELETONS=data/domains/ih-challenge/skeletons_train_r0.jsonl DEFENDER=models/Qwen3-4B-OT3 OUT=data/rl_pools/ihc_r0_4b.jsonl \
#     sbatch -J k-sean-ihc-synth-r0 $(place 1) scripts/ihc_synth.sbatch        (BUDGET=3 default; LIMIT=64 for a smoke)
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 6:00:00
#SBATCH --output=logs/ihc-synth-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs tmp/cache
SKELETONS="${SKELETONS:?set SKELETONS}"; DEFENDER="$(realpath "${DEFENDER:?set DEFENDER}")"; ATTACKER="$(realpath "${ATTACKER:-${DEFENDER}}")"
OUT="${OUT:?set OUT}"; BUDGET="${BUDGET:-3}"; LIMIT_ARG=""; [[ -n "${LIMIT:-}" ]] && LIMIT_ARG="--limit ${LIMIT}"
mkdir -p "$(dirname "${OUT}")"
NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
echo "[ihc-synth] skeletons=${SKELETONS} defender=${DEFENDER} attacker=${ATTACKER} budget=${BUDGET} out=${OUT} start=$(date +%s)"
srun --nodes=1 --ntasks=1 --export=ALL --container-image="${IMG}" --container-mounts="${MOUNT}" --container-workdir="${REPO}" bash -lc "
${FWD_ENV} ${PRELUDE} cd '${REPO}'; pids=()
for g in 0 1 2 3 4 5 6 7; do
  export CUDA_VISIBLE_DEVICES=\$g VLLM_PORT=\$((41000 + g*1000))
  $(cache_env '$g' | sed 's/\$SLURM_JOB_ID/'"${SLURM_JOB_ID}"'/g')
  python -m mopd.data.domains.ihc_attack_synth --skeletons '${SKELETONS}' --attacker '${ATTACKER}' --defender '${DEFENDER}' \
    --out '${OUT}' --budget ${BUDGET} --shard-rank \$g --num-shards 8 ${LIMIT_ARG} > logs/ihc-synth-${SLURM_JOB_ID}-shard\$g.log 2>&1 &
  pids+=(\$!)
done
rc=0; for p in \"\${pids[@]}\"; do wait \$p || rc=1; done
python -m mopd.data.domains.ihc_attack_synth --skeletons '${SKELETONS}' --attacker '${ATTACKER}' --defender '${DEFENDER}' --out '${OUT}' --merge
exit \$rc"
rc=$?; echo "[ihc-synth] finished rc=${rc} rows=$(wc -l < "${OUT}" 2>/dev/null) end=$(date +%s)"; exit ${rc}
