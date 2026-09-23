#!/bin/bash
# IH-Challenge attack synthesis for one training round (or for the held-out benchmark): 8 data-parallel vLLM shards on
# one node (training container + mopd venv), then a merge step. Attacker = models/Qwen3-4B (official, frozen) unless
# ATTACKER is given (ATTACKER=<defender> = the recipe's v1 self-play); defender = the current policy.
#   SKELETONS=data/domains/ih-challenge/skeletons_train_r0.jsonl DEFENDER=models/Qwen3-4B-OT3 OUT=data/rl_pools/ihc_r0_4b.jsonl \
#     sbatch -J k-sean-ihc-synth-r0 $(place 1) scripts/ihc_synth.sbatch        (BUDGET=3 default; LIMIT=64 for a smoke)
#SBATCH --exclude=ib-a100-cluster-a-n[033-036]   # user rule 2026-09-15 (CLI --exclude overrides; launchers pass a superset)
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 6:00:00
#SBATCH --output=logs/ihc-synth-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs tmp/cache
SKELETONS="${SKELETONS:?set SKELETONS}"; DEFENDER="$(realpath "${DEFENDER:?set DEFENDER}")"; ATTACKER="$(realpath "${ATTACKER:-models/Qwen3-4B}")"   # frozen official attacker (think off); ATTACKER=<defender> for self-play
OUT="${OUT:?set OUT}"; BUDGET="${BUDGET:-3}"; LIMIT_ARG=""; [[ -n "${LIMIT:-}" ]] && LIMIT_ARG="--limit ${LIMIT}"
TP="${TP:-1}"; NSH=$((8 / TP)); EXTRA="${EXTRA:-}"; [[ "${ATK_THINK:-0}" == "1" ]] && EXTRA="${EXTRA} --atk-think --atk-tokens ${ATK_TOKENS:-4096}"
mkdir -p "$(dirname "${OUT}")"
NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
echo "[ihc-synth] skeletons=${SKELETONS} defender=${DEFENDER} attacker=${ATTACKER} budget=${BUDGET} out=${OUT} start=$(date +%s)"
srun --nodes=1 --ntasks=1 --export=ALL --container-image="${IMG}" --container-mounts="${MOUNT}" --container-workdir="${REPO}" bash -lc "
${FWD_ENV} ${PRELUDE} cd '${REPO}'; pids=()
for ((i=0;i<${NSH};i++)); do
  g=\$i; lo=\$((i*${TP})); gpus=\$lo; for ((k=1;k<${TP};k++)); do gpus=\"\$gpus,\$((lo+k))\"; done
  export CUDA_VISIBLE_DEVICES=\$gpus VLLM_PORT=\$((41000 + i*1000))
  $(cache_env '$g' | sed 's/\$SLURM_JOB_ID/'"${SLURM_JOB_ID}"'/g')
  python -m mopd.data.domains.ihc_attack_synth --skeletons '${SKELETONS}' --attacker '${ATTACKER}' --defender '${DEFENDER}' \
    --out '${OUT}' --budget ${BUDGET} --shard-rank \$i --num-shards ${NSH} --tp ${TP} ${EXTRA} ${LIMIT_ARG} > logs/ihc-synth-${SLURM_JOB_ID}-shard\$i.log 2>&1 &
  pids+=(\$!)
done
rc=0; for p in \"\${pids[@]}\"; do wait \$p || rc=1; done
python -m mopd.data.domains.ihc_attack_synth --skeletons '${SKELETONS}' --attacker '${ATTACKER}' --defender '${DEFENDER}' --out '${OUT}' --merge
exit \$rc"
rc=$?; echo "[ihc-synth] finished rc=${rc} rows=$(wc -l < "${OUT}" 2>/dev/null) end=$(date +%s)"; exit ${rc}
