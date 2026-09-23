#!/bin/bash
# 6-benchmark OOD/general eval (AIME24/25/26, LiveCodeBench v6, IFEval, IFBench) on N nodes: each node runs 8 vLLM DP
# workers on its shard subset, node 0 merges + grades. Standard OT3 setting = 32k tokens, qwen-thinking preset.
#   MODEL=outputs/opd/<run>/checkpoint-200 TAG=full-opd-<run>-ck200-32k MAX_TOKENS=32768 MAX_MODEL_LEN=40960 \
#     sbatch -N 2 -J k-sean-eval6-<tag> --cpus-per-task=32 --mem=600G -t 4:00:00 $(prefer 2) scripts/eval_6bench.sbatch
#SBATCH --exclude=ib-a100-cluster-a-n[033-036]   # user rule 2026-09-15 (CLI --exclude overrides; launchers pass a superset)
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 4:00:00
#SBATCH --output=logs/eval6-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs
MODEL="${MODEL:?set MODEL}"; TAG="${TAG:?set TAG}"; NUM_NODES="${SLURM_NNODES:?}"
OUT="${OUT:-outputs/eval_6bench/${TAG}}"; mkdir -p "${OUT}"
ARGS="--model '$(realpath "${MODEL}")' --out '${OUT}' --gpus 0-7 --preset ${PRESET:-train} --max-tokens ${MAX_TOKENS:-32768} --max-model-len ${MAX_MODEL_LEN:-40960}"
ARGS="${ARGS} --benchmarks ${BENCHMARKS:-aime24,aime25,aime26,lcb_v6,ifeval,ifbench} --prompt-mode ${PROMPT_MODE:-chat} --thinking ${THINKING:-true} --num-nodes ${NUM_NODES}"
[[ -n "${N:-}" ]] && ARGS="${ARGS} --n ${N}"
[[ -n "${N_MATH:-}" ]] && ARGS="${ARGS} --n-math ${N_MATH}"
[[ -n "${N_CODE:-}" ]] && ARGS="${ARGS} --n-code ${N_CODE}"
[[ -n "${N_IF:-}" ]] && ARGS="${ARGS} --n-if ${N_IF}"
[[ -n "${LIMIT:-}" ]] && ARGS="${ARGS} --limit ${LIMIT}"
[[ -n "${LCB_START:-}" ]] && ARGS="${ARGS} --lcb-start-date ${LCB_START}"
[[ -n "${LCB_END:-}" ]] && ARGS="${ARGS} --lcb-end-date ${LCB_END}"
ARGS="${ARGS} ${EXTRA:-}"
echo "[eval6] nodes=${NUM_NODES} ${ARGS}"
srun --nodes="${NUM_NODES}" --ntasks-per-node=1 --wait=0 --kill-on-bad-exit=0 --export=ALL --container-image="${IMG}" --container-mounts="${MOUNT}" \
  --container-workdir="${REPO}" bash -lc "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; ${FWD_ENV} ${PRELUDE} cd '${REPO}'; \
  python -m mopd.eval.run_eval ${ARGS} --node-rank \$SLURM_NODEID" > "${OUT}/gen.log" 2>&1
rc=$?; echo "[eval6] generation exit ${rc}"
run_step "$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)" "" "${OUT}/eval.log" "python -m mopd.eval.run_eval ${ARGS} --aggregate-only"
rc2=$?; echo "[eval6] aggregate exit ${rc2}"
[[ -f "${OUT}/metrics.json" ]] && python3 -c "import json;m=json.load(open('${OUT}/metrics.json'));print({b:round(r['score'],2) for b,r in m['benchmarks'].items()}, m['domain_avg'])"
exit $(( rc + rc2 ))
