#!/bin/bash
# ONE-JOB evaluation: in-domain (MedQA/CaseHOLD/FinQA) + IF + OOD (math, code, safety, GPQA).
# N nodes x 8 DP vLLM workers; every GPU loads the student ONCE and serves all benchmarks with
# their own sampling protocol (mopd/eval/registry.py). Replaces the eval_domain + eval_6bench pair.
#   MODEL=outputs/opd/<run>/checkpoint-200 TAG=all-opd-<run>-ck200 \
#     sbatch -N 2 -J k-sean-evalall-<tag> $(prefer 2) scripts/eval_all.sbatch
#   BENCHMARKS=gpqa,harmbench,sycophancy,truthfulqa_mc ...      # safety+GPQA only
#   LIMIT=4 ...                                                 # smoke
#SBATCH --exclude=ib-a100-cluster-a-n[033-036]   # user rule 2026-09-15 (CLI --exclude overrides)
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 4:00:00
#SBATCH --output=logs/evalall-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs tmp/cache
MODEL="$(realpath "${MODEL:?set MODEL}")"; TAG="${TAG:?set TAG}"; NUM_NODES="${SLURM_NNODES:?}"
OUT="${OUT:-${REPO}/outputs/eval_all/${TAG}}"; mkdir -p "${OUT}"
BENCHMARKS="${BENCHMARKS:-medqa,casehold,finqa,ifeval,ifbench,aime24,aime25,aime26,lcb_v6,gpqa,harmbench,sycophancy,truthfulqa_mc}"
CLS="${HARMBENCH_CLS:-${REPO}/models/HarmBench-Llama-2-13b-cls}"

ARGS="--model '${MODEL}' --out '${OUT}' --benchmarks '${BENCHMARKS}' --gpus ${GPUS:-0-7}"
ARGS="${ARGS} --max-model-len ${MAX_MODEL_LEN:-40960} --num-nodes ${NUM_NODES} --seed ${SEED:-0}"
[[ -n "${LIMIT:-}" ]] && ARGS="${ARGS} --limit ${LIMIT}"
[[ -n "${N_OF:-}" ]] && ARGS="${ARGS} --n-of ${N_OF}"
[[ -n "${MAX_TOKENS_OF:-}" ]] && ARGS="${ARGS} --max-tokens-of ${MAX_TOKENS_OF}"
[[ -n "${CHUNK:-}" ]] && ARGS="${ARGS} --chunk ${CHUNK}"
ARGS="${ARGS} ${EXTRA:-}"
HEAD="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
echo "[evalall] nodes=${NUM_NODES} head=${HEAD} out=${OUT}"
echo "[evalall] ${ARGS}"

# ---- 1. generation + log-prob scoring: one engine per GPU, all benchmarks
srun --nodes="${NUM_NODES}" --ntasks-per-node=1 --wait=0 --kill-on-bad-exit=0 --export=ALL \
  --container-image="${IMG}" --container-mounts="${MOUNT}" --container-workdir="${REPO}" \
  bash -lc "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; export EVAL_DTYPE='${EVAL_DTYPE:-bfloat16}'; ${FWD_ENV} ${PRELUDE} cd '${REPO}'; \
  python -m mopd.eval.run_all ${ARGS} --node-rank \$SLURM_NODEID" > "${OUT}/gen.log" 2>&1
rc=$?; echo "[evalall] generation exit ${rc}"

# ---- 2. merge shards -> <bench>.gen.jsonl / <bench>.lp.jsonl  (CPU)
run_step "${HEAD}" "" "${OUT}/merge.log" "python -m mopd.eval.run_all ${ARGS} --merge-only"
rc2=$?; echo "[evalall] merge exit ${rc2}"

# ---- 3. HarmBench judge: the official 13B classifier, one GPU, same job
if [[ ",${BENCHMARKS}," == *",harmbench,"* ]]; then
  if [[ -d "${CLS}" ]]; then
    run_step "${HEAD}" "0" "${OUT}/judge.log" \
      "python -m mopd.eval.harmbench_judge --out '${OUT}' --cls '${CLS}' $( [[ -n "${LIMIT:-}" ]] && echo --limit ${LIMIT} )"
    echo "[evalall] judge exit $?"
  else
    echo "[evalall] WARNING: classifier ${CLS} missing -> harmbench stays unlabelled"
  fi
fi

# ---- 3b. StrongREJECT judge: the authors' fine-tuned evaluator (LoRA on gemma-2b), one GPU
if [[ ",${BENCHMARKS}," == *",strongreject,"* ]]; then
  run_step "${HEAD}" "0" "${OUT}/sr_judge.log" \
    "python -m mopd.eval.strongreject_judge --out '${OUT}' $( [[ -n "${LIMIT:-}" ]] && echo --limit ${LIMIT} )"
  echo "[evalall] strongreject judge exit $?"
fi

# ---- 3c. SORRY-Bench judge: the authors' fine-tuned 7B judge, one GPU
if [[ ",${BENCHMARKS}," == *",sorrybench,"* ]]; then
  run_step "${HEAD}" "0" "${OUT}/sb_judge.log" \
    "python -m mopd.eval.sorrybench_judge --out '${OUT}' $( [[ -n "${LIMIT:-}" ]] && echo --limit ${LIMIT} )"
  echo "[evalall] sorrybench judge exit $?"
fi

# ---- 4. grade everything -> metrics.json  (CPU; LCB grading is the slow part)
run_step "${HEAD}" "" "${OUT}/eval.log" "python -m mopd.eval.run_all ${ARGS} --aggregate-only"
rc3=$?; echo "[evalall] aggregate exit ${rc3}"
[[ -f "${OUT}/metrics.json" ]] && python3 -c "
import json; m=json.load(open('${OUT}/metrics.json'))
print({k:(None if v is None else round(v,2)) for k,v in m['scores'].items()})
print(m.get('group_avg'))"
exit $(( rc + rc2 + rc3 ))
