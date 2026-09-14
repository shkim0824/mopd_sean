#!/bin/bash
# Domain-benchmark eval (MedQA / CaseHOLD / FinQA / MedXpertQA / PubMedQA / ...; mopd.eval.domains.benches) on N nodes
# x 8 DP vLLM workers; node p runs global shards p*8..p*8+7, then one aggregate step scores the shards. T=1.0 default.
#   MODEL=outputs/opd/<run>/checkpoint-200 TAG=opd-<run>-ck200-dom3-T1 BENCHMARKS=medqa,casehold,finqa \
#     sbatch -N 2 -J k-sean-domeval-<tag> $(prefer 2) scripts/eval_domain.sbatch          (LIMIT=48 for a smoke)
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 8:00:00
#SBATCH --output=logs/domeval-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs tmp/cache
MODEL="$(realpath "${MODEL:?set MODEL}")"; TAG="${TAG:?set TAG}"; BENCHMARKS="${BENCHMARKS:?set BENCHMARKS}"
OUT="${OUT:-${REPO}/outputs/eval_domain/${TAG}}"; mkdir -p "${OUT}"
LIMIT_ARG=""; [[ -n "${LIMIT:-}" ]] && LIMIT_ARG="--limit ${LIMIT}"
NODES=${SLURM_JOB_NUM_NODES}; TOTAL=$((NODES * 8)); MAXTOK="${MAX_TOKENS:-26624}"; TEMP="${TEMP:-1.0}"
echo "[domeval] nodes=${NODES} shards=${TOTAL} model=${MODEL} benchmarks=${BENCHMARKS} T=${TEMP} start=$(date +%s)"
srun --kill-on-bad-exit=0 --ntasks=${NODES} --ntasks-per-node=1 --export=ALL --container-image="${IMG}" --container-mounts="${MOUNT}" \
  --container-workdir="${REPO}" bash -lc "
${FWD_ENV} ${PRELUDE} cd '${REPO}'
node=\${SLURM_PROCID:-0}; echo \"[domeval] node=\$node host=\$(hostname)\"
pids=()
for g in 0 1 2 3 4 5 6 7; do
  rank=\$((node*8+g))
  # per-job/per-rank compile caches (shared /tmp inductor caches got corrupted by concurrent jobs)
  export VLLM_CACHE_ROOT=${REPO}/tmp/cache/vllm_${SLURM_JOB_ID}_\$rank TORCHINDUCTOR_CACHE_DIR=${REPO}/tmp/cache/ind_${SLURM_JOB_ID}_\$rank TRITON_CACHE_DIR=${REPO}/tmp/cache/triton_${SLURM_JOB_ID}_\$rank XDG_CACHE_HOME=${REPO}/tmp/cache/xdg
  # distinct torch.distributed/vLLM port base per worker (8 simultaneous engine starts raced on one ephemeral port)
  export VLLM_PORT=\$((41000 + g*1000))
  CUDA_VISIBLE_DEVICES=\$g python -m mopd.eval.domains.run_domain_eval \
    --model '${MODEL}' --benchmarks '${BENCHMARKS}' --out '${OUT}' ${LIMIT_ARG} \
    --max-tokens ${MAXTOK} --temperature ${TEMP} \
    --shard-rank \$rank --num-shards ${TOTAL} > '${OUT}'/worker_\$rank.log 2>&1 &
  pids+=(\$!)
done
rc=0; for p in \"\${pids[@]}\"; do wait \$p || { rc=1; touch '${OUT}'/failed_node\$node; }; done
echo \"[domeval] node=\$node done rc=\$rc\"
t0=\$(date +%s)
while [ \$(( \$(ls '${OUT}'/shard*.jsonl.done 2>/dev/null | wc -l) + 8 * \$(ls '${OUT}'/failed_node* 2>/dev/null | wc -l) )) -lt ${TOTAL} ]; do
  [ \$(( \$(date +%s) - t0 )) -gt 14400 ] && { echo barrier-timeout; break; }; sleep 20
done
echo \"[domeval] node=\$node barrier passed\"; exit \$rc
"
rc=$?
run_step "$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)" "" "${OUT}/aggregate.log" \
  "python -m mopd.eval.domains.run_domain_eval --aggregate-only --benchmarks '${BENCHMARKS}' --out '${OUT}' ${LIMIT_ARG}"
[[ -f "${OUT}/metrics.json" ]] && python3 -c "import json;m=json.load(open('${OUT}/metrics.json'));print({b:{k:round(100*v,1) for k,v in r.items() if isinstance(v,float)} for b,r in m.items()})"
echo "[domeval] finished rc=${rc} end=$(date +%s)"; exit ${rc}
