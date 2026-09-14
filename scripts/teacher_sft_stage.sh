#!/bin/bash
# teacher_sft_stage.sh <gen_tag> <pool.jsonl> <out.jsonl> <config> [keep] [nodes]: build a TEACHER-SFT set from finished
# big-teacher shards (outputs/teacher_gen/<gen_tag>, scripts/gen_teacher.sbatch) and submit the teacher SFT.
#   default   : mopd.teacher.reject_filter  (verified-correct traces, <= keep per prompt, near-duplicate filter)
#   SELECT=1  : mopd.teacher.select_traces  (the C6 recipe: rank by enum_len, --max-think-chars 16000; SELECT_ARGS for more)
# The config's data.sources must point at <out.jsonl> (configs/sft/distill_*.yaml).
#   bash scripts/teacher_sft_stage.sh gen-med-35b-k4 data/distill_pools/med_pool.jsonl data/sft_distill/med_sft.jsonl configs/sft/distill_med_c6_4ep.yaml 2 4
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"; cd "${REPO}"
GEN="$1"; POOL="$2"; OUTF="$3"; CONFIG="$4"; KEEP="${5:-2}"; NODES="${6:-4}"
GD="${REPO}/outputs/teacher_gen/${GEN}"
NSH=$(ls "${GD}"/shard*.jsonl 2>/dev/null | wc -l); NDONE=$(ls "${GD}"/*.done 2>/dev/null | wc -l)
[[ "${NSH}" -ge 1 && "${NDONE}" -ge "${NSH}" ]] || { echo "ABORT: ${GD}: ${NDONE}/${NSH} shards finished"; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "ABORT: no config ${CONFIG}"; exit 1; }
mkdir -p "$(dirname "${OUTF}")"
export PYTHONPATH="${REPO}" DOMAINS_DATA="${REPO}/data/domains" DOMAINS_TP="${REPO}/third_party"
if [[ "${SELECT:-0}" == "1" ]]; then
  python3 -m mopd.teacher.select_traces --gen-dirs "${GD}" --pools "${POOL}" --out "${OUTF}" --keep "${KEEP}" ${SELECT_ARGS:-} 2>&1 | tail -15
else
  python3 -m mopd.teacher.reject_filter --gen-dir "${GD}" --pool "${POOL}" --out "${OUTF}" --keep "${KEEP}" 2>&1 | tail -15
fi
[[ -s "${OUTF}" ]] || { echo "ABORT: filter produced no rows (${OUTF})"; exit 1; }
N=$(wc -l < "${OUTF}"); echo "teacher-SFT rows: ${N} -> ${OUTF}"; [[ "${N}" -ge "${MIN_ROWS:-1000}" ]] || { echo "ABORT: fewer than ${MIN_ROWS:-1000} rows"; exit 1; }
grep -q "$(basename "${OUTF}")" "${CONFIG}" || echo "WARNING: ${CONFIG} does not reference $(basename "${OUTF}") in data.sources"
NAME="$(basename "${CONFIG}" .yaml)"
J=$(CONFIG="${CONFIG}" sbatch -N "${NODES}" $(prefer "${NODES}") -J "${JP}-sft-${NAME}" --cpus-per-task=32 --mem=600G -t 48:00:00 --parsable scripts/sft.sbatch)
echo "SFT-SUBMITTED ${J} ${NAME} (${N} rows, ${NODES} nodes)"
