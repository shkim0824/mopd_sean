#!/bin/bash
# sft_warmup_stage.sh <gen_tag> <pool.jsonl> <out_name> <n_rows> <config> [keep]: rejection-filter finished teacher shards
# (outputs/teacher_gen/<gen_tag>) into data/sft_warmup/<out_name>_accepted.jsonl, sample exactly <n_rows> (seed 42) into
# data/sft_warmup/<out_name>.jsonl, then submit the SFT (config must point data.sources at that file).
#   bash scripts/sft_warmup_stage.sh gen-law-teacher-k1 data/sft_warmup/law_gen_pool10k.jsonl law_sft6400 6400 configs/sft/warmup_law.yaml
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"; cd "${REPO}"
GEN="$1"; POOL="$2"; NAME="$3"; NROWS="$4"; CONFIG="$5"; KEEP="${6:-1}"
GD="${REPO}/outputs/teacher_gen/${GEN}"; NDONE=$(ls "${GD}"/*.done 2>/dev/null | wc -l); [[ "${NDONE}" -ge 1 ]] || { echo "ABORT: no finished shards in ${GD}"; exit 1; }
export PYTHONPATH="${REPO}" DOMAINS_DATA="${REPO}/data/domains" DOMAINS_TP="${REPO}/third_party"
python3 -m mopd.teacher.reject_filter --gen-dir "${GD}" --pool "${POOL}" --out "data/sft_warmup/${NAME}_accepted.jsonl" --keep "${KEEP}" --max-chars 70000 2>&1 | tail -15 || { echo "ABORT: reject_filter failed"; exit 1; }
N=$(wc -l < "data/sft_warmup/${NAME}_accepted.jsonl"); echo "accepted rows: ${N}"; [[ "${N}" -ge "${NROWS}" ]] || { echo "ABORT: fewer than ${NROWS} accepted rows (${N})"; exit 1; }
python3 - "${NAME}" "${NROWS}" <<'PY'
import json, random, sys, statistics
name, n = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(f"data/sft_warmup/{name}_accepted.jsonl")]
random.seed(42); random.shuffle(rows); rows = rows[:n]
with open(f"data/sft_warmup/{name}.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
L = [r["meta"]["teacher_chars"] for r in rows]
print(f"wrote {n} rows; teacher chars median {statistics.median(L):.0f} p90 {sorted(L)[int(0.9*len(L))]} max {max(L)}")
PY
J=$(CONFIG="${CONFIG}" sbatch -N "${NODES:-1}" $(prefer "${NODES:-1}") -J "${JP}-sft-${NAME}" --cpus-per-task=32 --mem=600G -t 24:00:00 --parsable scripts/sft.sbatch); echo "SFT-SUBMITTED ${J} ${NAME}"
