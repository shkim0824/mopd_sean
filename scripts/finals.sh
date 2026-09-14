#!/bin/bash
# finals.sh <run>: for a FINISHED OPD/MOPD run (outputs/opd/<run>/opd_done.json + final/): strip optimizer states from
# every checkpoint (weights kept) and submit the 6-bench eval of final/ (tag full-opd-<run>-final-32k).
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"
RUN="$1"; D="${REPO}/outputs/opd/${RUN}"; cd "${REPO}"
[[ -f "${D}/opd_done.json" ]] || { echo "ERROR: ${D}/opd_done.json missing (run not finished)"; exit 1; }
[[ -f "${D}/final/config.json" && -f "${D}/final/model.safetensors.index.json" ]] || { echo "ERROR: ${D}/final incomplete"; exit 1; }
if squeue -h -n "${JP}-opd-${RUN}" -o %T | grep -q .; then echo "ERROR: job ${JP}-opd-${RUN} still in queue; not stripping"; exit 1; fi
bash scripts/strip_optimizer.sh "${D}"
FT="full-opd-${RUN}-final-32k"
J=$(MODEL="${D}/final" TAG="${FT}" MAX_TOKENS=32768 MAX_MODEL_LEN=40960 sbatch -N 2 $(prefer 2) -J "${JP}-eval6-${FT}" --cpus-per-task=32 --mem=600G -t 12:00:00 --parsable scripts/eval_6bench.sbatch)
echo "FULLEVAL ${J} ${FT}"
