#!/bin/bash
# Delete per-job vLLM/inductor/triton compile caches under tmp/cache whose job is no longer in the queue
# (they are re-created on demand; 6,523 dirs / 1.2 TB had piled up by 2026-09-14). Dry run without --yes.
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; REPO="$(realpath "${_here}/../..")"; C="${REPO}/tmp/cache"
RUN=$(squeue -h -o "%i" | sort -u); n=0; b=0
for d in "${C}"/vllm_* "${C}"/ind_* "${C}"/triton_*; do
  [[ -d "${d}" ]] || continue; j=$(basename "${d}" | cut -d_ -f2)
  echo "${RUN}" | grep -qx "${j}" && continue
  n=$((n+1)); [[ "${1:-}" == "--yes" ]] && rm -rf "${d}"
done
echo "$([[ "${1:-}" == "--yes" ]] && echo removed || echo would-remove) ${n} finished-job cache dirs under ${C} (pass --yes to delete)"
