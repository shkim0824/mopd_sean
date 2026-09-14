#!/bin/bash
# [mopd_sean operator tool] laptop-side watcher; cluster paths point at $S/mopd_sean (outputs/eval_domain, outputs/eval_6bench, models/teacher-*)
# args: RUN -> exits when outputs/opd/RUN has opd_done.json + final weights AND job k-sean-opd-RUN has left the queue
RUN="$1"; R=/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean; fails=0
while true; do
  out=$(ssh -o ConnectTimeout=15 ib-a100-controller "[ -f $R/outputs/opd/$RUN/opd_done.json ] && [ -f $R/outputs/opd/$RUN/final/model.safetensors.index.json ] && ! squeue -h -n k-sean-opd-$RUN -o %T | grep -q . && echo READY; echo RC" 2>/dev/null)
  if ! echo "$out" | grep -q RC; then fails=$((fails+1)); [ $fails -ge 40 ] && { echo "### $RUN waiter UNREACHABLE"; exit 2; }; sleep 120; continue; fi
  fails=0
  if echo "$out" | grep -q READY; then echo "### READY-FOR-FINALS $RUN $(date '+%H:%M')"; exit 0; fi
  sleep 90
done
