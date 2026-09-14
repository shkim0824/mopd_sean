#!/bin/bash
# [mopd_sean operator tool] laptop-side watcher; cluster paths point at $S/mopd_sean (outputs/eval_domain, outputs/eval_6bench, models/teacher-*)
# args: JOBID TAG -> on completion print domain-eval metrics (accuracy / trunc) from mopd_domains/outputs/eval/<TAG>/metrics.json
J="$1"; TAG="$2"; H="ssh -o ConnectTimeout=15 ib-a100-controller"; R=/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean; fails=0
while true; do
  st=$($H "squeue -j $J -h -o %T 2>/dev/null; echo RC" 2>/dev/null)
  if ! echo "$st" | grep -q RC; then fails=$((fails+1)); [ $fails -ge 40 ] && { echo "### $TAG ($J) UNREACHABLE"; exit 2; }; sleep 200; continue; fi
  fails=0; state=$(echo "$st" | head -1)
  if [ -z "$state" ] || [ "$state" = "RC" ]; then
    $H "echo \"### DOMEVAL DONE $TAG ($J) state=\$(sacct -j $J -X -n -o State|tr -d ' ') elapsed=\$(sacct -j $J -X -n -o Elapsed|tr -d ' ')\"; python3 -c \"import json;m=json.load(open('$R/outputs/eval_domain/$TAG/metrics.json'));print({b:{k:round(v,4) for k,v in r.items() if isinstance(v,float)} for b,r in m.items()})\" 2>&1 | tail -1" 2>/dev/null
    exit 0
  fi
  sleep 300
done
