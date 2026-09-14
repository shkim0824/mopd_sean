#!/bin/bash
# [mopd_sean operator tool] laptop-side watcher; cluster paths point at $S/mopd_sean (outputs/eval_domain, outputs/eval_6bench, models/teacher-*)
# args: JOBID TAG -> on completion print 6-bench scores + domain averages from mopd/outputs/eval/<TAG>/metrics.json
J="$1"; TAG="$2"; H="ssh -o ConnectTimeout=15 ib-a100-controller"; R=/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean; fails=0
while true; do
  st=$($H "squeue -j $J -h -o %T 2>/dev/null; echo RC" 2>/dev/null)
  if ! echo "$st" | grep -q RC; then fails=$((fails+1)); [ $fails -ge 40 ] && { echo "### $TAG ($J) UNREACHABLE"; exit 2; }; sleep 200; continue; fi
  fails=0; state=$(echo "$st" | head -1)
  if [ -z "$state" ] || [ "$state" = "RC" ]; then
    $H "echo \"### FULL-EVAL DONE $TAG ($J) state=\$(sacct -j $J -X -n -o State|tr -d ' ') elapsed=\$(sacct -j $J -X -n -o Elapsed|tr -d ' ')\"; python3 -c \"
import json;m=json.load(open('$R/outputs/eval_domain/$TAG/metrics.json'));s={b:round(r['score'],1) for b,r in m['benchmarks'].items()}
math=sum(s[b] for b in ('aime24','aime25','aime26'))/3; print(s, 'domain_avg', {'math':math,'code':s.get('lcb_v6'),'if':(s.get('ifeval',0)+s.get('ifbench',0))/2})\" 2>&1 | tail -1" 2>/dev/null
    exit 0
  fi
  sleep 300
done
