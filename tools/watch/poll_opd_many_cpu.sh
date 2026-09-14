#!/bin/bash
# [mopd_sean operator tool] laptop-side watcher; cluster paths point at $S/mopd_sean (outputs/eval_domain, outputs/eval_6bench, models/teacher-*)
# Watch ALL OPD ckpt auto-eval logs + OPD eval results at once. Exits (=notifies) when a new submission line
# or a new opd-* metrics.json appears; prints what changed. Re-arm after each notification.
H="ssh -p 38483 -i /Users/sean.shk/.ssh/id_rsa_braincloud -o ConnectTimeout=20 -o BatchMode=yes bc-user@kap-instance.onkakao.net"; R=/data/ib-a100-cluster-a-pri-lmalign_942/personal/sean/mopd_sean
STATE=/private/tmp/claude-501/-Users-sean-shk/b02e2f5b-b5d8-4115-973a-39d3310221eb/scratchpad/opd_poll_state_cpu.txt; touch $STATE
while true; do
  out=$($H "cd $R; for f in logs/auto_eval_opd_*.log logs/auto_eval_sftw_*.log; do grep -H 'submitted evals\|submitted casehold\|submitted medqa\|submitted finals\|AUTO-EVAL DONE' \$f; done 2>/dev/null; for m in outputs/eval_domain/opd-*/metrics.json outputs/eval_domain/rl4-med-v[456]*/metrics.json outputs/eval_domain/sftw-*/metrics.json outputs/eval_domain/merged-*/metrics.json; do [ -f \$m ] && echo \"RESULT \$(dirname \$m | xargs basename) \$(tr -d '\n ' < \$m | cut -c1-90)\"; done 2>/dev/null; for m in /data/ib-a100-cluster-a-pri-lmalign_942/personal/sean/mopd_sean/outputs/eval_6bench/full-opd-*/metrics.json /data/ib-a100-cluster-a-pri-lmalign_942/personal/sean/mopd_sean/outputs/eval_6bench/full-sftw-*/metrics.json /data/ib-a100-cluster-a-pri-lmalign_942/personal/sean/mopd_sean/outputs/eval_6bench/full-merged-*/metrics.json; do [ -f \$m ] && echo \"FULL \$(dirname \$m | xargs basename) \$(python3 -c \"import json,sys;m=json.load(open(sys.argv[1]));print({b:round(r['score'],1) for b,r in m['benchmarks'].items()})\" \$m)\"; done 2>/dev/null" 2>/dev/null)
  [ -z "$out" ] && { sleep 300; continue; }
  new=$(echo "$out" | grep -vxF -f $STATE)
  if [ -n "$new" ]; then echo "$new" | cut -c1-220; echo "$out" > $STATE; exit 0; fi
  sleep 300
done
