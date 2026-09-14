#!/bin/bash
# [mopd_sean operator tool] laptop-side watcher; cluster paths point at $S/mopd_sean (outputs/eval_domain, outputs/eval_6bench, models/teacher-*)
# args: JOBID DOMAIN VARIANT TOPK [TRIES] -> report on end; if no outputs/opd/<d>-<v>/final, resubmit (auto-resume from last full ckpt), max 3
J="$1"; D="$2"; V="$3"; K="$4"; TRIES=${5:-0}; SN=${SN:-1}; NN=${NN:-2}
H="ssh -o ConnectTimeout=15 ib-a100-controller"; S=/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean; R=$S/mopd_sean; OUTD=$R/outputs/opd/$D-$V; fails=0
CFG=$R/configs/opd_${D}_${V}.yaml
if [ "$D" = law ]; then T="law:$R/models/teacher-law"; W="law=1.0"
elif [ "$D" = if ]; then T="if:$S/mopd_sean/models/teacher-if"; W="if=1.0"
elif [ "$D" = med ]; then T="med:$R/outputs/sft/sft_distill_med_c6_4ep/checkpoint-3368"; W="med=1.0"
elif [ "$D" = mopd ]; then T="law:$R/models/teacher-law,fin:$R/models/teacher-fin,if:$S/mopd_sean/models/teacher-if,med:$R/outputs/sft/sft_distill_med_c6_4ep/checkpoint-3368"; W="law=1.0,fin=1.0,if=1.0,med=1.0"; CFG=$R/configs/mopd_4dom_pg${V}.yaml; OUTD=$R/outputs/opd/mopd-4dom-$V
elif [ "$D" = mopdcfg ]; then CFG=$R/configs/$V.yaml; OUTD=$R/outputs/opd/$V; T="${TEACHERS_OVERRIDE:?}"; W="${WEIGHTS_OVERRIDE:?}"
else T="fin:$R/models/teacher-fin"; W="fin=1.0"; fi
while true; do
  st=$($H "squeue -j $J -h -o %T 2>/dev/null; echo RC" 2>/dev/null)
  if ! echo "$st" | grep -q RC; then fails=$((fails+1)); [ $fails -ge 40 ] && { echo "### opd-$D-$V ($J) UNREACHABLE"; exit 2; }; sleep 200; continue; fi
  fails=0; state=$(echo "$st" | head -1)
  if [ -z "$state" ] || [ "$state" = "RC" ]; then
    info=$($H "echo state=\$(sacct -j $J -X -n -o State|tr -d ' '|head -1) elapsed=\$(sacct -j $J -X -n -o Elapsed|tr -d ' '|head -1) ckpts=\$(ls $OUTD 2>/dev/null | grep -E 'checkpoint|final' | tr '\n' ','); grep -o \"'loss': [0-9.]*\|'mopd/student_teacher_kl': [0-9.]*\|'completions/mean_length': [0-9.]*\" $OUTD/train.log 2>/dev/null | tail -3 | tr '\n' ' '" 2>/dev/null | tr '\n' ' ')
    if echo "$info" | grep -q "final"; then echo "### OPD DONE opd-$D-$V ($J): $info"; exit 0; fi
    if [ "$TRIES" -lt 3 ]; then
      idle=$($H "sinfo -h -n ib-a100-cluster-a-n[041-066] -t idle -o %D | head -1" 2>/dev/null); idle=${idle:-0}
      EXC="ib-a100-cluster-a-n[001-040,250]"; { [ "$idle" -ge "$NN" ] || [ "${FORCE41:-0}" = "1" ]; } && EXC="ib-a100-cluster-a-n[001-040,067-258]"
      NJ=$($H "cd $R && TEACHERS='$T' WEIGHTS='$W' TOPK=$K TEACHER_GPU_UTIL=0.55 TEACHER_MAX_BATCHED=4096 TEACHER_MAX_LEN=32768 CONFIG=$CFG OVERRIDES='' STUDENT_NODES=$SN sbatch -N $NN -J k-sean-opd-$(basename $OUTD) --cpus-per-task=32 --mem=600G -t 48:00:00 --exclude='$EXC' --parsable scripts/opd.sbatch" 2>/dev/null | tail -1)
      echo "### opd-$D-$V ($J) ended without final: $info -> RESUBMITTED (auto-resume) as $NJ (try $((TRIES+1)), exclude=$EXC)"
      SN=$SN NN=$NN FORCE41=${FORCE41:-0} TEACHERS_OVERRIDE="${TEACHERS_OVERRIDE:-}" WEIGHTS_OVERRIDE="${WEIGHTS_OVERRIDE:-}" exec bash "$0" "$NJ" "$D" "$V" "$K" $((TRIES+1))
    fi
    echo "### opd-$D-$V ($J) ended without final, retries exhausted: $info"; exit 1
  fi
  sleep 600
done
