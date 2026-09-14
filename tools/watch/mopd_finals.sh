#!/bin/bash
# [mopd_sean operator tool] laptop-side watcher; cluster paths point at $S/mopd_sean (outputs/eval_domain, outputs/eval_6bench, models/teacher-*)
# args: RUN  -> strips optimizer states of outputs/opd/RUN and submits the final dom3 T0.6 eval (ck200 T1 + 6-bench come from the auto-eval loop)
RUN="$1"
ssh -o ConnectTimeout=25 ib-a100-controller 'S=/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean; R=$S/mopd_sean; cd $R || exit 1; echo MARK; date "+%H:%M KST"; d=outputs/opd/'"$RUN"'; J=$(sacct -n -X -o JobID,State,Elapsed --name=k-sean-opd-'"$RUN"' | tail -1); echo "job: $J"
if [ -f $d/opd_done.json ] && [ -f $d/final/model.safetensors.index.json ] && ! squeue -h -n k-sean-opd-'"$RUN"' -o %T | grep -q .; then before=$(du -sh $d | cut -f1); for ck in $d/checkpoint-*; do rm -rf $ck/global_step* $ck/optimizer.pt $ck/scheduler.pt $ck/rng_state_*.pth $ck/latest $ck/zero_to_fp32.py 2>/dev/null; done; echo "stripped: $before -> $(du -sh $d | cut -f1)"; else echo "NOT READY"; exit 3; fi
grep -ao "'"'"'completions/mean_length'"'"': [0-9.]*\|'"'"'completions/clipped_ratio'"'"': [0-9.]*\|'"'"'mopd/student_teacher_kl'"'"': [0-9.]*" $d/train.log | tail -3 | tr "\n" " "; echo
prefer(){ local idle; idle=$(sinfo -h -n ib-a100-cluster-a-n[041-066] -t idle -o "%D" 2>/dev/null | head -1); idle=${idle:-0}; if [ "$idle" -ge "$1" ]; then echo "--exclude=ib-a100-cluster-a-n[001-040,067-258]"; else echo "--exclude=ib-a100-cluster-a-n[001-040,250]"; fi; }
echo "T0.6 final eval DISABLED (user 2026-09-14: all performance at T=1.0; ck200 T1.0 + 6-bench come from the auto-eval loop)"
grep -o "submitted.*step 200.*" logs/auto_eval_opd_'"$RUN"'.log' 2>&1 | sed -n '/MARK/,$p'
