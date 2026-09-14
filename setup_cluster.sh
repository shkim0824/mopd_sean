#!/bin/bash
# Cluster-side setup of the consolidated repo: symlink farm into the original data/checkpoint trees (nothing is copied or moved)
# + rsync of the assets that are not part of the code tarball (benchmark data, nltk, NeMo venv extras). Re-runnable.
set -u
S=/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean; R=/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean; cd $R || exit 1
mkdir -p models/rl_steps data/raw outputs/sft outputs/rl logs/nemo tmp/cache
ln1(){ [ -e "$1" ] && ! [ -L "$1" ] && { echo "exists (real): $1"; return; }; ln -sfn "$2" "$1"; }
ln1 models/Qwen3-4B-OT3 /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/models/Qwen3-4B-OT3
ln1 models/Qwen3-4B /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/models/Qwen3-4B
ln1 models/Qwen3-4B-Base /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/models/Qwen3-4B-Base
ln1 models/Qwen3-1.7B /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/models/Qwen3-1.7B
ln1 models/Qwen3-1.7B-Base /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/models/Qwen3-1.7B-Base
ln1 models/Qwen3-1.7B-OT3 /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/outputs/sft_ot3/qwen3-1p7b-18k
ln1 models/teacher-law /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/models/rl2-law-distill-ck200
ln1 models/teacher-law-rlonly /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/models/rl-law-ck200
ln1 models/teacher-fin /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/models/rl-fin-ck200
ln1 models/teacher-if /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_rl/models/rl-if-ck200
ln1 models/teacher-med /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/results/sft_distill_med_c6_4ep/checkpoint-3368
ln1 models/merged-4teachers-uniform /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/models/merged-4teachers-uniform
ln1 models/merged-4teachers-w4411 /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/models/merged-4teachers-w4411
ln1 data/eval /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/data/eval
ln1 data/train /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/data/train
ln1 data/domains /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/domains
ln1 data/rl_pools /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/rl_pools
ln1 data/distill_pools /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/distill_pools
ln1 data/sft_distill /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/sft_distill
ln1 data/sft_distill_v2 /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/sft_distill_v2
ln1 data/sft_distill_v3 /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/sft_distill_v3
ln1 data/sft_med /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/sft_med
ln1 data/sft_med_o1 /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/sft_med_o1
ln1 data/sft_warmup /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/sft_warmup
ln1 data/rl3dom /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_rl/data/rl3dom
ln1 data/nemotron_math /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_rl/data/nemotron_math
ln1 data/raw/OpenThoughts3-1.2M /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/OpenThoughts3-1.2M
ln1 data/raw/Nemotron-3-Nano-RL-Blend /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/Nemotron-3-Nano-RL-Blend
ln1 data/raw/Nemotron-RL-Math-v2 /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/Nemotron-RL-Math-v2
ln1 data/raw/Nemotron-RL-Ultra-restored /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/Nemotron-RL-Ultra-restored
ln1 data/raw/Nemotron-RL-Ultra-Training-Blends /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/Nemotron-RL-Ultra-Training-Blends
ln1 data/raw/livecodebench /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/livecodebench
ln1 outputs/opd /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/outputs/opd
ln1 outputs/eval_domain /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/outputs/eval
ln1 outputs/eval_6bench /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/outputs/eval
ln1 outputs/teacher_gen /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/outputs/distill
ln1 outputs/sft/ot3_qwen3-4b /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/outputs/sft_ot3/qwen3-4b
ln1 outputs/sft/ot3_qwen3-1p7b-18k /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/outputs/sft_ot3/qwen3-1p7b-18k
ln1 logs/_orig_mopd /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/logs
ln1 logs/_orig_mopd_domains /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/logs
ln1 logs/_orig_mopd_rl /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_rl/logs
for r in $S/mopd_domains/results/*; do n=$(basename $r); case $n in grpo_*) ln1 outputs/rl/$n $r;; *) ln1 outputs/sft/$n $r;; esac; done
for r in $S/mopd_rl/results/grpo_*; do ln1 outputs/rl/$(basename $r) $r; done
for m in $S/mopd_domains/models/rl* $S/mopd_rl/models/rl-*; do ln1 models/rl_steps/$(basename $m) $m; done
rsync -a $S/mopd_domains/third_party/ third_party/            # benchmark data (1.3 GB)
rsync -a $S/mopd/env/nltk_data $S/mopd/env/tiktoken env/       # offline nltk + tiktoken assets
mkdir -p env/nemo_extras; rsync -a $S/mopd_rl/vendor $S/mopd_rl/third_party_py env/nemo_extras/   # python deps vendored for the NeMo container
echo "setup done: $(ls models | wc -l) model links, $(ls outputs/sft | wc -l) sft runs, $(ls outputs/rl | wc -l) rl runs"
