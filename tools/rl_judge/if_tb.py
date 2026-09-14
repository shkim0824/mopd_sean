import glob, os, statistics
S = "/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
dirs = sorted(glob.glob(S + "/mopd_sean/logs/nemo_if/exp_*/tensorboard"))
acc={}
for d in dirs:
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload(); tags=ea.Tags()["scalars"]
    for t in ["train/reward","train/baseline_reward/pct_0","train/baseline_reward/pct_1","train/approx_entropy","train/mean_gen_tokens_per_sample","validation/accuracy","train/truncation_rate"]:
        if t in tags:
            for e in ea.Scalars(t): acc.setdefault(t,{})[e.step]=e.value
print("dirs:",len(dirs),"steps logged:",len(acc.get("train/reward",{})))
def win(t,a,b):
    v=[acc[t][s] for s in range(a,b+1) if s in acc.get(t,{})]; return f"{statistics.mean(v):.3f}" if v else "-"
last=max(acc["train/reward"])
print("window | reward | pct0 | pct1 | entropy | gen_tok | trunc")
for a in range(1,last+1,25):
    b=min(a+24,last); print(f"{a:3d}-{b:3d} | {win('train/reward',a,b)} | {win('train/baseline_reward/pct_0',a,b)} | {win('train/baseline_reward/pct_1',a,b)} | {win('train/approx_entropy',a,b)} | {win('train/mean_gen_tokens_per_sample',a,b)} | {win('train/truncation_rate',a,b)}")
print("val:", [(s,round(v,3)) for s,v in sorted(acc.get("validation/accuracy",{}).items())])
