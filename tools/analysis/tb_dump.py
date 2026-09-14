import glob, os, re, sys
from tensorboard.backend.event_processing import event_accumulator as ea
base=sys.argv[1]; pat=re.compile(sys.argv[2]) if len(sys.argv)>2 else None
data={}
for exp in sorted(glob.glob(os.path.join(base,"exp_*"))):
    for f in glob.glob(os.path.join(exp,"**","events.*"), recursive=True):
        acc=ea.EventAccumulator(f, size_guidance={"scalars":0}); acc.Reload()
        for tag in acc.Tags()["scalars"]:
            for e in acc.Scalars(tag): data.setdefault(tag,{})[e.step]=e.value
print("TAGS:", sorted(data))
steps=[1,2,5,10,25,50,75,100,125,150,175,200]
print("tag".ljust(44)+" ".join(f"{s:>7}" for s in steps))
for t in sorted(data):
    if pat and not pat.search(t): continue
    d=data[t]; row=[]
    for s in steps:
        ks=[k for k in d if k<=s]; row.append(f"{d[max(ks)]:>7.3g}" if ks else "      -")
    print(t[:44].ljust(44)+" ".join(row))
