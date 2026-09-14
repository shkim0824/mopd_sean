import json,os
S="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean/outputs/opd"
runs={"law-pg (v1)":"law-pg","law-top64 (v1)":"law-top64","law-top16 (v1)":"law-top16","law-pg-v2":"law-pg-v2","law-top64-v2":"law-top64-v2","law-top16-v2":"law-top16-v2"}
steps=[30,50,65,70,75,78,80,84,88,90,94,98,100,105,110,125,150,175,200]
for name,d in runs.items():
    cks=sorted([int(x.split("-")[1]) for x in os.listdir(f"{S}/{d}") if x.startswith("checkpoint-")])
    st=json.load(open(f"{S}/{d}/checkpoint-{cks[-1]}/trainer_state.json"))
    lh={r["step"]:r for r in st["log_history"] if "completions/mean_length" in r}
    def g(s,k):
        r=lh.get(s)
        if r is None: return "-"
        return str(round(r[k])) if k.endswith("length") else ("%.2f" % r[k])
    print("==",name,"(last ckpt",cks[-1],", steps logged",len(lh),")")
    print("step    :"," ".join("%5s"%s for s in steps))
    print("mean_len:"," ".join("%5s"%g(s,"completions/mean_length") for s in steps))
    print("clipped :"," ".join("%5s"%g(s,"completions/clipped_ratio") for s in steps))
    print("KL      :"," ".join("%5s"%g(s,"mopd/student_teacher_kl") for s in steps))
