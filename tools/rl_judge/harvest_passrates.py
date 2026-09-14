import glob, json, os, re, sys, hashlib
from collections import defaultdict

S = "/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean"
dec = json.JSONDecoder()

files = []
for d in ("exp_002", "exp_003"):
    files += glob.glob(S + f"/mopd_sean/logs/nemo_math_v2/{d}/train_data_step*.jsonl")
def stepno(f): return int(re.search(r"step(\d+)\.jsonl", f).group(1))
files = sorted(set(files), key=stepno)
print(f"files: {len(files)}, steps {stepno(files[0])}..{stepno(files[-1])}", flush=True)

# per-prompt: key -> [n_ep1, c_ep1, n_all, c_all]; prompt text stored on first sight
stats = defaultdict(lambda: [0, 0, 0, 0])
texts = {}
for fi, f in enumerate(files):
    step = stepno(f)
    ep1 = step <= 58
    ep2 = 58 < step <= 116
    with open(f, buffering=1024*1024*8) as fh:
        for line in fh:
            q = line.find("\"", line.find("[["))
            try:
                prompt, _ = dec.raw_decode(line, q)
            except Exception:
                continue
            key = hashlib.md5(prompt.encode()).hexdigest()
            if key not in texts:
                texts[key] = prompt
            j = line.find("\"rewards\": [")
            if j < 0: continue
            j2 = line.find("]", j)
            try:
                r = float(line[j+12:j2])
            except Exception:
                continue
            st = stats[key]
            correct = 1 if r > 0.5 else 0
            if ep1: st[0] += 1; st[1] += correct
            elif ep2 and st[0] == 0:
                # stragglers never seen in epoch1 (drop_last): use epoch2 as their pass@8
                st[0] += 1; st[1] += correct
            st[2] += 1; st[3] += correct
    if (fi+1) % 20 == 0:
        print(f"  {fi+1}/{len(files)} files done, prompts so far {len(stats)}", flush=True)

print(f"unique prompts in logs: {len(stats)}", flush=True)

# map to source rows
src = [json.loads(l) for l in open(S + "/mopd_sean/data/rl3dom/math_train.jsonl")]
def norm(p):
    p = p.strip()
    if p.startswith("<|im_start|>user"):
        p = p[len("<|im_start|>user"):].lstrip("\n")
    for suf in ("<|im_end|>",):
        i = p.find(suf)
        if i >= 0: p = p[:i]
    return p.strip()
by_input = {}
for row in src: by_input[row["input"].strip()] = row
matched, unmatched = {}, []
for key, prompt in texts.items():
    p = norm(prompt)
    row = by_input.get(p)
    if row is None: unmatched.append(p[:80]); continue
    matched[key] = row
print(f"matched to source: {len(matched)}/{len(texts)}; source rows: {len(src)}", flush=True)
if unmatched: print("unmatched examples:", unmatched[:3], flush=True)

# histograms + band extraction (pass@8 from epoch1, band [0.2, 0.8] inclusive)
def hist(idx_n, idx_c, tag):
    buckets = {"=0": 0, "(0,0.2)": 0, "[0.2,0.8]": 0, "(0.8,1)": 0, "=1": 0}
    band = []
    for key, row in matched.items():
        n, c = stats[key][idx_n], stats[key][idx_c]
        if n == 0: continue
        pr = c / n
        if pr == 0: buckets["=0"] += 1
        elif pr < 0.2: buckets["(0,0.2)"] += 1
        elif pr <= 0.8: buckets["[0.2,0.8]"] += 1; band.append(row)
        elif pr < 1: buckets["(0.8,1)"] += 1
        else: buckets["=1"] += 1
    print(tag, buckets, "-> band survivors:", len(band), flush=True)
    return band

band_ep1 = hist(0, 1, "pass@8 (epoch1 policy):")
band_all = hist(2, 3, "pass rate (all ~24 samples):")

out = S + "/mopd_sean/data/rl3dom/math_train_band.jsonl"
seen = set()
with open(out, "w") as fo:
    for row in band_ep1:
        k = row["input"][:200]
        if k in seen: continue
        seen.add(k)
        fo.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"WROTE {out}: {len(seen)} rows", flush=True)

# sample counts sanity
import statistics
ns = [stats[k][2] for k in matched]
print("samples per prompt: min", min(ns), "median", statistics.median(ns), "max", max(ns), flush=True)
print("DONE", flush=True)
