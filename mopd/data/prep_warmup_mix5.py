"""5-domain SFT warm-up mix = the phase-3 mixed warm-up rows (data/sft_warmup/mix_lawsft3200_medB2ep.jsonl: law-sft 3200 + med B x2,
{messages, meta}) + N pre-rendered tau2 SFT rows ({prompt_text, completion}, tools in the prompt) sampled uniformly from
data/sft_distill/tau2_sft.jsonl. Both row formats are accepted by mopd/sft/dataset.py (encode_row)."""
import json, random, sys, collections
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6400
rng = random.Random(42)
tau = [l for l in open("data/sft_distill/tau2_sft.jsonl", encoding="utf-8")]
pick = rng.sample(tau, N)
rows = [l for l in open("data/sft_warmup/mix_lawsft3200_medB2ep.jsonl", encoding="utf-8")]
allr = rows + pick; rng.shuffle(allr)
out = "data/sft_warmup/mix_lawsft3200_medB2ep_tau%d.jsonl" % N
with open(out, "w", encoding="utf-8") as f:
    for l in allr: f.write(l if l.endswith("\n") else l + "\n")
sub = collections.Counter(json.loads(l).get("sub_domain") for l in pick)
json.dump({"base_mix_rows": len(rows), "tau_rows": N, "tau_sub_domains": dict(sub), "total": len(allr), "seed": 42}, open(out + ".stats.json", "w"), indent=1)
print("wrote", out, "rows", len(allr), "tau sub-domains", dict(sub))
