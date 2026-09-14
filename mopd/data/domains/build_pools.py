"""Distillation prompt pools — aggressive volume, SAME sources + decontamination as
the RL pools (mopd.data.domains.prep_rl_pools), plus stable ids.

  python -m mopd.data.domains.build_pools --out data/distill_pools --medmcqa-cap 40000

med: MedQA-USMLE train (all) + MedMCQA train random subsample (cap, default 40k)
law: CaseHOLD train (all) + barexam_qa.  housing_qa (yes/no) is EXCLUDED: a binary
     answer is right by chance 50% of the time, so rejection sampling on it keeps
     wrong-reasoning traces far too often.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

from mopd.data.domains.prep_rl_pools import build_eval_grams, contaminated, law_rows, med_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/distill_pools")
    p.add_argument("--medmcqa-cap", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    specs = (("med", med_rows, {"medmcqa_cap": args.medmcqa_cap}),
             ("law", law_rows, {"housing_cap": 0}))
    for domain, gen, kw in specs:
        eg = build_eval_grams(domain)
        rows = [r for r in gen(seed=args.seed, **kw) if r["meta"].get("kind") != "yesno"]
        random.Random(args.seed).shuffle(rows)
        kept, dropped = [], Counter()
        for r in rows:
            if contaminated(r["input"], eg):
                dropped[r["meta"]["source"]] += 1
            else:
                kept.append(r)
        for i, r in enumerate(kept):
            r["id"] = f"{domain}-{i:06d}"
        out_f = os.path.join(args.out, f"{domain}_pool.jsonl")
        with open(out_f, "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        srcs = Counter(r["meta"]["source"] for r in kept)
        print(f"[{domain}] kept {len(kept)} -> {out_f} | sources {dict(srcs)} | "
              f"dropped_contaminated {dict(dropped)}")
    print("BUILD_POOLS_DONE")


if __name__ == "__main__":
    main()
