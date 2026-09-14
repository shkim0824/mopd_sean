"""Medical SFT set: m23k (23.5k) + Medical-R1-Distill-Data (22k) → one jsonl in the
mopd SFT contract: {"prompt": <user msg>, "completion": "<think>\\n{trace}\\n</think>\\n\\n{answer}"}.

- m23k rows: prompt / reasoning / distilled_answer_string / answer_letter (MCQA).
  The prompt already contains the lettered options; we keep it verbatim and append
  the standard answer-format instruction used by our RL pools so SFT and RL agree.
- Medical-R1-Distill rows: question / reasoning (R1) / response + free-text gold.
- Dedup: exact-question dedup across the two sets, then 8-gram decontamination
  against the medical eval benches (MedQA test, MedXpertQA, PubMedQA).
  python -m mopd.data.domains.prep_med_sft --out data/sft_med
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Set

from mopd.data.domains.prep_rl_pools import build_eval_grams, contaminated
from mopd.eval.domains.benches import D, MCQA_SUFFIX


def norm_q(t: str) -> str:
    return " ".join(str(t).split()).lower()[:300]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sft_med")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import pyarrow.parquet as pq
    eg = build_eval_grams("med")
    seen: Set[str] = set()
    n_m23k = n_r1 = n_dupe = n_cont = 0
    out_f = open(os.path.join(args.out, "med_sft.jsonl"), "w")

    t = pq.read_table(sorted(glob.glob(os.path.join(D, "med_m23k/data/*.parquet")))[0])
    cols = {c: t.column(c).to_pylist() for c in
            ("prompt", "reasoning", "distilled_answer_string", "answer_letter")}
    for q, think, ans, letter in zip(cols["prompt"], cols["reasoning"],
                                     cols["distilled_answer_string"], cols["answer_letter"]):
        key = norm_q(q)
        if key in seen:
            n_dupe += 1; continue
        seen.add(key)
        user = str(q).strip() + MCQA_SUFFIX
        if contaminated(user, eg):
            n_cont += 1; continue
        completion = (f"<think>\n{str(think).strip()}\n</think>\n\n"
                      f"{str(ans).strip()}\n\nAnswer: {letter}")
        out_f.write(json.dumps({"prompt": user, "completion": completion,
                                "meta": {"source": "m23k"}}, ensure_ascii=False) + "\n")
        n_m23k += 1

    fs = [f for f in
          (sorted(glob.glob(os.path.join(D, "med_r1_distill/**/*.parquet"), recursive=True))
           or sorted(glob.glob(os.path.join(D, "med_r1_distill/*.json"), recursive=True)))
          if ".cache" not in f]
    for f in fs:
        if f.endswith(".parquet"):
            t = pq.read_table(f)
            recs = [ {c: t.column(c)[i].as_py() for c in t.column_names}
                     for i in range(t.num_rows) ]
        else:
            loaded = json.load(open(f))         # single JSON array (not JSONL)
            recs = loaded if isinstance(loaded, list) else loaded.get("data", [])
        for r in recs:
            lk = {k.lower(): k for k in r}
            q = r.get(lk.get("question", ""), None)
            think = r.get(lk.get("reasoning (reasoning_content)", ""), None) or \
                r.get(lk.get("reasoning", ""), None)
            resp = r.get(lk.get("response (content)", ""), None) or \
                r.get(lk.get("response", ""), None)
            if not (q and think and resp):
                continue
            key = norm_q(q)
            if key in seen:
                n_dupe += 1; continue
            seen.add(key)
            user = str(q).strip()
            if contaminated(user, eg):
                n_cont += 1; continue
            completion = f"<think>\n{str(think).strip()}\n</think>\n\n{str(resp).strip()}"
            out_f.write(json.dumps({"prompt": user, "completion": completion,
                                    "meta": {"source": "Medical-R1-Distill"}},
                                   ensure_ascii=False) + "\n")
            n_r1 += 1
    out_f.close()
    print(json.dumps({"m23k": n_m23k, "r1_distill": n_r1, "dupes": n_dupe,
                      "contaminated_dropped": n_cont,
                      "total": n_m23k + n_r1}))
    print("PREP_MED_SFT_DONE")


if __name__ == "__main__":
    main()
