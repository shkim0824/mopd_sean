"""IH-Challenge (safety domain) data preparation.

Source: openai/ih-challenge (apache-2.0), the copy under data/domains/ih-challenge/ (4 splits; the copy is made by
--import-from <dir>). Following lena's recipe (VerIH/RLVR/RECIPES.md §2.4, build_skeletons.py) only the
single-constraint (1,900) and multi-constraint (4,832) splits are used = 6,732 skeletons.

Deviations from the recipe, both deliberate:
  * a real HELD-OUT split: 10% of the skeletons (stratified by task_type, seed 42) are never trained on and form the
    in-domain benchmark (the recipe's test.parquet was the first 64 training rows);
  * the training rounds are drawn from a SHUFFLED pool (seed 42) so every round mixes single/multi tasks (the recipe's
    rounds followed file order: r0 all single-constraint, r2-r5 all composite, which the recipe itself flags as a confound).

Outputs (data/domains/ih-challenge/):
  skeletons_all.jsonl      6,732 rows: id, split, task_type, attack_level, privileged_level, defender_template,
                           attack_placeholder, attacker_meta_problem, attacker_problem, grader_code
  skeletons_train.jsonl    6,059 rows (rounds are slices of this file: --rounds N -> skeletons_train_r{k}.jsonl)
  skeletons_heldout.jsonl    673 rows
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil

SPLITS = ["single-constraint", "multi-constraint"]


def read_jsonl(p):
    with open(p) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(p, rows):
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_skeletons(ds_dir):
    out, n = [], 0
    for split in SPLITS:
        for r in read_jsonl(os.path.join(ds_dir, f"{split}.jsonl")):
            m = r["metadata"]
            out.append({
                "id": f"{split}-{n:05d}", "split": split, "task_type": m["task_type"],
                "attack_level": m["attack_level"], "privileged_level": m["privileged_level"],
                "defender_template": r["defender_problem_template"], "attack_placeholder": m["attack_placeholder"],
                "attacker_meta_problem": r["attacker_meta_problem"], "attacker_problem": r["attacker_problem"],
                "grader_code": m["grader_code_python"],
            })
            n += 1
    bad = sum(1 for r in out if sum(r["attack_placeholder"] in msg["content"] for msg in r["defender_template"]) != 1)
    print(f"skeletons: {len(out)} (single {sum(r['split'] == SPLITS[0] for r in out)}, multi {sum(r['split'] == SPLITS[1] for r in out)}); "
          f"rows with != 1 placeholder turn: {bad}")
    return out


def stratified_split(rows, heldout_frac, seed):
    rng = random.Random(seed)
    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r["task_type"]].append(r)
    train, held = [], []
    for t, items in sorted(by_type.items()):
        rng.shuffle(items)
        k = int(round(len(items) * heldout_frac))
        held.extend(items[:k]); train.extend(items[k:])
    rng.shuffle(train); rng.shuffle(held)
    return train, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("MOPD_DATA_DIR", "data"), "domains", "ih-challenge"))
    ap.add_argument("--import-from", default=None, help="directory holding the 4 split jsonl files (copied once)")
    ap.add_argument("--heldout-frac", type=float, default=0.10)
    ap.add_argument("--rounds", type=int, default=5, help="write skeletons_train_r{k}.jsonl slices")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.import_from:
        for f in os.listdir(a.import_from):
            if f.endswith(".jsonl") or f in ("README.md", ".gitattributes"):
                shutil.copy2(os.path.join(a.import_from, f), os.path.join(a.out_dir, f))
        print("imported dataset files from", a.import_from)
    skel = build_skeletons(a.out_dir)
    write_jsonl(os.path.join(a.out_dir, "skeletons_all.jsonl"), skel)
    train, held = stratified_split(skel, a.heldout_frac, a.seed)
    write_jsonl(os.path.join(a.out_dir, "skeletons_train.jsonl"), train)
    write_jsonl(os.path.join(a.out_dir, "skeletons_heldout.jsonl"), held)
    per = len(train) // a.rounds
    for k in range(a.rounds):
        lo = k * per; hi = len(train) if k == a.rounds - 1 else (k + 1) * per
        write_jsonl(os.path.join(a.out_dir, f"skeletons_train_r{k}.jsonl"), train[lo:hi])
    types = collections.Counter(r["task_type"] for r in held)
    print(f"train {len(train)} (rounds of {per}), heldout {len(held)} over {len(types)} task types; "
          f"heldout split mix: single {sum(r['split'] == SPLITS[0] for r in held)} / multi {sum(r['split'] == SPLITS[1] for r in held)}")
    json.dump({"source": "openai/ih-challenge (single-constraint + multi-constraint)", "skeletons": len(skel), "train": len(train),
               "heldout": len(held), "heldout_frac": a.heldout_frac, "rounds": a.rounds, "seed": a.seed,
               "heldout_task_types": dict(sorted(types.items()))}, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
