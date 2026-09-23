"""Merge per-shard tau2-bench Results files (disjoint task subsets of one split) into one Results.

    python -m mopd.eval.tau2_merge --out merged.json part0.json part1.json ...

Simulations are concatenated; tasks are unioned by id; `info` is taken from the first part (all
shards ran the same agent / user / seed / trials). The merged file is what tau2's own
compute_metrics scores in mopd/eval/tau2_metrics.py, so sharding across nodes changes nothing
about the metric.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("parts", nargs="+")
    a = ap.parse_args(argv)
    from tau2.data_model.simulation import Results
    parts = [Results.load(Path(p)) for p in a.parts]
    tasks, sims, seen = [], [], set()
    for r in parts:
        for t in r.tasks:
            if t.id not in seen:
                seen.add(t.id); tasks.append(t)
        sims.extend(r.simulations)
    merged = parts[0].model_copy(update={"tasks": tasks, "simulations": sims})
    out = Path(a.out)
    try:
        merged.save(out)
    except TypeError:
        out.write_text(merged.model_dump_json(indent=1))
    n_tasks = len({s.task_id for s in sims})
    print("[tau2 merge] %d part(s) -> %s: %d tasks, %d simulations" % (len(parts), out, n_tasks, len(sims)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
