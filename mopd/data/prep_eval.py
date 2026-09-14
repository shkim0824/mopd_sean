"""Build data/eval/{aime24,aime25,aime26,lcb_v6,ifeval,ifbench}/test.jsonl (+manifest.json).

Runs on the cpu-instance (HF reachable) against RAW snapshot downloads staged at
``--hf-root`` (``snapshot_download(repo_type="dataset", local_dir=<hf-root>/<org>__<name>)``)
so the exact upstream files are on disk; every output carries its sha256 + source.

Sources (standard ids, see docs/DATA.md):
  aime24  MathArena/aime_2024_I + MathArena/aime_2024_II   (cross-checked vs HuggingFaceH4/aime_2024)
  aime25  MathArena/aime_2025                                (cross-checked vs opencompass/AIME2025)
  aime26  MathArena/aime_2026                                (cross-checked vs math-ai/aime26)
  lcb_v6  livecodebench/code_generation_lite  release_v6 = test.jsonl..test6.jsonl (1055 rows)
          (the cluster already holds the identical 1055-row dump at data/livecodebench/test.jsonl
           built from the same loading-script files; --lcb-from reuses it)
  ifeval  google/IFEval (541)      ifbench  allenai/IFBench_test (300)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from mopd.common.io import iter_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402


def _pq(path_glob: str) -> List[Dict[str, Any]]:
    """rows of every parquet matching the glob; falls back to *.jsonl in the same tree
    (opencompass/AIME2025 and math-ai/aime26 ship jsonl, not parquet)."""
    import pyarrow.parquet as pq
    rows: List[Dict[str, Any]] = []
    for f in sorted(glob.glob(path_glob, recursive=True)):
        rows.extend(pq.read_table(f).to_pylist())
    if not rows:
        base = path_glob.split("**")[0]
        for f in sorted(glob.glob(os.path.join(base, "**", "*.jsonl"), recursive=True)):
            if "/.cache/" in f:
                continue
            rows.extend(iter_jsonl(f))
    return rows


def norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def build_aime(hf: str, out: str) -> Dict[str, Any]:
    specs = {
        "aime24": [("MathArena__aime_2024_I", 2024, "I"), ("MathArena__aime_2024_II", 2024, "II")],
        "aime25": [("MathArena__aime_2025", 2025, "I+II")],
        "aime26": [("MathArena__aime_2026", 2026, "I+II")],
    }
    cross = {
        "aime24": ("HuggingFaceH4__aime_2024", "problem", "answer"),
        "aime25": ("opencompass__AIME2025", "question", "answer"),
        "aime26": ("math-ai__aime26", "problem", "answer"),
    }
    report = {}
    for name, srcs in specs.items():
        rows = []
        for d, year, contest in srcs:
            for r in _pq(os.path.join(hf, d, "**", "*.parquet")):
                rows.append({"id": f"{name}_{contest.replace('+', '')}_{r['problem_idx']}", "problem": r["problem"],
                             "answer": str(r["answer"]).strip(), "year": year, "contest": contest,
                             "problem_idx": int(r["problem_idx"]), "source": d.replace("__", "/")})
        assert len(rows) == 30, (name, len(rows))
        for r in rows:
            assert re.fullmatch(r"\d{1,3}", r["answer"]) and 0 <= int(r["answer"]) <= 999, r
        # cross-check answers against an independent publisher when staged
        xd, xp, xa = cross[name]
        xrows = _pq(os.path.join(hf, xd, "**", "*.parquet")) if os.path.isdir(os.path.join(hf, xd)) else []
        mism = 0
        if xrows:
            xmap = {norm_text(x[xp]): str(x[xa]).strip() for x in xrows}
            for r in rows:
                k = norm_text(r["problem"])
                if k in xmap and int(xmap[k]) != int(r["answer"]):
                    mism += 1
        od = os.path.join(out, name)
        n = write_jsonl(os.path.join(od, "test.jsonl"), rows)
        man = {"n": n, "sources": [s[0].replace("__", "/") for s in srcs], "cross_check": xd.replace("__", "/") if xrows else None,
               "cross_check_mismatches": mism if xrows else None, "sha256": sha256_file(os.path.join(od, "test.jsonl"))}
        write_json(os.path.join(od, "manifest.json"), man)
        report[name] = man
        assert mism == 0
    return report


def build_if(hf: str, out: str) -> Dict[str, Any]:
    report = {}
    for name, d, key_field in (("ifeval", "google__IFEval", "key"), ("ifbench", "allenai__IFBench_test", "key")):
        rows = []
        pq_rows = _pq(os.path.join(hf, d, "**", "*.parquet"))
        if not pq_rows:  # jsonl fallback (IFBench repo ships jsonl)
            for f in glob.glob(os.path.join(hf, d, "**", "*.jsonl"), recursive=True):
                pq_rows.extend(iter_jsonl(f))
        for r in pq_rows:
            kws = r["kwargs"]
            if isinstance(kws, str):
                kws = json.loads(kws)
            kws = [{k: v for k, v in (k_ or {}).items() if v is not None} for k_ in kws]
            rows.append({"key": r[key_field], "prompt": r["prompt"], "instruction_id_list": list(r["instruction_id_list"]), "kwargs": kws})
        expected = {"ifeval": 541, "ifbench": 300}[name]
        assert len(rows) == expected, (name, len(rows))
        od = os.path.join(out, name)
        n = write_jsonl(os.path.join(od, "test.jsonl"), rows)
        ids = sorted({i for r in rows for i in r["instruction_id_list"]})
        man = {"n": n, "source": d.replace("__", "/"), "n_instruction_ids": len(ids), "sha256": sha256_file(os.path.join(od, "test.jsonl"))}
        write_json(os.path.join(od, "manifest.json"), man)
        report[name] = man
    return report


def build_lcb(hf: str, out: str, lcb_from: str = "") -> Dict[str, Any]:
    """release_v6 = the six jsonl files of livecodebench/code_generation_lite, in order."""
    od = os.path.join(out, "lcb_v6")
    os.makedirs(od, exist_ok=True)
    rows_out = 0
    dst = os.path.join(od, "test.jsonl")
    files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
    src_dir = os.path.join(hf, "livecodebench__code_generation_lite")
    with open(dst + ".tmp", "w") as fo:
        if lcb_from:
            # cluster dump: rows {task_type, base_question, ground_truth, lcb_row(JSON string)}
            for r in iter_jsonl(lcb_from):
                row = json.loads(r["lcb_row"]) if isinstance(r.get("lcb_row"), str) else r
                row.pop("id", None)
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_out += 1
        else:
            for k, fn in enumerate(files):
                with open(os.path.join(src_dir, fn)) as fi:
                    for line in fi:
                        if line.strip():
                            row = json.loads(line)
                            row["slice"] = f"v{k+1}"
                            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows_out += 1
    os.replace(dst + ".tmp", dst)
    assert rows_out == 1055, rows_out
    # per-slice counts & date range
    dates, diff, plat = [], {}, {}
    for r in iter_jsonl(dst):
        dates.append(r["contest_date"])
        diff[r["difficulty"]] = diff.get(r["difficulty"], 0) + 1
        plat[r["platform"]] = plat.get(r["platform"], 0) + 1
    man = {"n": rows_out, "source": "livecodebench/code_generation_lite", "version_tag": "release_v6",
           "date_range": [min(dates), max(dates)], "difficulty": diff, "platform": plat,
           "from": lcb_from or src_dir, "sha256": sha256_file(dst),
           "note": "private_test_cases kept as base64(zlib(pickle)) exactly like upstream; decoded at grading time"}
    write_json(os.path.join(od, "manifest.json"), man)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-root", required=True, help="dir of raw HF snapshots (<org>__<name>/...)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--lcb-from", default="", help="existing 1055-row LCB dump (cluster data/livecodebench/test.jsonl)")
    ap.add_argument("--only", default="aime,if,lcb")
    a = ap.parse_args()
    rep = {}
    if "aime" in a.only:
        rep.update(build_aime(a.hf_root, a.out))
    if "if" in a.only:
        rep.update(build_if(a.hf_root, a.out))
    if "lcb" in a.only:
        rep["lcb_v6"] = build_lcb(a.hf_root, a.out, a.lcb_from)
    write_json(os.path.join(a.out, "MANIFEST.json"), rep)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
