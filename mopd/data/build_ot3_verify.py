"""Build the OpenThoughts3 verification sidecar (run on the CPU-INSTANCE after
mopd_stage/dl_ot3_verify_sources.sh finishes).

The released open-thoughts/OpenThoughts3-1.2M carries NO ground truth (the build pipeline
strips it; the paper §4.5 deliberately skipped answer filtering). Ground truth is recovered by
joining back to the seed datasets, using the row-aligned unstripped superset
mlfoundations-dev/OpenThoughts3 (same 1.2M rows, 37 metadata columns — row-order verified
against the released set here):

  math    850k `ai2-adapt-dev/openmath-2-math`  -> nvidia/OpenMathInstruct-2 `problem` ->
          `expected_answer` (join on normalised question text; GT for augmented problems is
          Llama-405B majority-vote — treat as slightly noisy)
  code    125,840 `nvidia/OpenCodeReasoning` -> per unstripped `dataset`:
            code_contests (113,472): deepmind/code_contests, join on normalised description
                                     (no index in OCR); tests = public+private+generated
            taco (5,456) / apps (4,480): BAAI/TACO / codeparrot/apps by `split`+`index`
            "-" (2,432): unknown origin -> ungradeable
          all normalised into the LCB input_output dict {"inputs","outputs","fn_name"?}
  science  59,904 organic-chemistry (42,144 SCP-116K `extract_solution` + 17,760 PDF
          `matched_solution`/`extracted_answer_choices`): free-form reference only (LLM-judge
          material, NOT machine-gradeable) -> stored for completeness
  ungradeable: stackexchange_codegolf 124,160 (byte-count golf, no tests),
          stackexchange-physics 40,096 (no answers anywhere)

Outputs (keyed by qhash = sha1 of whitespace-normalised question text), written to --out:
  math_answers.jsonl        {qhash, answer, problem_source}
  code_tests.jsonl.gz       {qhash, input_output, origin}    (gz: generated_tests are large)
  science_refs.jsonl        {qhash, reference, choices?, subsource}
  MANIFEST.json             join-coverage stats

Usage (cpu-instance, envs/mopd_cpu):
  cd $C/mopd && PYTHONPATH=. python -m mopd.data.build_ot3_verify \
      --superset $S/mopd_stage/hf/mlfoundations-dev__OpenThoughts3 \
      --released $C/data/OpenThoughts3-1.2M \
      --omi2 $S/mopd_stage/hf/nvidia__OpenMathInstruct-2 \
      --code-contests $S/mopd_stage/hf/deepmind__code_contests \
      --taco $S/mopd_stage/hf/BAAI__TACO --apps $S/mopd_stage/hf/codeparrot__apps \
      --out data/train/ot3_verify
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import time
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional

def norm_q(text: str) -> str:
    return " ".join((text or "").split())


def qhash(text: str) -> str:
    return hashlib.sha1(norm_q(text).encode("utf-8")).hexdigest()


def _tables(path_glob: str, columns: Optional[List[str]] = None) -> Iterator[Any]:
    import pyarrow.parquet as pq
    files = sorted(glob.glob(path_glob, recursive=True))
    assert files, f"no parquet at {path_glob}"
    for f in files:
        try:
            yield pq.read_table(f, columns=columns)
        except Exception as e:  # column missing in some shard layouts
            raise RuntimeError(f"{f}: {e}") from e


# ----------------------------------------------------------------------------- superset spine
SUP_COLS = ["instruction_seed", "_source", "_domain", "dataset", "split", "index", "id",
            "extract_solution", "matched_solution", "extracted_answer_choices", "domain"]


def read_superset(sup_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in _tables(os.path.join(sup_dir, "data", "*.parquet"), None):
        cols = [c for c in SUP_COLS if c in t.column_names]
        rows.extend(t.select(cols).to_pylist())
    return rows


def verify_alignment(sup_rows, released_dir: str, n_check: int = 500) -> int:
    """Released row i question == superset row i instruction_seed (agent-verified at 6 offsets;
    re-verified here on n_check evenly spaced rows)."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(released_dir, "data", "*.parquet")))
    sizes = []
    for f in files:
        sizes.append(pq.ParquetFile(f).metadata.num_rows)
    total = sum(sizes)
    assert total == len(sup_rows), f"row count mismatch: released {total} vs superset {len(sup_rows)}"
    step = max(1, total // n_check)
    checked = 0
    offs = list(range(0, total, step))
    # group offsets by file
    import bisect
    cum = [0]
    for s in sizes:
        cum.append(cum[-1] + s)
    by_file: Dict[int, List[int]] = {}
    for o in offs:
        fi = bisect.bisect_right(cum, o) - 1
        by_file.setdefault(fi, []).append(o - cum[fi])
    for fi, local_offs in by_file.items():
        t = pq.read_table(files[fi], columns=["conversations"])
        conv = t.column("conversations").to_pylist()
        for lo in local_offs:
            q = conv[lo][0]["value"]
            assert norm_q(q) == norm_q(sup_rows[cum[fi] + lo]["instruction_seed"]), \
                f"alignment broken at global row {cum[fi] + lo}"
            checked += 1
    return checked


# ----------------------------------------------------------------------------- joins
def build_math(sup_rows, omi2_dir: str, out_path: str) -> Dict[str, Any]:
    gold: Dict[str, str] = {}
    src: Dict[str, str] = {}
    conflicts = 0
    # data/train-*.parquet = the full 14M split ONLY (train_1M-* is a subset of it; reading both
    # would double-count)
    for t in _tables(os.path.join(omi2_dir, "data", "train-*.parquet"), ["problem", "expected_answer", "problem_source"]):
        for r in t.to_pylist():
            h = qhash(r["problem"])
            if h in gold and gold[h] != r["expected_answer"]:
                conflicts += 1
                continue
            gold[h] = r["expected_answer"]
            src[h] = r.get("problem_source") or ""
    want = {}
    for r in sup_rows:
        if r.get("_source") == "ai2-adapt-dev/openmath-2-math":
            want[qhash(r["instruction_seed"])] = True
    n_hit = 0
    with open(out_path, "w") as f:
        for h in want:
            if h in gold:
                f.write(json.dumps({"qhash": h, "answer": gold[h], "problem_source": src[h]}) + "\n")
                n_hit += 1
    return {"unique_math_q": len(want), "joined": n_hit, "coverage": round(n_hit / max(1, len(want)), 4),
            "omi2_rows_conflicting_answers": conflicts}


def _cc_tests(row: Dict[str, Any], cap: Optional[int]) -> Dict[str, Any]:
    ins: List[str] = []
    outs: List[str] = []
    for k in ("public_tests", "private_tests", "generated_tests"):
        t = row.get(k) or {}
        ins.extend(t.get("input") or [])
        outs.extend(t.get("output") or [])
    if cap and len(ins) > cap:
        ins, outs = ins[:cap], outs[:cap]
    return {"inputs": ins, "outputs": outs, "fn_name": None}


def build_code(sup_rows, cc_dir: str, taco_dir: str, apps_dir: str, out_path: str,
               cap_tests: Optional[int]) -> Dict[str, Any]:
    ocr = [r for r in sup_rows if r.get("_source") == "nvidia/OpenCodeReasoning"]
    st = Counter(r.get("dataset") for r in ocr)
    written: Dict[str, bool] = {}
    n = Counter()
    f = gzip.open(out_path, "wt")

    def emit(h: str, io: Dict[str, Any], origin: str):
        if h in written or not io["inputs"]:
            n["skipped_dup_or_empty"] += 1
            return
        f.write(json.dumps({"qhash": h, "input_output": io, "origin": origin}) + "\n")
        written[h] = True
        n[origin] += 1

    # --- code_contests: join by normalised description
    cc_map: Dict[str, Dict[str, Any]] = {}
    for t in _tables(os.path.join(cc_dir, "data", "*.parquet"),
                     ["description", "public_tests", "private_tests", "generated_tests"]):
        for r in t.to_pylist():
            cc_map[qhash(r["description"])] = r
    for r in ocr:
        if r.get("dataset") == "code_contests":
            h = qhash(r["instruction_seed"])
            hit = cc_map.get(h)
            if hit is not None:
                emit(h, _cc_tests(hit, cap_tests), "code_contests")
            else:
                n["cc_miss"] += 1
    del cc_map

    # --- taco / apps: index joins (index = row offset within the origin split)
    def load_indexed(root: str, want: List[Dict[str, Any]], origin: str):
        by_split: Dict[str, List[Dict[str, Any]]] = {}
        for r in want:
            by_split.setdefault(str(r.get("split")), []).append(r)
        for split, rows in by_split.items():
            qs: List[str] = []
            ios: List[str] = []
            if origin == "taco":  # BAAI/TACO "ALL" config: ALL/{split}-*.parquet
                import pyarrow.parquet as pq
                files = sorted(glob.glob(os.path.join(root, "ALL", f"{split}-*.parquet")))
                for fp in files:
                    t = pq.read_table(fp, columns=["question", "input_output"])
                    qs.extend(t.column("question").to_pylist())
                    ios.extend(t.column("input_output").to_pylist())
            else:  # codeparrot/apps ships {split}.jsonl
                fp = os.path.join(root, f"{split}.jsonl")
                if os.path.exists(fp):
                    with open(fp) as fh:
                        for line in fh:
                            r2 = json.loads(line)
                            qs.append(r2.get("question") or "")
                            ios.append(r2.get("input_output") or "")
            if not qs:
                n[f"{origin}_no_files_{split}"] += len(rows)
                continue
            for r in rows:
                try:
                    i = int(r.get("index"))
                except (TypeError, ValueError):
                    n[f"{origin}_no_index"] += 1
                    continue
                if not (0 <= i < len(ios)):
                    n[f"{origin}_index_oob"] += 1
                    continue
                # sanity: question text must match the OCR seed, else the index join is wrong
                if norm_q(qs[i])[:300] != norm_q(r["instruction_seed"])[:300]:
                    n[f"{origin}_index_text_mismatch"] += 1
                    continue
                try:
                    io = json.loads(ios[i]) if ios[i] else None
                except Exception:
                    n[f"{origin}_bad_io"] += 1
                    continue
                if not io or not io.get("inputs"):
                    n[f"{origin}_empty_io"] += 1
                    continue
                io = {"inputs": io["inputs"], "outputs": io["outputs"], "fn_name": io.get("fn_name")}
                if cap_tests and len(io["inputs"]) > cap_tests:
                    io = {"inputs": io["inputs"][:cap_tests], "outputs": io["outputs"][:cap_tests],
                          "fn_name": io.get("fn_name")}
                emit(qhash(r["instruction_seed"]), io, origin)

    load_indexed(taco_dir, [r for r in ocr if r.get("dataset") == "taco"], "taco")
    load_indexed(apps_dir, [r for r in ocr if r.get("dataset") == "apps"], "apps")
    f.close()
    uniq = len({qhash(r["instruction_seed"]) for r in ocr})
    return {"ocr_rows_by_dataset": dict(st), "unique_code_q": uniq, "joined": len(written),
            "coverage_of_joinable": round(len(written) / max(1, len({qhash(r['instruction_seed']) for r in ocr if r.get('dataset') in ('code_contests', 'taco', 'apps')})), 4),
            "detail": dict(n)}


def build_science(sup_rows, out_path: str) -> Dict[str, Any]:
    n = Counter()
    seen: Dict[str, bool] = {}
    with open(out_path, "w") as f:
        for r in sup_rows:
            if r.get("_source") not in ("organic-chemistry-questions",):
                continue
            h = qhash(r["instruction_seed"])
            if h in seen:
                continue
            ref = r.get("extract_solution") or r.get("matched_solution")
            choices = r.get("extracted_answer_choices")
            if isinstance(choices, list) and choices == ["NO CHOICES AVAILABLE"]:
                choices = None
            if not ref or ref == "NO ANSWER DETECTED":
                n["no_reference"] += 1
                continue
            sub = "scp116k" if r.get("extract_solution") else "pdf"
            f.write(json.dumps({"qhash": h, "reference": ref, "choices": choices, "subsource": sub},
                               ensure_ascii=False) + "\n")
            seen[h] = True
            n[sub] += 1
    return {"joined": len(seen), "detail": dict(n)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--superset", required=True)
    p.add_argument("--released", required=True)
    p.add_argument("--omi2", required=True)
    p.add_argument("--code-contests", required=True)
    p.add_argument("--taco", required=True)
    p.add_argument("--apps", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cap-tests", type=int, default=0, help="0 = keep all tests per problem")
    args = p.parse_args(argv)
    cap = args.cap_tests or None
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    sup = read_superset(args.superset)
    print(f"[verify-build] superset rows: {len(sup)} ({time.time()-t0:.0f}s)", flush=True)
    checked = verify_alignment(sup, args.released)
    print(f"[verify-build] row alignment verified on {checked} samples", flush=True)
    m = build_math(sup, args.omi2, os.path.join(args.out, "math_answers.jsonl"))
    print("[verify-build] math:", json.dumps(m), flush=True)
    c = build_code(sup, args.code_contests, args.taco, args.apps,
                   os.path.join(args.out, "code_tests.jsonl.gz"), cap)
    print("[verify-build] code:", json.dumps(c), flush=True)
    s = build_science(sup, os.path.join(args.out, "science_refs.jsonl"))
    print("[verify-build] science:", json.dumps(s), flush=True)
    manifest = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "alignment_checked": checked,
                "cap_tests": cap, "math": m, "code": c, "science": s,
                "ungradeable_sources": ["stackexchange_codegolf", "stackexchange-physics"],
                "seconds": round(time.time() - t0)}
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[verify-build] DONE", flush=True)


if __name__ == "__main__":
    main()
