"""Code grading = the official LiveCodeBench harness, unchanged.

* extraction: LAST fenced code block in the response (LCB
  ``extraction_utils.extract_code`` for every chat LM style: the text between
  the second-to-last and last lines containing ``` ``` ```), after stripping
  the thinking block.
* execution: ``third_party.lcb_runner.testing_util.run_test(sample, test, timeout)``
  in a forked subprocess (LCB ``check_correctness``), where ``sample`` is
  ``{"input_output": json.dumps({"inputs", "outputs", "fn_name"})}``.
  stdin problems compare stripped lines with Decimal tolerance; call-based
  problems (``fn_name``) compare ``json.loads`` outputs exactly. Per-test
  timeout 6 s (LCB default); global timeout (timeout+1)*n_tests+5 like LCB.
* pass = every test returns > 0 (LCB ``np.all(gen > 0)``).

Three test-case formats are normalised into the LCB ``input_output`` dict:
  - LCB rows: ``public_test_cases`` + ``private_test_cases`` (json or
    base64(zlib(pickle(json)))) lists of {input, output, testtype},
    ``metadata.func_name`` -> ``fn_name``   (== ``CodeGenerationProblem.get_evaluation_sample``)
  - DeepCoder ``taco`` rows: ``tests`` is a TACO dict {inputs, outputs, fn_name?}
  - DeepCoder ``primeintellect`` / ``lcbv5`` rows: ``tests`` is a list of
    {type|testtype, input, output} (+ optional fn_name / metadata.func_name)
"""
from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import pickle
import re
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

from mopd.graders.math_grader import strip_thinking

DEFAULT_TIMEOUT = 6  # LCB --timeout default (seconds per test)


# ----------------------------------------------------------------------------- extraction
_BROKEN_TAIL_FENCE = re.compile(r"\n\s*`{1,2}\s*$")


def _repair_trailing_fence(text: str) -> str:
    """2026-09-01: models whose response ends "``<eos>" (one-token-broken closing fence —
    observed after math-only off-policy distillation, likely presence_penalty-tipped) leave
    the last block unclosed; line-based extraction then returns block+junk (pyexec) or ""
    (LCB) and the sample auto-fails. Repair ONLY a trailing line of 1-2 backticks into a
    proper ```. Content-preserving; a no-op for every well-formed response."""
    return _BROKEN_TAIL_FENCE.sub("\n```", text.rstrip())


def extract_code(response: str) -> str:
    """LCB ``extract_code`` for chat models: last ``` fenced block (line based)."""
    text = _repair_trailing_fence(strip_thinking(response))
    lines = text.split("\n")
    idx = [i for i, line in enumerate(lines) if "```" in line]
    if len(idx) < 2:
        return ""
    return "\n".join(lines[idx[-2] + 1 : idx[-1]])


# ----------------------------------------------------------------------------- test normalisation
def _decode_private(s: Any) -> list:
    if not isinstance(s, str):
        return list(s or [])
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


def lcb_row_to_input_output(row: Dict[str, Any]) -> Dict[str, Any]:
    """== lcb_runner CodeGenerationProblem.get_evaluation_sample()['input_output'] (as dict)."""
    pub = row.get("public_test_cases") or []
    if isinstance(pub, str):
        pub = json.loads(pub)
    priv = _decode_private(row.get("private_test_cases") or [])
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta) if meta else {}
    tests = list(pub) + list(priv)
    return {
        "inputs": [t["input"] for t in tests],
        "outputs": [t["output"] for t in tests],
        "fn_name": meta.get("func_name", None),
    }


def generic_tests_to_input_output(tests: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """DeepCoder / PrimeIntellect / TACO test-case payloads -> LCB input_output dict."""
    if isinstance(tests, str):
        tests = json.loads(tests)
    fn_name = None
    if metadata:
        fn_name = metadata.get("func_name") or metadata.get("fn_name")
    if isinstance(tests, dict):  # TACO style
        io = {"inputs": list(tests["inputs"]), "outputs": list(tests["outputs"]),
              "fn_name": tests.get("fn_name") or fn_name}
    elif isinstance(tests, list):
        inputs, outputs = [], []
        for t in tests:
            inputs.append(t["input"])
            outputs.append(t["output"])
            tt = t.get("testtype") or t.get("type")
            if tt == "functional" and fn_name is None:
                fn_name = (t.get("metadata") or {}).get("func_name")
            if fn_name is None and t.get("fn_name"):
                fn_name = t["fn_name"]
        io = {"inputs": inputs, "outputs": outputs, "fn_name": fn_name}
    else:
        raise ValueError(f"unknown tests payload type {type(tests)}")
    # TACO stores call-based inputs as lists already; LCB stores them as
    # newline-joined json strings.  run_test handles both (it json.loads
    # strings for call-based problems and passes lists through).
    return io


# ----------------------------------------------------------------------------- execution
def _run_in_child(input_output: Dict[str, Any], code: str, timeout: int, debug: bool, result: list, meta: list, box: str):
    # sandbox (2026-09-01): the box is created+removed by the UNRESTRICTED parent (run_tests);
    # the child only chdirs in and drops privileges, so generated open()/ctypes/truncate cannot
    # touch the repo, and the child cannot delete the box itself (needs write on its parent).
    os.chdir(box)
    from mopd.graders.sandbox import enter_sandbox
    enter_sandbox(box)  # Landlock + seccomp allowlist; blocks absolute-path/ctypes/truncate access
    from third_party.lcb_runner.testing_util import run_test  # imported in the child
    res, md = run_test({"input_output": json.dumps(input_output)}, test=code, debug=debug, timeout=timeout)
    result.append(res)
    meta.append(md)


def run_tests(input_output: Dict[str, Any], code: str, timeout: int = DEFAULT_TIMEOUT, debug: bool = False):
    """LCB ``check_correctness``: run in a child process with a global timeout;
    returns (per-test result list, metadata). A killed child => all -1."""
    import shutil
    import tempfile
    n = len(input_output["inputs"])
    manager = mp.Manager()
    result, meta = manager.list(), manager.list()
    box = tempfile.mkdtemp(prefix="lcbexec_")
    p = mp.Process(target=_run_in_child, args=(input_output, code, timeout, debug, result, meta, box))
    p.start()
    p.join(timeout=(timeout + 1) * n + 5)
    try:
        if p.is_alive():
            p.kill()
        if not result:
            return [-1] * n, {"error": "global timeout / crash"}
        return list(result[0]) if isinstance(result[0], (list, tuple)) else list(result), dict(meta[0]) if meta else {}
    finally:
        shutil.rmtree(box, ignore_errors=True)


def grade_code(input_output: Dict[str, Any], code: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Grade ONE program against all tests. ``passed`` iff every test > 0."""
    if not code.strip():
        return {"passed": False, "results": [], "meta": {"error": "no code block"}}
    results, meta = run_tests(input_output, code, timeout)
    passed = len(results) > 0 and all((r is True) or (isinstance(r, (int, float)) and r > 0) for r in results)
    return {"passed": bool(passed), "results": results, "meta": meta}


def grade_response(input_output: Dict[str, Any], response: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    out = grade_code(input_output, extract_code(response), timeout)
    return out


# ----------------------------------------------------------------------------- batch (RL rewards / eval)
def _grade_one(args):
    i, io, code, timeout = args
    return i, grade_code(io, code, timeout)


def grade_batch(items: Sequence[Dict[str, Any]], num_workers: Optional[int] = None,
                timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """items: [{"input_output": dict, "code": str}] -> list of grade dicts in order.
    Each item is graded in its own subprocess (LCB semantics); ``num_workers``
    processes run concurrently (LCB default 12-16; RL rewards use the trainer
    node's spare CPUs)."""
    if not num_workers:  # CPU-bound: all cores the job may use (cgroup affinity, not the host count) minus 2
        try:
            ncpu = len(os.sched_getaffinity(0))
        except Exception:
            ncpu = os.cpu_count() or 4
        num_workers = max(1, ncpu - 2)
    out: List[Optional[Dict[str, Any]]] = [None] * len(items)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        futs = [ex.submit(_grade_one, (i, it["input_output"], it["code"], timeout)) for i, it in enumerate(items)]
        for f in as_completed(futs):
            i, g = f.result()
            out[i] = g
    return out  # type: ignore


_FENCE_RE = re.compile(r"```")


def has_code_block(response: str) -> bool:
    return len(_FENCE_RE.findall(strip_thinking(response))) >= 2
