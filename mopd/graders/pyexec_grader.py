"""Function-level Python grading for HumanEval / MBPP / AceCode-style asserts.

Mechanics follow the OFFICIAL graders:

* HumanEval (github.com/openai/human-eval ``execution.py``): the program is the
  string concatenation ``code + test + f"check({entry_point})"`` executed in one
  namespace; pass iff it raises nothing.  The official harness concatenates
  ``prompt + completion`` (base-model completion convention); for CHAT models the
  standard (EvalPlus ``sanitize``/leaderboard, Qwen/DeepSeek model cards) is that
  the model rewrites the WHOLE function inside a markdown fence -> we extract the
  last fenced block and, if it does not define ``entry_point``, fall back to
  prepending the original ``prompt`` (completion-style answer).
* MBPP (Austin et al. 2021, google-research/mbpp): generated code + the task's
  ``test_list`` assertions run together; any failing assert = fail.
  EvalPlus MBPP+ ships an extended ``test`` script with the same semantics.
* AceCode-89K RL rewards: same as MBPP (list of assert strings).

Execution isolation == our LCB harness: a forked child process per program with a
hard join-timeout, stdout/stderr swallowed.  The official HumanEval timeout is
3 s; MBPP original uses no explicit per-assert timeout — we default to 6 s per
program (LCB convention) and expose the knob.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

from mopd.graders.math_grader import strip_thinking

DEFAULT_TIMEOUT = 6  # seconds per program (official HumanEval: 3)


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
    """Last fenced code block (line-based, == LCB ``extract_code``); if the response
    has no complete fence, fall back to the raw de-thinked text when it looks like
    bare code (starts with import/def/class) else ''."""
    text = _repair_trailing_fence(strip_thinking(response))
    lines = text.split("\n")
    idx = [i for i, line in enumerate(lines) if "```" in line]
    if len(idx) >= 2:
        return "\n".join(lines[idx[-2] + 1: idx[-1]])
    if len(idx) == 1:  # opening fence, truncated before closing
        cand = "\n".join(lines[idx[0] + 1:])
        return cand
    if re.match(r"^\s*(import |from |def |class )", text):
        return text
    return ""


def _defines(code: str, name: str) -> bool:
    return re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(name)}\s*\(", code, re.M) is not None or \
        re.search(rf"^\s*{re.escape(name)}\s*=", code, re.M) is not None or \
        re.search(rf"^\s*class\s+{re.escape(name)}\b", code, re.M) is not None


# ----------------------------------------------------------------------------- execution
def _child_exec(program: str, box: str, q):
    """Runs in a forked, short-lived child. HARD LESSON (2026-09-01): an earlier version executed
    with cwd = the REPO and no filesystem guard; graded model-generated programs littered the repo
    root and one of them recursively DELETED the working tree (configs/outputs/scripts/... of
    mopd_nt). Now: per-program throwaway tmpdir + the official human-eval reliability_guard set."""
    import faulthandler
    import io
    import shutil
    import subprocess
    import sys
    import tempfile

    faulthandler.disable()
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    # 1) sandbox cwd: box is created (and later removed) by the UNRESTRICTED parent; the child
    #    only chdirs in — every file the program creates lands here, never the repo
    os.chdir(box)
    # 1b) kernel-enforced allowlist (Landlock + seccomp, mopd/graders/sandbox.py): unlike the
    #     monkeypatch guard below it also blocks ABSOLUTE-path reads/writes, ctypes syscalls and
    #     truncate(2). box gets RW; box's PARENT stays denied, so the child cannot rmdir the box
    #     itself (the parent does that) nor touch sibling boxes.
    from mopd.graders.sandbox import enter_sandbox
    enter_sandbox(box)
    # 2) reliability guard (== official human-eval execution.py, adapted): destructive /
    #    process-spawning primitives disabled INSIDE THE CHILD ONLY (the fork isolates the parent)
    import builtins
    builtins.input = lambda *a, **k: ""
    builtins.exit = builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = os.system = os.remove = os.removedirs = os.rmdir = os.unlink = None  # type: ignore
    os.rename = os.renames = os.replace = os.truncate = os.fchdir = os.chdir = None  # type: ignore
    os.chmod = os.chown = os.chroot = os.setuid = os.fork = os.forkpty = os.killpg = None  # type: ignore
    shutil.move = shutil.chown = None  # type: ignore
    subprocess.Popen = subprocess.run = subprocess.call = subprocess.check_call = None  # type: ignore
    subprocess.check_output = None  # type: ignore
    sys.modules["ipdb"] = None  # type: ignore
    sys.modules["joblib"] = None  # type: ignore
    sys.modules["psutil"] = None  # type: ignore
    shutil.rmtree = None  # type: ignore
    g: Dict[str, Any] = {"__name__": "__main__"}
    try:
        exec(compile(program, "<solution>", "exec"), g)
        q.put(("pass", ""))
    except BaseException as e:  # noqa: BLE001 - any failure (assert, exit, recursion) = fail
        q.put(("fail", f"{type(e).__name__}: {e}"[:300]))


def run_program(program: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Execute one self-contained test program in a forked child. pass iff clean exit.
    The throwaway box is created and removed HERE (the parent is not Landlock-restricted); the
    child only chdirs into it and drops privileges, so it can neither escape the box nor delete
    it (deleting the box needs write on its parent, which the child is denied)."""
    import shutil
    import tempfile
    box = tempfile.mkdtemp(prefix="pyexec_")
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child_exec, args=(program, box, q))
    p.start()
    p.join(timeout=timeout)
    try:
        if p.is_alive():
            p.kill()
            p.join()
            return {"passed": False, "error": "timeout"}
        try:
            status, err = q.get_nowait()
        except Exception:
            return {"passed": False, "error": f"crash (exitcode {p.exitcode})"}
        return {"passed": status == "pass", "error": err}
    finally:
        shutil.rmtree(box, ignore_errors=True)


# ----------------------------------------------------------------------------- program builders
def humaneval_program(row: Dict[str, Any], code: str) -> Optional[str]:
    """row: {prompt, test, entry_point}. code: the extracted fenced block."""
    if not code.strip():
        return None
    ep = row["entry_point"]
    if not _defines(code, ep):
        # completion-style answer (function body only): official concatenation
        code = row["prompt"] + "\n" + code
    return code + "\n\n" + row["test"] + f"\n\ncheck({ep})\n"


def mbpp_program(row: Dict[str, Any], code: str) -> Optional[str]:
    """row: {test_list, test_setup_code?, test_imports?} (google-research/mbpp) or
    {test} (evalplus/mbppplus extended script)."""
    if not code.strip():
        return None
    parts = [code]
    if row.get("test"):  # MBPP+ full test script (defines assertion loop itself)
        parts.append(row["test"])
    else:
        setup = row.get("test_setup_code") or ""
        if isinstance(row.get("test_imports"), (list, tuple)):
            setup = "\n".join(list(row["test_imports"]) + ([setup] if setup else []))
        if setup:
            parts.append(setup)
        parts.append("\n".join(row["test_list"]))
    return "\n\n".join(parts) + "\n"


def asserts_program(asserts: Sequence[str], code: str) -> Optional[str]:
    """AceCode-style: generated code + assert statements."""
    if not code.strip():
        return None
    return code + "\n\n" + "\n".join(asserts) + "\n"


def check_program(code: str, test_code: str, entry_point: str, prelude: str = "") -> Optional[str]:
    """LeetCodeDataset-style: [imports prelude +] generated code + ``def check(candidate)``
    test suite + ``check(<entry_point>)`` (entry_point may be an expression like
    ``Solution().twoSum``)."""
    if not code.strip():
        return None
    parts = []
    if prelude:
        parts.append(prelude)
    parts += [code, test_code, f"check({entry_point})"]
    return "\n\n".join(parts) + "\n"


# ----------------------------------------------------------------------------- batch
def _grade_one(args):
    i, program, timeout = args
    if program is None:
        return i, {"passed": False, "error": "no code block"}
    return i, run_program(program, timeout)


def grade_batch(programs: Sequence[Optional[str]], num_workers: Optional[int] = None,
                timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """programs: list of self-contained test programs (or None = no code)."""
    from mopd.graders.sandbox import landlock_abi
    if landlock_abi() <= 0 and os.environ.get("MOPD_SANDBOX", "1") not in ("0", "off", "false"):
        print("[pyexec] WARNING: Landlock unavailable on this kernel — grading children fall back "
              "to the python-level guard only (writes are tmpdir-contained by chdir, but "
              "absolute-path/ctypes access is NOT kernel-blocked)", flush=True)
    if not num_workers:
        try:
            ncpu = len(os.sched_getaffinity(0))
        except Exception:
            ncpu = os.cpu_count() or 4
        num_workers = max(1, ncpu - 2)
    out: List[Optional[Dict[str, Any]]] = [None] * len(programs)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        futs = [ex.submit(_grade_one, (i, pr, timeout)) for i, pr in enumerate(programs)]
        for f in as_completed(futs):
            i, g = f.result()
            out[i] = g
    return out  # type: ignore
