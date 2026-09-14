"""Sandbox canary for mopd_poc graders (2026-09-01 fix port). Run BEFORE any large grading run:
(1) writes by graded code land in a throwaway tmpdir, never the observation cwd;
(2) destructive primitives (os.remove / os.system / shutil.rmtree) are nulled in the child;
(3) normal grading and timeouts still work; (4) code_grader (LCB) child is cwd-isolated too.
NOTE: grade_batch uses a spawn pool -> the __main__ guard below is REQUIRED."""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mopd.graders import code_grader, pyexec_grader  # noqa: E402


def main():
    obs = tempfile.mkdtemp(prefix="canary_obs_")
    os.chdir(obs)
    sentinel = os.path.join(obs, "SENTINEL.txt")
    open(sentinel, "w").write("precious")
    before = set(os.listdir(obs))

    # A) destructive + write-escaping program (all three ops must be blocked by the guard)
    prog_a = (
        "import os, shutil\n"
        "open('junk_canary.txt', 'w').write('x')\n"
        "blocked = 0\n"
        "try:\n"
        f"    os.remove({sentinel!r})\n"
        "except TypeError:\n"
        "    blocked += 1\n"
        "try:\n"
        f"    os.system('touch {obs}/escaped_by_system')\n"
        "except TypeError:\n"
        "    blocked += 1\n"
        "try:\n"
        f"    shutil.rmtree({obs!r})\n"
        "except TypeError:\n"
        "    blocked += 1\n"
        "assert blocked == 3, f'guard not fully applied: {blocked}/3'\n"
    )
    prog_b = "def f(x):\n    return x + 1\nassert f(1) == 2\n"          # B) normal grading
    prog_c = "while True:\n    pass\n"                                   # C) timeout
    # E-H) KERNEL-level checks (Landlock): the python guard cannot block these — the kernel must.
    prog_e = (                                                           # absolute-path write
        "try:\n"
        f"    open({os.path.join(obs, 'abs_attack.txt')!r}, 'w').write('x')\n"
        "    raise SystemExit('landlock did not block absolute write')\n"
        "except PermissionError:\n    pass\n"
    )
    prog_f = (                                                           # absolute-path read
        "try:\n"
        f"    open({sentinel!r}).read()\n"
        "    raise SystemExit('landlock did not block absolute read')\n"
        "except PermissionError:\n    pass\n"
    )
    prog_g = (                                                           # ctypes syscall bypass
        "import ctypes\n"
        "libc = ctypes.CDLL(None)\n"
        f"r = libc.unlink({os.fsencode(sentinel)!r})\n"
        "assert r != 0, 'ctypes unlink SUCCEEDED — kernel guard missing'\n"
    )
    prog_h = "import json\nopen(json.__file__).read(10)\n"               # allowlisted RO read works
    victim = os.path.join(obs, "victim.txt")                             # I) ctypes truncate(2) gap
    open(victim, "w").write("IMPORTANT DATA")
    prog_i = (
        "import ctypes\n"
        "libc = ctypes.CDLL(None)\n"
        f"libc.truncate({os.fsencode(victim)!r}, 0)\n"
    )
    res = pyexec_grader.grade_batch([prog_a, prog_b, prog_c, prog_e, prog_f, prog_g, prog_h, prog_i],
                                    num_workers=4, timeout=4)
    assert res[0]["passed"], f"guard canary program failed: {res[0]}"
    assert res[1]["passed"], f"normal grading broken: {res[1]}"
    assert not res[2]["passed"] and "time" in (res[2].get("error") or "").lower(), f"timeout broken: {res[2]}"
    assert res[3]["passed"], f"LANDLOCK absolute-write not blocked: {res[3]}"
    assert res[4]["passed"], f"LANDLOCK absolute-read not blocked: {res[4]}"
    assert res[5]["passed"], f"LANDLOCK ctypes-unlink not blocked: {res[5]}"
    assert res[6]["passed"], f"allowlisted python RO read broken: {res[6]}"
    assert not os.path.exists(os.path.join(obs, "abs_attack.txt")), "abs_attack.txt materialized!"
    assert open(victim).read() == "IMPORTANT DATA", f"SECCOMP truncate(2) gap EXPLOITED: victim zeroed"

    # D) code_grader (LCB stdin/stdout) — its write must not land in the observation cwd
    item = {"input_output": {"inputs": ["hello\n"], "outputs": ["hello\n"]},
            "code": "open('lcb_junk.txt','w').write('x')\nprint(input())"}
    r = code_grader.grade_batch([item], num_workers=1, timeout=6)
    assert r[0]["passed"], f"lcb canary program did not pass: {r[0]}"

    after = set(os.listdir(obs)) - {"victim.txt"}
    assert open(sentinel).read() == "precious", "SENTINEL modified!"
    leaked = after - before
    assert not leaked, f"files leaked into observation cwd: {leaked}"
    boxes_before = len([d for d in os.listdir("/tmp") if d.startswith("pyexec_")])
    pyexec_grader.grade_batch(["open('x','w').write('y')"] * 30, num_workers=6, timeout=4)
    boxes_after = len([d for d in os.listdir("/tmp") if d.startswith("pyexec_")])
    assert boxes_after <= boxes_before, f"tmpbox leak: {boxes_before} -> {boxes_after} pyexec_ dirs in /tmp"
    from mopd.graders.sandbox import landlock_abi
    print(f"CANARY ALL OK (landlock ABI {landlock_abi()}): writes contained, guard blocks rm/system/rmtree, "
          "LANDLOCK blocks abs-write/abs-read/ctypes-unlink, SECCOMP blocks ctypes-truncate, "
          "no tmpbox leak, normal grading + timeout intact, lcb child isolated")
    os.chdir("/")
    shutil.rmtree(obs, ignore_errors=True)


if __name__ == "__main__":
    main()
