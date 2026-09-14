"""OT3 grader unit test with a fabricated sidecar (no real data needed).
  PYTHONPATH=. python tests/test_ot3_grader.py"""
import gzip
import json
import os
import shutil
import tempfile

from mopd.graders.ot3_grader import OT3Grader, qhash

Q_MATH = "What is 2+2?\n\nGive your answer."
Q_CODE = "Read an integer n and print n*2."
Q_SCI = "Why is the sky blue?"
Q_GOLF = "Shortest code that prints hello."


def _fake_sidecar(d):
    with open(os.path.join(d, "math_answers.jsonl"), "w") as f:
        f.write(json.dumps({"qhash": qhash(Q_MATH), "answer": "4", "problem_source": "test"}) + "\n")
    with gzip.open(os.path.join(d, "code_tests.jsonl.gz"), "wt") as f:
        f.write(json.dumps({"qhash": qhash(Q_CODE),
                            "input_output": {"inputs": ["3\n", "10\n"], "outputs": ["6\n", "20\n"],
                                             "fn_name": None},
                            "origin": "test"}) + "\n")
    with open(os.path.join(d, "science_refs.jsonl"), "w") as f:
        f.write(json.dumps({"qhash": qhash(Q_SCI), "reference": "Rayleigh scattering",
                            "choices": None, "subsource": "test"}) + "\n")


def main():
    d = tempfile.mkdtemp(prefix="ot3v_")
    try:
        _fake_sidecar(d)
        g = OT3Grader(d)
        assert g.coverage() == {"math": 1, "code": 1, "science_ref": 1}
        # math: right, wrong, whitespace-insensitive lookup
        r = g.grade(Q_MATH, "<think>easy</think>\n\nThe answer is $\\boxed{4}$.")
        assert r["gradeable"] and r["passed"], r
        assert not g.grade(Q_MATH, "\\boxed{5}")["passed"]
        assert g.grade("What  is 2+2?   Give your answer.", "\\boxed{4}")["passed"]  # normalised match
        # code: stdin/stdout pass + fail + no fence
        ok = "```python\nn = int(input())\nprint(n * 2)\n```"
        bad = "```python\nn = int(input())\nprint(n + 2)\n```"
        r = g.grade(Q_CODE, f"<think>x</think>\n\n{ok}")
        assert r["gradeable"] and r["passed"], r
        assert not g.grade(Q_CODE, bad)["passed"]
        assert not g.grade(Q_CODE, "no code")["passed"]
        # science / unknown -> not gradeable
        r = g.grade(Q_SCI, "whatever")
        assert not r["gradeable"] and r["passed"] is None and "Rayleigh" in r["detail"]["reference"]["reference"]
        assert g.grade(Q_GOLF, "x")["kind"] == "none"
        # batch (mixed kinds, order preserved)
        outs = g.grade_batch([Q_MATH, Q_CODE, Q_SCI], ["\\boxed{4}", ok, "hm"], num_workers=2)
        assert outs[0]["passed"] and outs[1]["passed"] and outs[2]["passed"] is None
        # max_tests cap
        g2 = OT3Grader(d, max_tests=1)
        assert g2.grade(Q_CODE, ok)["passed"]
        print("ALL OK")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
