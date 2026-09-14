"""Fast CPU tests for the three graders (run: python tests/test_graders_quick.py)."""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mopd.graders import math_grader as mg
from mopd.graders import if_grader as ig
from mopd.graders import code_grader as cg

def test_math():
    assert mg.last_boxed(r"so \boxed{ \frac{1}{2} } done") == r"\frac{1}{2}"
    assert mg.last_boxed(r"\boxed{a} then \boxed{\left(1,2\right)}") == r"\left(1,2\right)"
    assert mg.grade("<think>\\boxed{99}</think>\n\nAnswer: \\boxed{073}", "73")
    assert not mg.grade("<think>\\boxed{73}</think>\n\nno box here", "73")
    assert mg.grade(r"final \boxed{\frac{1}{2}}", r"\frac12")
    assert mg.grade(r"\boxed{0.5}", r"\frac{1}{2}")
    assert not mg.grade(r"\boxed{74}", "73")
    assert mg.grade(r"\boxed{[0, \frac{1}{2}]}", r"[0, \frac{1}{2}]")
    assert mg.grade("<think>", "5") is False
    assert mg.grade(r"\boxed{x^2+1}", r"1+x^2")
    # MATH-500 formatting-only gold answers (\$ literal dollar, \! thin space)
    assert mg.grade(r"final \boxed{32348}", r"\$32,\!348")
    assert mg.grade(r"final \boxed{\$18.90}", r"\$18.90")
    assert mg.grade(r"\boxed{18.90}", r"\$18.90")
    print("math ok")

def test_if():
    row = {"prompt": "Write a 300+ word summary of the wikipedia page. Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.",
           "instruction_id_list": ["punctuation:no_comma", "detectable_format:number_highlighted_sections", "length_constraints:number_words"],
           "kwargs": [{}, {"num_highlights": 3}, {"relation": "at least", "num_words": 300}]}
    body = ("*sec one* " + "word " * 320 + "*sec two* *sec three*")
    e = ig.evaluate_example("ifeval", row["prompt"], row["instruction_id_list"], row["kwargs"], "<think>ignore, commas,,,</think>\n" + body)
    assert e["prompt_strict"], e
    e2 = ig.evaluate_example("ifeval", row["prompt"], row["instruction_id_list"], row["kwargs"], body.replace("word ", "word, ", 1))
    assert not e2["strict"][0] and e2["loose"][0] is False
    e3 = ig.evaluate_example("ifeval", row["prompt"], row["instruction_id_list"], row["kwargs"], "Sure, here it is:\n" + body)
    assert not e3["strict"][0] and e3["loose"][0]
    gt = "[{'instruction_id': ['detectable_format:sentence_hyphens', 'last_word:last_word_answer'], 'kwargs': [None, {'last_word': 'brief'}]}]"
    assert ig.ifevalg_reward("<think>x</think>\nHello world.-This is brief", gt) == 1.0
    assert ig.ifevalg_reward("Hello world. This is brief", gt) == 0.5
    assert ig.ifevalg_reward("Hello world. This is brief", gt, all_or_nothing=True) == 0.0
    assert ig.ifevalg_reward("", gt) == 0.0
    assert len(ig.registry("ifbench")) == 83, len(ig.registry("ifbench"))
    assert len(ig.registry("ifeval")) == 25
    assert len(ig.registry("ifevalg")) == 54, len(ig.registry("ifevalg"))
    print("if ok")

def test_ifbench_official_reproduction():
    """Reproduce the official repo's eval/eval_results_*.jsonl (294 GPT responses) exactly."""
    base = os.path.join(ROOT, "..", "tp", "IFBench")
    if not os.path.isdir(base):
        print("skip ifbench reproduction (no tp/IFBench)"); return
    rows = {json.loads(l)["prompt"]: json.loads(l) for l in open(os.path.join(base, "data", "IFBench_test.jsonl"))}
    for kind in ("strict", "loose"):
        ref = [json.loads(l) for l in open(os.path.join(base, "eval", f"eval_results_{kind}.jsonl"))]
        matched = [r for r in ref if r["prompt"] in rows]
        mism = 0
        for r in matched:
            row = rows[r["prompt"]]
            e = ig.evaluate_example("ifbench", row["prompt"], row["instruction_id_list"], row["kwargs"], r["response"], strip_think=False)
            if r["follow_instruction_list"] != e[kind]:
                mism += 1
                # the only known difference: ratio:sentence_type / sentence_balance depend on the
                # Punkt sentence splitter; upstream results were produced with the legacy punkt
                # pickle, we load punkt_tab (nltk>=3.9) -> 1/290 edge-case split differs.
                assert set(row["instruction_id_list"]) <= {"ratio:sentence_type", "ratio:sentence_balance"}, (row["instruction_id_list"], r["follow_instruction_list"], e[kind])
        print(f"ifbench {kind}: {mism} mismatches / {len(matched)} matched of {len(ref)} ref rows")
        assert len(matched) > 250 and mism <= 1

def test_code():
    io = {"inputs": ["3\n1 2 3\n"], "outputs": ["6\n"], "fn_name": None}
    good = "```python\nn=int(input()); print(sum(map(int,input().split())))\n```"
    bad = "```python\nprint(5)\n```"
    tle = "```python\nwhile True: pass\n```"
    assert cg.grade_response(io, "<think>```python\nprint(0)\n```</think>\nfinal:\n" + good)["passed"]
    assert not cg.grade_response(io, bad)["passed"]
    r = cg.grade_response(io, tle, timeout=2); assert not r["passed"]
    io2 = {"inputs": ["[1, 2, 3]\n2"], "outputs": ["3"], "fn_name": "add"}
    fn = "```python\nclass Solution:\n    def add(self, a, b):\n        return sum(a) - b - 1\n```"
    assert cg.grade_response(io2, fn)["passed"]
    assert cg.extract_code("no code") == ""
    io3 = {"inputs": ["x\n"], "outputs": ["1.0\n"], "fn_name": None}
    assert cg.grade_response(io3, "```python\nprint(1)\n```")["passed"]
    out = cg.grade_batch([{"input_output": io, "code": cg.extract_code(good)}, {"input_output": io, "code": cg.extract_code(bad)}], num_workers=2)
    assert out[0]["passed"] and not out[1]["passed"]
    taco = json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]})
    assert cg.generic_tests_to_input_output(taco)["fn_name"] is None
    pi = json.dumps([{"type": "stdin_stdout", "input": "1 2\n", "output": "3\n"}])
    assert cg.generic_tests_to_input_output(pi)["inputs"] == ["1 2\n"]
    lcb5 = json.dumps([{"testtype": "functional", "input": "[1]", "output": "1"}])
    assert cg.generic_tests_to_input_output(lcb5, {"func_name": "f"})["fn_name"] == "f"
    print("code ok")

def test_pyexec():
    from mopd.graders import pyexec_grader as pg
    # extraction: last fenced block; unfenced bare code; nothing
    assert pg.extract_code("text\n```python\nx=1\n```\nmore\n```python\ny=2\n```") == "y=2"
    assert pg.extract_code("def f():\n    return 1").startswith("def f")
    assert pg.extract_code("no code here") == ""
    # HumanEval: full-function rewrite AND completion-style bodies both grade
    he = {"prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
          "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
          "entry_point": "add"}
    full = "```python\ndef add(a, b):\n    return a + b\n```"
    body = "```python\n    return a + b\n```"
    wrong = "```python\ndef add(a, b):\n    return a - b\n```"
    assert pg.run_program(pg.humaneval_program(he, pg.extract_code(full)))["passed"]
    assert pg.run_program(pg.humaneval_program(he, pg.extract_code(body)))["passed"]
    assert not pg.run_program(pg.humaneval_program(he, pg.extract_code(wrong)))["passed"]
    assert pg.humaneval_program(he, "") is None
    # MBPP: test_list asserts (+ setup code) and MBPP+ extended `test` script
    mb = {"test_list": ["assert mul(2, 3) == 6", "assert mul(0, 5) == 0"], "test_setup_code": ""}
    ok = "```python\ndef mul(a, b):\n    return a * b\n```"
    assert pg.run_program(pg.mbpp_program(mb, pg.extract_code(ok)))["passed"]
    assert not pg.run_program(pg.mbpp_program(mb, "def mul(a,b):\n    return a+b"))["passed"]
    mbp = {"test": "assert mul(2, 3) == 6\nassert mul(4, 4) == 16"}
    assert pg.run_program(pg.mbpp_program(mbp, pg.extract_code(ok)))["passed"]
    # asserts (AceCode RL rewards) + timeout + batch
    assert pg.run_program(pg.asserts_program(["assert inc(1) == 2"], "def inc(x):\n    return x + 1"))["passed"]
    r = pg.run_program("while True: pass", timeout=2)
    assert not r["passed"] and r["error"] == "timeout"
    out = pg.grade_batch([pg.mbpp_program(mb, pg.extract_code(ok)), None], num_workers=2)
    assert out[0]["passed"] and not out[1]["passed"]
    # thinking hygiene: fenced block inside <think> is ignored
    assert pg.extract_code("<think>```python\nbad\n```</think>\n" + ok) == "def mul(a, b):\n    return a * b"
    print("pyexec ok")


if __name__ == "__main__":
    test_math(); test_if(); test_ifbench_official_reproduction(); test_code(); test_pyexec(); print("ALL OK")
