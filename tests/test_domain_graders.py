"""Gold-answer verification for every domain grader (grader-incident rule:
graders must score gold answers 100% and perturbed answers 0% BEFORE any big run).
CPU-only.  PYTHONPATH=. python tests/test_graders.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mopd.graders.domains.extract import (extract_letter, extract_number,
                                          extract_yesno_maybe)
from mopd.graders.domains.graders import (fin_numbers_equal, grade_fin_numeric,
                                          grade_legalbench_task, grade_mcqa,
                                          grade_pubmedqa_batch, grade_tatqa_batch)


def wrap(ans: str) -> str:
    return f"<think>some long reasoning here</think>\n\nAnswer: {ans}"


def test_extract():
    assert extract_letter(wrap("C"), 4) == "C"
    assert extract_letter("<think>x</think>The answer is (B).", 4) == "B"
    assert extract_letter("<think>x</think>\\boxed{J}", 10) == "J"
    assert extract_letter("<think>x</think>I think D. Answer: A", 4) == "A"
    assert extract_yesno_maybe(wrap("maybe")) == "maybe"
    assert extract_yesno_maybe("<think>maybe yes</think>Final: no.") == "no"
    assert extract_number(wrap("14.1%")) == 14.1
    assert extract_number("<think>x</think>\\boxed{-3,200.5}") == -3200.5
    assert extract_number("<think>x</think>The value is $(1,000)") == -1000.0
    print("extract OK")


def test_mcqa():
    ok = bad = 0
    for i in range(200):
        gold = random.choice("ABCDE")
        assert grade_mcqa(wrap(gold), gold, 5)["correct"]; ok += 1
        wrong = random.choice([c for c in "ABCDE" if c != gold])
        assert not grade_mcqa(wrap(wrong), gold, 5)["correct"]; bad += 1
    print(f"mcqa OK ({ok} gold pass / {bad} perturbed fail)")


def test_pubmedqa():
    golds = ["yes", "no", "maybe", "yes", "no"]
    res = grade_pubmedqa_batch([wrap(g) for g in golds], golds)
    assert res["accuracy"] == 1.0 and res["macro_f1"] == 1.0
    res2 = grade_pubmedqa_batch([wrap("no") for _ in golds], golds)
    assert res2["accuracy"] < 1.0
    print("pubmedqa OK")


def test_fin_numeric():
    # DocMath official comparator semantics
    assert fin_numbers_equal(0.18004, 0.18004)
    assert fin_numbers_equal(0.18, 0.18004)          # 0.15% rel tolerance
    assert fin_numbers_equal(18.004, 0.18004)        # x100 percent rescue
    assert not fin_numbers_equal(0.19, 0.18004)
    assert grade_fin_numeric(wrap("14.1%"), "14.1")["correct"]
    assert grade_fin_numeric(wrap("yes"), "yes")["correct"]
    assert not grade_fin_numeric(wrap("13.0"), "14.1")["correct"]
    print("fin_numeric OK")


def test_tatqa():
    items = [
        {"response": wrap("1,203.5 million"), "gold_answer": "1203.5",
         "gold_scale": "million", "answer_type": "arithmetic", "uid": "a"},
        {"response": wrap("42"), "gold_answer": "42", "gold_scale": "",
         "answer_type": "count", "uid": "b"},
    ]
    res = grade_tatqa_batch(items)
    assert res["em"] == 1.0, res
    bad = grade_tatqa_batch([{"response": wrap("999"), "gold_answer": "42",
                              "gold_scale": "", "answer_type": "count", "uid": "c"}])
    assert bad["em"] == 0.0
    print("tatqa OK", res)


def test_legalbench():
    # abercrombie: official exact-match balanced-accuracy task, labels are classes
    answers = ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"]
    gens = [f"<think>x</think>\nAnswer: {a}" for a in answers]
    res = grade_legalbench_task("abercrombie", gens, answers)
    assert res["score"] == 1.0, res
    res_bad = grade_legalbench_task("abercrombie",
                                    [gens[1], gens[0], gens[3], gens[2], gens[4]],
                                    answers)
    assert res_bad["score"] < 1.0
    print("legalbench OK", res)


if __name__ == "__main__":
    test_extract()
    test_mcqa()
    test_pubmedqa()
    test_fin_numeric()
    test_tatqa()
    test_legalbench()
    print("ALL GRADER TESTS OK")
