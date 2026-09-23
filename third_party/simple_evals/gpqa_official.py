"""VERBATIM excerpts of openai/simple-evals (MIT) — the GPQA protocol.

Source: https://github.com/openai/simple-evals (clone of 2026-09-16; commit in
``data/eval/gpqa/manifest.json``).  The authors' own repo (``idavidrein/gpqa``)
ships baseline prompting experiments and NO scoring code, so simple-evals is the
reproducible reference and the source of the GPQA-Diamond numbers people quote.

  * ``common.py::QUERY_TEMPLATE_MULTICHOICE``   -- the whole zero-shot prompt
  * ``common.py::ANSWER_PATTERN_MULTICHOICE``   -- the ONE extraction regex
  * ``gpqa_eval.py``                            -- choice order + permutation:
        choices = [Correct, Incorrect 1, Incorrect 2, Incorrect 3]
        choices = [choices[i] for i in row["permutation"]]     # random.Random(0)
        correct_answer = "ABCD"[choices.index(row["Correct Answer"])]
        match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
        score = 1.0 if (match.group(1) if match else None) == correct_answer else 0.0

Extraction has NO fallback on purpose: a response that ignores the format scores 0
rather than being rescued.  That strictness is part of the published number, and a
rescue heuristic would move our scores in the generous direction — the worse
direction to be wrong in.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# --- common.py:15 ----------------------------------------------------------- VERBATIM
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# --- common.py:26 ----------------------------------------------------------- VERBATIM
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"

LETTERS = "ABCD"


def format_multichoice_question(row: Dict[str, Any]) -> str:
    """``common.py::format_multichoice_question`` = template.format(**row) with
    row = dict(A=..., B=..., C=..., D=..., Question=...)."""
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def gpqa_prompt(question: str, choices: List[str]) -> str:
    if len(choices) != 4:
        raise ValueError("GPQA expects exactly 4 choices, got %d" % len(choices))
    return format_multichoice_question(dict(Question=question, A=choices[0], B=choices[1],
                                            C=choices[2], D=choices[3]))


def gpqa_extract(text: str) -> Optional[str]:
    m = re.search(ANSWER_PATTERN_MULTICHOICE, text or "")
    return m.group(1).upper() if m else None
