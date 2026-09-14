"""Math answer grading = the NeMo-Skills / Math-Verify convention.

* extraction: the LAST ``\\boxed{...}`` in the response, brace-matched
  (NeMo-Skills ``math_grader.search_boxed`` semantics; lighteval's
  ``boxed_match_priority=0`` behaves the same for boxed answers).
* comparison: ``math_verify.verify(parse(gold), parse(pred))`` on both sides
  wrapped in ``$...$`` when they carry no LaTeX environment (NeMo-Skills does
  exactly this so bare strings such as ``73`` / ``073`` parse), with an integer
  fast path for AIME-style answers (0-999).

The thinking block (``<think> ... </think>``) is stripped before extraction so a
boxed expression inside the reasoning can never be graded.
"""
from __future__ import annotations

import re
from typing import Optional

from math_verify import LatexExtractionConfig, parse, verify

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove closed <think>...</think> blocks; if only an opening tag exists the
    answer never finished thinking -> treat everything after the tag as answer
    text (it will almost always fail extraction, which is the intended score)."""
    if not text:
        return ""
    out = _THINK_RE.sub("", text)
    if "<think>" in out and "</think>" not in out:
        out = out.split("<think>", 1)[1]
    return out.strip()


def last_boxed(text: str) -> Optional[str]:
    """Return the content of the last \\boxed{...} / \\fbox{...} with balanced
    braces, or None. Mirrors NeMo-Skills' ``search_boxed`` (rfind + brace walk)."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = idx
    depth = 0
    start = None
    while i < len(text):
        c = text[i]
        if c == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i].strip()
        i += 1
    # ``\boxed 5`` (no braces) is used by some models
    m = re.match(r"\\(?:boxed|fbox)\s+([^\s$]+)", text[idx:])
    return m.group(1) if m else None


def _clean_fmt(s: str) -> str:
    """Drop pure-formatting LaTeX that Math-Verify cannot parse: thin spaces ``\\!``
    and literal dollar signs ``\\$`` (MATH answers like ``\\$32,\\!348``)."""
    return s.replace("\\!", "").replace("\\$", "")


def _wrap_latex(s: str) -> str:
    s = _clean_fmt(s.strip())
    if "$" in s or "\\boxed" in s or "\\(" in s or "\\[" in s:
        return s
    return f"${s}$"


def _as_int(s: str) -> Optional[int]:
    s = _clean_fmt(s.strip()).replace(",", "").replace("$", "").replace("\\", "")
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d+\.0+", s):
        return int(float(s))
    return None


def extract_answer(response: str) -> Optional[str]:
    return last_boxed(strip_thinking(response))


def is_equiv(gold: str, pred: Optional[str], timeout_seconds: int = 10) -> bool:
    """Symbolic/numeric equivalence of a gold answer string and an extracted
    prediction (already boxed-extracted). None -> False."""
    if pred is None:
        return False
    gi, pi = _as_int(gold), _as_int(pred)
    if gi is not None and pi is not None:
        return gi == pi
    try:
        g = parse(_wrap_latex(gold), extraction_config=[LatexExtractionConfig()],
                  parsing_timeout=timeout_seconds)
        p = parse(_wrap_latex(pred), extraction_config=[LatexExtractionConfig()],
                  parsing_timeout=timeout_seconds)
        if not g or not p:
            return False
        return bool(verify(g, p, timeout_seconds=timeout_seconds))
    except Exception:
        return False


def grade(response: str, gold: str) -> bool:
    """Full pipeline: strip thinking -> last boxed -> verify against gold."""
    return is_equiv(str(gold), extract_answer(response))
