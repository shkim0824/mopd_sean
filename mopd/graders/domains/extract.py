"""Answer extraction from thinking-model responses (post-</think> text).

Conventions follow the official eval harnesses of the target benchmarks:
- letters:  "Answer: X" / "answer is (X)" / \\boxed{X} / last standalone capital letter
  (MedXpertQA eval + LegalBench normalization + our existing mopd math/if contract)
- yes/no/maybe: PubMedQA official expects one of {yes,no,maybe} (evaluation.py)
- numbers: last \\boxed{...} number, else "Answer: <num>", else last number token
  (DocMath-Eval utils/evaluation_utils.py extraction convention)
"""
from __future__ import annotations

import re
from typing import Optional

THINK_CLOSE = "</think>"


def strip_thinking(text: str) -> str:
    """Return the post-think segment; if no close tag, return the raw text
    (matches mopd.graders.math_grader.strip_thinking behaviour)."""
    if THINK_CLOSE in text:
        return text.split(THINK_CLOSE)[-1]
    return text


_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANS_LINE = re.compile(r"(?:answer|정답)\s*(?:is)?\s*[:\-]?\s*\(?([A-Ja-j])\)?(?![a-zA-Z])",
                       re.IGNORECASE)
_LONE_LETTER = re.compile(r"(?<![A-Za-z])([A-J])(?![A-Za-z])")


def extract_letter(response: str, n_options: int = 10) -> Optional[str]:
    """Extract a single option letter (A..) from the post-think text."""
    valid = {chr(ord("A") + i) for i in range(n_options)}
    text = strip_thinking(response).strip()
    for m in reversed(_BOXED.findall(text)):
        cand = m.strip().strip("()").upper()
        if cand in valid:
            return cand
    for m in reversed(list(_ANS_LINE.finditer(text))):
        cand = m.group(1).upper()
        if cand in valid:
            return cand
    tail = text[-200:]
    cands = [c.upper() for c in _LONE_LETTER.findall(tail) if c.upper() in valid]
    if cands:
        return cands[-1]
    return None


_YNM = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)


def extract_yesno_maybe(response: str, allow_maybe: bool = True) -> Optional[str]:
    text = strip_thinking(response)
    for m in reversed(_BOXED.findall(text)):
        w = m.strip().lower()
        if w in ("yes", "no") or (allow_maybe and w == "maybe"):
            return w
    hits = [w.lower() for w in _YNM.findall(text)]
    if not allow_maybe:
        hits = [w for w in hits if w != "maybe"]
    return hits[-1] if hits else None


_NUM = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?%?")


def _clean_number(tok: str) -> Optional[float]:
    tok = tok.strip().replace("$", "").replace(",", "")
    neg = tok.startswith("(") and tok.endswith(")")  # accounting negatives
    tok = tok.strip("()")
    pct = tok.endswith("%")
    tok = tok.rstrip("%")
    try:
        v = float(tok)
    except ValueError:
        return None
    if neg:
        v = -v
    # NOTE: percent sign kept out of the value; comparator handles x100 rescue
    return v


def extract_number(response: str) -> Optional[float]:
    text = strip_thinking(response)
    for m in reversed(_BOXED.findall(text)):
        for tok in reversed(_NUM.findall(m)):
            v = _clean_number(tok)
            if v is not None:
                return v
    ans = re.split(r"(?:answer|answer is|final answer)\s*[:\-]?", text,
                   flags=re.IGNORECASE)
    for seg in reversed(ans[1:] or []):
        for tok in _NUM.findall(seg[:120]):
            v = _clean_number(tok)
            if v is not None:
                return v
    toks = _NUM.findall(text[-300:])
    for tok in reversed(toks):
        v = _clean_number(tok)
        if v is not None:
            return v
    return None


def extract_text_answer(response: str) -> str:
    """Free-text final answer (LegalBench-style): last non-empty line post-think."""
    text = strip_thinking(response).strip()
    m = re.split(r"(?:answer|final answer)\s*[:\-]\s*", text, flags=re.IGNORECASE)
    if len(m) > 1:
        return m[-1].strip().split("\n")[0].strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else ""
