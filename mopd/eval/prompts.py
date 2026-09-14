"""Benchmark prompts = the official / harness-standard ones, verbatim.

math  (AIME)  : Qwen3 model-card instruction (also used by MathArena / DeepSeek-R1 card)
                  "{problem}\\n\\nPlease reason step by step, and put your final answer within \\boxed{}."
                alt ``nemo``: NeMo-Skills generic/math
                  "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\\n\\n{problem}"
code  (LCB)   : LiveCodeBench ``lcb_runner/prompts/code_generation.py``
                  system = SYSTEM_MESSAGE_GENERIC, user = get_generic_question_template_answer(problem)
                (this generic template does NOT contain the "You will NOT return anything except
                 for the program" sentence that Qwen3 removed for thinking mode)
if    (IFEval / IFBench): the raw prompt as the single user turn (lm-eval-harness ``doc_to_text: prompt``,
                IFBench ``generate_responses.py``), no system prompt.
"""
from __future__ import annotations

from typing import Any, Dict, List

MATH_QWEN_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
MATH_NEMO_PREFIX = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n"

# --- verbatim from lcb_runner/prompts/code_generation.py -------------------------------------
LCB_SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question (problem specification) and "
    "will generate a correct Python program that matches the specification and passes all tests."
)
LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem and enclose your "
    "code within delimiters."
)
LCB_FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
    "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when "
    "the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def lcb_user_prompt(question_content: str, starter_code: str = "") -> str:
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def math_messages(problem: str, style: str = "qwen") -> List[Dict[str, str]]:
    if style == "qwen":
        return [{"role": "user", "content": problem.strip() + MATH_QWEN_SUFFIX}]
    if style == "nemo":
        return [{"role": "user", "content": MATH_NEMO_PREFIX + problem.strip()}]
    raise ValueError(style)


def code_messages(question_content: str, starter_code: str = "", system: bool = True) -> List[Dict[str, str]]:
    user = lcb_user_prompt(question_content, starter_code or "")
    if system:
        return [{"role": "system", "content": LCB_SYSTEM_MESSAGE_GENERIC}, {"role": "user", "content": user}]
    # single-turn variant (DEFAULT): system text folded into the user message. The MoT/gpt-oss SFT data has no
    # system turns; with a real system turn the SFT'd Qwen3-4B answered LCB prompts with an immediate <|im_end|>.
    return [{"role": "user", "content": LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + user}]


def if_messages(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def raw_completion_prompt(messages: List[Dict[str, str]]) -> str:
    """Base-model probe: no chat template at all — the user text followed by
    a bare newline (what a pretrained LM would see in a plain document)."""
    return "\n\n".join(m["content"] for m in messages if m["role"] != "system") + "\n\n"
