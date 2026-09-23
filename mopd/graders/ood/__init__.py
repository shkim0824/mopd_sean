"""Graders for the phase-3 OOD benchmarks (GPQA + the three safety sets).

Every scoring rule imports the official code vendored under ``third_party/``:
  gpqa        -> third_party/simple_evals/gpqa_official.py   (openai/simple-evals)
  harmbench   -> third_party/harmbench/harmbench_official.py (centerforaisafety/HarmBench)
  truthfulqa  -> third_party/truthfulqa/truthfulqa_official.py (sylinrl/TruthfulQA)
  sycophancy  -> anthropics/evals sycophancy/README.md protocol (log-prob comparison)
"""
