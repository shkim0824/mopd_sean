# mopd_domains — dataset & benchmark selection (2026-09-07)

Selected via 3 parallel deep-research sweeps (all HF ids / repos verified live).
User decisions applied: medical = SFT + RL; law & finance = **RL-only** (no public
English long-CoT SFT exists for either — verified: Legal-R1 repo empty, Unilaw-R1
Chinese/unreleased, Fin-R1-Data HTTP 401 private; self-distillation was declined).

## Medical
| Role | Dataset | Size | Grading / notes |
|---|---|---|---|
| SFT | `UCSC-VLAA/m23k-tokenized` | 23,493 | R1 traces ≤29.7k chars; MCQA-verifiable; m1-paper-proven |
| SFT | `FreedomIntelligence/Medical-R1-Distill-Data` | 22,000 | full R1 traces ≤83k chars, Apache-2.0 |
| RL | `GBaker/MedQA-USMLE-4-options` train | 10,178 | letter exact-match (Med-RLVR precedent, arXiv 2502.19655) |
| RL | `openlifescienceai/medmcqa` train subsample | 20k cap | letter exact-match, single-choice only |
| Eval | MedQA test | 1,273 | accuracy, letter match |
| Eval | `TsinghuaC3I/MedXpertQA` Text | 2,460 | 10-option accuracy; official MIT eval repo |
| Eval | PubMedQA PQA-L test (official repo data) | 500 | official evaluation.py: accuracy + macro-F1 over yes/no/maybe |
| Rejected | HuatuoGPT-o1 SFT / MedReason (traces ≤1.5k tok — shorten thinking models), HealthBench (GPT-4.1 judge required) |

## Law (RL-only)
| Role | Dataset | Size | Grading / notes |
|---|---|---|---|
| RL backbone | `casehold/casehold` train (reglab splits ONLY) | 42.5k | 5-way letter match; NEVER mix with lex_glue case_hold (different split boundaries → leakage) |
| RL | `HolySaint/MBE-exam-questions` + `reglab/barexam_qa` (deduped) | ~2.4k | 4-option letter match |
| RL | `reglab/housing_qa` (statutes in-context, capped) | ≤4k | Yes/No match (binary — capped share to limit reward hacking) |
| Eval | `LEXam-Benchmark/LEXam` mcq_4_choices, EN subset | subset of 1,655 | exact match; released 2025 → uncontaminated; ICLR'26 |
| Eval | CaseHOLD test | 5,314 | macro-F1 (official reglab metric) + accuracy |
| Eval | `nguha/legalbench` exact-match tasks via OFFICIAL evaluation.py | 150+ tasks | balanced accuracy (+ task specials: sara_numeric ±10% etc.); `rule_qa` excluded (official: manual-only) |
| Rejected | MMLU professional_law (same bar-prep universe as RL pools → contaminated), stindardlogic-100k etc. (unverified synthetic, no answer keys) |

## Finance (RL-only)
| Role | Dataset | Size | Grading / notes |
|---|---|---|---|
| RL | FinQA train (official repo czyssrs/FinQA) | 6,251 | numeric; reward comparator = DocMath official `compare_two_numbers` (0.15% rel. tol + percent/scale rescue) |
| RL | TAT-QA train, arithmetic+count only | ~7k | numeric + scale |
| (alt) | `TheFinAI/Fino1_RL_train_9301` | 9,301 | curated equivalent pool, kept as fallback |
| Eval | FinQA test | 1,147 | answer accuracy w/ DocMath comparator (official evaluate.py grades DSL programs — N/A for free-text; community convention documented) |
| Eval | TAT-QA test (gold public since 2024) | 1,669 | OFFICIAL tatqa_metric.py EM + numeracy F1 |
| Eval | DocMath-Eval | pending | HF repo **gated (click-through)** — accept terms on HF to enable; official 0.15% scorer already vendored. Fallback 3rd bench: FinanceReasoning (repo cloned) |
| Rejected | FinanceBench (judge-graded), FinEval (Chinese + hidden labels), ConvFinQA (test withheld), flare-cfa (license ambiguity — internal signal only) |

## Cross-cutting rules (encoded in prep)
- 8-gram decontamination of every RL pool against its domain's eval prompts.
- val = last 100 rows per pool (mopd_rl convention).
- All graders port/import OFFICIAL code (vendored in `third_party/`); gold-answer
  100% + perturbation-fail tests required before any large run (grader-incident rule).
- Clean-room: implemented without reading any mtm code.
