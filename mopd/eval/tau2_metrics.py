"""Score a tau2-bench results file with tau2's OWN metrics code and write the unified metrics.json.

Runs in the tau2 venv (python 3.13):
    python -m mopd.eval.tau2_metrics --results <results.json> --out outputs/eval_all/<tag> \
        --model <path> --user-llm <name> --agent-args '<json>' --domain telecom --split base

  score                     pass^1 x 100 on every task of the split (official number)
  pass_hat_k[k]             tau2.metrics.agent_metrics.pass_hat_k, k = 1..num_trials, mean over tasks
  *_clean                   the same on the tasks NOT present in the AReaL SFT data
                            (data/tau2/contaminated_telecom_task_ids.json, written by the prep script)
  avg_reward, n_tasks, n_trials, n_sims, termination reasons, infra errors, mean agent turns

Nothing here re-implements a metric: pass^k and success come from tau2.metrics.agent_metrics.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--user-llm", default="")
    ap.add_argument("--user-args", default="{}")
    ap.add_argument("--agent-args", default="{}")
    ap.add_argument("--domain", default="telecom")
    ap.add_argument("--split", default="base")
    ap.add_argument("--contaminated", default="data/tau2/contaminated_telecom_task_ids.json")
    ap.add_argument("--vllm-flags", default="")
    ap.add_argument("--agent-impl", default="llm_agent", help="llm_agent = official prompt; llm_agent_areal = AReaL training prompt (NON-official)")
    a = ap.parse_args(argv)

    from tau2.data_model.simulation import Results
    from tau2.metrics.agent_metrics import compute_metrics, is_successful

    res = Results.load(Path(a.results))
    m = compute_metrics(res)
    bench = "tau2_%s" % a.domain
    bench_note = "" if a.agent_impl == "llm_agent" else " [NON-official prompt]"

    def as_dict(metrics):
        d = metrics.__dict__ if hasattr(metrics, "__dict__") else dict(metrics)
        out = {}
        for k, v in d.items():
            try:
                json.dumps(v); out[k] = v
            except TypeError:
                out[k] = str(v)
        return out

    sims = res.simulations
    n_tasks = len({s.task_id for s in sims})
    trials = collections.Counter(s.task_id for s in sims)
    n_trials = max(trials.values()) if trials else 0
    term = collections.Counter(str(getattr(getattr(s, "termination_reason", "?"), "value", getattr(s, "termination_reason", "?"))) for s in sims)
    rewards = [float(s.reward_info.reward) if s.reward_info is not None else 0.0 for s in sims]
    n_turns = [len(getattr(s, "messages", []) or []) for s in sims]
    succ = sum(1 for r in rewards if is_successful(r))

    # tau2's compute_metrics EXCLUDES simulations that ended in infrastructure_error (and tasks with no
    # simulation at all). For a small-context model that is exactly the failure mode we care about, so
    # a conservative number is reported alongside the official one: every task of the split counts,
    # and a task whose only simulations errored or are missing counts as failed.
    # pass^1 with every task of the split in the denominator and every errored / missing trial counted as
    # a failure: mean over tasks of (#successful trials / n_trials)
    split_ids = [t.id for t in res.tasks]
    def _is_infra(s):
        tr = getattr(s, "termination_reason", "")
        return str(getattr(tr, "value", tr)).lower().endswith("infrastructure_error")
    n_infra = sum(1 for s in sims if _is_infra(s))
    succ_by_task = collections.Counter()
    for s in sims:
        r = s.reward_info.reward if s.reward_info is not None else None
        if (r is not None) and is_successful(float(r)):
            succ_by_task[s.task_id] += 1
    scored_tasks = {s.task_id for s in sims if not _is_infra(s)}
    n_unscored_tasks = sum(1 for t in split_ids if t not in scored_tasks)
    n_split = max(len(split_ids), 1); k1 = max(n_trials, 1)
    per_task_success = {t: succ_by_task[t] / k1 for t in split_ids}
    score_all_tasks = 100.0 * sum(per_task_success.values()) / n_split
    entry = {
        "score": 100.0 * float(m.pass_hat_ks.get(1, 0.0)) if isinstance(m.pass_hat_ks, dict) else None,
        "score_all_tasks": score_all_tasks,
        "n_infrastructure_error_sims": n_infra,
        "n_tasks_without_scored_sim": n_unscored_tasks,
        "metric": "pass^1 (tau2-bench, %s split)%s" % (a.split, bench_note),
        "pass_hat_k": {str(k): 100.0 * float(v) for k, v in (m.pass_hat_ks or {}).items()},
        "avg_reward": 100.0 * float(m.avg_reward),
        "success_rate_sims": 100.0 * succ / max(len(sims), 1),
        "n_tasks": n_tasks, "n_trials": n_trials, "n_sims": len(sims),
        "termination_reasons": dict(term),
        "mean_messages_per_sim": (sum(n_turns) / len(n_turns)) if n_turns else None,
        "agent_metrics_raw": as_dict(m),
        "higher_is_better": True,
    }
    # contamination-free subset (same official function on the filtered simulations/tasks)
    bad = set()
    if os.path.exists(a.contaminated):
        bad = set(json.load(open(a.contaminated)))
    if bad:
        keep_sims = [s for s in sims if s.task_id not in bad]
        keep_tasks = [t for t in res.tasks if t.id not in bad]
        if keep_sims:
            sub = res.model_copy(update={"simulations": keep_sims, "tasks": keep_tasks})
            mc = compute_metrics(sub)
            entry["clean"] = {
                "excluded_task_ids": sorted(bad & {s.task_id for s in sims}),
                "n_tasks": len({s.task_id for s in keep_sims}),
                "pass_hat_k": {str(k): 100.0 * float(v) for k, v in (mc.pass_hat_ks or {}).items()},
                "avg_reward": 100.0 * float(mc.avg_reward),
                "score_clean": 100.0 * float(mc.pass_hat_ks.get(1, 0.0)),
            }

    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(a.results, os.path.join(a.out, "%s.results.json" % bench))
    scores = {bench: entry["score"], "%s_all_tasks" % bench: entry["score_all_tasks"]}
    for k, v in entry["pass_hat_k"].items():
        if k != "1":
            scores["%s_pass%s" % (bench, k)] = v
    if "clean" in entry:
        scores["%s_clean" % bench] = entry["clean"]["score_clean"]
    metrics = {
        "model": a.model, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "runner": "tau2-bench (official harness)",
        "benchmarks": {bench: entry}, "scores": scores,
        "protocol": {
            bench: {
                "harness": "sierra-research/tau2-bench v1.0.1, tau2 run --domain %s --task-split-name %s" % (a.domain, a.split),
                "agent": a.agent_impl, "user": "user_simulator (dual-control)",
                "official": a.agent_impl == "llm_agent",
                "prompt_note": ("official tau2 LLMAgent system prompt" if a.agent_impl == "llm_agent" else
                                "NON-OFFICIAL: agent system prompt replaced by the AReaL training prompt (instructions + main policy, no tech-support manual); tasks/user sim/grading untouched"),
                "agent_llm": "student via vLLM OpenAI server", "agent_llm_args": json.loads(a.agent_args or "{}"),
                "user_llm": a.user_llm, "user_llm_args": json.loads(a.user_args or "{}"),
                "vllm_flags": a.vllm_flags, "num_trials": n_trials, "seed": 300, "max_steps": 200,
            }
        },
        "protocol_source": "tau2_metrics.py (written at scoring time from the run's arguments)",
    }
    p = os.path.join(a.out, "metrics.json")
    json.dump(metrics, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)
    print(json.dumps({"scores": scores, "n_tasks": n_tasks, "n_trials": n_trials, "n_sims": len(sims),
                      "termination": dict(term)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
