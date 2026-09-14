"""Resume policy shared by SFT / RL / MOPD.

find_resume_checkpoint(output_dir, mode):
  mode "auto"  -> the latest COMPLETE ``checkpoint-N`` under output_dir (has trainer_state.json AND
                  the optimizer/DeepSpeed state), or None when the run never checkpointed
  mode "none"  -> None (fresh run even if checkpoints exist)
  mode <path>  -> that checkpoint

A checkpoint is complete iff ``trainer_state.json`` exists and either a DeepSpeed ``global_step*``
dir or ``optimizer.pt`` is present — a job killed mid-save leaves a partial dir that HF would
crash on; those are renamed to ``checkpoint-N.partial`` so they never get picked up (and are
cleaned by save_total_limit logic on the next save).
"""
from __future__ import annotations

import glob
import os
import re


def _complete(d: str) -> bool:
    if not os.path.isfile(os.path.join(d, "trainer_state.json")):
        return False
    if glob.glob(os.path.join(d, "global_step*")):
        # DeepSpeed writes the model states last; require at least one rank file
        return bool(glob.glob(os.path.join(d, "global_step*", "*model_states.pt"))) or \
            bool(glob.glob(os.path.join(d, "global_step*", "*optim_states.pt")))
    return os.path.isfile(os.path.join(d, "optimizer.pt")) or os.path.isfile(os.path.join(d, "scheduler.pt"))


def list_checkpoints(output_dir: str):
    out = []
    for d in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        m = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(d))
        if m and os.path.isdir(d):
            out.append((int(m.group(1)), d))
    return sorted(out)


def find_resume_checkpoint(output_dir: str, mode: str = "auto"):
    if mode in (None, "", "none", "false", "False"):
        return None
    if mode != "auto":
        assert os.path.isdir(mode), f"resume checkpoint not found: {mode}"
        return mode
    latest = None
    for step, d in list_checkpoints(output_dir):
        if _complete(d):
            latest = d
        else:
            # partial (killed mid-save) -> quarantine so HF never loads it; only rank 0 renames
            if os.environ.get("RANK", "0") in ("0", ""):
                try:
                    os.rename(d, d + ".partial")
                except OSError:
                    pass
    return latest
