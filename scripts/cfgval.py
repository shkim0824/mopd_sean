#!/usr/bin/env python3
"""stdlib-only reader for the simple YAML configs in configs/ (the batch host may lack PyYAML).
usage: cfgval.py <config.yaml> <section.key> [...]   -> one value per line ('' if absent)
       cfgval.py <config.yaml> data.paths.KEYS          -> the domain names listed under data.paths (space separated)
       cfgval.py --check-paths [--include-legacy]       -> every path-like value in configs/**/*.yaml must exist (rc 1 otherwise)"""
import glob, os, re, sys

PATH_KEYS = ("base", "student", "cache_dir", "output", "checkpoint_dir", "model_name", "data_path", "log_dir", "nltk_data",
             "mopd_rl_root", "worker_py_executable", "worker_pythonpath", "law", "fin", "if", "med", "math", "code", "mix")
OUTPUT_KEYS = ("output", "checkpoint_dir", "log_dir")


def check_paths(include_legacy=False):
    """Scan the raw yaml text (NeMo configs nest deeper than load() follows) for path-like values under PATH_KEYS.
    Relative paths are resolved against the repo root; output-like keys only need their parent directory."""
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    canon = os.path.join(os.path.dirname(root), "mopd_sean")   # NeMo configs carry absolute canonical paths; map them onto a staged copy
    pat = re.compile(r"^\s*(" + "|".join(PATH_KEYS) + r"):\s*\"?([^\s\"#]+)\"?", re.M)
    checked, bad = 0, []
    for f in sorted(glob.glob(os.path.join(root, "configs", "**", "*.yaml"), recursive=True)):
        rel = os.path.relpath(f, root)
        if os.path.basename(f) in ("accelerate_zero2.yaml", "eval.yaml") or (not include_legacy and "/legacy_trl/" in rel):
            continue
        for m in pat.finditer(open(f).read()):
            key, val = m.group(1), m.group(2)
            if val in ("null", "auto", "none", "true", "false") or val.startswith(("${", "http", "file://")) or "/" not in val:
                continue
            for part in val.split(":"):
                if root != canon and part.startswith(canon + "/"): part = root + part[len(canon):]
                q = part if part.startswith("/") else os.path.join(root, part); checked += 1
                if key in OUTPUT_KEYS:
                    if not os.path.isdir(os.path.dirname(q.rstrip("/"))): bad.append((rel, key, part, "parent-missing"))
                elif not os.path.exists(q):
                    bad.append((rel, key, part, "missing"))
    print(f"[cfgval] checked {checked} paths in configs/ ({'incl.' if include_legacy else 'excl.'} legacy_trl): {len(bad)} problems")
    for b in bad: print("  ", *b)
    return 1 if bad else 0


def load(path):
    out, sec, sub = {}, None, None
    for line in open(path):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^(\s*)([A-Za-z_][\w-]*):\s*(.*?)\s*(#.*)?$", line.rstrip("\n"))
        if not m:
            continue
        indent, key, val = len(m.group(1)), m.group(2), m.group(3)
        if indent == 0:
            sec, sub = key, None
            out.setdefault(sec, {})
            if val:
                out[sec] = val
        elif indent == 2 and sec is not None and isinstance(out.get(sec), dict):
            sub = key if not val else None
            out[sec][key] = val.strip().strip('"').strip("'") if val else {}
        elif indent == 4 and sec is not None and sub is not None and isinstance(out[sec].get(sub), dict):
            out[sec][sub][key] = val.strip().strip('"').strip("'")
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check-paths"]:
        sys.exit(check_paths(include_legacy="--include-legacy" in sys.argv))
    cfg = load(sys.argv[1])
    for q in sys.argv[2:]:
        parts = q.split(".")
        if len(parts) == 3 and parts[2] == "KEYS":
            v = cfg.get(parts[0], {}).get(parts[1], {})
            print(" ".join(v.keys()) if isinstance(v, dict) else "")
            continue
        s, _, k = q.partition(".")
        v = cfg.get(s, {})
        print(v.get(k, "") if isinstance(v, dict) else v)
