"""YAML config + ``--override key=value`` (dotted keys) -> plain dict / attribute access."""
from __future__ import annotations

import argparse
import copy
import json
from typing import Any, Dict, List, Optional, Sequence

import yaml


class Cfg(dict):
    """dict with attribute access and nested get('a.b.c')."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def get_path(self, path: str, default=None):
        cur: Any = self
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    def set_path(self, path: str, value):
        cur = self
        parts = path.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value


def _coerce(value: str, old: Any):
    if isinstance(old, bool):
        return value.lower() in ("1", "true", "yes", "y")
    if isinstance(old, int) and not isinstance(old, bool):
        try:
            return int(value)
        except ValueError:
            return float(value)
    if isinstance(old, float):
        return float(value)
    if isinstance(old, list):
        return [x for x in value.split(",") if x != ""]
    if old is None:
        # best-effort literal
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def load_config(path: Optional[str], overrides: Sequence[str] = (), base: Optional[Dict[str, Any]] = None) -> Cfg:
    cfg: Dict[str, Any] = copy.deepcopy(base or {})
    if path:
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        _deep_update(cfg, loaded)
    c = Cfg(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value: {ov}")
        k, v = ov.split("=", 1)
        c.set_path(k, _coerce(v, c.get_path(k)))
    return c


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def add_config_args(p: argparse.ArgumentParser, default_config: Optional[str] = None):
    p.add_argument("--config", default=default_config)
    p.add_argument("--override", action="append", default=[], help="key.path=value (repeatable)")
    return p


def dump(cfg: Cfg) -> str:
    return yaml.safe_dump(json.loads(json.dumps(cfg)), sort_keys=False)
