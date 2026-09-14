"""Teacher prefill client: per-token log-probabilities of a (prompt+completion)
token sequence under a frozen teacher served by ``vllm serve``.

Request (== verl ``AsyncTeacherLLMServerManager`` / TRL ``AsyncDistillationTrainer``):
  POST {url}/v1/completions  {"model": name, "prompt": <token ids>, "max_tokens": 1,
                              "temperature": 1.0, "prompt_logprobs": K, "logprobs": 0}
  -> choices[0].prompt_logprobs : list (len = n_prompt_tokens) of
       None (position 0)  |  {token_id(str): {"logprob": float, "rank": int, "decoded_token": str}, ...}
     For every position the dict ALWAYS contains the actual prompt token, plus the
     top-K teacher tokens when K > 0  (vLLM semantics: ``prompt_logprobs=0`` -> actual token only).
     The server must run with ``--max-logprobs >= K``.

What we return per sequence (completion positions only):
  logp   : teacher log-prob of the sampled completion token     -> policy-gradient advantage (Eq. 3)
  topk   : (ids[T,K], logps[T,K]) teacher top-K at each position -> top-k loss (Eq. 5)

Routing: one client per domain, replicas round-robin, thread pool for concurrency,
retries with backoff, hard timeout. Servers are plain HTTP on bare short hostnames
-> proxies must be unset in the job (see infra memory).
"""
from __future__ import annotations

import itertools
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


class TeacherPool:
    def __init__(self, endpoints: Dict[str, List[str]], served_names: Optional[Dict[str, str]] = None,
                 top_k: int = 0, max_workers: int = 64, timeout: float = 600.0, retries: int = 3,
                 logprobs_mode_check: bool = True):
        self.endpoints = {d: list(u) for d, u in endpoints.items()}
        self.names = served_names or {d: d for d in endpoints}
        self.top_k = top_k
        self.timeout = timeout
        self.retries = retries
        self._rr = {d: itertools.count() for d in endpoints}
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._session = threading.local()
        self.stats = {"requests": 0, "failures": 0, "seconds": 0.0, "tokens": 0}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ http
    def _sess(self) -> requests.Session:
        s = getattr(self._session, "s", None)
        if s is None:
            s = requests.Session()
            s.trust_env = False  # never pick up http_proxy from the container env
            self._session.s = s
        return s

    def healthcheck(self) -> Dict[str, bool]:
        out = {}
        for d, urls in self.endpoints.items():
            for u in urls:
                try:
                    out[u] = self._sess().get(f"{u}/v1/models", timeout=10).status_code == 200
                except Exception:
                    out[u] = False
        return out

    def wait_ready(self, timeout: float = 3600.0, interval: float = 10.0):
        t0 = time.time()
        while True:
            h = self.healthcheck()
            if all(h.values()):
                return h
            if time.time() - t0 > timeout:
                raise RuntimeError(f"teacher servers not ready: {h}")
            time.sleep(interval)

    def _url(self, domain: str) -> str:
        urls = self.endpoints[domain]
        return urls[next(self._rr[domain]) % len(urls)]

    def _post(self, domain: str, token_ids: Sequence[int], top_k: int) -> List[Optional[Dict[str, Any]]]:
        body = {"model": self.names[domain], "prompt": list(map(int, token_ids)), "max_tokens": 1,
                "temperature": 1.0, "prompt_logprobs": int(top_k), "logprobs": 0, "seed": 0}
        last = None
        for attempt in range(self.retries):
            url = self._url(domain)
            t0 = time.time()
            try:
                r = self._sess().post(f"{url}/v1/completions", json=body, timeout=self.timeout)
                if r.status_code != 200:
                    raise RuntimeError(f"{url} -> {r.status_code}: {r.text[:300]}")
                pl = r.json()["choices"][0]["prompt_logprobs"]
                with self._lock:
                    self.stats["requests"] += 1
                    self.stats["seconds"] += time.time() - t0
                    self.stats["tokens"] += len(token_ids)
                return pl
            except Exception as e:  # noqa: BLE001
                last = e
                with self._lock:
                    self.stats["failures"] += 1
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"teacher prefill failed for domain={domain} after {self.retries} tries: {last}")

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _parse(pl: List[Optional[Dict[str, Any]]], token_ids: Sequence[int], start: int, top_k: int
               ) -> Tuple[List[float], Optional[Tuple[List[List[int]], List[List[float]]]]]:
        """completion positions = [start, len). Returns per-token logp of the actual
        token and (optionally) fixed-width top-k ids/logps (padded with the actual token)."""
        logps: List[float] = []
        tk_ids: List[List[int]] = []
        tk_lp: List[List[float]] = []
        for pos in range(start, len(token_ids)):
            entry = pl[pos] if pos < len(pl) else None
            tid = int(token_ids[pos])
            if not entry:  # position 0 has no logprob (only possible if start == 0)
                logps.append(0.0)
                if top_k:
                    tk_ids.append([-1] * top_k)
                    tk_lp.append([0.0] * top_k)
                continue
            actual = entry.get(str(tid))
            lp = float(actual["logprob"]) if actual is not None else -1e4
            logps.append(lp)
            if top_k:
                # vLLM returns the top-K teacher tokens plus the actual token (rank may be > K).
                # T_t (paper Eq. 5) = the K highest-probability teacher tokens; the actual token is
                # in it iff its rank <= K.  Sort by logprob and keep K.
                items = sorted(((int(k), float(v["logprob"])) for k, v in entry.items()), key=lambda kv: -kv[1])[:top_k]
                ids = [kv[0] for kv in items]
                lps = [kv[1] for kv in items]
                while len(ids) < top_k:  # server returned fewer than K: pad with id -1 (masked out in the loss)
                    ids.append(-1)
                    lps.append(0.0)
                tk_ids.append(ids)
                tk_lp.append(lps)
        return logps, ((tk_ids, tk_lp) if top_k else None)

    # ------------------------------------------------------------------ public
    def prefill_one(self, domain: str, token_ids: Sequence[int], completion_start: int, top_k: Optional[int] = None):
        k = self.top_k if top_k is None else top_k
        pl = self._post(domain, token_ids, k)
        return self._parse(pl, token_ids, completion_start, k)

    def prefill_batch(self, domains: Sequence[str], token_id_lists: Sequence[Sequence[int]],
                      completion_starts: Sequence[int], top_k: Optional[int] = None):
        futs = [self._pool.submit(self.prefill_one, d, ids, s, top_k)
                for d, ids, s in zip(domains, token_id_lists, completion_starts)]
        return [f.result() for f in futs]

    def flush_stats(self) -> Dict[str, float]:
        with self._lock:
            s = dict(self.stats)
            self.stats = {"requests": 0, "failures": 0, "seconds": 0.0, "tokens": 0}
        out = {"teacher/requests": s["requests"], "teacher/failures": s["failures"],
               "teacher/sec_per_request": s["seconds"] / max(s["requests"], 1),
               "teacher/tokens_per_request": s["tokens"] / max(s["requests"], 1)}
        return out


def load_endpoints(path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """teacher_endpoints.json written by the sbatch (from placement.py --emit json)."""
    p = json.load(open(path))
    endpoints = p["endpoints"] if "endpoints" in p else p
    names = {s["domain"]: s["served_name"] for s in p.get("servers", [])} if isinstance(p, dict) else {}
    return endpoints, names
