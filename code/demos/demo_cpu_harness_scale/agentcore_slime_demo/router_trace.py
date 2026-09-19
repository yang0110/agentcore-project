"""Per-episode ROUTER tracing + inference-engine PINNING for the scaling demos.

The async-RL rollout has four replicated tiers an episode flows through, and the demos want to
PROVE each episode is sticky to ONE path through them (except inference, which slime normally
load-balances per request — see below):

    harness    (W warm workers)     — which worker DROVE the episode      (sticky per episode)
    inference  (E SGLang engines)    — which engine SERVED its /generate    (PINNED here)
    sandbox    (one session/episode) — which sandbox held its files         (sticky per episode)
    runtime    (K AgentCore Runtimes)— which Runtime ARN the sandbox used   (round-robin, sticky)

This module does two things:

1. PIN INFERENCE. slime's SGLang router load-balances every /generate across the E engines
   (cache-aware), so a single episode's turns can hit different engines. To make inference sticky
   too, we discover the engine base-URLs once (the router's worker registry) and target ONE engine
   directly, chosen deterministically from the episode's key. If the registry endpoint is
   unavailable we fall back to the router URL (unpinned — still fully RECORDED, just load-balanced).

2. RECORD THE PATH. `emit()` writes ONE greppable `[router]` line per episode tying the episode key
   to all four tiers' routing identity. metrics.py parses these into `<log>.router.csv` and asserts
   stickiness (one worker / engine / runtime / sandbox-session per episode). The `[sbx-op]` records
   already carry `session=<sandbox-session>` + `arn=<runtime>`, so the `[router]` line is the JOIN
   that maps that sandbox session to its worker + pinned engine.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
import zlib

_ENGINE_LOCK = threading.Lock()
_ENGINE_CACHE: list[str] | None = None


def _http_get_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (trusted local router)
        raw = r.read()
    return json.loads(raw.decode()) if raw else None


def _extract_urls(data) -> list[str]:
    """Pull worker base-URLs out of the several shapes sgl-router's registry endpoints return:
    {"urls": [...]}, {"workers": [{"url": ...}]}, ["http://..", ...], or /get_loads' {url: load}."""
    if isinstance(data, dict):
        if isinstance(data.get("urls"), list):
            return [str(u) for u in data["urls"]]
        if isinstance(data.get("workers"), list):
            return [w.get("url") for w in data["workers"] if isinstance(w, dict) and w.get("url")]
        http_keys = [k for k in data.keys() if str(k).startswith("http")]
        if http_keys:
            return [str(k) for k in http_keys]
    if isinstance(data, list):
        out: list[str] = []
        for w in data:
            if isinstance(w, str):
                out.append(w)
            elif isinstance(w, dict) and w.get("url"):
                out.append(str(w["url"]))
        return out
    return []


def engine_urls(router_url: str) -> list[str]:
    """The E SGLang engine base-URLs behind slime's router, discovered once and cached process-wide.

    Tries the router's worker-registry endpoints (this sgl-router build serves GET /workers and
    /get_loads); falls back to [router_url] so pinning degrades gracefully to router LB (still
    recorded). Override with ROUTER_ENGINE_URLS (comma-separated) to skip discovery entirely."""
    global _ENGINE_CACHE
    if _ENGINE_CACHE is not None:
        return _ENGINE_CACHE
    with _ENGINE_LOCK:
        if _ENGINE_CACHE is not None:
            return _ENGINE_CACHE
        override = os.environ.get("ROUTER_ENGINE_URLS", "").strip()
        urls: list[str] = []
        if override:
            urls = [u.strip() for u in override.split(",") if u.strip()]
        else:
            router = router_url.rstrip("/")
            for ep in ("/workers", "/list_workers", "/get_loads"):
                try:
                    urls = _extract_urls(_http_get_json(f"{router}{ep}"))
                except Exception:  # noqa: BLE001 — any failure => try the next endpoint / fallback
                    urls = []
                if urls:
                    break
        if not urls:
            urls = [router_url.rstrip("/")]  # graceful fallback: unpinned, record-only
        urls = [u if str(u).startswith("http") else f"http://{u}" for u in urls]
        _ENGINE_CACHE = sorted(set(urls))
        return _ENGINE_CACHE


def pin_engine(key: str, router_url: str) -> tuple[str, int, int]:
    """Deterministically map an episode/worker `key` to ONE engine.

    Returns (engine_url, engine_idx, num_engines). Uses crc32 (stable across processes, unlike the
    salted builtin hash) so the same key always lands on the same engine. num_engines==1 means
    discovery failed and we are recording against the router unpinned."""
    urls = engine_urls(router_url)
    idx = zlib.crc32(key.encode()) % len(urls)
    return urls[idx], idx, len(urls)


def emit(**fields) -> None:
    """One greppable per-episode `[router] k=v ...` record (single write so concurrent episode
    coroutines never interleave). None values are dropped; str values are stripped of spaces."""
    parts = []
    for k, v in fields.items():
        if v is None:
            continue
        parts.append(f"{k}={str(v).replace(' ', '_')}")
    sys.stdout.write(f"[router] ts={time.time():.3f} " + " ".join(parts) + "\n")
    sys.stdout.flush()


def arn_tail(arn: str | None) -> str:
    """Short, stable id for a Runtime ARN (matches [sbx-op]'s arn= field)."""
    if not arn:
        return "?"
    return arn.rsplit("/", 1)[-1]


def _unwrap(sb):
    """Reach the real sandbox under a wrapper (demo_cpu_quiesce's KeptSandbox, etc.), so the
    join fields below come from the object that actually owns the AgentCore session/ARN."""
    for attr in ("_sandbox", "sandbox", "inner", "_inner", "_wrapped"):
        inner = getattr(sb, attr, None)
        if inner is not None and inner is not sb:
            return inner
    return sb


def session_of(sb) -> str:
    """The sandbox SESSION id stamped on every [sbx-op] (session=…) — the episode join key.
    Digs through a wrapper if the outer object doesn't carry it."""
    for obj in (sb, _unwrap(sb)):
        v = getattr(obj, "_session_id", None) or getattr(obj, "_sub", None)
        if v:
            return str(v)
    return "?"


def arn_of(sb) -> str:
    """The episode's Runtime ARN tail (matches [sbx-op] arn=), through a wrapper if needed."""
    for obj in (sb, _unwrap(sb)):
        v = getattr(obj, "_arn", None)
        if v:
            return arn_tail(v)
    return "?"


def sandbox_id_of(sb) -> str:
    for obj in (sb, _unwrap(sb)):
        v = getattr(obj, "sandbox_id", None)
        if v:
            return str(v)
    return "?"
