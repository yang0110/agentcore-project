"""Give one sandbox sub-session a FRESH environment on a REUSED (persistent) instance.

Both farms keep the AgentCore Instance alive across many episodes (all under one shared
``runtimeSessionId``, each carrying a per-episode ``sub_session``) to avoid re-paying the ~60-80 s
Instance cold-boot per episode. The GPU farm ALSO keeps one warm actor *process* alive per slice
(warm CUDA context). Reuse is the point — but it means episode state can leak between episodes that
share the instance. This module centralizes the fix.

What already isolates episodes today: each sub_session runs in its OWN ``{root}/{sub}`` dir (unique
uuid), and every ``exec`` spawns a FRESH subprocess. So files-in-the-workdir and in-checker Python
state never cross episodes. The ONLY remaining leak vector is writes to shared paths OUTSIDE the
session dir — ``$HOME``, ``$TMPDIR`` / ``/tmp``, and the write caches (pip / HF / torch). ``fresh_env``
redirects those into the session dir so a write in episode A cannot be seen by episode B.

This env redirection is hygiene, not a boundary: the CPU farm (``sandbox_cpu_farm``) additionally runs
every exec in a per-episode Landlock jail (``sandbox_farm_jail``; plus its own uid where the container
has CAP_SETUID, which AgentCore does not), which is what stops a DELIBERATE read.

The one nuance is caches. Compiled-CUDA-kernel and downloaded-model caches are content-addressed and
read-mostly, so SHARING them across episodes is both safe (identical regardless of episode) and a big
speedup (no recompile / re-download per episode). So:

  * CPU farm  -> ``isolate_caches=True``  : nothing warm to protect, isolate everything (full freshness).
  * GPU farm  -> ``isolate_caches=False`` : redirect HOME/TMPDIR (write isolation) but keep the caches
                                            on a SHARED path so the warm kernel cache survives.
"""
from __future__ import annotations

import os
import shutil

# A single shared cache root for the GPU path (isolate_caches=False). We must set these EXPLICITLY:
# because fresh_env redirects HOME per session, the default ``~/.cache`` would otherwise move into the
# session dir too and silently lose the warm cache. Pointing them here keeps the cache shared+warm.
_SHARED_CACHE_ROOT = os.environ.get("SANDBOX_SHARED_CACHE", "/workspace/_shared_cache")


def fresh_env(session_dir: str, extra: dict | None = None, *, isolate_caches: bool = True) -> tuple[str, dict]:
    """Return ``(workdir, env)`` for running one sub-session's op with a fresh environment.

    ``HOME`` / ``TMPDIR`` are ALWAYS redirected into ``session_dir`` (write isolation). Caches are
    redirected there too when ``isolate_caches`` (full freshness), or pinned to a shared warm root
    when not (GPU: keep the compiled-kernel cache across episodes).
    """
    home = os.path.join(session_dir, "home")
    tmp = os.path.join(session_dir, "tmp")
    for p in (home, tmp):
        os.makedirs(p, exist_ok=True)
    env = {**os.environ, **(extra or {})}
    env.update({
        "HOME": home,
        "TMPDIR": tmp, "TMP": tmp, "TEMP": tmp,
        "PYTHONPYCACHEPREFIX": os.path.join(tmp, "pyc"),
    })
    if isolate_caches:
        cache = os.path.join(home, ".cache")
        os.makedirs(cache, exist_ok=True)
        env.update({
            "XDG_CACHE_HOME": cache,
            "PIP_CACHE_DIR": os.path.join(cache, "pip"),
            "HF_HOME": os.path.join(cache, "hf"),
            "TORCH_HOME": os.path.join(cache, "torch"),
        })
    else:
        # Keep caches SHARED + warm despite the per-session HOME (must be explicit — see module note).
        for sub in ("pip", "hf", "torch", "xdg"):
            os.makedirs(os.path.join(_SHARED_CACHE_ROOT, sub), exist_ok=True)
        env.update({
            "XDG_CACHE_HOME": os.path.join(_SHARED_CACHE_ROOT, "xdg"),
            "PIP_CACHE_DIR": os.path.join(_SHARED_CACHE_ROOT, "pip"),
            "HF_HOME": os.path.join(_SHARED_CACHE_ROOT, "hf"),
            "TORCH_HOME": os.path.join(_SHARED_CACHE_ROOT, "torch"),
        })
    return session_dir, env


def wipe_session(session_dir: str) -> None:
    """Remove a sub-session's dir on close so its state cannot linger on the reused instance."""
    shutil.rmtree(session_dir, ignore_errors=True)
