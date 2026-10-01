"""CPU sandbox farm — many episodes SHARE one persistent CPU Instance, each with a FRESH environment.

This is the CPU analogue of ``sandbox_ray_farm.RayGpuFarm``. The client (``RayFarmRuntimeSandbox``,
SANDBOX=agentcore-ray-farm) invokes every episode under ONE shared ``runtimeSessionId`` carrying a
per-episode ``sub_session`` — so all episodes land on the SAME warm CPU instance and the ~60-80 s
capacity-provider cold-boot is paid ONCE for the whole run instead of once per episode.

Unlike the GPU farm there is NO GPU to pack and NO warm CUDA context to preserve, so there is no Ray
actor pool and no fractional-GPU gate. Freshness is simpler and STRONGER here:

  * each ``sub_session`` gets its own ``{root}/{sub}`` working dir (created fresh on ``open``),
  * every ``exec`` runs in a FRESH subprocess with a fully isolated environment — HOME, TMPDIR and
    ALL caches (pip / HF / torch / xdg / pyc) redirected INTO the session dir (``fresh_env`` with
    ``isolate_caches=True``), so nothing an episode writes can leak to another episode, and
  * ``close`` wipes the session dir, so no state lingers on the reused instance.

Concurrency needs no gate: distinct sub_sessions are independent subprocesses on distinct dirs, so
they run side by side. The class only tracks open sessions for ``describe``/hygiene.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import threading

try:  # package import (normal) and flat import (server run as a bare script in the image)
    from .sandbox_farm_env import fresh_env, wipe_session
except ImportError:  # pragma: no cover
    from sandbox_farm_env import fresh_env, wipe_session


class CpuFarm:
    """A per-instance registry of sub-session workdirs. One CpuFarm per served CPU instance."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self.root = workspace_root or os.environ.get("SANDBOX_FARM_ROOT", "/workspace/_farm")
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.Lock()
        self._open: dict[str, str] = {}   # sub_session -> its dir

    @classmethod
    def from_env(cls) -> "CpuFarm":
        return cls()

    def start(self) -> "CpuFarm":
        return self

    def describe(self) -> dict:
        with self._lock:
            return {"kind": "cpu", "root": self.root, "open_sessions": len(self._open)}

    def _session_dir(self, sub: str) -> str:
        return os.path.join(self.root, sub)

    def _abs(self, sub: str, path: str) -> str:
        base = self._session_dir(sub)
        return path if os.path.isabs(path) else os.path.join(base, path)

    # ---- op protocol (mirrors sandbox_ray_farm.RayGpuFarm, but CPU-only + full env freshness) -----
    def open(self, sub: str) -> dict:
        with self._lock:
            d = self._session_dir(sub)
            wipe_session(d)          # a fresh start even if the id was somehow reused
            os.makedirs(d, exist_ok=True)
            self._open[sub] = d
        return {"ok": True, "sub_session": sub, "workdir": d}

    def close(self, sub: str) -> dict:
        with self._lock:
            d = self._open.pop(sub, None) or self._session_dir(sub)
        wipe_session(d)              # nothing this episode wrote survives on the reused instance
        return {"ok": True, "sub_session": sub, "wiped": True}

    def exec(self, sub: str, command: str, env: dict | None, timeout: int) -> dict:
        # Fresh, fully-isolated environment for this episode's subprocess (HOME/TMPDIR/caches all in
        # the session dir). We keep the caller's explicit `env` as exports too, matching the GPU farm.
        workdir, run_env = fresh_env(self._session_dir(sub), env, isolate_caches=True)
        exports = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in (env or {}).items())
        full = f"cd {shlex.quote(workdir)}; {exports}bash -lc {shlex.quote(command)}"
        try:
            proc = subprocess.run(full, shell=True, capture_output=True, text=True,
                                  timeout=int(timeout), env=run_env)
            return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        except subprocess.TimeoutExpired as e:
            return {"exit_code": 124, "stdout": e.stdout or "", "stderr": "timeout"}

    def write_file(self, sub: str, path: str, text: str) -> dict:
        dst = self._abs(sub, path)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "w") as f:
            f.write(text or "")
        return {"ok": True}

    def read_file(self, sub: str, path: str) -> dict:
        src = self._abs(sub, path)
        try:
            with open(src) as f:
                return {"ok": True, "text": f.read()}
        except FileNotFoundError:
            return {"ok": False, "error": f"not found: {path}"}
