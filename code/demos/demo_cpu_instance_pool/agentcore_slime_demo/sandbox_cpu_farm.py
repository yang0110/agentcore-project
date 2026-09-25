"""CPU sandbox farm — many episodes SHARE one persistent CPU Instance, each with a FRESH, ISOLATED environment.

This is the CPU analogue of ``sandbox_ray_farm.RayGpuFarm``. The client (``RayFarmRuntimeSandbox``,
SANDBOX=agentcore-ray-farm) invokes every episode under ONE shared ``runtimeSessionId`` carrying a
per-episode ``sub_session`` — so all episodes land on the SAME warm CPU instance and the ~60-80 s
capacity-provider cold-boot is paid ONCE for the whole run instead of once per episode.

Isolation between sub_sessions on the shared instance (the server runs as root in the image; every
sub_session's code runs as its OWN unprivileged uid):

  * **user**: ``open`` leases a free slot and the sub_session runs as uid/gid ``UID_BASE + slot``
    (default 20000+), no supplementary groups. Different uids => no ptrace, no signals, no reading
    another episode's ``/proc/<pid>/environ``.
  * **filesystem**: ``{root}/{sub}`` is created fresh, owned by that uid, mode 0700; the farm root is
    root-owned 0711 (traversable, NOT listable). So episode B can neither list the root nor open
    anything under A's dir (EACCES). ``write_file`` output is chowned to the uid; ``write_file`` /
    ``read_file`` (done by the root server) refuse any path that resolves outside the session dir
    (``..``, absolute paths, symlinks). Every exec runs with ``umask 077``.
  * **env**: HOME, TMPDIR and ALL caches (pip / HF / torch / xdg / pyc) point INTO the session dir
    (``fresh_env``, ``isolate_caches=True``); the server's ``AWS_*`` variables are NOT inherited.
  * **resources**: each exec runs under ``prlimit`` — ``NPROC`` (per uid, so per sub_session: caps a
    fork bomb), ``FSIZE`` (largest file), ``CORE=0``; no address-space cap (it breaks torch/numpy
    allocators); plus the per-op wall-clock ``timeout``.
  * **lifetime**: ``close`` (and an exec timeout) SIGKILLs EVERY process of the uid — including
    daemonised / setsid'd children — then removes the dir and any files the uid left in ``/tmp``,
    ``/var/tmp`` and ``/dev/shm``, then frees the slot. Nothing an episode started survives it.

What this does NOT isolate (same kernel, same network namespace, no per-episode cgroup): see the
"Isolation guarantees" table in the demo README. Use a microVM per session when that matters.

If the server is NOT root (``os.geteuid() != 0``), uid switching is impossible: the farm falls back to
the dir + env freshness only and ``describe()`` reports ``uid_isolation: False``.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time

try:  # package import (normal) and flat import (server run as a bare script in the image)
    from .sandbox_farm_env import fresh_env, wipe_session
except ImportError:  # pragma: no cover
    from sandbox_farm_env import fresh_env, wipe_session

# sub_session ids become directory names: no slashes, no "..", bounded length.
_SUB_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}$")
_SHARED_TMP = ("/tmp", "/var/tmp", "/dev/shm")


def _become_subreaper() -> None:
    """Orphans of killed episodes re-parent to THIS server (not to a PID 1 that may never reap), so
    ``kill_uid`` can reap them. Without it their zombies keep the uid alive and count toward NPROC."""
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)   # PR_SET_CHILD_SUBREAPER
    except Exception:  # pragma: no cover - non-Linux
        pass


def _uid_pids(uid: int) -> list[tuple[int, int, str]]:
    """``(pid, ppid, state)`` of every process whose real or effective uid is ``uid`` (from /proc)."""
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/status") as f:
                st = dict(line.split(":", 1) for line in f if ":" in line)
            real, eff = st["Uid"].split()[:2]
            if uid in (int(real), int(eff)):
                out.append((int(name), int(st["PPid"]), st["State"].split()[0]))
        except (FileNotFoundError, ProcessLookupError, PermissionError, KeyError, ValueError):
            continue
    return out


def kill_uid(uid: int, rounds: int = 20) -> int:
    """SIGKILL every process of ``uid`` until none is left (forks can race one sweep), reaping the
    zombies that re-parented to us. Returns #killed."""
    killed, me = 0, os.getpid()
    for _ in range(rounds):
        procs = _uid_pids(uid)
        live = [p for p, _, state in procs if state != "Z"]
        for pid, ppid, state in procs:
            if state == "Z" and ppid == me:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
        if not live:
            break
        for pid in live:
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                pass
        time.sleep(0.05)
    return killed


class CpuFarm:
    """A per-instance registry of sub-session workdirs + uid slots. One CpuFarm per served CPU instance."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self.root = workspace_root or os.environ.get("SANDBOX_FARM_ROOT", "/workspace/_farm")
        self.uid_isolation = os.geteuid() == 0 and os.environ.get("SANDBOX_FARM_UID_ISOLATION", "1") == "1"
        self.uid_base = int(os.environ.get("SANDBOX_FARM_UID_BASE", "20000"))
        self.max_slots = int(os.environ.get("SANDBOX_FARM_MAX_SLOTS", "256"))
        self.nproc = int(os.environ.get("SANDBOX_FARM_NPROC", "256"))
        self.fsize = int(os.environ.get("SANDBOX_FARM_FSIZE_MB", "1024")) * 1024 * 1024
        self._prlimit = shutil.which("prlimit")
        if self.uid_isolation:
            _become_subreaper()
        os.makedirs(self.root, exist_ok=True)
        if self.uid_isolation:
            os.chown(self.root, 0, 0)
            os.chmod(self.root, 0o711)   # traversable (to reach your own dir), not listable
        self._lock = threading.Lock()
        self._open: dict[str, str] = {}   # sub_session -> its dir
        self._slot: dict[str, int] = {}   # sub_session -> uid slot

    @classmethod
    def from_env(cls) -> "CpuFarm":
        return cls()

    def start(self) -> "CpuFarm":
        return self

    def describe(self) -> dict:
        with self._lock:
            return {"kind": "cpu", "root": self.root, "open_sessions": len(self._open),
                    "uid_isolation": self.uid_isolation, "prlimit": bool(self._prlimit)}

    # ---- paths ---------------------------------------------------------------------------------
    def _session_dir(self, sub: str) -> str:
        if not _SUB_RE.match(sub or ""):
            raise ValueError(f"bad sub_session id: {sub!r}")
        return os.path.join(self.root, sub)

    def _abs(self, sub: str, path: str) -> str:
        """Resolve ``path`` inside the session dir; refuse anything (.., abs, symlink) that escapes it."""
        base = os.path.realpath(self._session_dir(sub))
        dst = os.path.realpath(path if os.path.isabs(path) else os.path.join(base, path))
        if dst != base and not dst.startswith(base + os.sep):
            raise PermissionError(f"path escapes the sub_session dir: {path}")
        return dst

    def _uid(self, sub: str) -> int | None:
        slot = self._slot.get(sub)
        return None if slot is None else self.uid_base + slot

    def _chown(self, path: str, uid: int | None) -> None:
        if uid is not None:
            os.chown(path, uid, uid)

    def _scrub_uid(self, uid: int) -> None:
        """Kill the uid's processes and drop anything it left in the world-writable tmp dirs."""
        kill_uid(uid)
        for d in _SHARED_TMP:
            try:
                entries = os.listdir(d)
            except OSError:
                continue
            for name in entries:
                p = os.path.join(d, name)
                try:
                    if os.lstat(p).st_uid != uid:
                        continue
                    if os.path.isdir(p) and not os.path.islink(p):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        os.unlink(p)
                except OSError:
                    pass

    # ---- op protocol (mirrors sandbox_ray_farm.RayGpuFarm, but CPU-only + full env freshness) -----
    def open(self, sub: str) -> dict:
        with self._lock:
            d = self._session_dir(sub)
            if self.uid_isolation and sub not in self._slot:
                used = set(self._slot.values())
                free = next((s for s in range(self.max_slots) if s not in used), None)
                if free is None:
                    return {"ok": False, "error": f"no free uid slot (max {self.max_slots})"}
                self._slot[sub] = free
            uid = self._uid(sub)
            if uid is not None:
                self._scrub_uid(uid)     # a slot is reused: nothing of its previous owner may remain
            wipe_session(d)              # a fresh start even if the id was somehow reused
            os.makedirs(d, exist_ok=True)
            if uid is not None:
                os.chmod(d, 0o700)
                self._chown(d, uid)
                for p in ("home", "home/.cache", "tmp"):
                    os.makedirs(os.path.join(d, p), exist_ok=True)
                    self._chown(os.path.join(d, p), uid)
            self._open[sub] = d
        return {"ok": True, "sub_session": sub, "workdir": d, "uid": uid}

    def close(self, sub: str) -> dict:
        with self._lock:
            d = self._open.pop(sub, None) or self._session_dir(sub)
            uid = self._uid(sub)
        killed = 0
        if uid is not None:
            killed = kill_uid(uid)   # before the rmtree, so no process keeps writing into the dir
            self._scrub_uid(uid)
        wipe_session(d)              # nothing this episode wrote survives on the reused instance
        with self._lock:
            self._slot.pop(sub, None)
        return {"ok": True, "sub_session": sub, "wiped": True, "killed": killed}

    def exec(self, sub: str, command: str, env: dict | None, timeout: int) -> dict:
        # Fresh, fully-isolated environment for this episode's subprocess (HOME/TMPDIR/caches all in
        # the session dir); the caller's explicit `env` is merged into it by fresh_env.
        d = self._session_dir(sub)
        uid = self._uid(sub)
        workdir, run_env = fresh_env(d, env, isolate_caches=True)
        if uid is not None:
            for k in [k for k in run_env if k.startswith("AWS_") and k not in (env or {})]:
                run_env.pop(k)       # the server's role credentials are not the episode's
            for p in (run_env["HOME"], run_env["TMPDIR"], run_env["XDG_CACHE_HOME"]):
                self._chown(p, uid)  # fresh_env may have (re)created them as root
            run_env["USER"] = run_env["LOGNAME"] = f"sbx{uid}"
        # The workdir goes in via cwd= and the env via env=, NOT on the command line: other uids can
        # read /proc/<pid>/cmdline, and the sub_session id is the key to this episode's farm ops.
        full = f"umask 077; bash -lc {shlex.quote(command)}"
        argv = ["bash", "-c", full]
        kw: dict = {}
        if uid is not None:
            kw = {"user": uid, "group": uid, "extra_groups": [], "start_new_session": True}
            if self._prlimit:
                argv = [self._prlimit, f"--nproc={self.nproc}", f"--fsize={self.fsize}", "--core=0",
                        "--", *argv]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=int(timeout),
                                  env=run_env, cwd=workdir, **kw)
            return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        except subprocess.TimeoutExpired as e:
            if uid is not None:
                kill_uid(uid)        # the timed-out command's whole process tree, not just bash
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return {"exit_code": 124, "stdout": out, "stderr": "timeout"}

    def write_file(self, sub: str, path: str, text: str) -> dict:
        try:
            dst = self._abs(sub, path)
        except PermissionError as e:
            return {"ok": False, "error": str(e)}
        uid = self._uid(sub)
        parent = os.path.dirname(dst)
        base = os.path.realpath(self._session_dir(sub))
        missing = []
        p = parent
        while p != base and p.startswith(base + os.sep) and not os.path.exists(p):
            missing.append(p)
            p = os.path.dirname(p)
        os.makedirs(parent, exist_ok=True)
        for m in missing:
            self._chown(m, uid)
        with open(dst, "w") as f:
            f.write(text or "")
        self._chown(dst, uid)
        return {"ok": True}

    def read_file(self, sub: str, path: str) -> dict:
        try:
            src = self._abs(sub, path)
            with open(src) as f:
                return {"ok": True, "text": f.read()}
        except FileNotFoundError:
            return {"ok": False, "error": f"not found: {path}"}
        except PermissionError as e:
            return {"ok": False, "error": str(e)}
