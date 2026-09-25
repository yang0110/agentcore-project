"""CPU sandbox farm — many episodes SHARE one persistent CPU Instance, each with a FRESH, ISOLATED environment.

This is the CPU analogue of ``sandbox_ray_farm.RayGpuFarm``. The client (``RayFarmRuntimeSandbox``,
SANDBOX=agentcore-ray-farm) invokes every episode under ONE shared ``runtimeSessionId`` carrying a
per-episode ``sub_session`` — so all episodes land on the SAME warm CPU instance and the ~60-80 s
capacity-provider cold-boot is paid ONCE for the whole run instead of once per episode.

Isolation between sub_sessions on the shared instance. There are two layers, and the farm picks what
the container allows (``describe()["mode"]``):

  * **Landlock jail (always, when the kernel has it)** — every exec runs under ``sandbox_farm_jail.py``:
    a Landlock domain that allows read+exec of the system dirs and read+write of ONLY this episode's
    dir (no sibling dirs, no ``/tmp``/``/dev/shm``/``/workspace``); TCP connect/bind denied
    (``SANDBOX_FARM_NET=deny``); signals and abstract unix sockets scoped to the episode; ptrace and
    ``/proc/<pid>/environ`` of other episodes denied. The wrapper is a subreaper outside the domain:
    it kills the WHOLE tree (setsid'd daemons included) when the command exits or times out, above
    ``SANDBOX_FARM_NPROC`` processes or above ``SANDBOX_FARM_MEM_MB`` RSS.
  * **own uid (only with CAP_SETUID/SETGID/CHOWN/KILL)** — the sub_session also runs as uid
    ``UID_BASE + slot`` (default 20000+), its dir owned by that uid with mode 0700, farm root 0711,
    ``prlimit --nproc/--fsize``, and ``close`` kills every process of the uid and scrubs its files in
    ``/tmp``, ``/var/tmp``, ``/dev/shm``.

**On AgentCore only the Landlock layer is available.** Measured live on 2026-09-25: the server is uid 0 in
a user namespace with CapEff = CapBnd = 0 and NoNewPrivs=1, so setuid and chown are EPERM, and every
sub_session shares one uid. The uid layer is used where the container keeps the capabilities (local
docker, the agentcore_p4 dev box).

In both modes: HOME, TMPDIR and ALL caches (pip / HF / torch / xdg / pyc) point INTO the session dir
(``fresh_env``, ``isolate_caches=True``). The server's ``AWS_*`` variables are NOT inherited.
``write_file``/``read_file`` (done by the server) refuse any path that resolves outside the session dir
(``..``, absolute paths, symlinks). Every exec runs with ``umask 077``. ``close`` kills the episode's
running execs, then removes its dir.

What this does NOT isolate (same kernel, same network namespace — UDP is not covered by Landlock — no
per-episode cgroup): see the "Isolation guarantees" table in the demo README. Use a microVM per session
when that matters.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

try:  # package import (normal) and flat import (server run as a bare script in the image)
    from .sandbox_farm_env import fresh_env, wipe_session
except ImportError:  # pragma: no cover
    from sandbox_farm_env import fresh_env, wipe_session

# sub_session ids become directory names: no slashes, no "..", bounded length.
_SUB_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}$")
_SHARED_TMP = ("/tmp", "/var/tmp", "/dev/shm")
_JAIL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_farm_jail.py")
_NEEDED_CAPS = (0, 5, 6, 7)   # CAP_CHOWN, CAP_KILL, CAP_SETGID, CAP_SETUID


def has_uid_caps() -> bool:
    """True iff this process can actually switch uid and chown (root in a capability-less user
    namespace, as on AgentCore, cannot, even though ``geteuid() == 0``)."""
    try:
        with open("/proc/self/status") as f:
            eff = next(int(line.split()[1], 16) for line in f if line.startswith("CapEff:"))
    except (OSError, StopIteration, ValueError):
        return False
    return all(eff >> bit & 1 for bit in _NEEDED_CAPS)


def landlock_abi() -> int:
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        return max(0, int(libc.syscall(444, None, ctypes.c_size_t(0), ctypes.c_uint32(1))))
    except Exception:  # pragma: no cover - non-Linux
        return 0


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
        self.uid_isolation = (os.geteuid() == 0 and has_uid_caps()
                              and os.environ.get("SANDBOX_FARM_UID_ISOLATION", "1") == "1")
        self.landlock_abi = landlock_abi() if os.environ.get("SANDBOX_FARM_LANDLOCK", "1") == "1" else 0
        self.net = os.environ.get("SANDBOX_FARM_NET", "deny")
        self.mem_mb = int(os.environ.get("SANDBOX_FARM_MEM_MB", "4096"))
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
        os.chmod(self.root, 0o711)       # traversable (to reach your own dir), not listable
        self._lock = threading.Lock()
        self._running: dict[str, set] = {}   # sub_session -> its live jail wrappers (Popen)
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
                    "mode": self.mode, "uid_isolation": self.uid_isolation,
                    "landlock_abi": self.landlock_abi, "net": self.net, "nproc": self.nproc,
                    "mem_mb": self.mem_mb, "prlimit": bool(self._prlimit and self.uid_isolation)}

    @property
    def mode(self) -> str:
        layers = (["uid"] if self.uid_isolation else []) + (["landlock"] if self.landlock_abi else [])
        return "+".join(layers) or "dir-only"

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
            os.chmod(d, 0o700)
            for p in ("home", "home/.cache", "tmp"):
                os.makedirs(os.path.join(d, p), exist_ok=True)
            if uid is not None:
                for p in ("", "home", "home/.cache", "tmp"):
                    self._chown(os.path.join(d, p), uid)
            self._open[sub] = d
        return {"ok": True, "sub_session": sub, "workdir": d, "uid": uid, "mode": self.mode}

    def close(self, sub: str) -> dict:
        with self._lock:
            d = self._open.pop(sub, None) or self._session_dir(sub)
            uid = self._uid(sub)
            running = list(self._running.pop(sub, ()))
        killed = self._stop(running)  # its in-flight execs: each jail kills its own tree
        if uid is not None:
            killed += kill_uid(uid)  # before the rmtree, so no process keeps writing into the dir
            self._scrub_uid(uid)
        wipe_session(d)              # nothing this episode wrote survives on the reused instance
        with self._lock:
            self._slot.pop(sub, None)
        return {"ok": True, "sub_session": sub, "wiped": True, "killed": killed}

    @staticmethod
    def _stop(procs: list, grace: float = 5.0) -> int:
        """SIGTERM jail wrappers (each then SIGKILLs its whole tree and exits); SIGKILL stragglers."""
        for p in procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.monotonic() + grace
        for p in procs:
            try:
                p.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                p.kill()
        return len(procs)

    def exec(self, sub: str, command: str, env: dict | None, timeout: int) -> dict:
        # Fresh, fully-isolated environment for this episode's subprocess (HOME/TMPDIR/caches all in
        # the session dir); the caller's explicit `env` is merged into it by fresh_env.
        d = self._session_dir(sub)
        uid = self._uid(sub)
        workdir, run_env = fresh_env(d, env, isolate_caches=True)
        for k in [k for k in run_env if k.startswith("AWS_") and k not in (env or {})]:
            run_env.pop(k)           # the server's role/region settings are not the episode's
        if uid is not None:
            for p in (run_env["HOME"], run_env["TMPDIR"], run_env["XDG_CACHE_HOME"]):
                self._chown(p, uid)  # fresh_env may have (re)created them as root
            run_env["USER"] = run_env["LOGNAME"] = f"sbx{uid}"
        # The workdir goes in via cwd= and the env via env=, NOT on the command line: other episodes can
        # read /proc/<pid>/cmdline, and the sub_session id is the key to this episode's farm ops.
        run_env["SANDBOX_JAIL_DIR"] = d
        full = f"umask 077; bash -lc {shlex.quote(command)}"
        argv = [sys.executable, _JAIL, f"--nproc={self.nproc}", f"--mem-mb={self.mem_mb}",
                f"--fsize={self.fsize}", f"--net={self.net}"]
        argv += ["--require-landlock"] if self.landlock_abi else ["--no-landlock"]
        argv += ["--", "bash", "-c", full]
        kw: dict = {"start_new_session": True}
        if uid is not None:
            kw.update(user=uid, group=uid, extra_groups=[])
            if self._prlimit:
                argv = [self._prlimit, f"--nproc={self.nproc}", f"--fsize={self.fsize}", "--core=0",
                        "--", *argv]
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env=run_env, cwd=workdir, **kw)
        with self._lock:
            self._running.setdefault(sub, set()).add(proc)
        try:
            out, err = proc.communicate(timeout=int(timeout))
            return {"exit_code": proc.returncode, "stdout": out, "stderr": err}
        except subprocess.TimeoutExpired:
            self._stop([proc])       # the jail kills the timed-out command's whole process tree
            if uid is not None:
                kill_uid(uid)
            out, _ = proc.communicate()
            return {"exit_code": 124, "stdout": out or "", "stderr": "timeout"}
        finally:
            with self._lock:
                self._running.get(sub, set()).discard(proc)

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
