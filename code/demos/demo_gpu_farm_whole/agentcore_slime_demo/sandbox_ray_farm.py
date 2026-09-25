"""`sandbox_ray_farm` — many concurrent sandboxes sharing ONE g6 GPU via Ray fractional GPUs.

This is Demo B4's SERVER-SIDE backend, and the third answer to "where does the KernelBench
checker's GPU come from?":

  * Demo B / B3  — the sandbox is its OWN AgentCore Runtime #2, and AgentCore maps 1 session ->
                   1 isolated microVM/instance. That is clean isolation but does NOT share a GPU:
                   each concurrent rollout wants its own g6 instance, so N in-flight episodes need
                   N g6 instances. On a capacity-starved g6 fleet that is exactly what stalled B3.
  * Demo B4 (here) — keep the sandbox on a REMOTE g6 AgentCore Runtime #2 (like B2/B3 — "all of
                   series B uses the remote g6 instance"), but pack SEVERAL concurrent sandboxes
                   onto that ONE g6 instance and let **Ray split its single GPU** between them.

Where this runs (the important correction over the earlier draft)
------------------------------------------------------------------
This farm runs INSIDE the g6 AgentCore Runtime #2 container (driven by ``sandbox_server_agent``),
NOT on the slime trainer box. The trainer box gives ALL 8 of its GPUs to slime (4 train + 4
infer), exactly like B2/B3 — the shared GPU is the *remote g6's* GPU, reached over
``invoke_agent_runtime``. Because AgentCore maps 1 ``runtimeSessionId`` -> 1 instance, every B4
episode invokes the g6 under ONE shared ``runtimeSessionId`` (so they all land on the same g6
instance) and is distinguished by a per-episode ``sub_session`` carried in the op payload. The
server leases one fractional-GPU Ray actor per ``sub_session`` from this farm, so up to
``max_sandboxes`` episodes run on the single g6 GPU concurrently. See ``RayFarmRuntimeSandbox``
(the client half, in ``agentcore_runtime_sandbox``), which Demo B4 selects via
``SANDBOX=agentcore-ray-farm`` so the SAME in-process ``run_episode`` hook
(``agentcore_slime_demo.generate.generate``, shared with Demo B) drives it — the farm is the only
difference between the two demos.

Why a software memory cap on TOP of Ray's fraction
--------------------------------------------------
Ray's ``num_gpus=0.25`` is a *scheduling* fraction only: it tells Ray "at most 4 of these fit on a
GPU", sets the actor's ``CUDA_VISIBLE_DEVICES`` to the GPU it packed the actor onto, and stops.
It does NOT enforce any memory or compute limit — the actors then race for the whole device, and
one runaway kernel can OOM its neighbours. L4 / L40S are Ada Lovelace: **no MIG**, so there is no
hardware partition to fall back on. So we also enforce a per-sandbox memory cap in **software**
(``torch.cuda.set_per_process_memory_fraction``). The catch is that the KernelBench checker runs as
a *subprocess* (``python3 run_checks.py``), and that call only affects the process that makes it —
so we drop a tiny ``sitecustomize.py`` into each sandbox's working dir (Python auto-imports
``sitecustomize`` from the script's own directory at startup) which reads ``ENV_GPU_MEM_FRACTION``
and applies the cap inside the checker process itself. Ray does the packing; sitecustomize isolates.

Isolation between co-resident sub_sessions
------------------------------------------
Every slot runs its checker subprocesses as its OWN unprivileged uid (``B4_SANDBOX_UID_BASE`` +
slot, default 20000+). Its session dir is owned by that uid with mode 0700, and the farm root and the
per-actor dirs are 0711 (traversable, not listable). ``write_file`` / ``read_file`` resolve the path
and refuse anything outside the session dir, symlinks included. On close the actor SIGKILLs every
process of that uid (plus each exec's process group), then removes the dir. HOME and the compile
caches are per uid, so one slot cannot plant a compiled kernel that ANOTHER slot loads. Episodes that
follow each other on the SAME slot share its uid and cache; ``SANDBOX_ISOLATE_HOME=1`` moves HOME and
every cache into the session dir, which is wiped on close, and removes that last path too (at the cost
of recompiling each episode's kernels).

The uid switch needs CAP_SETUID/SETGID/CHOWN/KILL. Docker's default root has them (the local farm).
The AgentCore container on the g6 is uid 0 with an EMPTY capability set (see sandbox_server_agent), so
there the farm detects the missing caps and falls back to one shared uid: separate dirs, path
confinement and process-group kill still apply, but nothing OS-level stops one checker from reading
another's dir. ``ping``/``open`` report ``uid_isolation`` so the log shows which mode ran.

``ray`` and ``torch`` are imported lazily so this module still loads for offline tooling/tests (and
so it imports cleanly on the trainer box, which never runs the farm — only the g6 does).
"""

from __future__ import annotations

import contextlib
import math
import os
import queue
import shlex
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

from .sandbox import ExecResult, FileContent  # noqa: F401 — re-exported for type users

# Host-wide lockfile serializing CUDA kernel COMPILATION across co-resident checker subprocesses.
# Lives in /dev/shm (shared across all actor processes on the one g6 host, same tmpfs as the
# default workspace_root). See _SITECUSTOMIZE for why only the *compile* is gated.
_NVCC_LOCK_PATH = "/dev/shm/b4-nvcc.lock"

# Per-slot unprivileged uids (slot i runs as UID_BASE + i) and the root of their per-uid HOMEs.
_UID_BASE = int(os.environ.get("B4_SANDBOX_UID_BASE", "20000"))
_HOME_ROOT = os.environ.get("B4_SANDBOX_HOME_ROOT", "/var/tmp/b4-home")
# CAP_CHOWN=0, CAP_KILL=5, CAP_SETGID=6, CAP_SETUID=7: all four are needed to hand a slot its own uid.
_NEEDED_CAPS = (0, 5, 6, 7)


def _can_switch_uid() -> bool:
    """True when this process may run children as another uid (root with the needed capabilities).
    False on the AgentCore g6 container (uid 0, CapEff=0) and for any non-root offline run."""
    if os.environ.get("B4_SANDBOX_UID_ISOLATION", "auto") == "0" or os.geteuid() != 0:
        return False
    try:
        with open("/proc/self/status") as f:
            caps = next(int(l.split()[1], 16) for l in f if l.startswith("CapEff:"))
    except (OSError, StopIteration, ValueError):
        return False
    return all(caps & (1 << c) for c in _NEEDED_CAPS)


def _uid_pids(uid: int) -> list[int]:
    """PIDs of every live process whose real uid is ``uid`` (read from /proc). Zombies are skipped:
    they hold no resources, cannot be killed, and vanish once their parent reaps them."""
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/status") as f:
                st = dict(l.split(":", 1) for l in f if ":" in l)
            ruid = int(st["Uid"].split()[0])
            state = st["State"].split()[0]
        except (OSError, KeyError, ValueError, IndexError):
            continue
        if ruid == uid and state != "Z":
            pids.append(int(d))
    return pids


def _chown_tree(path: str, uid: int) -> None:
    os.chown(path, uid, uid)
    for dirpath, dirnames, filenames in os.walk(path):
        for n in dirnames + filenames:
            with contextlib.suppress(FileNotFoundError):
                os.chown(os.path.join(dirpath, n), uid, uid, follow_symlinks=False)

# Auto-imported by every `python3` the sandbox runs (Python imports `sitecustomize` from the
# script's own directory, which is the per-session workdir). It (1) applies the per-sandbox memory
# cap and (2) serializes nvcc compiles INSIDE the checker subprocess — pieces Ray's fractional
# `num_gpus` does not give us. Each block is a no-op unless its env var is set (so a CPU-only smoke
# run is unaffected).
_SITECUSTOMIZE = """\
# Auto-generated by sandbox_ray_farm: cap this process's GPU share + serialize kernel compiles.
import os as _os

_frac = _os.environ.get("ENV_GPU_MEM_FRACTION")
if _frac:
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            # cuda:0 == the ONE device Ray exposed to this actor via CUDA_VISIBLE_DEVICES.
            _torch.cuda.set_per_process_memory_fraction(min(float(_frac), 1.0), 0)
    except Exception:
        pass

# Serialize CUDA kernel COMPILATION (only) across all co-resident checker subprocesses via a
# host-wide file lock. Co-resident sandboxes may still RUN kernels concurrently on the shared GPU;
# what wedges a non-MIG L4 is many nvcc / cpp_extension builds firing at once (observed: the g6
# goes unresponsive under sustained concurrent compiles). So we gate just the build: the lock is
# held only for the load_inline/load call and released before the kernel is executed/timed.
_lockpath = _os.environ.get("ENV_NVCC_LOCK")
if _lockpath:
    try:
        import fcntl as _fcntl
        import functools as _functools
        from torch.utils import cpp_extension as _cext

        def _serialize(_fn):
            @_functools.wraps(_fn)
            def _wrapped(*a, **k):
                # Read-only open: the lockfile is created 0644 by the (root) actor, and flock() works
                # on a read-only fd, so the unprivileged checker uid can take it without write access.
                try:
                    _f = open(_lockpath)
                except OSError:                       # no lockfile: build unserialized, never fail
                    return _fn(*a, **k)
                try:
                    _fcntl.flock(_f, _fcntl.LOCK_EX)   # one nvcc build at a time, host-wide
                    return _fn(*a, **k)
                finally:
                    _fcntl.flock(_f, _fcntl.LOCK_UN)
                    _f.close()
            return _wrapped

        _cext.load_inline = _serialize(_cext.load_inline)
        _cext.load = _serialize(_cext.load)
    except Exception:
        pass
"""


def _ray_get(obj_ref):
    import ray

    return ray.get(obj_ref)


class _SandboxActorImpl:
    """The body of one sandbox actor: a long-lived process holding a warm CUDA context on its
    fractional slice of the g6 GPU, serving exec/write_file/read_file against per-session dirs.

    One instance == one concurrent-sandbox *slot* (Ray gives it ``num_gpus`` of the device). It is
    reused across episodes: each episode leases it, runs in an isolated ``{root}/{session}`` dir,
    and clears that dir on release, so the warm CUDA context (and the actor's Python process)
    survive — the whole point of a *pool of long-lived sessions* over per-episode launches.

    Inherits the container ENV, so on the g6 image the volume-installed torch (SANDBOX_PYLIB on
    PYTHONPATH) and its driver (LD_LIBRARY_PATH=/usr/lib64) — both baked into the image ENV — are
    visible to this actor process AND to the checker subprocess it spawns.
    """

    def __init__(self, actor_id: str, *, mem_fraction: float, workspace_root: str, warm: bool = True,
                 slot: int = 0) -> None:
        self.actor_id = actor_id
        self.mem_fraction = float(mem_fraction)
        self.root = workspace_root
        os.makedirs(self.root, exist_ok=True)
        # Ray sets this to the (possibly shared) physical GPU it packed this actor onto.
        self.cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.isolate_home = os.environ.get("SANDBOX_ISOLATE_HOME") == "1"
        # This slot's own uid, or None when the container can't switch uids (the g6: zero caps).
        self.uid = _UID_BASE + int(slot) if _can_switch_uid() else None
        self.home = os.path.join(_HOME_ROOT, f"uid-{self.uid}" if self.uid is not None else f"slot-{slot}")
        self._pgids: dict[str, set[int]] = {}   # session -> process groups of its execs
        # The lockfile must exist before an unprivileged checker opens it read-only.
        with contextlib.suppress(OSError):
            os.close(os.open(_NVCC_LOCK_PATH, os.O_CREAT | os.O_RDONLY, 0o644))
        os.makedirs(self.home, exist_ok=True)
        if self.uid is not None:
            # Farm root + actor dir: traversable (x) but not listable/readable by the slot uids.
            for d in (os.path.dirname(self.root), self.root, _HOME_ROOT):
                os.chmod(d, 0o711)
            os.chown(self.home, self.uid, self.uid)
            os.chmod(self.home, 0o700)
        self.warm_error = ""
        if warm:
            self._warm_cuda()

    def _warm_cuda(self) -> None:
        """Create the CUDA context once so the first checker call isn't cold, and cap the actor's
        OWN memory (subprocess caps are applied separately by sitecustomize)."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(min(self.mem_fraction, 1.0), 0)
                torch.zeros(1, device="cuda")
                torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 — a missing/incompatible torch must not kill the actor
            self.warm_error = f"{type(e).__name__}: {e}"

    # ---- op protocol (mirrors sandbox_server_agent.handle, but keyed by sub-session) ----------
    def ping(self) -> dict:
        return {
            "ok": True,
            "actor": self.actor_id,
            "cuda_visible": self.cuda_visible,
            "mem_fraction": self.mem_fraction,
            "warm_error": self.warm_error,
            "uid": self.uid if self.uid is not None else os.geteuid(),
            "uid_isolation": self.uid is not None,
            "isolate_home": self.isolate_home,
        }

    def _session_dir(self, session_id: str) -> str:
        if not session_id or "/" in session_id or session_id in (".", ".."):
            raise ValueError(f"bad sub_session id {session_id!r}")
        d = os.path.join(self.root, session_id)
        if not os.path.isdir(d):
            os.makedirs(os.path.join(d, "tmp"), exist_ok=True)
            # Drop the memory-cap hook so every `python3` launched from this dir inherits the cap.
            with open(os.path.join(d, "sitecustomize.py"), "w") as f:
                f.write(_SITECUSTOMIZE)
            if self.uid is not None:
                _chown_tree(d, self.uid)
                os.chmod(d, 0o700)
        return d

    def _abs(self, session_id: str, path: str) -> str:
        """Resolve ``path`` inside the session dir; refuse anything that escapes it (``..``, an
        absolute path elsewhere, or a symlink the checker planted pointing at another session)."""
        base = os.path.realpath(self._session_dir(session_id))
        p = os.path.realpath(path if os.path.isabs(path) else os.path.join(base, path))
        if p != base and not p.startswith(base + os.sep):
            raise PermissionError(f"path {path!r} is outside this sub_session")
        return p

    def _run_env(self, workdir: str, env: dict | None) -> dict:
        run_env = {**os.environ, **(env or {})}
        if self.isolate_home:
            # Full freshness: HOME, TMPDIR and EVERY cache (incl. compiled CUDA extensions) live in
            # the session dir, which close() wipes. No state crosses episodes, even on the same slot.
            try:
                from .sandbox_farm_env import fresh_env
            except ImportError:
                from sandbox_farm_env import fresh_env
            _, run_env = fresh_env(workdir, run_env, isolate_caches=True)
            run_env["TORCH_EXTENSIONS_DIR"] = os.path.join(run_env["XDG_CACHE_HOME"], "torch_extensions")
            if self.uid is not None:
                _chown_tree(workdir, self.uid)
        else:
            # Default: a per-uid HOME that persists across this slot's episodes, so the compiled-
            # kernel cache stays warm, but is private to the slot (0700, its own uid).
            cache = os.path.join(self.home, ".cache")
            run_env.update({
                "HOME": self.home, "XDG_CACHE_HOME": cache,
                "TORCH_EXTENSIONS_DIR": os.path.join(cache, "torch_extensions"),
                "TMPDIR": os.path.join(workdir, "tmp"),
            })
        if self.uid is not None:
            run_env.update({"USER": f"sbx{self.uid}", "LOGNAME": f"sbx{self.uid}"})
        return run_env

    def exec(self, session_id: str, command: str, env: dict | None, timeout: int) -> dict:
        workdir = self._session_dir(session_id)
        # Inject the fractional-GPU memory cap + keep the session dir first on PYTHONPATH so the
        # subprocess's sitecustomize (the cap) is the one that loads. The rest of PYTHONPATH (e.g.
        # SANDBOX_PYLIB on the g6 image) is preserved AFTER it, so the checker's torch still
        # resolves. CUDA_VISIBLE_DEVICES is inherited from the actor (set by Ray), pinning the
        # subprocess to the same shared device.
        run_env = self._run_env(workdir, env)
        run_env["ENV_GPU_MEM_FRACTION"] = f"{self.mem_fraction:.4f}"
        # Serialize nvcc compiles across co-resident sandboxes (default on; B4_SERIALIZE_COMPILE=0
        # to A/B against the old always-concurrent behaviour that wedged the non-MIG L4).
        if os.environ.get("B4_SERIALIZE_COMPILE", "1") != "0":
            run_env["ENV_NVCC_LOCK"] = _NVCC_LOCK_PATH
        run_env["PYTHONPATH"] = workdir + (":" + run_env["PYTHONPATH"] if run_env.get("PYTHONPATH") else "")
        exports = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in (env or {}).items())
        full = f"cd {shlex.quote(workdir)}; {exports}bash -lc {shlex.quote(command)}"
        # Drop to the slot's uid in the child. subprocess's user=/group=/extra_groups= do the same
        # setgroups([]) + setgid + setuid as a preexec_fn would, but in C after fork: a Python
        # preexec_fn is unsafe here because the Ray actor process is multi-threaded.
        drop = ({"user": self.uid, "group": self.uid, "extra_groups": [], "umask": 0o077}
                if self.uid is not None else {})
        # Own process group per exec, so a timeout or close() can kill the whole tree.
        proc = subprocess.Popen(full, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=run_env, start_new_session=True, **drop)
        self._pgids.setdefault(session_id, set()).add(proc.pid)
        try:
            out, err = proc.communicate(timeout=int(timeout))
            return {"exit_code": proc.returncode, "stdout": out, "stderr": err}
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            return {"exit_code": 124, "stdout": out or "", "stderr": "timeout"}

    def write_file(self, session_id: str, path: str, text: str) -> dict:
        dst = self._abs(session_id, path)
        parent = os.path.dirname(dst)
        if not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
            if self.uid is not None:
                _chown_tree(self._session_dir(session_id), self.uid)
        with open(dst, "w") as f:
            f.write(text or "")
        if self.uid is not None:
            os.chown(dst, self.uid, self.uid)
        return {"ok": True}

    def read_file(self, session_id: str, path: str) -> dict:
        try:
            with open(self._abs(session_id, path)) as f:
                return {"text": f.read()}
        except FileNotFoundError:
            return {"text": "", "error": "not found"}

    def _kill_session_procs(self, session_id: str) -> int:
        """SIGKILL everything the episode started: each exec's process group, then (when the slot has
        its own uid) every remaining process of that uid, e.g. a `setsid nohup` that left the group."""
        killed = 0
        for pgid in self._pgids.pop(session_id, set()):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
                killed += 1
        if self.uid is not None:
            seen: set[int] = set()
            for _ in range(50):                   # re-sweep: a dying parent may have just forked
                pids = _uid_pids(self.uid)
                if not pids:
                    break
                for pid in pids:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
                        if pid not in seen:
                            seen.add(pid)
                            killed += 1
                time.sleep(0.1)
        return killed

    def clear_session(self, session_id: str) -> dict:
        """Kill the episode's processes and wipe its working dir (called when the sub-session closes)
        so the reused actor starts the next episode clean. The warm CUDA context + the actor process
        persist."""
        killed = self._kill_session_procs(session_id)
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(os.path.join(self.root, session_id))
        left = len(_uid_pids(self.uid)) if self.uid is not None else 0
        return {"ok": True, "killed": killed, "left_procs": left}


@dataclass
class _FarmStats:
    checkouts: int = 0
    total_wait_s: float = 0.0
    max_wait_s: float = 0.0


class RayGpuFarm:
    """A fixed pool of ``max_sandboxes`` Ray actors, each holding ``gpu_fraction`` of the g6 GPU.

    Runs INSIDE the g6 AgentCore Runtime #2 (see ``sandbox_server_agent``). SYNCHRONOUS + thread
    safe on purpose: the AgentCore entrypoint is a plain sync function that AgentCore may dispatch
    on several threads for the same session (that concurrency is exactly what lets several episodes
    share the one GPU), so the gate is a ``queue.Queue`` of free actor handles guarded by a lock —
    not asyncio.

    The pool size IS the concurrency gate: ``open(sub_session)`` takes one free actor and blocks
    when all are in flight, so at most ``max_sandboxes`` episodes touch the GPU at once. Ray only
    schedules an actor if ``gpu_fraction`` of some GPU is free, so ``max_sandboxes`` must not exceed
    ``floor(farm_gpus / gpu_fraction)`` or the surplus actors sit PENDING forever. ``from_env``
    computes a safe default (and on a single-L4 g6, ``farm_gpus`` defaults to the detected GPU
    count). Actors are *named* in a Ray namespace with ``get_if_exists`` so a rebuild converges on
    the same set rather than doubling GPU pressure.
    """

    def __init__(
        self,
        *,
        gpu_fraction: float = 0.5,
        max_sandboxes: int | None = None,
        farm_gpus: float | None = None,
        per_sandbox_mem_fraction: float | None = None,
        workspace_root: str = "/dev/shm/b4-sbx",
        warm: bool = True,
        namespace: str = "b4_farm",
        num_cpus_per_actor: float = 1.0,
    ) -> None:
        if not (0.0 < gpu_fraction <= 1.0):
            raise ValueError("gpu_fraction must be in (0, 1]")
        self.gpu_fraction = gpu_fraction
        self.farm_gpus = farm_gpus            # None => detect from ray at build time
        self._max_sandboxes_req = max_sandboxes
        self.max_sandboxes = int(max_sandboxes) if max_sandboxes else 0  # finalized in _build_actors
        # By default each sandbox may use its Ray fraction of the device's memory (matches the
        # scheduling fraction). Override to over/under-commit memory independently of scheduling.
        self.per_sandbox_mem_fraction = (
            per_sandbox_mem_fraction if per_sandbox_mem_fraction is not None else gpu_fraction
        )
        self.workspace_root = workspace_root
        self.warm = warm
        self.namespace = namespace
        self.num_cpus_per_actor = num_cpus_per_actor
        self._actors: list = []
        self._free: queue.Queue = queue.Queue()
        self._leases: dict[str, object] = {}   # sub_session -> actor handle
        self._lock = threading.Lock()
        self._started = False
        self.stats = _FarmStats()

    @classmethod
    def from_env(cls) -> "RayGpuFarm":
        """Build from B4_* env vars baked into the g6 sandbox image so it can be tuned at deploy
        time without a code change: B4_GPU_FRACTION (0.5), B4_FARM_GPUS (default = detected GPUs),
        B4_MAX_SANDBOXES (default floor(farm_gpus/frac)), B4_PER_SANDBOX_MEM_FRACTION (default =
        fraction), B4_WORKSPACE_ROOT (/dev/shm/b4-sbx), B4_WARM_CUDA (1)."""
        frac = float(os.environ.get("B4_GPU_FRACTION", "0.5"))
        farm_gpus = os.environ.get("B4_FARM_GPUS", "").strip()
        max_sb = os.environ.get("B4_MAX_SANDBOXES", "").strip()
        mem = os.environ.get("B4_PER_SANDBOX_MEM_FRACTION", "").strip()
        return cls(
            gpu_fraction=frac,
            farm_gpus=float(farm_gpus) if farm_gpus else None,
            max_sandboxes=int(max_sb) if max_sb else None,
            per_sandbox_mem_fraction=float(mem) if mem else None,
            workspace_root=os.environ.get("B4_WORKSPACE_ROOT", "/dev/shm/b4-sbx"),
            warm=os.environ.get("B4_WARM_CUDA", "1").lower() not in ("0", "", "false", "no"),
        )

    def start(self) -> "RayGpuFarm":
        """Create the actors and fill the free queue (idempotent, thread-safe)."""
        if self._started:
            return self
        with self._lock:
            if self._started:
                return self
            self._build_actors()
            for a in self._actors:
                self._free.put(a)
            self._started = True
        return self

    def _build_actors(self) -> None:
        import ray

        if not ray.is_initialized():
            # Start / attach to the g6's LOCAL single-node Ray cluster (this process is the only
            # user of it — the trainer box's cluster is a different machine reached via invoke).
            # Pass num_gpus EXPLICITLY when farm_gpus is known (B4_FARM_GPUS): the g6 sandbox image
            # is python:slim with NO nvidia-smi, so Ray's GPU autodetection returns 0 and the
            # num_gpus<1 actors would sit PENDING forever (open() would block). Forcing the count
            # makes Ray schedule them onto the injected L4. On a box WITH nvidia-smi (the offline
            # smoke, or when farm_gpus is None) autodetection is left to work as before.
            init_kwargs = {"namespace": self.namespace, "ignore_reinit_error": True}
            if self.farm_gpus is not None:
                init_kwargs["num_gpus"] = self.farm_gpus
            ray.init(**init_kwargs)
        # Default farm_gpus to however many GPUs this g6 actually has (usually 1 L4).
        if self.farm_gpus is None:
            try:
                self.farm_gpus = float(ray.cluster_resources().get("GPU", 1.0)) or 1.0
            except Exception:  # noqa: BLE001
                self.farm_gpus = 1.0
        fit = max(1, math.floor(self.farm_gpus / self.gpu_fraction))
        self.max_sandboxes = int(self._max_sandboxes_req) if self._max_sandboxes_req else fit
        RemoteActor = ray.remote(num_gpus=self.gpu_fraction, num_cpus=self.num_cpus_per_actor)(
            _SandboxActorImpl
        )
        for i in range(self.max_sandboxes):
            actor = RemoteActor.options(
                name=f"b4-sbx-{i}", namespace=self.namespace, get_if_exists=True, max_concurrency=1,
            ).remote(
                f"b4-sbx-{i}",
                mem_fraction=self.per_sandbox_mem_fraction,
                workspace_root=os.path.join(self.workspace_root, f"actor-{i}"),
                warm=self.warm,
                slot=i,
            )
            self._actors.append(actor)

    # ---- lease lifecycle (called by the server per op) ----------------------------------------
    def open(self, sub_session: str, *, timeout: float | None = None) -> dict:
        """Lease a free actor for ``sub_session`` (blocks on the gate when all are in flight)."""
        import time

        self.start()
        with self._lock:
            if sub_session in self._leases:
                return {"ok": True, "reused": True}
        t0 = time.perf_counter()
        actor = self._free.get(timeout=timeout)   # blocks -> the concurrency gate
        waited = time.perf_counter() - t0
        with self._lock:
            self._leases[sub_session] = actor
            self.stats.checkouts += 1
            self.stats.total_wait_s += waited
            self.stats.max_wait_s = max(self.stats.max_wait_s, waited)
        info = _ray_get(actor.ping.remote())      # verify the actor is live before the episode
        info.update({"ok": True, "waited_s": round(waited, 4)})
        return info

    def close(self, sub_session: str) -> dict:
        with self._lock:
            actor = self._leases.pop(sub_session, None)
        res = {"ok": True}
        if actor is not None:
            with contextlib.suppress(Exception):
                res.update(_ray_get(actor.clear_session.remote(sub_session)))
            self._free.put(actor)
        return res

    def _actor_for(self, sub_session: str):
        with self._lock:
            actor = self._leases.get(sub_session)
        if actor is None:
            # Forgiving: an op arrived before/without an explicit open — lease one now.
            self.open(sub_session)
            with self._lock:
                actor = self._leases.get(sub_session)
        return actor

    def exec(self, sub_session: str, command: str, env: dict | None, timeout: int) -> dict:
        return _ray_get(self._actor_for(sub_session).exec.remote(sub_session, command, env or {}, int(timeout)))

    def write_file(self, sub_session: str, path: str, text: str) -> dict:
        return _ray_get(self._actor_for(sub_session).write_file.remote(sub_session, path, text))

    def read_file(self, sub_session: str, path: str) -> dict:
        return _ray_get(self._actor_for(sub_session).read_file.remote(sub_session, path))

    def describe(self) -> str:
        return (
            f"RayGpuFarm: {self.max_sandboxes} sandbox actor(s) x {self.gpu_fraction:.3g} GPU each "
            f"on {self.farm_gpus} g6 GPU(s) | per-sandbox mem cap={self.per_sandbox_mem_fraction:.3g} "
            f"| root={self.workspace_root}"
        )
