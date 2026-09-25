#!/usr/bin/env python3
"""Per-exec jail for one CPU-farm sub_session: Landlock + subreaper + per-episode watchdog (stdlib only).

Why this exists: on an AgentCore capacity-provider Instance the sandbox container is "root" in a user
namespace with EVERY capability dropped (CapEff = CapBnd = 0), NoNewPrivs=1 and a seccomp filter —
measured live 2026-09-25. So setuid/chown/unshare/cgroups are all EPERM and every sub_session runs as
the SAME uid. What the kernel still allows an unprivileged process is **Landlock** (ABI 7 on the
Instance's 6.18 kernel), and that is what separates the episodes:

  * **filesystem** — the jailed command may read+execute only the system dirs (``--ro``), read+write
    only ITS session dir and a few /dev files. Everything else is EACCES, including sibling episode
    dirs, ``/tmp``, ``/var/tmp``, ``/dev/shm``, ``/root`` and ``/workspace``.
  * **network** (ABI >= 4, ``--net deny``) — every TCP ``connect``/``bind`` is refused, so an episode
    can't reach the server's own port 8080, the instance-metadata endpoint or the internet (UDP is
    not covered by Landlock).
  * **signals / abstract unix sockets** (ABI >= 6) — scoped to the episode's own Landlock domain, so
    it can't signal or connect to another episode (or this wrapper). Landlock also blocks ``ptrace``
    and ``/proc/<pid>/{environ,cwd,mem,…}`` across domains.

Around that, this wrapper (which is OUTSIDE the domain) is a ``PR_SET_CHILD_SUBREAPER``: every
descendant — even a setsid'd, double-forked daemon — stays in its process tree. It polls the tree and
SIGKILLs all of it when the command exits (nothing outlives its exec), when the tree exceeds
``--nproc`` processes or ``--mem-mb`` RSS, or on SIGTERM (the server's timeout / close path).

    SANDBOX_JAIL_DIR=D python3 sandbox_farm_jail.py [--ro /usr:/etc:…] [--nproc N] [--mem-mb M]
                                 [--fsize B] [--net deny|allow] [--no-landlock] -- bash -c '…'

The session dir comes in via the environment, NOT argv: ``/proc/<pid>/cmdline`` is world-readable,
and the dir name is the sub_session id, which is the key to that episode's farm ops.

Exit code: the command's own, or 137 when the watchdog killed the tree (reason on stderr), or 126
when the Landlock policy could not be applied (``--require-landlock``).
"""
from __future__ import annotations

import argparse
import ctypes
import os
import resource
import signal
import struct
import sys
import time

# ---- Landlock (include/uapi/linux/landlock.h); syscall numbers are the same on x86_64 and arm64 ----
_SYS_CREATE, _SYS_ADD_RULE, _SYS_RESTRICT = 444, 445, 446
_RULE_PATH_BENEATH, _RULE_NET_PORT = 1, 2
_FS_EXECUTE, _FS_WRITE_FILE, _FS_READ_FILE, _FS_READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_FS_BASE = (1 << 13) - 1                      # ABI 1: EXECUTE … MAKE_SYM
_FS_REFER, _FS_TRUNCATE, _FS_IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
_NET_BIND_TCP, _NET_CONNECT_TCP = 1 << 0, 1 << 1
_SCOPE_ABSTRACT_UNIX, _SCOPE_SIGNAL = 1 << 0, 1 << 1
_PR_SET_NO_NEW_PRIVS, _PR_SET_CHILD_SUBREAPER = 38, 36

_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


def landlock_abi() -> int:
    """The kernel's Landlock ABI version (0 = unavailable)."""
    v = _libc.syscall(_SYS_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    return max(0, int(v))


def _fs_handled(abi: int) -> int:
    h = _FS_BASE
    if abi >= 2:
        h |= _FS_REFER
    if abi >= 3:
        h |= _FS_TRUNCATE
    if abi >= 5:
        h |= _FS_IOCTL_DEV
    return h


def apply_landlock(rw: list[str], ro: list[str], dev_files: list[str], net_deny: bool) -> int:
    """Restrict THIS process (and all its future children) to the policy. Returns the ABI used."""
    abi = landlock_abi()
    if abi < 1:
        raise OSError("landlock unavailable")
    fs = _fs_handled(abi)
    net = (_NET_BIND_TCP | _NET_CONNECT_TCP) if (net_deny and abi >= 4) else 0
    scoped = (_SCOPE_ABSTRACT_UNIX | _SCOPE_SIGNAL) if abi >= 6 else 0
    attr = struct.pack("=QQQ", fs, net, scoped)[: 8 if abi < 4 else 16 if abi < 6 else 24]
    buf = ctypes.create_string_buffer(attr, len(attr))
    rs = _libc.syscall(_SYS_CREATE, buf, ctypes.c_size_t(len(attr)), ctypes.c_uint32(0))
    if rs < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    file_bits = (_FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE | _FS_TRUNCATE | _FS_IOCTL_DEV) & fs

    def allow(path: str, access: int) -> None:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return                               # a missing optional path is simply not granted
        try:
            if not os.path.isdir(path):
                access &= file_bits
            rule = ctypes.create_string_buffer(struct.pack("=Qi", access, fd), 12)
            if _libc.syscall(_SYS_ADD_RULE, ctypes.c_int(rs), ctypes.c_int(_RULE_PATH_BENEATH), rule,
                             ctypes.c_uint32(0)) < 0:
                raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
        finally:
            os.close(fd)

    for p in ro:
        allow(p, (_FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR) & fs)
    for p in dev_files:
        allow(p, (_FS_READ_FILE | _FS_WRITE_FILE | _FS_TRUNCATE | _FS_IOCTL_DEV) & fs)
    for p in rw:
        allow(p, fs)
    _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    if _libc.syscall(_SYS_RESTRICT, ctypes.c_int(rs), ctypes.c_uint32(0)) < 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    os.close(rs)
    return abi


# ---- process-tree watchdog -------------------------------------------------------------------------
def _procs() -> dict[int, tuple[int, int, str]]:
    """pid -> (ppid, rss_bytes, state) for every visible process."""
    out = {}
    page = os.sysconf("SC_PAGE_SIZE")
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                rest = f.read().rsplit(")", 1)[1].split()
            out[int(name)] = (int(rest[1]), int(rest[21]) * page, rest[0])
        except (OSError, IndexError, ValueError):
            continue
    return out


def _tree(root: int) -> tuple[list[int], int]:
    """Descendants of ``root`` (not root itself) and their summed RSS."""
    procs = _procs()
    kids: dict[int, list[int]] = {}
    for pid, (ppid, _, _) in procs.items():
        kids.setdefault(ppid, []).append(pid)
    seen, stack, rss = [], [root], 0
    while stack:
        for c in kids.get(stack.pop(), []):
            seen.append(c)
            stack.append(c)
            rss += procs[c][1]
    return seen, rss


def _kill_tree(me: int) -> None:
    for _ in range(50):
        pids, _ = _tree(me)
        if not pids:
            return
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _reap()
        time.sleep(0.02)


def _reap() -> list[tuple[int, int]]:
    done = []
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return done
        if pid == 0:
            return done
        done.append((pid, status))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.environ.pop("SANDBOX_JAIL_DIR", ""),
                    help="the sub_session dir (read+write); default $SANDBOX_JAIL_DIR")
    ap.add_argument("--ro", default="/usr:/bin:/sbin:/lib:/lib64:/etc:/opt:/app:/proc:/sys")
    ap.add_argument("--dev", default="/dev/null:/dev/zero:/dev/full:/dev/random:/dev/urandom:/dev/tty")
    ap.add_argument("--nproc", type=int, default=256)
    ap.add_argument("--mem-mb", type=int, default=0, help="kill the tree above this RSS (0 = off)")
    ap.add_argument("--fsize", type=int, default=1 << 30)
    ap.add_argument("--net", choices=("deny", "allow"), default="deny")
    ap.add_argument("--poll", type=float, default=0.1)
    ap.add_argument("--require-landlock", action="store_true")
    ap.add_argument("--no-landlock", action="store_true", help="watchdog + lifetime only")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd or not a.dir:
        ap.error("need a session dir ($SANDBOX_JAIL_DIR or --dir) and a command")
    me = os.getpid()
    _libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    stop = {"why": ""}

    def on_term(signum, _frame):  # noqa: ANN001
        stop["why"] = stop["why"] or f"signal {signum}"

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    rfd, wfd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(rfd)
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (a.fsize, a.fsize))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            os.umask(0o077)
            abi = 0
            try:
                if a.no_landlock:
                    raise OSError("disabled (--no-landlock)")
                abi = apply_landlock([a.dir], [p for p in a.ro.split(":") if p],
                                     [p for p in a.dev.split(":") if p], a.net == "deny")
            except OSError as e:
                if a.require_landlock and not a.no_landlock:
                    os.write(wfd, f"landlock: {e}".encode())
                    os._exit(126)
            os.environ["SANDBOX_JAIL"] = f"landlock-abi{abi}" if abi else "none"
            os.close(wfd)
            os.execvp(cmd[0], cmd)
        except BaseException as e:  # noqa: BLE001
            try:
                os.write(wfd, f"jail: {e}".encode())
            except OSError:
                pass
            os._exit(126)
    os.close(wfd)
    code = None
    while code is None:
        for pid, status in _reap():
            if pid == child:
                code = os.waitstatus_to_exitcode(status)
        if code is not None:
            break
        if not stop["why"]:
            pids, rss = _tree(me)
            if len(pids) > a.nproc:
                stop["why"] = f"process cap: {len(pids)} > {a.nproc}"
            elif a.mem_mb and rss > a.mem_mb * 1024 * 1024:
                stop["why"] = f"memory cap: {rss >> 20} MB > {a.mem_mb} MB"
        if stop["why"]:
            _kill_tree(me)
            code = 137
            break
        time.sleep(a.poll)
    _kill_tree(me)            # nothing the command started outlives it
    err = os.read(rfd, 4096).decode(errors="replace")
    if err:
        sys.stderr.write(err + "\n")
    if stop["why"]:
        sys.stderr.write(f"[jail] killed the episode's process tree: {stop['why']}\n")
    code = 128 - code if code < 0 else code
    return code


if __name__ == "__main__":
    sys.exit(main())
