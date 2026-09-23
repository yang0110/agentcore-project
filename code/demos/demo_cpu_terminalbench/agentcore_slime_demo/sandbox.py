"""Sandbox backends for the demo.

Design principle (Agent Lightning v1.0, "Towards Harnessed Agentic RL", arXiv 2608.17528):
agent *execution* is fully decoupled from training. The sandbox is where the agent's tool
actions actually run. We expose the same minimal contract as slime's `Sandbox` Protocol
(repos/slime/slime/agent/sandbox.py:27-59) so this code can drop into slime's agent stack:

    ExecResult = (exit_code: int, stdout: str, stderr: str)
    async exec(cmd, *, timeout, check) -> ExecResult
    async write_file(path, content) -> None
    async read_file(path) -> str

Two implementations:
  * LocalSandbox      — subprocess in a temp dir. No AWS, no GPU. Used for offline smoke/tests.
  * AgentCoreSandbox  — AWS Bedrock AgentCore Code Interpreter (managed microVM). Real backend.

They are interchangeable; the harness/rollout code never knows which one it holds.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import tempfile
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

ExecResult = tuple[int, str, str]  # (exit_code, stdout, stderr)
FileContent = str | bytes | Path


# --- Multi-Runtime fan-out (rollout-throughput ablation) --------------------------------------
# To boost rollout throughput we can spread concurrent episodes across SEVERAL sandbox Runtimes
# instead of hammering one. Set AGENTCORE_SANDBOX_RUNTIME_ARNS to a comma-separated list of
# Runtime ARNs; make_sandbox() assigns each new episode's sandbox to the next ARN round-robin.
# The number of ARNs in that list IS the ablation knob (sweep 1,2,4,...). Falls back to the
# single AGENTCORE_SANDBOX_RUNTIME_ARN when the list is unset (backward compatible).
import itertools as _itertools
import threading as _threading

_ARN_LOCK = _threading.Lock()
_ARN_CYCLE = None  # itertools.cycle over the current ARN list
_ARN_LIST: list[str] | None = None  # remembered list to detect env changes


def _runtime_arns() -> list[str]:
    listed = os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARNS", "").strip()
    if listed:
        return [a.strip() for a in listed.split(",") if a.strip()]
    single = os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN", "").strip()
    return [single] if single else []


def _next_runtime_arn() -> str | None:
    """Round-robin the configured sandbox Runtime ARNs (thread-safe). None if none configured."""
    global _ARN_CYCLE, _ARN_LIST
    arns = _runtime_arns()
    if not arns:
        return None
    with _ARN_LOCK:
        if _ARN_LIST != arns:
            _ARN_LIST = arns
            _ARN_CYCLE = _itertools.cycle(arns)
        return next(_ARN_CYCLE)


def _runtime_arn_for_image(image: str | None) -> str | None:
    """Capability-routing by container image (the mixtask custom-image case).

    A heterogeneous batch may contain a family whose sandbox needs a DIFFERENT image than the
    generic CPU one — e.g. the `datasci` family needs pandas, which is absent from the plain CPU
    sandbox image. Each task carries an `image` tag (task.image); this maps that tag to a Runtime
    ARN via AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE, a JSON object like:

        {"cpu": "arn:aws:bedrock-agentcore:...:runtime/acs_sandbox_cpu-...",
         "datasci": "arn:aws:bedrock-agentcore:...:runtime/acs_sandbox_datasci-..."}

    If the tag has a mapped ARN, route there; otherwise fall back to the round-robin default pool
    (_next_runtime_arn). Backward compatible: with the env unset every image falls through to the
    round-robin, i.e. today's single-image behavior — so demos with one image need change nothing."""
    if image:
        raw = os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE", "").strip()
        if raw:
            try:
                mapping = json.loads(raw)
            except Exception:  # noqa: BLE001 — malformed map => fall back to the round-robin pool
                mapping = {}
            arn = mapping.get(image)
            if arn and str(arn).strip():
                return str(arn).strip()
    return _next_runtime_arn()


@runtime_checkable
class Sandbox(Protocol):
    """Provider-agnostic sandbox contract — signature-compatible with slime's
    `slime.agent.sandbox.Sandbox` (repos/slime/slime/agent/sandbox.py:27-59), so any
    implementation here is a drop-in for slime's `E2BSandbox`.

    ``write_file`` accepts in-memory content (str/bytes) or a host Path to stream in.
    ``user`` selects the in-sandbox user (root/agent); ``idempotent`` is a retry hint.
    """

    sandbox_id: str

    async def __aenter__(self) -> "Sandbox": ...
    async def __aexit__(self, exc_type, exc, tb) -> None: ...
    async def exec(
        self, cmd: str, *, user: str = "root", env: dict[str, str] | None = None,
        timeout: int = 120, check: bool = False, idempotent: bool = True,
    ) -> ExecResult: ...
    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...
    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...


class LocalSandbox:
    """Runs commands in a throwaway local directory. For offline dev/CI only.

    NOT isolated — do not use for untrusted code in production. It exists so the whole
    multi-turn loop can be exercised without AWS credentials or a GPU.
    """

    def __init__(self, workdir: str = "/workspace") -> None:
        self.sandbox_id = f"local-{uuid.uuid4().hex[:8]}"
        self._workdir = workdir
        self._root: Path | None = None

    async def __aenter__(self) -> "LocalSandbox":
        self._tmp = tempfile.TemporaryDirectory(prefix="acdemo-")
        self._root = Path(self._tmp.name)
        (self._root / self._workdir.lstrip("/")).mkdir(parents=True, exist_ok=True)
        return self

    async def __aexit__(self, *exc) -> None:
        if getattr(self, "_tmp", None) is not None:
            self._tmp.cleanup()

    def _abs(self, path: str) -> Path:
        p = path if os.path.isabs(path) else os.path.join(self._workdir, path)
        return self._root / p.lstrip("/")

    async def exec(
        self, cmd: str, *, user: str = "root", env: dict[str, str] | None = None,
        timeout: int = 120, check: bool = False, idempotent: bool = True,
    ) -> ExecResult:
        # `user` is ignored locally (no privilege switching without root); `env` is honored.
        proc_env = {**os.environ, **(env or {})}
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(self._abs("")),
            env=proc_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            code = proc.returncode or 0
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            code, out, err = 124, b"", b"timeout"
        stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
        if check and code != 0:
            raise RuntimeError(f"exec failed ({code}): {stderr}")
        return (code, stdout, stderr)

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        dst = self._abs(sandbox_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, Path):
            dst.write_bytes(content.read_bytes())
        elif isinstance(content, bytes):
            dst.write_bytes(content)
        else:
            dst.write_text(content)

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        return self._abs(sandbox_path).read_text()


class AgentCoreSandbox:
    """AWS Bedrock AgentCore Code Interpreter as a slime-compatible Sandbox.

    Uses the managed microVM code-interpreter (one instance == one isolated session ==
    one rollout). The boto3-backed SDK client is synchronous, so calls are offloaded with
    asyncio.to_thread. Field shapes (structuredContent.stdout/.stderr/.exitCode; writeFiles
    'content' list; readFiles 'paths') mirror the AWS samples repo
    (agentcore-samples/.../01-code-interpreter/); pin your SDK version and smoke-test.
    """

    def __init__(self, region: str | None = None, workdir: str = "",
                 session_timeout_seconds: int = 900) -> None:
        # workdir="" (default) means "use the session's native writable CWD". The managed
        # Code Interpreter runs as an unprivileged user (genesis1ptools) whose CWD
        # (/opt/amazon/genesis1p-tools/var) IS writable and persists across invokes, whereas
        # /workspace is NOT writable there — verified live (P0). writeFiles with a RELATIVE
        # path lands in that CWD and exec runs from the same CWD, so relative paths just work.
        # Pass an explicit writable absolute workdir only for a custom Runtime container
        # (Option B), where you control the filesystem/user.
        self.sandbox_id = f"ac-{uuid.uuid4().hex}"
        self._region = region or os.environ.get("AGENTCORE_REGION", "us-west-2")
        self._workdir = workdir
        self._timeout = session_timeout_seconds
        self._client = None

    async def __aenter__(self) -> "AgentCoreSandbox":
        # Imported lazily so the offline path never needs the AWS SDK installed.
        from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

        self._client = CodeInterpreter(self._region)
        await asyncio.to_thread(
            self._client.start, name=self.sandbox_id, session_timeout_seconds=self._timeout
        )
        if self._workdir:                       # only create/enter an explicit workdir
            await self.exec(f"mkdir -p {shlex.quote(self._workdir)}")
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.stop)
            self._client = None

    # Transport blips where the session is still alive: safe to retry the SAME op in place.
    _TRANSPORT_TRANSIENT = ("throttl", "unavailable", "internalserver", "timed out",
                            "connection", "read timeout", "reset by peer")
    # The session is genuinely gone: in-sandbox state is lost, so retrying here is wrong.
    # Let it propagate — recovery.resilient_episode restarts the whole episode on a fresh
    # session (and re-writes the workspace). See recovery.is_recoverable.
    _SESSION_DEAD = ("terminated", "not found", "notfound", "expired", "session is not")

    _RPC_RETRIES = 3
    _RPC_BACKOFF_SEC = 0.5

    def _call(self, tool: str, args: dict) -> dict:
        import time as _time

        last: Exception | None = None
        for attempt in range(self._RPC_RETRIES):
            try:
                resp = self._client.invoke(tool, args)  # code_interpreter_client.py:442-470
                for event in resp["stream"]:
                    return event["result"]
                return {"isError": True, "structuredContent": {}}
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if any(m in msg for m in self._SESSION_DEAD):
                    raise  # -> episode-level recovery
                if any(m in msg for m in self._TRANSPORT_TRANSIENT) and attempt + 1 < self._RPC_RETRIES:
                    last = e
                    _time.sleep(self._RPC_BACKOFF_SEC * (2 ** attempt))
                    continue
                raise
        assert last is not None
        raise last

    async def healthy(self) -> bool:
        """True if the session is READY (health-check via get_session)."""
        try:
            info = await asyncio.to_thread(self._client.get_session)
            return info.get("status") == "READY"
        except Exception:
            return False

    def _abs(self, path: str) -> str:
        if os.path.isabs(path) or not self._workdir:
            return path            # native-CWD mode: keep relative paths relative to CWD
        return f"{self._workdir}/{path}"

    async def exec(
        self, cmd: str, *, user: str = "root", env: dict[str, str] | None = None,
        timeout: int = 120, check: bool = False, idempotent: bool = True,
    ) -> ExecResult:
        # `user`: Code Interpreter has no privilege model — run as its fixed user, best-effort
        # `su` when a non-root user is requested (needed by slime's ClaudeCodeHarness, which
        # runs the agent as 'agent'). For the real Claude Code path prefer a custom Runtime
        # container (demo-plan.md Option B), which supports full multi-user Linux.
        exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in (env or {}).items())
        cd = f"cd {shlex.quote(self._workdir)}; " if self._workdir else ""   # else stay in native CWD
        inner = f"{cd}{exports}timeout {int(timeout)} bash -lc {shlex.quote(cmd)}"
        full = inner if user in ("", "root") else f"su {shlex.quote(user)} -c {shlex.quote(inner)}"
        res = await asyncio.to_thread(self._call, "executeCommand", {"command": full})
        sc = res.get("structuredContent", {})
        code = int(sc.get("exitCode", 1))
        stdout, stderr = sc.get("stdout", ""), sc.get("stderr", "")
        if check and code != 0:
            raise RuntimeError(f"exec failed ({code}): {stderr}")
        return (code, stdout, stderr)

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            text = content.read_text()
        elif isinstance(content, bytes):
            text = content.decode(errors="replace")
        else:
            text = content
        await asyncio.to_thread(
            self._call, "writeFiles", {"content": [{"path": self._abs(sandbox_path), "text": text}]}
        )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        res = await asyncio.to_thread(self._call, "readFiles", {"paths": [self._abs(sandbox_path)]})
        # readFiles returns {"content":[{"type":"resource","resource":{"uri","text",...}}]}
        # (verified live). Pull the text of the first resource; fall back to raw for safety.
        content = res.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    resource = item.get("resource")
                    if isinstance(resource, dict) and "text" in resource:
                        return resource["text"]
                    if "text" in item:
                        return item["text"]
        return content if isinstance(content, str) else json.dumps(content)


def make_sandbox(kind: str | None = None, **kwargs) -> Sandbox:
    """Factory. kind ∈ {"local","agentcore","agentcore-runtime"}; default from SANDBOX env.

    * "local"             — LocalSandbox (offline subprocess).
    * "agentcore"/"aws"    — AgentCoreSandbox (managed Code Interpreter; co-located target).
    * "agentcore-runtime"  — AgentCoreRuntimeSandbox: the sandbox as its OWN AgentCore Runtime,
                             driven over invoke_agent_runtime (the disaggregated two-Runtime
                             target, docs/harness-sandbox-decoupling.md §7). UNTESTED scaffolding.
    * "agentcore-ray-farm" — RayFarmRuntimeSandbox (Demo B4): same harness as "agentcore-runtime",
                             but several episodes SHARE one g6 instance whose GPU a Ray farm splits
                             (shared runtimeSessionId + per-episode sub_session).
    """
    # Default is AWS: every demo runs against AgentCore. "local" (LocalSandbox) is retained only
    # for offline unit tests / CI and must be requested explicitly (SANDBOX=local); it is never
    # the default path any longer.
    kind = (kind or os.environ.get("SANDBOX", "agentcore")).lower()
    # `image` is a capability tag (task.image) consumed ONLY by the runtime backend for
    # image-aware ARN routing; pop it here so the other backends never see the extra kwarg.
    image = kwargs.pop("image", None)
    if kind == "local":
        return LocalSandbox(**{k: v for k, v in kwargs.items() if k == "workdir"})
    if kind in ("agentcore", "aws"):
        return AgentCoreSandbox(**kwargs)
    if kind in ("agentcore-runtime", "runtime"):
        # Imported here so the offline/Code-Interpreter paths never load the new backend.
        from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox

        # Pick the Runtime ARN. If a caller passed an explicit runtime_arn we honor it; otherwise
        # route by the task's `image` tag (capability-routing — the custom-image family lands on
        # its own pandas Runtime) with a round-robin fallback across the default pool.
        if "runtime_arn" not in kwargs:
            arn = _runtime_arn_for_image(image)
            if arn is not None:
                kwargs["runtime_arn"] = arn
        return AgentCoreRuntimeSandbox(**kwargs)
    if kind in ("agentcore-ray-farm", "ray-farm", "b4"):
        # Demo B4: SAME in-process run_episode harness as "agentcore-runtime" (generate.generate
        # drives it identically) — the ONLY difference is the sandbox. Here the sandbox is a
        # sub_session slot on ONE shared g6 instance whose single GPU a Ray farm splits across
        # several concurrent episodes (server-side sandbox_ray_farm.RayGpuFarm). shared_session_id
        # comes from B4_SHARED_RUNTIME_SESSION (set once per job in run.sh so every episode lands on
        # the SAME g6), and a per-episode sub_session is minted per sandbox. Deliberately NOT part of
        # the ARN round-robin: B4 targets one instance and packs many episodes onto its one GPU.
        from .agentcore_runtime_sandbox import RayFarmRuntimeSandbox

        return RayFarmRuntimeSandbox(**kwargs)
    raise ValueError(f"unknown sandbox kind: {kind!r}")
