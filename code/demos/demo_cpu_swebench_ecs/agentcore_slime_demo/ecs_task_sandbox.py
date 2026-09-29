"""`EcsTaskSandbox`: one episode = one fresh ECS task on EC2 (awsvpc: its own ENI + network namespace).

The ECS counterpart of `AgentCoreRuntimeSandbox`, and a drop-in for it: same `exec` / `write_file` /
`read_file` / `_invoke` API, same op payloads, served by the SAME in-image server
(deploy/acs_swe_http_server.py on :8080, `POST /invocations`). The images are the amd64 wrapper images
deploy/deploy_swebench.py pushes (shared with demo_cpu_swebench_eks); each image has its own task
definition (deploy/ecs_deploy.py). Only the session lifecycle and the transport change:

    AgentCore                                     ECS on EC2 (this file)
    runtimeSessionId -> own arm64 microVM         RunTask -> own container (runc, all caps dropped, no task
                                                  role, IMDS blocked) with its own ENI, on a shared m5 node
    InvokeAgentRuntime (AWS data plane)           HTTP straight to the task's ENI IP (awsvpc, same VPC)
    StopRuntimeSession                            StopTask
    idle reclaim ~15 min -> heartbeat needed      no idle reclaim (heartbeat = liveness ping); entrypoint `timeout` caps a leak

Isolation: a plain container — the node kernel is SHARED by every task on the node (weaker than
AgentCore's microVM and than gVisor on EKS). What ECS adds: its own ENI + security group per task (the
task SG admits :8080 from the trainer and lets the task dial back only the adapter ports — the "egress
none" of the AgentCore demo), CAP_* all dropped, no task IAM role and ECS_AWSVPC_BLOCK_IMDS on the node,
so the agent gets no AWS credentials; one task per episode, never reused.

Open = RunTask -> PROVISIONING (managed scaling adds a node if none has room) -> PENDING (image pull,
skipped when the node has it cached: ECS_IMAGE_PULL_BEHAVIOR=prefer-cached + launch pre-pull) -> RUNNING
-> our /ping on the ENI IP answers. `open_sent_epoch` is taken BEFORE RunTask, so the pool's open_ms /
pull_boot_ms mean the same thing as on AgentCore; `open_phases` splits it further (provision_ms,
pull_ms, task_running_ms, server_ready_ms, node).

All waiting tasks of a process are polled by ONE background thread with batched DescribeTasks (100
tasks per call), so 64 pre-opened episodes cost one API call per ECS_POLL_SEC, not 64.

Knobs (env): ECS_CLUSTER (acs-swe-ecs)  ECS_CAPACITY_PROVIDER (acs-swe-ecs-cp)  ECS_SUBNETS (comma list)
ECS_SECURITY_GROUPS (comma list; the task SG)  ECS_OPEN_TIMEOUT (1200 s; includes waiting for a node
with room)  ECS_POLL_SEC (1.0)  ECS_RUNTASK_RETRY_SEC (5; RunTask `failures`, e.g. capacity, are retried
until the open timeout).
"""
from __future__ import annotations

import asyncio
import os
import secrets
import sys
import threading
import time
import urllib.request

from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox, _http_invoke_sync, _timed_invoke

STARTED_BY = "acs-swe"                  # RunTask startedBy (<= 36 chars): ecs_deploy.py down/status find ours by it


class TaskOpenError(RuntimeError):
    pass


_clients: dict[str, object] = {}
_clients_lock = threading.Lock()


def ecs_client():
    region = os.environ.get("AWS_REGION", "us-west-2")
    with _clients_lock:
        if region not in _clients:
            import boto3
            from botocore.config import Config

            _clients[region] = boto3.client("ecs", region_name=region, config=Config(
                retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=64))
        return _clients[region]


def _csv(name: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


class _TaskWatcher:
    """One poller thread per cluster: batched DescribeTasks for every task some opener waits on."""

    _by_cluster: dict[str, "_TaskWatcher"] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, cluster: str) -> "_TaskWatcher":
        with cls._lock:
            if cluster not in cls._by_cluster:
                cls._by_cluster[cluster] = cls(cluster)
            return cls._by_cluster[cluster]

    def __init__(self, cluster: str) -> None:
        self.cluster = cluster
        self._watch: set[str] = set()
        self._state: dict[str, dict] = {}
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None

    def watch(self, arn: str) -> None:
        with self._cv:
            self._watch.add(arn)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name=f"ecs-watch-{self.cluster}", daemon=True)
                self._thread.start()

    def unwatch(self, arn: str) -> None:
        with self._cv:
            self._watch.discard(arn)
            self._state.pop(arn, None)

    def wait_change(self, arn: str, seen: dict | None, timeout: float) -> dict | None:
        """Block until the task's described state differs from `seen` (or timeout); return the latest."""
        with self._cv:
            self._cv.wait_for(lambda: self._state.get(arn) is not seen, timeout=timeout)
            return self._state.get(arn)

    def _loop(self) -> None:
        poll = float(os.environ.get("ECS_POLL_SEC", "1.0"))
        while True:
            with self._cv:
                arns = sorted(self._watch)
            if not arns:
                time.sleep(poll)
                continue
            for i in range(0, len(arns), 100):
                chunk = arns[i:i + 100]
                try:
                    resp = ecs_client().describe_tasks(cluster=self.cluster, tasks=chunk)
                except Exception as e:  # noqa: BLE001 — transient API error: keep polling
                    sys.stderr.write(f"[ecs_task_sandbox] describe_tasks failed: {type(e).__name__}: {e}\n")
                    continue
                with self._cv:
                    for t in resp.get("tasks", []):
                        if t["taskArn"] in self._watch:
                            self._state[t["taskArn"]] = t
                    for f in resp.get("failures", []):      # MISSING: the task is gone
                        if f.get("arn") in self._watch:
                            self._state[f["arn"]] = {"taskArn": f["arn"], "lastStatus": "MISSING",
                                                     "stoppedReason": f.get("reason", "")}
                    self._cv.notify_all()
            time.sleep(poll)


def _epoch(t: dict, key: str) -> float | None:
    v = t.get(key)
    return v.timestamp() if hasattr(v, "timestamp") else None


def _eni_ip(t: dict) -> str | None:
    for att in t.get("attachments") or []:
        if att.get("type") == "ElasticNetworkInterface" and att.get("status") == "ATTACHED":
            for kv in att.get("details") or []:
                if kv.get("name") == "privateIPv4Address":
                    return kv.get("value")
    return None


def _stop_detail(t: dict) -> str:
    cont = "; ".join(f"{c.get('name')}: exit={c.get('exitCode')} {c.get('reason', '')}"
                     for c in t.get("containers") or [] if c.get("reason") or c.get("exitCode") is not None)
    return f"{t.get('stopCode', '')} {t.get('stoppedReason', '')} {cont}".strip()[:400]


class EcsTaskSandbox(AgentCoreRuntimeSandbox):
    """Sandbox backed by one brand-new ECS task. exec/write_file/read_file are inherited."""

    def __init__(self, *, task_definition: str, workdir: str = "", session_id: str | None = None,
                 cluster: str | None = None, read_timeout: int | None = None) -> None:
        super().__init__(runtime_arn=None, workdir=workdir, read_timeout=read_timeout,
                         session_id=session_id or f"ecsep-{secrets.token_hex(16)}")
        self.task_definition = task_definition
        self.cluster = cluster or os.environ.get("ECS_CLUSTER", "acs-swe-ecs")
        self.task_arn: str | None = None
        self.sandbox_id = "ecs-pending"
        self._url: str | None = None
        self.open_phases: dict = {}

    def _run_task_sync(self, deadline: float) -> str:
        subnets, sgs = _csv("ECS_SUBNETS"), _csv("ECS_SECURITY_GROUPS")
        if not subnets or not sgs:
            raise TaskOpenError("ECS_SUBNETS / ECS_SECURITY_GROUPS unset — run.sh reads them from "
                                "swebench/ecs_network.json (written by `deploy/ecs_deploy.py up`)")
        retry = float(os.environ.get("ECS_RUNTASK_RETRY_SEC", "5"))
        while True:
            resp = ecs_client().run_task(
                cluster=self.cluster, taskDefinition=self.task_definition, count=1,
                capacityProviderStrategy=[{"capacityProvider": os.environ.get(
                    "ECS_CAPACITY_PROVIDER", "acs-swe-ecs-cp"), "weight": 1}],
                networkConfiguration={"awsvpcConfiguration": {
                    "subnets": subnets, "securityGroups": sgs, "assignPublicIp": "DISABLED"}},
                placementStrategy=[{"type": "spread", "field": "instanceId"}],   # spread episodes over nodes
                startedBy=STARTED_BY, enableECSManagedTags=True, propagateTags="TASK_DEFINITION",
                tags=[{"key": "acs/session", "value": self._session_id[-40:]}])
            if resp.get("tasks"):
                return resp["tasks"][0]["taskArn"]
            why = "; ".join(f"{f.get('reason')} {f.get('detail', '')}".strip() for f in resp.get("failures", []))
            if time.time() + retry > deadline:
                raise TaskOpenError(f"RunTask {self.task_definition} kept failing: {why}")
            time.sleep(retry)

    def _wait_ready_sync(self) -> dict:
        timeout = float(os.environ.get("ECS_OPEN_TIMEOUT", "1200"))
        t0 = self.open_sent_epoch
        deadline = t0 + timeout
        self.task_arn = self._run_task_sync(deadline)
        self.sandbox_id = f"ecs-{self.task_arn.rsplit('/', 1)[-1]}"
        watcher = _TaskWatcher.get(self.cluster)
        watcher.watch(self.task_arn)
        try:
            t, t_running = None, None
            while True:
                t = watcher.wait_change(self.task_arn, t, timeout=max(0.1, min(5.0, deadline - time.time())))
                st = (t or {}).get("lastStatus", "")
                if st in ("DEACTIVATING", "STOPPING", "DEPROVISIONING", "STOPPED", "MISSING"):
                    raise TaskOpenError(f"task {self.sandbox_id} {st} before ready: {_stop_detail(t)}")
                if st == "RUNNING" and _eni_ip(t):
                    t_running = time.time()
                    break
                if time.time() > deadline:
                    raise TaskOpenError(f"task {self.sandbox_id} not RUNNING after {timeout:.0f}s (last {st or '?'})")
        finally:
            watcher.unwatch(self.task_arn)
        ip = _eni_ip(t)
        # RUNNING = the container process started; the server needs a moment more to bind :8080.
        while True:
            try:
                with urllib.request.urlopen(f"http://{ip}:8080/ping", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                pass
            if time.time() > deadline:
                raise TaskOpenError(f"task {self.sandbox_id} RUNNING at {ip} but /ping never answered")
            time.sleep(0.25)
        now = time.time()
        pull0, pull1 = _epoch(t, "pullStartedAt"), _epoch(t, "pullStoppedAt")
        return {"task_ip": ip, "node": (t.get("containerInstanceArn") or "?").rsplit("/", 1)[-1],
                "provision_ms": round(((pull0 or t_running) - t0) * 1000.0, 1),
                "pull_ms": round((pull1 - pull0) * 1000.0, 1) if pull0 and pull1 else 0.0,
                "task_running_ms": round((t_running - t0) * 1000.0, 1),
                "server_ready_ms": round((now - t0) * 1000.0, 1)}

    async def __aenter__(self) -> "EcsTaskSandbox":
        self.open_sent_epoch = time.time()
        try:
            self.open_phases = await asyncio.to_thread(self._wait_ready_sync)
            self._url = f"http://{self.open_phases['task_ip']}:8080"
            self.open_reply = await self._invoke({"op": "ping", "workdir": self._workdir})
        except BaseException:
            await self.__aexit__(None, None, None)   # `async with` skips __aexit__ on a failed open: no leak
            raise
        self.open_recv_epoch = time.time()
        return self

    async def __aexit__(self, *exc) -> None:
        if not self.task_arn:
            return
        status = "ok"
        try:
            await asyncio.to_thread(ecs_client().stop_task, cluster=self.cluster, task=self.task_arn,
                                    reason="acs-swe episode done")
        except Exception as e:  # noqa: BLE001 — teardown: log; the entrypoint timeout reclaims a leak
            status = "gone" if "not found" in str(e).lower() else type(e).__name__
        sys.stdout.write(f"[sbx-op] ts={time.time():.3f} op=stop_session ms=0.0 arn={self.sandbox_id} "
                         f"session={self._session_id} status={status}\n")
        sys.stdout.flush()
        self.task_arn = None

    def _invoke_sync(self, payload: dict) -> dict:
        if not self._url:
            raise RuntimeError(f"task {self.sandbox_id} is not open")
        return _timed_invoke(
            lambda: _http_invoke_sync(self._url, self._session_id, payload, self._read_timeout),
            op=str(payload.get("op")), arn=self.sandbox_id, session=self._session_id,
            what=f"ecs task invoke op={payload.get('op')}")
