"""`EksPodSandbox`: one episode = one fresh Kubernetes pod on EKS, run under gVisor (RuntimeClass `gvisor`).

The EKS counterpart of `AgentCoreRuntimeSandbox`, and a drop-in for it: same `exec` / `write_file` /
`read_file` / `_invoke` API, same op payloads, served by the SAME in-image server
(deploy/acs_swe_http_server.py on :8080, `POST /invocations`). The images are the ones the AgentCore
demo pushes (same ECR repos, same recipe tag). Only the session lifecycle and the transport change:

    AgentCore                                     EKS (this file)
    runtimeSessionId -> own m5 Instance           pod -> own gVisor sandbox (user-space kernel) on a
                                                  shared m5 node; several pods per node
    InvokeAgentRuntime (AWS data plane)           HTTP straight to the pod IP (VPC CNI: pods get VPC IPs)
    StopRuntimeSession                            DELETE pod (grace 0)
    idle reclaim ~15 min -> heartbeat needed      no idle reclaim (heartbeat = liveness ping); activeDeadlineSeconds caps a leak

Isolation: gVisor intercepts every syscall in its own user-space kernel (runsc), so the agent's code
never talks to the node kernel directly; one pod per episode, never reused, so no state leaks between
episodes. It is weaker than AgentCore's one-machine-per-session (pods share a node's CPU, memory
bandwidth and disk) and stronger than plain runc containers. Network: the `acs-swe` namespace's
NetworkPolicy (deploy/eks/sandbox-namespace.yaml) lets a pod receive :8080 from the trainer and dial
back only the trainer's adapter ports — the "egress none" of the AgentCore demo.

Hardening on top: the container keeps root (SWE-bench's /testbed and conda env are root-owned) but
drops ALL capabilities, cannot gain privileges (no setuid escalation) and runs the RuntimeDefault
seccomp profile (enforced on the runc A/B; gVisor filters syscalls itself). Each pod gets a random
ACS_SANDBOX_TOKEN and the server rejects any /invocations without it, so reaching :8080 is not enough.

Open = create pod -> scheduled -> image pulled (cached per node after the first pull) -> runsc starts
the container -> readiness probe on /ping -> our first ping. `open_sent_epoch` is taken BEFORE the
create, so the pool's open_ms / pull_boot_ms mean the same thing as on AgentCore; `open_phases` splits
it further (schedule_ms, pod_ready_ms, node).

Knobs (env): EKS_CLUSTER (acs-swe-eks)  EKS_NAMESPACE (acs-swe)  EKS_RUNTIME_CLASS (gvisor; "" = runc,
for A/B only)  EKS_POD_CPU (1500m)  EKS_POD_MEMORY (6Gi)  EKS_POD_MEMORY_LIMIT (8Gi)
EKS_POD_EPHEMERAL (10Gi)  EKS_POD_MAX_LIFETIME (7200 s)  EKS_OPEN_TIMEOUT (1200 s; includes waiting
for a node with room)  EKS_PULL_FAIL_SEC (180 s of ErrImagePull/ImagePullBackOff = failed open)
EKS_POLL_SEC (1.0).
"""
from __future__ import annotations

import asyncio
import os
import secrets
import sys
import time

from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox, _http_invoke_sync, _timed_invoke
from .eks_api import KubeError, get_api

_FATAL_WAITING = {"InvalidImageName", "CreateContainerConfigError", "CreateContainerError",
                  "RunContainerError", "ErrImageNeverPull"}
_PULL_WAITING = {"ErrImagePull", "ImagePullBackOff"}


class PodOpenError(RuntimeError):
    pass


SECURITY_CONTEXT = {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]},
                    "seccompProfile": {"type": "RuntimeDefault"}}


def pod_manifest(name: str, image: str, session_id: str, token: str) -> dict:
    env = os.environ.get
    runtime_class = env("EKS_RUNTIME_CLASS", "gvisor")
    spec = {
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": 0,
        "activeDeadlineSeconds": int(env("EKS_POD_MAX_LIFETIME", "7200")),
        "automountServiceAccountToken": False,     # the agent gets no Kubernetes credentials
        "enableServiceLinks": False,
        "containers": [{
            "name": "sandbox",
            "image": image,                        # ECR URI pinned by digest
            "imagePullPolicy": "IfNotPresent",
            "ports": [{"containerPort": 8080}],
            "env": [{"name": "ACS_SANDBOX_TOKEN", "value": token}],   # server pops it at start
            "securityContext": SECURITY_CONTEXT,
            "readinessProbe": {"httpGet": {"path": "/ping", "port": 8080},
                               "periodSeconds": 1, "failureThreshold": 600},
            "resources": {
                "requests": {"cpu": env("EKS_POD_CPU", "1500m"), "memory": env("EKS_POD_MEMORY", "6Gi"),
                             "ephemeral-storage": env("EKS_POD_EPHEMERAL", "10Gi")},
                "limits": {"memory": env("EKS_POD_MEMORY_LIMIT", "8Gi")},
            },
        }],
    }
    if runtime_class:
        spec["runtimeClassName"] = runtime_class   # its scheduling.nodeSelector keeps pods on runsc nodes
    return {"apiVersion": "v1", "kind": "Pod",
            "metadata": {"name": name, "labels": {"app": "acs-swe-sandbox", "acs/session": session_id[-40:]}},
            "spec": spec}


class EksPodSandbox(AgentCoreRuntimeSandbox):
    """Sandbox backed by one brand-new gVisor pod. exec/write_file/read_file are inherited."""

    def __init__(self, *, image: str, workdir: str = "", session_id: str | None = None,
                 namespace: str | None = None, read_timeout: int | None = None) -> None:
        super().__init__(runtime_arn=None, workdir=workdir, read_timeout=read_timeout,
                         session_id=session_id or f"eksep-{secrets.token_hex(16)}")
        self.image = image
        self.namespace = namespace or os.environ.get("EKS_NAMESPACE", "acs-swe")
        self.pod = f"swe-{self._session_id[-24:].lower()}"
        self.sandbox_id = f"eks-{self.pod}"
        self._url: str | None = None
        self._created = False
        self.open_phases: dict = {}
        self._token = secrets.token_hex(32)

    def _path(self, name: str = "") -> str:
        return f"/api/v1/namespaces/{self.namespace}/pods" + (f"/{name}" if name else "")

    def _wait_ready_sync(self) -> dict:
        api = get_api()
        timeout = float(os.environ.get("EKS_OPEN_TIMEOUT", "1200"))
        pull_fail = float(os.environ.get("EKS_PULL_FAIL_SEC", "180"))
        poll = float(os.environ.get("EKS_POLL_SEC", "1.0"))
        t0, t_sched, pull_bad_since = time.time(), None, None
        while True:
            pod = api.request("GET", self._path(self.pod))
            st = pod.get("status") or {}
            if t_sched is None and (pod.get("spec") or {}).get("nodeName"):
                t_sched = time.time()
            if st.get("phase") in ("Failed", "Succeeded"):
                raise PodOpenError(f"pod {self.pod} {st.get('phase')}: {st.get('reason')} {st.get('message')}")
            waiting = [((c.get("state") or {}).get("waiting") or {}) for c in st.get("containerStatuses") or []]
            for w in waiting:
                if w.get("reason") in _FATAL_WAITING:
                    raise PodOpenError(f"pod {self.pod} {w['reason']}: {w.get('message', '')[:300]}")
            if any(w.get("reason") in _PULL_WAITING for w in waiting):
                pull_bad_since = pull_bad_since or time.time()
                if time.time() - pull_bad_since > pull_fail:
                    raise PodOpenError(f"pod {self.pod} cannot pull {self.image}: {waiting}")
            else:
                pull_bad_since = None
            ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in st.get("conditions") or [])
            if ready and st.get("podIP"):
                now = time.time()
                return {"node": pod["spec"].get("nodeName", "?"), "pod_ip": st["podIP"],
                        "schedule_ms": round(((t_sched or now) - t0) * 1000.0, 1),
                        "pod_ready_ms": round((now - t0) * 1000.0, 1)}
            if time.time() - t0 > timeout:
                why = [c.get("message") for c in st.get("conditions") or [] if c.get("status") != "True"]
                raise PodOpenError(f"pod {self.pod} not ready after {timeout:.0f}s ({st.get('phase')}; {why})")
            time.sleep(poll)

    async def __aenter__(self) -> "EksPodSandbox":
        api = get_api()
        self.open_sent_epoch = time.time()
        await asyncio.to_thread(api.request, "POST", self._path(),
                                pod_manifest(self.pod, self.image, self._session_id, self._token))
        self._created = True
        self.open_phases = await asyncio.to_thread(self._wait_ready_sync)
        self._url = f"http://{self.open_phases['pod_ip']}:8080"
        self.open_reply = await self._invoke({"op": "ping", "workdir": self._workdir})
        self.open_recv_epoch = time.time()
        return self

    async def __aexit__(self, *exc) -> None:
        if not self._created:
            return
        status = "ok"
        try:
            await asyncio.to_thread(get_api().request, "DELETE", self._path(self.pod),
                                    {"gracePeriodSeconds": 0, "propagationPolicy": "Background"})
        except KubeError as e:
            status = "gone" if e.status == 404 else f"KubeError{e.status}"
        except Exception as e:  # noqa: BLE001 — teardown: log; activeDeadlineSeconds reclaims a leak
            status = type(e).__name__
        sys.stdout.write(f"[sbx-op] ts={time.time():.3f} op=stop_session ms=0.0 arn=eks:{self.pod} "
                         f"session={self._session_id} status={status}\n")
        sys.stdout.flush()
        self._created = False

    def _invoke_sync(self, payload: dict) -> dict:
        if not self._url:
            raise RuntimeError(f"pod {self.pod} is not open")
        return _timed_invoke(
            lambda: _http_invoke_sync(self._url, self._session_id, payload, self._read_timeout,
                                      {"X-Acs-Sandbox-Token": self._token}),
            op=str(payload.get("op")), arn=f"eks:{self.pod}", session=self._session_id,
            what=f"eks pod invoke op={payload.get('op')}")
