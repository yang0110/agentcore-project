"""`Ec2EpisodeSandbox` — ONE fresh EC2 instance per Terminal-Bench episode (the m5 tier's per_episode mode).

TB_M5_MODE=per_episode (run.sh). Same docker:// routes, same in-image server and compose rewrite as the
shared mode (DockerComposeSandbox), but the unit of isolation is a whole VM instead of a compose project:

    open   RunInstances (the baked AMI: Docker + compose agent, no task images; M5_EP_INSTANCE_TYPE,
           default m5.xlarge, 40 GB gp3; IMDSv2 hop 1; ECR-pull-only instance role) with a fresh agent
           secret in its user-data -> wait for the agent -> upload the task dir (no tests/solution) and
           its ECR build record -> `up` (docker login + pull ONLY this task's images from ECR, compose up)
           -> ping the client's server
    ops    straight to the client container, as in shared mode
    close  TerminateInstances (nothing to tear down inside: the VM goes)

The cold start (launch ~30-60 s + agent + pull + up) is what the look-ahead pre-open hides; at most
TB_M5_MAX_INSTANCES live at once (the pool's docker gate). Config: TB_M5_EPISODE (JSON written by
`deploy/m5_host.py bake-ami`) and tasks/docker_records/<task>.{json,tar.gz} (deploy_docker_tasks.py).
Every instance is tagged Name=<TB_M5_EPISODE.tag> so a crashed run's leftovers can be terminated
(`deploy/m5_host.py gc-episodes`, also run once at pool start).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import gzip
import json
import os
import secrets
import sys
import time

from .docker_compose_sandbox import DockerComposeSandbox, agent_call

_RECORDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "terminalbench", "tasks",
                        "docker_records")


def episode_config() -> dict:
    raw = os.environ.get("TB_M5_EPISODE", "").strip()
    if not raw:
        raise RuntimeError("TB_M5_MODE=per_episode needs TB_M5_EPISODE (run.sh reads deploy/m5_host.json; "
                           "bake it with: python3 deploy/m5_host.py roles && python3 deploy/m5_host.py bake-ami)")
    return json.loads(raw)


def _ec2(region: str):
    import boto3
    from botocore.config import Config

    return boto3.client("ec2", region_name=region, config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))


def terminate_tagged(region: str, tag: str) -> int:
    """Terminate every per-episode instance (crash recovery). Returns how many."""
    ec2 = _ec2(region)
    ids = [i["InstanceId"] for r in ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [tag]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])["Reservations"]
        for i in r["Instances"]]
    if ids:
        ec2.terminate_instances(InstanceIds=ids)
    return len(ids)


# Every m5 this process launched (per-episode AND warm) and has not terminated yet. A launch still in
# flight when its open is cancelled returns its id only to this set, and the exit hook terminates
# whatever is left in it (warm instances too) — on top of 5_run.sh's gc-episodes and the dead-man switch.
_LAUNCHED: set[str] = set()


def episode_user_data(token: str, env: dict | None = None) -> bytes:
    """User-data of a launched episode instance (m5 or g6): install its agent secret, then restart the
    agent with the dead-man switch (+ `env`) as a systemd drop-in. The switch is armed from the agent's
    FIRST start, not by a trainer call once the agent answers: an instance whose trainer died during
    boot still terminates itself."""
    env = {"ACS_IDLE_SHUTDOWN_S": int(float(os.environ.get("TB_M5_IDLE_SHUTDOWN_S", "1800"))), **(env or {})}
    lines = "".join(f"Environment={k}={v}\\n" for k, v in env.items())
    script = ("#!/bin/bash\nset -e\nexec > /var/log/acs-episode.log 2>&1\numask 077\n"
              f"echo '{token}' > /srv/acs/token\n"
              "mkdir -p /etc/systemd/system/acs-compose-agent.service.d\n"
              f"printf '[Service]\\n{lines}' > /etc/systemd/system/acs-compose-agent.service.d/idle.conf\n"
              "systemctl daemon-reload\nsystemctl restart acs-compose-agent\n")
    return gzip.compress(script.encode())


def _terminate_ids(region: str, ids) -> None:
    ids = sorted(set(ids))
    if ids:
        _ec2(region).terminate_instances(InstanceIds=ids)
    _LAUNCHED.difference_update(ids)


class Ec2EpisodeSandbox(DockerComposeSandbox):
    """A fresh EC2 instance per episode, running the task's compose project; terminated on close."""

    def __init__(self, route: str, *, region: str, workdir: str = "", session_id: str | None = None,
                 read_timeout: int | None = None, cfg: dict | None = None) -> None:
        super().__init__(route, workdir=workdir, session_id=session_id, read_timeout=read_timeout)
        self._region = region
        self._cfg = cfg or episode_config()
        self._agent_port = int(self._cfg.get("agent_port", 18300))
        self._token = secrets.token_urlsafe(32)
        self._instance_id: str | None = None
        self._abandoned = False   # closed before the launch returned: the launch terminates its instance
        self.sandbox_id = f"ec2ep-{self._episode}"

    def _label(self) -> str:
        return f"ec2:{self._task}"

    def _launch_sync(self) -> tuple[str, str]:
        c = self._cfg
        inst = _ec2(self._region).run_instances(
            ImageId=c["ami"], InstanceType=c.get("instance_type", "m5.2xlarge"), MinCount=1, MaxCount=1,
            UserData=episode_user_data(self._token),
            IamInstanceProfile={"Name": c["profile"]},
            NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": c["subnet"], "Groups": [c["sg"]],
                                "AssociatePublicIpAddress": False}],
            BlockDeviceMappings=[{"DeviceName": "/dev/xvda", "Ebs": {
                "VolumeSize": int(c.get("disk_gb", 40)), "VolumeType": "gp3", "DeleteOnTermination": True}}],
            MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 1, "HttpEndpoint": "enabled"},
            InstanceInitiatedShutdownBehavior="terminate",
            TagSpecifications=[{"ResourceType": "instance", "Tags": [
                {"Key": "Name", "Value": c.get("tag", "acs-tbh-m5-ep")},
                {"Key": "acs-episode", "Value": self._session_id}, {"Key": "acs-task", "Value": self._task}]}],
        )["Instances"][0]
        _LAUNCHED.add(inst["InstanceId"])
        if self._abandoned:        # the open was cancelled while RunInstances ran: nobody will close it
            with contextlib.suppress(Exception):
                _terminate_ids(self._region, [inst["InstanceId"]])
            raise RuntimeError(f"{inst['InstanceId']}: launched after its open was cancelled; terminated")
        return inst["InstanceId"], inst["PrivateIpAddress"]

    async def _wait_agent(self, deadline_s: float) -> None:
        end = time.monotonic() + deadline_s
        while True:
            try:   # 403 until user-data installs this instance's secret; refused until the agent is up
                rep = await asyncio.to_thread(lambda: agent_call(self._agent_host, self._agent_port, {"op": "health"},
                                                                 timeout=10, token=self._token))
                if rep.get("ok"):
                    return
            except Exception:  # noqa: BLE001 — still booting
                pass
            if time.monotonic() > end:
                raise RuntimeError(f"{self._instance_id}: compose agent not ready after {deadline_s:.0f}s")
            await asyncio.sleep(2)

    async def _arm_idle_shutdown(self) -> None:
        """Per-episode instances: the same dead-man switch (an older baked agent may lack the op)."""
        try:
            await asyncio.to_thread(lambda: agent_call(self._agent_host, self._agent_port, {
                "op": "set_idle_shutdown", "seconds": float(os.environ.get("TB_M5_IDLE_SHUTDOWN_S", "1800"))},
                timeout=30, token=self._token))
        except Exception:  # noqa: BLE001 — the trainer-side terminate + gc-episodes still apply
            pass

    async def _upload(self) -> None:
        with open(os.path.join(_RECORDS, f"{self._task}.json")) as f:
            rec = json.load(f)
        with open(os.path.join(_RECORDS, f"{self._task}.tar.gz"), "rb") as f:
            data = f.read()
        b64, chunk, key = base64.b64encode(data).decode(), 3 * 2**20, secrets.token_hex(6)
        call = lambda p: agent_call(self._agent_host, self._agent_port, p, timeout=300, token=self._token)  # noqa: E731
        for i in range(0, max(1, len(b64)), chunk):
            await asyncio.to_thread(call, {"op": "put_tar", "dest": f"task:{self._task}", "key": key,
                                           "b64": b64[i:i + chunk], "more": i + chunk < len(b64)})
        await asyncio.to_thread(call, {"op": "set_record", "task": self._task, "rec": rec})

    async def __aenter__(self) -> "Ec2EpisodeSandbox":
        t0, sent = time.perf_counter(), time.time()
        self._open_t0 = t0                        # the parent's single [sbx-op] op=open record covers it all
        try:
            self._instance_id, self._agent_host = await asyncio.to_thread(self._launch_sync)
            t_launch = time.perf_counter()
            await self._wait_agent(float(os.environ.get("TB_M5_BOOT_WAIT_S", "600")))
            await self._arm_idle_shutdown()
            t_agent = time.perf_counter()
            await self._upload()
            await super().__aenter__()            # up (pull + compose up) + ping; prints [sbx-op] op=open
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        self.open_sent_epoch = sent               # the cold start the episode pays starts at RunInstances
        self.open_reply.setdefault("boot", {}).update(
            instance_id=self._instance_id, launch_api_ms=round((t_launch - t0) * 1000.0, 1),
            instance_boot_ms=round((t_agent - t_launch) * 1000.0, 1),
            ec2_open_ms=round((time.perf_counter() - t0) * 1000.0, 1))
        return self

    async def __aexit__(self, *exc) -> None:
        self._abandoned = True
        iid, self._client_url = self._instance_id, None
        if not iid:
            return
        t0, status = time.perf_counter(), "ok"
        try:
            await asyncio.to_thread(_terminate_ids, self._region, [iid])
            self._instance_id = None
        except Exception as e:  # noqa: BLE001 — teardown: gc-episodes / the next pool start is the fallback
            status = type(e).__name__
        sys.stdout.write(f"[sbx-op] ts={time.time():.3f} op=stop_session ms={(time.perf_counter() - t0) * 1000.0:.1f} "
                         f"arn={self._label()} session={self._session_id} status={status}\n")
        sys.stdout.flush()


# ============================================================================ warm pool (TB_M5_MODE=warm)
# ONE instance serves ONE session at a time, but is KEPT after the episode instead of terminated: the
# agent runs `down -v` and a VERIFIED reset (reset_check: no containers / volumes / non-default networks /
# episode dirs left). The episode only ever touched its own compose project — containers can't reach the
# host — so a clean check means the next episode starts from the same state a fresh instance would. A
# failed check (or failed health) terminates the instance; it is never reused. An idle instance that
# already holds a task's images is preferred for that task (no pull: ~6 s open instead of ~35-60 s).
import threading as _threading

_AGENT_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "m5_agent", "compose_agent.py")


def _current_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class _WarmInstance:
    def __init__(self, iid: str, ip: str, token: str) -> None:
        self.iid, self.ip, self.token = iid, ip, token
        self.busy, self.tasks, self.episodes = True, set(), 0
        self.loop = _current_loop()            # the event loop whose session holds it (while busy)


class WarmPool:
    """Process-wide pool of warm m5 instances (thread-safe: the train and eval event loops share it).

    Every busy instance and every launch in progress is charged to the event loop that holds it, so a
    superseded loop's holdings can be reclaimed (reclaim): eval orphans the fully-async worker's loop,
    and sessions it pre-opened would otherwise stay busy forever and fill the cap (train48 hung at the
    step-20 eval with all 24 instances held by the orphaned loop)."""

    def __init__(self) -> None:
        self._lock = _threading.Lock()
        self._insts: list[_WarmInstance] = []
        self._launching: dict = {}             # loop -> launches in progress

    def stats(self) -> dict:
        with self._lock:
            return {"warm_instances": len(self._insts), "warm_busy": sum(i.busy for i in self._insts)}

    def _take(self, task: str, cap: int):
        """(instance, reused) or (None, launch?) — prefer an idle instance that already has this task."""
        loop = _current_loop()
        with self._lock:
            idle = [i for i in self._insts if not i.busy]
            for i in sorted(idle, key=lambda i: task not in i.tasks):
                i.busy, i.loop = True, loop
                return i, True
            if len(self._insts) + sum(self._launching.values()) < cap:
                self._launching[loop] = self._launching.get(loop, 0) + 1
                return None, True
            return None, False

    def _added(self, inst: _WarmInstance | None) -> None:
        loop = _current_loop()
        with self._lock:
            if self._launching.get(loop, 0) > 0:   # 0 if this loop's launches were reclaimed meanwhile
                self._launching[loop] -= 1
            if inst is not None:
                self._insts.append(inst)

    def reclaim(self, loop) -> list[str]:
        """Forget every instance (and launch slot) held by `loop`, a loop that will never close them;
        returns their ids for the caller to terminate. They are not reused: their compose projects are
        still up, so their state is unknown."""
        with self._lock:
            gone = [i for i in self._insts if i.busy and i.loop is loop]
            self._insts = [i for i in self._insts if i not in gone]
            self._launching.pop(loop, None)
        return [i.iid for i in gone]

    def _drop(self, inst: _WarmInstance) -> None:
        with self._lock:
            if inst in self._insts:
                self._insts.remove(inst)

    def _free(self, inst: _WarmInstance) -> None:
        with self._lock:
            inst.busy = False


_WARM = WarmPool()


def _terminate_warm_at_exit() -> None:
    """Best effort: a finished run leaves no warm instance behind (5_run.sh and the next pool start
    also terminate every tagged instance)."""
    ids = [i.iid for i in _WARM._insts] + list(_LAUNCHED)
    if ids:
        try:
            _terminate_ids(os.environ.get("AWS_REGION", "us-west-2"), ids)
        except Exception:  # noqa: BLE001
            pass


def reclaim_warm(loop, region: str) -> None:
    """Free the cap held by a superseded event loop's warm sessions; terminate those instances."""
    ids = _WARM.reclaim(loop)
    if not ids:
        return
    sys.stdout.write(f"[demo] warm_reclaim instances={len(ids)} reason=loop_superseded ids={','.join(ids)}\n")
    sys.stdout.flush()
    try:
        _terminate_ids(region, ids)
    except Exception as e:  # noqa: BLE001 — the dead-man switch and gc-episodes are the backstop
        sys.stdout.write(f"[demo] warm_reclaim terminate failed: {e}\n")


import atexit as _atexit  # noqa: E402

_atexit.register(_terminate_warm_at_exit)


class Ec2WarmSandbox(Ec2EpisodeSandbox):
    """A warm m5 instance for ONE episode at a time; verified reset afterwards instead of terminate."""

    def __init__(self, route: str, **kw) -> None:
        super().__init__(route, **kw)
        self._inst: _WarmInstance | None = None
        self.sandbox_id = f"ec2warm-{self._episode}"

    def _label(self) -> str:
        return f"ec2warm:{self._task}"

    async def _ensure_agent(self) -> None:
        """Instances from the baked AMI may run an older agent: push this one (it restarts itself)."""
        import re

        call = lambda p, t=60: agent_call(self._agent_host, self._agent_port, p, timeout=t, token=self._token)  # noqa: E731
        with open(_AGENT_SRC, encoding="utf-8") as f:
            src = f.read()
        m = re.search(r'AGENT_VERSION = "([^"]+)"', src)
        want = m.group(1) if m else None
        h = await asyncio.to_thread(call, {"op": "health"})
        if want and h.get("version") != want:
            await asyncio.to_thread(call, {"op": "self_update", "b64": base64.b64encode(src.encode()).decode()})
            await asyncio.sleep(3)
            await self._wait_agent(120)
        # Dead-man switch: if the trainer stops calling (box died, run killed), the instance shuts itself
        # down, which TERMINATES it (InstanceInitiatedShutdownBehavior=terminate).
        await asyncio.to_thread(call, {"op": "set_idle_shutdown",
                                       "seconds": float(os.environ.get("TB_M5_IDLE_SHUTDOWN_S", "1800"))})

    async def __aenter__(self) -> "Ec2WarmSandbox":
        cap = int(os.environ.get("TB_M5_MAX_INSTANCES", "24"))
        t0, sent = time.perf_counter(), time.time()
        self._open_t0 = t0
        while True:
            inst, ok = _WARM._take(self._task, cap)
            if inst is not None:
                try:   # an idle instance may have shut itself down (dead-man switch) or died: check first
                    await asyncio.to_thread(lambda: agent_call(inst.ip, self._agent_port, {"op": "health"},
                                                               timeout=10, token=inst.token))
                    break
                except Exception:  # noqa: BLE001
                    _WARM._drop(inst)
                    sys.stdout.write(f"[demo] warm_retire instance={inst.iid} reason=unhealthy_on_reuse "
                                     f"episodes={inst.episodes}\n")
                    await asyncio.to_thread(_terminate_ids, self._region, [inst.iid])
                    continue
            if ok:
                break
            await asyncio.sleep(2)                 # all warm instances busy and at the cap
        launched = inst is None
        try:
            if launched:
                self._instance_id, self._agent_host = await asyncio.to_thread(self._launch_sync)
                inst = _WarmInstance(self._instance_id, self._agent_host, self._token)
                _WARM._added(inst)
                self._inst = inst                   # from here a failure retires it (drops it from the pool)
                await self._wait_agent(float(os.environ.get("TB_M5_BOOT_WAIT_S", "600")))
                await self._ensure_agent()
            else:
                self._instance_id, self._agent_host, self._token = inst.iid, inst.ip, inst.token
            self._inst = inst
            if self._task not in inst.tasks:       # task dir + ECR record, once per instance
                await self._upload()
                inst.tasks.add(self._task)
            await DockerComposeSandbox.__aenter__(self)     # up (pull only missing images) + ping
        except BaseException:
            if launched and inst is None:
                _WARM._added(None)
            await self._retire("open_failed")
            raise
        inst.episodes += 1
        self.open_sent_epoch = sent
        self.open_reply.setdefault("boot", {}).update(instance_id=inst.iid, warm_reused=int(not launched),
                                                      instance_episodes=inst.episodes, **_WARM.stats())
        self.demo_fields = {"warm_reused": int(not launched), "instance_episodes": inst.episodes}
        return self

    async def _retire(self, why: str) -> None:
        inst, self._inst = self._inst, None
        self._abandoned = True               # a launch still in flight terminates its own instance
        if inst is None:
            if self._instance_id:            # launched but never registered
                await asyncio.to_thread(_terminate_ids, self._region, [self._instance_id])
            return
        _WARM._drop(inst)
        try:
            await asyncio.to_thread(_terminate_ids, self._region, [inst.iid])
        finally:
            sys.stdout.write(f"[demo] warm_retire instance={inst.iid} reason={why} episodes={inst.episodes}\n")
            sys.stdout.flush()

    async def __aexit__(self, *exc) -> None:
        inst = self._inst
        if inst is None:
            return
        t0, status = time.perf_counter(), "ok"
        call = lambda p, t=330: agent_call(inst.ip, self._agent_port, p, timeout=t, token=inst.token)  # noqa: E731
        try:
            rep = await asyncio.to_thread(call, {"op": "down", "episode": self._episode})
            chk = await asyncio.to_thread(call, {"op": "reset_check"}, 600)
            if rep.get("ok") and chk.get("clean"):
                self._inst = None
                _WARM._free(inst)                  # verified clean: next episode may use it
            else:
                status = "reset_dirty"
                sys.stdout.write(f"[demo] warm_reset_dirty instance={inst.iid} problems="
                                 f"{str(chk.get('problems'))[:200].replace(' ', '_')}\n")
                await self._retire("reset_dirty")
        except Exception as e:  # noqa: BLE001 — an unreachable/failed agent: never reuse that instance
            status = type(e).__name__
            await self._retire(status)
        sys.stdout.write(f"[sbx-op] ts={time.time():.3f} op=stop_session ms={(time.perf_counter() - t0) * 1000.0:.1f} "
                         f"arn={self._label()} session={self._session_id} status={status}\n")
        sys.stdout.flush()
