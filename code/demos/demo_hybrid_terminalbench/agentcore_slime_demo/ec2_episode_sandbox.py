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


class Ec2EpisodeSandbox(DockerComposeSandbox):
    """A fresh EC2 instance per episode, running the task's compose project; terminated on close."""

    def __init__(self, route: str, *, region: str, workdir: str = "", session_id: str | None = None,
                 read_timeout: int | None = None) -> None:
        super().__init__(route, workdir=workdir, session_id=session_id, read_timeout=read_timeout)
        self._region = region
        self._cfg = episode_config()
        self._agent_port = int(self._cfg.get("agent_port", 18300))
        self._token = secrets.token_urlsafe(32)
        self._instance_id: str | None = None
        self.sandbox_id = f"ec2ep-{self._episode}"

    def _label(self) -> str:
        return f"ec2:{self._task}"

    def _launch_sync(self) -> tuple[str, str]:
        c = self._cfg
        script = ("#!/bin/bash\nset -e\nexec > /var/log/acs-episode.log 2>&1\numask 077\n"
                  f"echo '{self._token}' > /srv/acs/token\nsystemctl restart acs-compose-agent\n")
        inst = _ec2(self._region).run_instances(
            ImageId=c["ami"], InstanceType=c.get("instance_type", "m5.xlarge"), MinCount=1, MaxCount=1,
            UserData=gzip.compress(script.encode()),
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
        iid, self._client_url = self._instance_id, None
        if not iid:
            return
        t0, status = time.perf_counter(), "ok"
        try:
            await asyncio.to_thread(lambda: _ec2(self._region).terminate_instances(InstanceIds=[iid]))
            self._instance_id = None
        except Exception as e:  # noqa: BLE001 — teardown: gc-episodes / the next pool start is the fallback
            status = type(e).__name__
        sys.stdout.write(f"[sbx-op] ts={time.time():.3f} op=stop_session ms={(time.perf_counter() - t0) * 1000.0:.1f} "
                         f"arn={self._label()} session={self._session_id} status={status}\n")
        sys.stdout.flush()
