"""`Ec2GpuSandbox` — KernelBench on PLAIN EC2 g6 instances (opt-in: TB_KB_SUBSTRATE=ec2), a warm GPU-slot pool.

The default KernelBench substrate stays the AgentCore g6 Runtime (route `gpu:<arn>`): one session = one
whole managed instance, cold-booted per episode (~2.2 min open). This substrate (route `gpuec2:<task>`)
is the m5 warm pool (ec2_episode_sandbox.Ec2WarmSandbox) carried over to GPUs:

    instance  launched from the g6 AMI (deploy/g6_host.py bake-ami: Deep Learning Base OSS Nvidia AMI +
              Docker + NVIDIA container toolkit + the compose agent, KernelBench image pre-pulled) and
              KEPT across episodes; tagged Name=<cfg.tag> for gc; dead-man switch as on the m5
    slot      ONE host GPU. A g6.xlarge has 1 slot, a g6.12xlarge 4, a g6.48xlarge 8: each slot runs
              one episode at a time, so a multi-GPU instance runs several episodes side by side
    open      take a free slot (pack: the busiest instance with a free slot first) or launch an instance
              -> `up` the task's compose project with the client pinned to that GPU
              (compose device reservation) and a 1/slots share of the vCPUs / memory -> ping
    close     `down -v` -> reset_check SCOPED to the project + "no process left on that GPU" -> the slot
              is free again. A dirty check or a dead agent retires the WHOLE instance (its other
              episodes fail and retry elsewhere); an instance whose slots stayed free for
              TB_G6_IDLE_RETIRE_S is terminated

Capacity: the launch walks cfg.instance_types x cfg.subnets (one per AZ) on InsufficientInstanceCapacity,
so g6 / g5 / g6e and every AZ are tried in order; TB_G6_CAPACITY_RESERVATION_ID targets an On-Demand
Capacity Reservation or (TB_G6_CAPACITY_BLOCK=1) a Capacity Block, which pins type + AZ.

Isolation is WEAKER than AgentCore's: episodes on one instance share its kernel and GPU driver (each in
its own container, own GPU, no host access — the m5's firewall + compose safety checks). Fine for
KernelBench, which only writes and runs a kernel; not the one-machine-per-session boundary.

Config: TB_G6_EC2 (JSON written by `deploy/g6_host.py bake-ami`, run.sh exports it). Knobs:
TB_G6_MAX_INSTANCES=16 (live instances), TB_G6_IDLE_RETIRE_S=900, TB_G6_BOOT_WAIT_S=900,
TB_G6_LAUNCH_WAIT_S=900 (keep re-sweeping the candidates this long while all are starved),
TB_G6_LAUNCH_CONC=2 (RunInstances calls in flight, process-wide),
TB_M5_IDLE_SHUTDOWN_S (the shared dead-man switch).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

from .docker_compose_sandbox import DockerComposeSandbox, agent_call
from .ec2_episode_sandbox import (_LAUNCHED, Ec2WarmSandbox, _current_loop, _ec2, _terminate_ids,
                                  episode_user_data)

G6EC2_PREFIX = "gpuec2:"

# Launch errors that mean "this type / AZ has nothing right now": try the next candidate.
_CAPACITY_ERRORS = ("InsufficientInstanceCapacity", "Unsupported", "InsufficientCapacity",
                    "InsufficientHostCapacity", "InsufficientReservedInstanceCapacity")


# Process-wide launch hygiene (a look-ahead opens dozens of episodes at once): at most TB_G6_LAUNCH_CONC
# RunInstances calls in flight, and a type/AZ that just answered "no capacity" is skipped by EVERY launcher
# for _STARVED_S. The first run without these got the account throttled (RequestLimitExceeded) within
# 2 minutes: 16 launchers x 18 candidates, back to back.
_LAUNCH_SEM = threading.BoundedSemaphore(int(os.environ.get("TB_G6_LAUNCH_CONC", "2")))
_STARVED: dict = {}                      # (type, az) -> monotonic time until which it is skipped
_STARVED_LOCK = threading.Lock()
_STARVED_S = 60.0
_THROTTLE_CODES = ("RequestLimitExceeded", "Throttling", "ThrottlingException")


def _starved(key) -> bool:
    with _STARVED_LOCK:
        return _STARVED.get(key, 0.0) > time.monotonic()


def _mark_starved(key) -> None:
    with _STARVED_LOCK:
        _STARVED[key] = time.monotonic() + _STARVED_S


def _launch_client(region: str):
    """RunInstances WITHOUT botocore retries: EC2 answers InsufficientInstanceCapacity with HTTP 500, which
    the retrying client re-sent up to 8 times with backoff (~55 s per starved candidate in the first probe)
    instead of moving on to the next type / AZ."""
    import boto3
    from botocore.config import Config

    return boto3.client("ec2", region_name=region, config=Config(retries={"max_attempts": 1, "mode": "standard"}))


def is_g6ec2_route(route: str) -> bool:
    return isinstance(route, str) and route.startswith(G6EC2_PREFIX)


def g6_config() -> dict:
    import json

    raw = os.environ.get("TB_G6_EC2", "").strip()
    if not raw or raw == '""':
        raise RuntimeError("a gpuec2: route needs TB_G6_EC2 (run.sh reads deploy/g6_host.json; build it with: "
                           "python3 deploy/g6_host.py image && python3 deploy/g6_host.py bake-ami)")
    return json.loads(raw)


_SHAPES: dict = {}
_SHAPES_LOCK = threading.Lock()


def instance_shape(region: str, itype: str) -> dict:
    """{gpus, vcpus, mem_mib} of an instance type (DescribeInstanceTypes, cached)."""
    with _SHAPES_LOCK:
        if itype in _SHAPES:
            return _SHAPES[itype]
    t = _ec2(region).describe_instance_types(InstanceTypes=[itype])["InstanceTypes"][0]
    shape = {"gpus": sum(g.get("Count", 0) for g in (t.get("GpuInfo") or {}).get("Gpus", [])) or 1,
             "vcpus": t["VCpuInfo"]["DefaultVCpus"], "mem_mib": t["MemoryInfo"]["SizeInMiB"]}
    with _SHAPES_LOCK:
        _SHAPES[itype] = shape
    return shape


def _emit(**kv) -> None:
    sys.stdout.write("[demo] " + " ".join(f"{k}={v}" for k, v in kv.items()) + "\n")
    sys.stdout.flush()


class _GpuInstance:
    def __init__(self, iid: str, ip: str, token: str, itype: str, az: str, gpus: int) -> None:
        self.iid, self.ip, self.token, self.itype, self.az, self.gpus = iid, ip, token, itype, az, gpus
        self.free: set[int] = set(range(gpus))
        self.held: dict[int, object] = {}      # gpu -> the event loop whose session holds it
        self.lost: set[int] = set()            # slots of a superseded loop: never reused
        self.draining = False                  # no new episodes; terminated once its slots are idle
        self.tasks: set[str] = set()
        self.episodes = 0
        self.launched = time.time()
        self.idle_since = time.time()


class GpuSlotPool:
    """Process-wide pool of warm g6 instances, handed out ONE GPU slot at a time (thread-safe: the train
    and eval event loops share it). As in the m5 WarmPool, every held slot and every launch in progress
    is charged to its event loop, so a superseded loop's holdings can be reclaimed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._insts: list[_GpuInstance] = []
        self._launching: dict = {}             # loop -> launches in progress

    def stats(self) -> dict:
        with self._lock:
            return {"g6_instances": len(self._insts),
                    "g6_slots_busy": sum(len(i.held) for i in self._insts),
                    "g6_slots": sum(i.gpus for i in self._insts)}

    def take(self, cap: int, idle_retire_s: float):
        """((instance, gpu) | None, may_launch, [instances to terminate])."""
        loop, now = _current_loop(), time.time()
        with self._lock:
            stale = [i for i in self._insts if not i.held and not i.draining and idle_retire_s > 0
                     and now - i.idle_since > idle_retire_s]
            self._insts = [i for i in self._insts if i not in stale]
            open_ = [i for i in self._insts if i.free and not i.draining]
            if open_:
                inst = min(open_, key=lambda i: (len(i.free), i.launched))   # pack the busiest first
                gpu = min(inst.free)
                inst.free.discard(gpu)
                inst.held[gpu] = loop
                return (inst, gpu), False, stale
            if len(self._insts) + sum(self._launching.values()) < cap:
                self._launching[loop] = self._launching.get(loop, 0) + 1
                return None, True, stale
            return None, False, stale

    def added(self, inst: _GpuInstance | None) -> int | None:
        """Register a freshly launched instance; the launcher takes its slot 0. Returns that gpu."""
        loop = _current_loop()
        with self._lock:
            if self._launching.get(loop, 0) > 0:
                self._launching[loop] -= 1
            if inst is None:
                return None
            inst.free.discard(0)
            inst.held[0] = loop
            self._insts.append(inst)
            return 0

    def free(self, inst: _GpuInstance, gpu: int) -> bool:
        """Slot verified clean. True if the instance should now be terminated (draining and idle)."""
        with self._lock:
            inst.held.pop(gpu, None)
            if gpu not in inst.lost:
                inst.free.add(gpu)
            if not inst.held:
                inst.idle_since = time.time()
            if inst.draining and not inst.held and inst in self._insts:
                self._insts.remove(inst)
                return True
            return False

    def drop(self, inst: _GpuInstance) -> None:
        with self._lock:
            if inst in self._insts:
                self._insts.remove(inst)

    def reclaim(self, loop) -> list[str]:
        """Forget every slot held by `loop` (a loop that will never close them). Their projects are still
        up, so those slots are never reused: an instance held ONLY by `loop` is returned for termination,
        one shared with live episodes is drained (terminated when they finish)."""
        gone = []
        with self._lock:
            self._launching.pop(loop, None)
            for i in list(self._insts):
                mine = [g for g, lp in i.held.items() if lp is loop]
                if not mine:
                    continue
                for g in mine:
                    i.held.pop(g)
                    i.lost.add(g)
                i.draining = True
                if not i.held:
                    self._insts.remove(i)
                    gone.append(i)
        return [i.iid for i in gone]

    def all_ids(self) -> list[str]:
        with self._lock:
            return [i.iid for i in self._insts]


_HI_WAITING = [0]       # opens a pool worker is blocked on, waiting for a slot (see __aenter__)
_POOL = GpuSlotPool()


def reclaim_g6(loop, region: str) -> None:
    ids = _POOL.reclaim(loop)
    if not ids:
        return
    _emit(g6ec2_reclaim=len(ids), reason="loop_superseded", ids=",".join(ids))
    try:
        _terminate_ids(region, ids)
    except Exception as e:  # noqa: BLE001 — the dead-man switch and gc are the backstop
        _emit(g6ec2_reclaim_terminate_failed=type(e).__name__)


def _terminate_at_exit() -> None:
    ids = _POOL.all_ids()
    if ids:
        try:
            _terminate_ids(os.environ.get("AWS_REGION", "us-west-2"), ids)
        except Exception:  # noqa: BLE001
            pass


import atexit as _atexit  # noqa: E402

_atexit.register(_terminate_at_exit)


class Ec2GpuSandbox(Ec2WarmSandbox):
    """ONE GPU slot of a warm g6 instance for ONE episode; scoped verified reset afterwards."""

    def __init__(self, route: str, *, region: str, workdir: str = "", session_id: str | None = None,
                 read_timeout: int | None = None) -> None:
        cfg = g6_config()
        task = route[len(G6EC2_PREFIX):]
        # The docker:// form only carries the task: the host is the slot's instance, set at open.
        super().__init__(f"docker://0.0.0.0:{int(cfg.get('agent_port', 18300))}/{task}", region=region,
                         workdir=workdir, session_id=session_id, read_timeout=read_timeout, cfg=cfg)
        self._ginst: _GpuInstance | None = None
        self._gpu: int | None = None
        self.sandbox_id = f"g6ec2-{self._episode}"

    def _label(self) -> str:
        return f"g6ec2:{self._task}"

    # ---- launch with capacity fallback: instance types x AZ subnets (or a reservation) -------------
    def _candidates(self) -> tuple[list[tuple[str, str, str]], dict]:
        c = self._cfg
        subnets = c.get("subnets") or [{"id": c["subnet"], "az": c.get("az", "?")}]
        extra: dict = {}
        crid = os.environ.get("TB_G6_CAPACITY_RESERVATION_ID", "").strip()
        if crid:   # a reservation fixes type + AZ: launch only there
            cr = _ec2(self._region).describe_capacity_reservations(
                CapacityReservationIds=[crid])["CapacityReservations"][0]
            extra["CapacityReservationSpecification"] = {"CapacityReservationTarget": {"CapacityReservationId": crid}}
            if os.environ.get("TB_G6_CAPACITY_BLOCK") == "1":
                extra["InstanceMarketOptions"] = {"MarketType": "capacity-block"}
            sn = [s for s in subnets if s["az"] == cr["AvailabilityZone"]]
            if not sn:
                raise RuntimeError(f"capacity reservation {crid} is in {cr['AvailabilityZone']}, which has no "
                                   f"g6 subnet (TB_G6_AZS at bake-ami time)")
            return [(cr["InstanceType"], sn[0]["id"], sn[0]["az"])], extra
        # Rotate the AZ order per launch so a fleet spreads (and one starved AZ isn't always tried first).
        k = int(time.time()) % len(subnets)
        subnets = subnets[k:] + subnets[:k]
        types = c.get("instance_types") or [c.get("instance_type", "g6.xlarge")]
        return [(t, s["id"], s["az"]) for t in types for s in subnets], extra

    def _launch_sync(self) -> tuple[str, str]:
        c = self._cfg
        cands, extra = self._candidates()
        end = time.monotonic() + float(os.environ.get("TB_G6_LAUNCH_WAIT_S", "900"))
        tries, sweeps = 0, 0
        while True:     # sweep every candidate; if all are starved, wait and sweep again
            got = self._sweep(c, cands, extra)
            sweeps += 1
            tries += got[1]
            if got[0] is not None:
                inst, itype, az, shape = got[0]
                break
            if self._abandoned or time.monotonic() > end:
                raise RuntimeError(f"no g6 capacity after {sweeps} sweep(s): {', '.join(got[2])}")
            time.sleep(min(30.0, 5.0 * sweeps))
        _LAUNCHED.add(inst["InstanceId"])
        if self._abandoned:
            _terminate_ids(self._region, [inst["InstanceId"]])
            raise RuntimeError(f"{inst['InstanceId']}: launched after its open was cancelled; terminated")
        self._launch_meta = {"instance_type": itype, "az": az, "gpus": shape["gpus"], "launch_tries": tries,
                             "launch_sweeps": sweeps, "vcpus": shape["vcpus"], "mem_mib": shape["mem_mib"]}
        return inst["InstanceId"], inst["PrivateIpAddress"]

    def _sweep(self, c: dict, cands, extra: dict):
        """One pass over the candidates: ((inst, type, az, shape) | None, calls made, capacity errors)."""
        import random

        from botocore.exceptions import ClientError

        errors, calls = [], 0
        for itype, subnet, az in cands:
            if self._abandoned:
                return None, calls, errors + ["cancelled"]
            if _starved((itype, az)):
                errors.append(f"{itype}/{az}:recently_starved")
                continue
            shape = instance_shape(self._region, itype)
            try:
                with _LAUNCH_SEM:
                    calls += 1
                    inst = self._run_instances(c, itype, subnet, shape, extra)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in _THROTTLE_CODES:          # back off; the caller's next sweep retries
                    errors.append(f"{itype}/{az}:{code}")
                    time.sleep(random.uniform(3.0, 10.0))
                    return None, calls, errors
                if code not in _CAPACITY_ERRORS:
                    raise
                _mark_starved((itype, az))
                errors.append(f"{itype}/{az}:{code}")
                continue
            return (inst, itype, az, shape), calls, errors
        return None, calls, errors

    def _run_instances(self, c: dict, itype: str, subnet: str, shape: dict, extra: dict) -> dict:
        return _launch_client(self._region).run_instances(
            ImageId=c["ami"], InstanceType=itype, MinCount=1, MaxCount=1,
            UserData=episode_user_data(self._token, {"ACS_MAX_PROJECTS": shape["gpus"]}),
            IamInstanceProfile={"Name": c["profile"]},
            NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": subnet, "Groups": [c["sg"]],
                                "AssociatePublicIpAddress": False}],
            BlockDeviceMappings=[{"DeviceName": c.get("root_device", "/dev/xvda"), "Ebs": {
                "VolumeSize": int(c.get("disk_gb", 100)), "VolumeType": "gp3", "DeleteOnTermination": True}}],
            MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 1, "HttpEndpoint": "enabled"},
            InstanceInitiatedShutdownBehavior="terminate",
            TagSpecifications=[{"ResourceType": "instance", "Tags": [
                {"Key": "Name", "Value": c.get("tag", "acs-tbh-g6ec2")},
                {"Key": "acs-task", "Value": self._task}]}],
            **extra)["Instances"][0]

    def _slot_limits(self, inst: _GpuInstance) -> dict:
        """This slot's share of the instance: 1/gpus of the vCPUs and ~90% of the memory."""
        shape = instance_shape(self._region, inst.itype)
        return {"gpu": self._gpu, "cpus": max(1, shape["vcpus"] // inst.gpus),
                "mem": f"{int(shape['mem_mib'] * 0.9 / inst.gpus)}m",
                "shm": os.environ.get("TB_G6_SHM", "2g")}

    async def __aenter__(self) -> "Ec2GpuSandbox":
        cap = int(os.environ.get("TB_G6_MAX_INSTANCES", "16"))
        idle_retire = float(os.environ.get("TB_G6_IDLE_RETIRE_S", "900"))
        t0, sent = time.perf_counter(), time.time()
        self._open_t0 = t0
        # Slots go to opens a worker is WAITING on first (`_tb_prio["hi"]`, set by the pool's claim; a
        # sandbox opened outside the pool counts as high): a look-ahead pre-open yields while one waits.
        # A high wait has a deadline, so a full pool fails the open instead of hanging the run forever.
        slot_wait = float(os.environ.get("TB_G6_SLOT_WAIT_S", "900"))
        hi_since, counted = None, False
        try:
            while True:
                hi = bool((getattr(self, "_tb_prio", None) or {"hi": True}).get("hi"))
                if hi and not counted:
                    _HI_WAITING[0] += 1
                    counted, hi_since = True, time.monotonic()
                if hi and time.monotonic() - hi_since > slot_wait:
                    raise RuntimeError(f"no EC2 GPU slot within {slot_wait:.0f}s (pool at {cap} instances, "
                                       f"{_HI_WAITING[0]} worker opens waiting)")
                if not hi and _HI_WAITING[0] > 0:
                    await asyncio.sleep(2)
                    continue
                got, may_launch, stale = _POOL.take(cap, idle_retire)
                for i in stale:
                    _emit(g6ec2_retire=i.iid, instance_type=i.itype, reason="idle", episodes=i.episodes,
                          alive_s=round(time.time() - i.launched, 1))
                    await asyncio.to_thread(_terminate_ids, self._region, [i.iid])
                if got is not None:
                    inst, gpu = got
                    try:   # an idle instance may have shut itself down (dead-man switch) or died
                        await asyncio.to_thread(lambda: agent_call(inst.ip, self._agent_port, {"op": "health"},
                                                                   timeout=10, token=inst.token))
                        break
                    except Exception:  # noqa: BLE001
                        _POOL.drop(inst)
                        _emit(g6ec2_retire=inst.iid, instance_type=inst.itype, reason="unhealthy_on_reuse",
                              episodes=inst.episodes, alive_s=round(time.time() - inst.launched, 1))
                        await asyncio.to_thread(_terminate_ids, self._region, [inst.iid])
                        continue
                if may_launch:
                    inst = gpu = None
                    break
                await asyncio.sleep(2)             # every slot busy and at the instance cap
        finally:
            if counted:
                _HI_WAITING[0] -= 1
        launched = inst is None
        agent_ok = not launched
        try:
            if launched:
                tl = time.perf_counter()
                self._instance_id, self._agent_host = await asyncio.to_thread(self._launch_sync)
                m = self._launch_meta
                inst = _GpuInstance(self._instance_id, self._agent_host, self._token, m["instance_type"], m["az"],
                                    m["gpus"])
                gpu = _POOL.added(inst)
                self._ginst, self._gpu = inst, gpu      # from here a failure retires it
                _emit(g6ec2_launch=inst.iid, instance_type=inst.itype, az=inst.az, gpus=inst.gpus,
                      launch_tries=m["launch_tries"], launch_sweeps=m["launch_sweeps"], launch_api_ms=round((time.perf_counter() - tl) * 1000.0, 1))
                await self._wait_agent(float(os.environ.get("TB_G6_BOOT_WAIT_S", "900")))
                await self._ensure_agent()
                agent_ok = True
            else:
                self._instance_id, self._agent_host, self._token = inst.iid, inst.ip, inst.token
            self._ginst, self._gpu = inst, gpu
            if self._task not in inst.tasks:       # task dir + ECR record, once per instance
                await self._upload()
                inst.tasks.add(self._task)
            self._up_extra = self._slot_limits(inst)
            await DockerComposeSandbox.__aenter__(self)     # up (image pre-pulled in the AMI) + ping
        except BaseException:
            self._abandoned = True                 # a launch still in flight terminates its own instance
            if self._ginst is None:                # the launch itself failed / was cancelled
                if launched:
                    _POOL.added(None)
                if self._instance_id:
                    await asyncio.to_thread(_terminate_ids, self._region, [self._instance_id])
            elif not agent_ok:                     # never booted: nothing to release
                await self._retire("boot_failed")
            else:   # release the slot like a close: clean -> reusable, else the instance is retired
                await self.__aexit__(None, None, None)
            raise
        inst.episodes += 1
        self.open_sent_epoch = sent
        self.open_reply.setdefault("boot", {}).update(
            instance_id=inst.iid, instance_type=inst.itype, gpu_slot=gpu, warm_reused=int(not launched),
            instance_episodes=inst.episodes, **_POOL.stats())
        self.demo_fields = {"warm_reused": int(not launched), "instance_episodes": inst.episodes,
                            "instance_type": inst.itype, "gpu_slot": gpu}
        return self

    async def _retire(self, why: str) -> None:
        inst, self._ginst = self._ginst, None
        self._abandoned = True
        if inst is None:
            if self._instance_id:            # launched but never registered
                await asyncio.to_thread(_terminate_ids, self._region, [self._instance_id])
            return
        _POOL.drop(inst)
        try:
            await asyncio.to_thread(_terminate_ids, self._region, [inst.iid])
        finally:
            _emit(g6ec2_retire=inst.iid, instance_type=inst.itype, reason=why, episodes=inst.episodes,
                  alive_s=round(time.time() - inst.launched, 1))

    async def __aexit__(self, *exc) -> None:
        inst, gpu = self._ginst, self._gpu
        if inst is None:
            return
        t0, status = time.perf_counter(), "ok"
        call = lambda p, t=330: agent_call(inst.ip, self._agent_port, p, timeout=t, token=inst.token)  # noqa: E731
        try:
            rep = await asyncio.to_thread(call, {"op": "down", "episode": self._episode})
            chk = await asyncio.to_thread(call, {"op": "reset_check", "episode": self._episode, "gpu": gpu}, 600)
            if rep.get("ok") and chk.get("clean"):
                self._ginst = None
                if _POOL.free(inst, gpu):          # a drained instance whose last live slot just ended
                    _emit(g6ec2_retire=inst.iid, instance_type=inst.itype, reason="drained",
                          episodes=inst.episodes, alive_s=round(time.time() - inst.launched, 1))
                    await asyncio.to_thread(_terminate_ids, self._region, [inst.iid])
            else:
                status = "reset_dirty"
                _emit(g6ec2_reset_dirty=inst.iid, gpu=gpu,
                      problems=str(chk.get("problems"))[:200].replace(" ", "_"))
                await self._retire("reset_dirty")
        except Exception as e:  # noqa: BLE001 — an unreachable/failed agent: never reuse that instance
            status = type(e).__name__
            await self._retire(status)
        sys.stdout.write(f"[sbx-op] ts={time.time():.3f} op=stop_session ms={(time.perf_counter() - t0) * 1000.0:.1f} "
                         f"arn={self._label()} session={self._session_id} status={status}\n")
        sys.stdout.flush()
