#!/usr/bin/env python3
"""Probe the AgentCore microVM image-size cap (quota L-0A9E32B3, 2048 MB) and cold boot vs image size.

The quota text says "maximum container size" without saying compressed or on-disk, and our deploy
scripts only ever gated on it themselves (compressed ECR size > 2048 MB -> skipped, never sent to
AWS). This probe asks AWS directly. It builds arm64 variants of one tiny, already-wrapped
Terminal-Bench image (acs_http_server with boot timestamps) plus a filler layer:

    zeros3g    3 GB of zeros      -> compressed ~ few MB, on disk ~3 GB   (does the cap count on-disk?)
    rand2300   2300 MB random     -> compressed ~ on disk ~2.3 GB         (does the cap count compressed?)
    rand1500   1500 MB random     -> ~1.5 GB                              (cold boot at SWE-bench size)
    rand1000   1000 MB random     -> ~1.0 GB                              (cold boot at SWE-bench size)
    rand1890..rand2000             -> 1951 / 2001 / 2041 / 2061 MB         (is the cap 2048 MiB or 2048 MB?)

For each: push to ECR (records imageSizeInBytes), CreateAgentRuntime (VPC, like the TB pool), wait
READY / *FAILED (records failureReason), then open fresh sessions — one alone, then a burst of
PROBE_BURST at once — and time them (open_s, pull_boot_s, warm_hit, as probe_coldstart.py).
Writes swebench/capprobe.csv. Runtimes + the ECR repo are deleted at the end unless CAPPROBE_KEEP=1.

    python3 probe_image_cap.py                        # all variants
    CAPPROBE_VARIANTS=zeros3g,rand2300 python3 probe_image_cap.py
"""
from __future__ import annotations

import asyncio
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "deploy"))
import agentcore_deploy as d  # noqa: E402
from agentcore_slime_demo.agentcore_runtime_sandbox import AgentCoreRuntimeSandbox  # noqa: E402

VARIANTS = {"zeros3g": ("zero", 3072), "rand2300": ("rand", 2300), "rand1500": ("rand", 1500),
            "rand1000": ("rand", 1000),
            # bracket the exact threshold: compressed = filler + ~61.5 MB base
            "rand1890": ("rand", 1890), "rand1940": ("rand", 1940), "rand1980": ("rand", 1980),
            "rand2000": ("rand", 2000)}
BASE_TASK = os.environ.get("CAPPROBE_BASE", "adaptive-rejection-sampler")   # 61 MB TB image
REPO = f"{d.PREFIX}-capprobe"
BURST = int(os.environ.get("PROBE_BURST", "4"))
WORK = Path(os.environ.get("CAPPROBE_WORK", "/tmp/acs_capprobe"))
OUT = HERE / "swebench" / os.environ.get("CAPPROBE_OUT", "capprobe.csv")


def base_uri() -> str:
    tb = HERE.parent / "demo_cpu_tasks" / "terminalbench" / "tasks" / "images.csv"
    row = next(r for r in csv.DictReader(tb.open()) if r["task"] == BASE_TASK)
    return f"{d.account()}.dkr.ecr.{d.REGION}.amazonaws.com/{row['repo']}@{row['digest']}"


def build_push(name: str, kind: str, mb: int, base: str, registry: str) -> dict:
    ctx = WORK / name
    ctx.mkdir(parents=True, exist_ok=True)
    filler = ctx / "filler.bin"
    if not filler.exists() or filler.stat().st_size != mb * 2**20:
        src = "/dev/urandom" if kind == "rand" else "/dev/zero"
        subprocess.run(f"head -c {mb * 2**20} {src} > {filler}", shell=True, check=True)
    (ctx / "Dockerfile").write_text(f"FROM {base}\nCOPY filler.bin /opt/filler.bin\n")
    uri = f"{registry}/{REPO}:{name}"
    t0 = time.perf_counter()
    d.sh(f"docker buildx build --builder acsbuilder --platform linux/arm64 --provenance=false "
         f"-t {uri} --push {ctx} >{ctx}/build.log 2>&1")
    det = boto3.client("ecr", region_name=d.REGION).describe_images(
        repositoryName=REPO, imageIds=[{"imageTag": name}])["imageDetails"][0]
    return {"variant": name, "filler_mb": mb, "filler": kind,
            "ecr_compressed_mb": round(det["imageSizeInBytes"] / 2**20, 1),
            "push_s": round(time.perf_counter() - t0, 1), "uri": f"{registry}/{REPO}@{det['imageDigest']}"}


def create_runtime(row: dict, xrole: str, network: dict) -> None:
    control = boto3.client("bedrock-agentcore-control", region_name=d.REGION)
    name = f"{d.PREFIX}_capprobe_{row['variant']}"
    d._reuse_or_clear(control, name)
    t0 = time.perf_counter()
    try:
        r = control.create_agent_runtime(
            agentRuntimeName=name, agentRuntimeArtifact={"containerConfiguration": {"containerUri": row["uri"]}},
            roleArn=xrole, protocolConfiguration={"serverProtocol": "HTTP"}, networkConfiguration=network,
            lifecycleConfiguration=d.runtime_lifecycle(d.MICROVM_MAX_LIFETIME),
            environmentVariables={"AWS_REGION": d.REGION})
    except Exception as e:  # noqa: BLE001 — a synchronous rejection IS the answer we're after
        row.update(create="rejected", reason=f"{type(e).__name__}: {e}"[:300])
        return
    row.update(runtime_id=r["agentRuntimeId"], arn=r["agentRuntimeArn"])
    while True:
        st = control.get_agent_runtime(agentRuntimeId=r["agentRuntimeId"])
        if st["status"] == "READY" or "FAILED" in st["status"]:
            break
        time.sleep(10)
    row.update(create=st["status"], reason=(st.get("failureReason") or "")[:300],
               create_s=round(time.perf_counter() - t0, 1))


async def open_one(arn: str) -> dict:
    sb = AgentCoreRuntimeSandbox(runtime_arn=arn, region=d.REGION, workdir="")
    t0, res = time.time(), {}
    try:
        await sb.__aenter__()
        boot = (sb.open_reply or {}).get("boot") or {}
        sent, srv = sb.open_sent_epoch, boot.get("server_start_epoch")
        res["open_s"] = round(sb.open_recv_epoch - sent, 2)
        if srv:
            res["pull_boot_s"], res["warm_hit"] = round(max(0.0, srv - sent), 2), int(srv < sent)
    except Exception as e:  # noqa: BLE001
        res.update(error=f"{type(e).__name__}: {e}"[:200], open_s=round(time.time() - t0, 2))
    finally:
        try:
            await sb.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
    return res


async def open_sessions(row: dict) -> None:
    first = [await open_one(row["arn"])]
    burst = await asyncio.gather(*[open_one(row["arn"]) for _ in range(BURST)])
    opens = first + list(burst)
    fmt = lambda r: (r.get("error") or f"{r['open_s']}s" + (  # noqa: E731
        f"/boot{r['pull_boot_s']}s/{'warm' if r.get('warm_hit') else 'cold'}" if "pull_boot_s" in r else ""))
    row["sessions_ok"] = f"{sum('error' not in r for r in opens)}/{len(opens)}"
    row["opens"] = " ".join(fmt(r) for r in opens)
    cold = [r["pull_boot_s"] for r in opens if r.get("warm_hit") == 0 and "pull_boot_s" in r]
    row["cold_pull_boot_s_max"] = max(cold) if cold else ""
    row["cold_n"] = len(cold)


def cleanup(rows: list[dict]) -> None:
    control = boto3.client("bedrock-agentcore-control", region_name=d.REGION)
    for r in rows:
        if r.get("runtime_id"):
            try:
                control.delete_agent_runtime(agentRuntimeId=r["runtime_id"])
            except Exception as e:  # noqa: BLE001
                d.err(f"delete {r['variant']}: {e}")
    try:
        boto3.client("ecr", region_name=d.REGION).delete_repository(repositoryName=REPO, force=True)
    except Exception as e:  # noqa: BLE001
        d.err(f"delete repo {REPO}: {e}")


def main() -> None:
    names = [v for v in os.environ.get("CAPPROBE_VARIANTS", ",".join(VARIANTS)).split(",") if v]
    registry = d.ensure_ecr(REPO).rsplit("/", 1)[0]
    d.sh(f"aws ecr get-login-password --region {d.REGION} | docker login -u AWS --password-stdin {registry} >/dev/null")
    base = base_uri()
    rows = []
    for n in names:
        rows.append(build_push(n, *VARIANTS[n], base, registry))
        d.err(f"[push] {rows[-1]}")
    xrole = d.exec_role(f"{d.PREFIX}-exec-role")
    network = d.vpc_network("nat")
    d.open_dialback(network["networkModeConfig"]["securityGroups"][0])
    for r in rows:
        create_runtime(r, xrole, network)
        d.err(f"[create] {r['variant']}: {r.get('create')} {r.get('reason', '')}")
    for r in rows:
        if r.get("create") == "READY":
            asyncio.run(open_sessions(r))
            d.err(f"[open] {r['variant']}: {r['sessions_ok']}  {r['opens']}")
    OUT.parent.mkdir(exist_ok=True)
    fields = ["variant", "filler", "filler_mb", "ecr_compressed_mb", "create", "reason", "create_s",
              "sessions_ok", "cold_n", "cold_pull_boot_s_max", "opens", "push_s", "uri", "arn", "runtime_id"]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")
    for r in rows:
        print(f"{r['variant']:9} ecr={r['ecr_compressed_mb']:>7} MB  create={r.get('create')}  "
              f"sessions={r.get('sessions_ok', '-')}  {r.get('opens', r.get('reason', ''))}")
    if os.environ.get("CAPPROBE_KEEP") != "1":
        cleanup(rows)


if __name__ == "__main__":
    main()
