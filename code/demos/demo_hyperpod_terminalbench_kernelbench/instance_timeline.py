#!/usr/bin/env python3
"""Record every EC2 instance a run launched (m5 episode/warm hosts AND the g6 Instances AgentCore runs).

    python3 instance_timeline.py logs/train48-hybrid_20261005_005629.log   # -> <log>.instances.csv

The g6 Instances behind the KernelBench Runtime are launched by AgentCore through acs-operator-role
in OUR account, so their RunInstances / TerminateInstances calls are in our CloudTrail even though
describe-instances does not list them. The window is the run's sessions.csv span (+/- slack, since
an Instance outlives its session until AgentCore reaps it). CloudTrail lookup keeps 90 days.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys

import boto3

KEEP = {"m5.xlarge", "g6.xlarge"}


def main(log: str, region: str = os.environ.get("AWS_REGION", "us-west-2")) -> None:
    base = log[:-4] if log.endswith(".log") else log
    with open(base + ".sessions.csv") as f:
        rows = list(csv.DictReader(f))
    t0 = min(float(r["start_ts"]) for r in rows) - 600
    t1 = max(float(r["end_ts"]) for r in rows) + 3600
    utc = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc)  # noqa: E731
    ct = boto3.client("cloudtrail", region_name=region)
    launch, term = {}, {}
    for name in ("RunInstances", "TerminateInstances"):
        for page in ct.get_paginator("lookup_events").paginate(
                LookupAttributes=[{"AttributeKey": "EventName", "AttributeValue": name}],
                StartTime=utc(t0), EndTime=utc(t1)):
            for e in page["Events"]:
                ev, ts = json.loads(e["CloudTrailEvent"]), e["EventTime"].timestamp()
                for it in ((ev.get("responseElements") or {}).get("instancesSet") or {}).get("items", []):
                    if name == "RunInstances":
                        if it.get("instanceType") in KEEP:
                            launch[it["instanceId"]] = (ts, it["instanceType"], ev["userIdentity"].get("arn", ""))
                    else:
                        term[it["instanceId"]] = min(ts, term.get(it["instanceId"], ts))
    out = base + ".instances.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "type", "launch_ts", "terminate_ts", "launched_by"])
        for iid, (ts, ty, who) in sorted(launch.items(), key=lambda kv: kv[1][0]):
            w.writerow([iid, ty, f"{ts:.0f}", f"{term[iid]:.0f}" if iid in term else "", who.split("/")[-2] if "/" in who else who])
    n = {ty: sum(v[1] == ty for v in launch.values()) for ty in KEEP}
    print(f"{out}: {n}, {sum(i not in term for i in launch)} without a terminate in the window")


if __name__ == "__main__":
    main(*sys.argv[1:])
