#!/usr/bin/env python3
"""Create the warmstart probe's sandbox Runtime: a managed arm64 microVM Runtime with a SHORT idle timeout.

The image is the ordinary stdlib CPU sandbox, the shared acs-sandbox-cpu:latest already in ECR (no build). The only thing special
is `lifecycleConfiguration.idleRuntimeSessionTimeout`: IDLE_S seconds (default 60, the API minimum)
instead of the 900 s default. That makes the heartbeat experiment in warmstart_demo.py cheap: a
session left silent for ~2 min is reclaimed by AgentCore (the "no heartbeat" arm), while the
heartbeat arm's no-op invokes every IDLE_S/2 s keep it — and its files — alive.

The demo's vendored deploy/agentcore_deploy.py is identical in every demo and its managed_runtime() takes no
lifecycle settings, so this script calls create_agent_runtime itself (same role, same network).

    python3 warmstart/deploy_warmstart.py            # or: bash warmstart/run_warmstart.sh
    # prints: export AGENTCORE_SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/acs_warmstart_cpu-...

Image: IMAGE_URI_cpu, else IMAGE_URI, else the
shared <acct>.dkr.ecr.<region>.amazonaws.com/acs-sandbox-cpu:latest (same server, already pushed).
Idempotent: an existing Runtime of the same name is reused; FORCE_RECREATE=1 deletes it first;
--cleanup deletes it.
Real AWS mutation: one managed serverless Runtime (billed per session).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deploy"))  # the demo's vendored agentcore_deploy.py

import boto3  # noqa: E402

import agentcore_deploy as d  # noqa: E402

NAME = f"{d.PREFIX}_warmstart_cpu"
IDLE_S = int(os.environ.get("IDLE_S", "60"))


def _image_uri() -> str:
    uri = os.environ.get("IMAGE_URI_cpu") or os.environ.get("IMAGE_URI")
    return uri or f"{d.account()}.dkr.ecr.{d.REGION}.amazonaws.com/{d.PREFIX}-sandbox-cpu:latest"


def main() -> int:
    if "--cleanup" in sys.argv[1:]:
        d.delete_runtime(NAME)
        return 0
    control = boto3.client("bedrock-agentcore-control", region_name=d.REGION)
    if os.environ.get("FORCE_RECREATE") == "1":
        d.delete_runtime(NAME)
    arn = d._reuse_or_clear(control, NAME)
    if arn:
        d.err(f"reusing runtime {NAME} (FORCE_RECREATE=1 to pick up a new image / IDLE_S)")
    else:
        uri = _image_uri()
        d.err(f"creating {NAME} from {uri} with idleRuntimeSessionTimeout={IDLE_S}s")
        resp = control.create_agent_runtime(
            agentRuntimeName=NAME,
            roleArn=d.exec_role(f"{d.PREFIX}-exec-role"),
            agentRuntimeArtifact={"containerConfiguration": {"containerUri": uri}},
            protocolConfiguration={"serverProtocol": "HTTP"},
            networkConfiguration={"networkMode": "PUBLIC"},
            environmentVariables={"AWS_REGION": d.REGION, "SANDBOX_WORKDIR": "/workspace"},
            lifecycleConfiguration={"idleRuntimeSessionTimeout": IDLE_S, "maxLifetime": 3600},
        )
        d._wait_runtime(control, resp["agentRuntimeId"])
        arn = resp["agentRuntimeArn"]
    rid = arn.rsplit("/", 1)[-1]
    lc = control.get_agent_runtime(agentRuntimeId=rid).get("lifecycleConfiguration", {})
    d.err(f"{NAME} lifecycle: {lc}")
    print(f"export AGENTCORE_SANDBOX_RUNTIME_ARN={arn}")
    print(f"export WARMSTART_IDLE_S={lc.get('idleRuntimeSessionTimeout', IDLE_S)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
