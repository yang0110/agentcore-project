#!/usr/bin/env python3
"""Provision demo_cpu_branching's CLI-capable CPU sandbox Runtime (real Claude Code CLI baked in).

Builds ONE custom sandbox image (node22 + @anthropic-ai/claude-code baked, see Dockerfile.sandbox.cli)
for arm64 (the managed serverless fleet is arm64-only), pushes it to its own ECR repo, and registers
ONE MANAGED AgentCore Runtime in networkMode=VPC.

Unlike the imagepool demo, branching's CLI must DIAL BACK to slime's AnthropicAdapter running on THIS
training box (the CLI issues Anthropic API calls that the adapter proxies to the local policy). That
dial-back is the ONLY inbound path we open, and it stays WITHIN THE VPC with NO public port:

  * networkMode=VPC (private egress via the shared VPC endpoints; no public network).
  * A single SG->SG ingress rule: allow TCP ADAPTER_PORT (default 18021) on the HOST security group
    FROM SOURCE = the runtime security group. Source is a security-group id, never a CIDR — so the
    port is reachable only from the sandbox runtime's ENIs inside this VPC, never from the internet.

Reuses the vendored deploy/agentcore_deploy.py helpers (account/ensure_ecr/_ensure_buildx/exec_role/
managed_runtime/default_vpc/ensure_vpc_endpoints) so it shares the SAME exec role + buildx builder +
VPC endpoints as the rest of the fleet. Idempotent: existing repo/Runtime/SG-rule are reused;
FORCE_RECREATE=1 deletes the Runtime first so it picks up a new image/env.

    python3 deploy_branching.py
    # prints: export AGENTCORE_SANDBOX_RUNTIME_ARN=<cli-runtime-arn>
    #         export ADAPTER_PUBLIC_HOST=<this box's private VPC IP>
    python3 deploy_branching.py --cleanup     # 6_cleanup.sh: delete the Runtime AND revoke the SG rule

Env: HOST_SG (host security group to open 18021 on; auto-detected via IMDS if unset),
     ADAPTER_PORT (default 18021), SKIP_BUILD=1 to reuse the pushed :latest and only (re)register.
Real AWS mutation: 1 ECR repo + image, 1 managed serverless Runtime, 1 SG ingress rule (--cleanup undoes
the last two).
"""
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.abspath(os.path.join(HERE, ".."))          # code/demos/demo_cpu_branching
sys.path.insert(0, HERE)  # this demo's vendored deploy/agentcore_deploy.py (no code/deploy dependency)

import agentcore_deploy as d  # noqa: E402

REGION = d.REGION
PREFIX = d.PREFIX
REPO = f"{PREFIX}-sandbox-cli"
DOCKERFILE = "Dockerfile.sandbox.cli"
PLATFORM = "linux/arm64"
ADAPTER_PORT = int(os.environ.get("ADAPTER_PORT", "18021"))


def _imds(path: str) -> str:
    """Best-effort IMDSv2 lookup (returns '' on failure)."""
    try:
        tok = urllib.request.urlopen(urllib.request.Request(
            "http://169.254.169.254/latest/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"}), timeout=2).read().decode()
        req = urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{path}",
                                     headers={"X-aws-ec2-metadata-token": tok})
        return urllib.request.urlopen(req, timeout=2).read().decode().strip()
    except Exception:
        return ""


def build_push() -> str:
    acct = d.account()
    registry = f"{acct}.dkr.ecr.{REGION}.amazonaws.com"
    uri = d.ensure_ecr(REPO) + ":latest"
    if os.environ.get("SKIP_BUILD") == "1":
        d.err(f"SKIP_BUILD=1 — reusing already-pushed {uri}")
        return uri
    d._ensure_buildx()
    d.sh(f"aws ecr get-login-password --region {REGION} | "
         f"docker login --username AWS --password-stdin {registry}")
    d.sh(f"docker buildx build --builder {d.BUILDX_BUILDER} --platform {PLATFORM} "
         f"--provenance=false --sbom=false -f {HERE}/{DOCKERFILE} -t {uri} --push {DEMO_DIR}")
    d.err(f"pushed {uri} ({PLATFORM})")
    return uri


def _host_sg() -> str:
    """HOST_SG, else the first security group on this box's primary ENI (via IMDS)."""
    host_sg = os.environ.get("HOST_SG", "")
    if not host_sg:
        # primary-ENI security groups; take the first (inbound rules are the union across attached SGs)
        mac = _imds("network/interfaces/macs/").splitlines()[0].rstrip("/") if _imds("network/interfaces/macs/") else ""
        sgids = _imds(f"network/interfaces/macs/{mac}/security-group-ids") if mac else ""
        host_sg = sgids.splitlines()[0] if sgids else ""
    if not host_sg:
        d.die("could not determine HOST_SG (set HOST_SG=sg-... to the host security group to open "
              f"tcp {ADAPTER_PORT} from the runtime SG)")
    return host_sg


def open_adapter_ingress(runtime_sg: str) -> None:
    """Add the ONE SG->SG ingress rule: host SG allows TCP ADAPTER_PORT FROM the runtime SG.

    Source is the runtime security-group id (never a CIDR): the adapter port is reachable only from
    the sandbox runtime's ENIs inside this VPC, never publicly. Idempotent (ignore Duplicate)."""
    host_sg = _host_sg()
    import subprocess
    rule = (f"aws ec2 authorize-security-group-ingress --region {REGION} --group-id {host_sg} "
            f"--protocol tcp --port {ADAPTER_PORT} --source-group {runtime_sg}")
    d.err(f"$ {rule}")
    p = subprocess.run(rule, shell=True, capture_output=True, text=True)  # tolerate Duplicate
    if p.returncode == 0:
        d.err(f"opened ingress tcp {ADAPTER_PORT} on {host_sg} from runtime SG {runtime_sg} (SG->SG, in-VPC only)")
    elif "InvalidPermission.Duplicate" in p.stderr or "already exists" in p.stderr:
        d.err(f"ingress tcp {ADAPTER_PORT} on {host_sg} from {runtime_sg} already present")
    else:
        d.err(f"WARN: could not add ingress (rc={p.returncode}): {p.stderr.strip()}")


def revoke_adapter_ingress(runtime_sg: str) -> None:
    """Undo open_adapter_ingress (6_cleanup.sh). Idempotent: a missing rule is not an error."""
    import subprocess
    host_sg = _host_sg()
    rule = (f"aws ec2 revoke-security-group-ingress --region {REGION} --group-id {host_sg} "
            f"--protocol tcp --port {ADAPTER_PORT} --source-group {runtime_sg}")
    d.err(f"$ {rule}")
    p = subprocess.run(rule, shell=True, capture_output=True, text=True)
    if p.returncode == 0:
        d.err(f"revoked ingress tcp {ADAPTER_PORT} on {host_sg} from runtime SG {runtime_sg}")
    elif "InvalidPermission.NotFound" in p.stderr:
        d.err(f"ingress tcp {ADAPTER_PORT} on {host_sg} from {runtime_sg} already absent")
    else:
        d.err(f"WARN: could not revoke ingress (rc={p.returncode}): {p.stderr.strip()}")


def cleanup() -> int:
    d.delete_runtime(f"{PREFIX}_sandbox_cli")
    _, _, runtime_sg = d.default_vpc()
    revoke_adapter_ingress(runtime_sg)
    return 0


def main() -> int:
    if "--cleanup" in sys.argv[1:]:
        return cleanup()
    xrole = d.exec_role(f"{PREFIX}-exec-role")
    env = {"AWS_REGION": REGION, "SANDBOX_WORKDIR": "/workspace"}

    vpc_id, subnets, runtime_sg = d.default_vpc()
    d.err(f"default VPC {vpc_id}; runtime SG {runtime_sg}")
    ep_subnets = d.ensure_vpc_endpoints(vpc_id, subnets, runtime_sg)
    if not ep_subnets:
        d.die("no subnet is served by all required VPC endpoints; cannot place VPC-mode runtime")
    vpc_net = {"networkMode": "VPC",
               "networkModeConfig": {"securityGroups": [runtime_sg], "subnets": ep_subnets}}

    uri = build_push()
    if os.environ.get("FORCE_RECREATE") == "1":
        d.delete_runtime(f"{PREFIX}_sandbox_cli")   # waits until the name is free
    arn = d.managed_runtime(f"{PREFIX}_sandbox_cli", uri, xrole, env, vpc_net)
    d.err(f"cli sandbox runtime -> {arn}")

    open_adapter_ingress(runtime_sg)
    host_ip = _imds("local-ipv4") or "<this-box-private-vpc-ip>"

    print("\nexport AGENTCORE_SANDBOX_RUNTIME_ARN='" + arn + "'")
    print("export ADAPTER_PUBLIC_HOST='" + host_ip + "'")
    print(f"# CLI sandbox Runtime ready; adapter dial-back allowed on tcp {ADAPTER_PORT} (SG->SG, in-VPC)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
