#!/usr/bin/env python3
"""demo_cpu_bigimage — safe-networking preflight for an AgentCore Runtime dial-back deploy.

This is the checklist in ../../docs/agentcore-networking.md turned into executable checks. It reads a
JSON description of your intended networking and FAILS loudly on the unsafe patterns (public ingress,
CIDR-based rules where an SG reference belongs, missing PrivateLink endpoints, uncovered AZs, …) so
you catch them BEFORE `CreateAgentRuntime` — where a VPC misconfig shows up only as an opaque
`CREATE_FAILED / statusReason=null / 0-byte log group`.

Offline by default (no AWS, no deps): it validates a config file. Ship your real config as JSON, or
start from the built-in samples:

    python3 netcheck.py --write-samples          # writes netcheck_sample_{good,bad}.json here
    python3 netcheck.py                           # checks the GOOD sample -> exits 0
    python3 netcheck.py --config netcheck_sample_bad.json   # -> prints FAILs, exits 1
    python3 netcheck.py --config my_deploy.json

Config shape (see the samples): a Runtime that needs the adapter dial-back must be networkMode=VPC,
reach the adapter via an SG->SG rule (never a CIDR / 0.0.0.0/0), sit behind the required PrivateLink
endpoints with PrivateDnsEnabled, expose tcp/443 to the VPC CIDR on the endpoint SG, use one subnet
per AZ, and only use AZs covered by EVERY endpoint.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Interface endpoints + the S3 GATEWAY endpoint a VPC-mode Runtime must reach (see the networking doc).
REQUIRED_ENDPOINTS = {"ecr.api", "ecr.dkr", "logs", "sts", "bedrock-agentcore", "s3"}

_GOOD = {
    "network_mode": "VPC",
    "adapter": {"host": "PRIVATE-IP-f99864bc", "port": 18021},
    "runtime_security_group": "sg-runtime",
    "adapter_ingress_rules": [
        {"port": 18021, "source_type": "security_group", "source": "sg-runtime"},
    ],
    "vpc_endpoints": sorted(REQUIRED_ENDPOINTS),
    "private_dns_enabled": True,
    "endpoint_sg_ingress": [{"port": 443, "source_type": "cidr", "source": "172.31.0.0/16"}],
    "subnets": [
        {"id": "subnet-a", "az": "us-west-2a"},
        {"id": "subnet-b", "az": "us-west-2b"},
    ],
    "endpoint_az_coverage": {
        "bedrock-agentcore": ["us-west-2a", "us-west-2b", "us-west-2c"],
        "ecr.api": ["us-west-2a", "us-west-2b", "us-west-2c", "us-west-2d"],
    },
}

_BAD = {
    "network_mode": "PUBLIC",                      # FAIL: dial-back needs VPC
    "adapter": {"host": "54.190.12.7", "port": 18021},   # FAIL: public adapter IP
    "runtime_security_group": "sg-runtime",
    "adapter_ingress_rules": [
        {"port": 18021, "source_type": "cidr", "source": "0.0.0.0/0"},   # FAIL: world-open
    ],
    "vpc_endpoints": ["ecr.api", "ecr.dkr", "logs"],   # FAIL: missing sts/bedrock-agentcore/s3
    "private_dns_enabled": False,                       # FAIL
    "endpoint_sg_ingress": [{"port": 443, "source_type": "cidr", "source": "0.0.0.0/0"}],  # FAIL
    "subnets": [
        {"id": "subnet-a", "az": "us-west-2a"},
        {"id": "subnet-b", "az": "us-west-2a"},        # FAIL: two subnets in one AZ
        {"id": "subnet-d", "az": "us-west-2d"},        # FAIL: AZ not covered by bedrock-agentcore
    ],
    "endpoint_az_coverage": {
        "bedrock-agentcore": ["us-west-2a", "us-west-2b", "us-west-2c"],
    },
}


def _is_private(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False  # a hostname — can't verify here; treated as "not provably private"


def _world_open(source: str) -> bool:
    return source.strip() in ("0.0.0.0/0", "::/0")


class Report:
    def __init__(self) -> None:
        self.fail: list[str] = []
        self.warn: list[str] = []
        self.ok: list[str] = []

    def check(self, cond: bool, ok_msg: str, fail_msg: str) -> None:
        (self.ok if cond else self.fail).append(ok_msg if cond else fail_msg)


def validate(cfg: dict) -> Report:
    r = Report()
    needs_dialback = bool(cfg.get("adapter"))
    mode = str(cfg.get("network_mode", "")).upper()

    if not needs_dialback:
        r.ok.append("No adapter dial-back configured -> pull-only (IAM/SigV4), no inbound to check.")
        return r

    # 1. dial-back requires VPC
    r.check(mode == "VPC",
            "networkMode=VPC (dial-back has no internet route; reaches AWS via PrivateLink).",
            f"networkMode is {mode or 'unset'} but a dial-back adapter is configured -> must be VPC.")

    # 2. adapter reached via SG->SG, never CIDR / world-open
    rt_sg = cfg.get("runtime_security_group")
    for rule in cfg.get("adapter_ingress_rules", []) or []:
        st, src, port = rule.get("source_type"), str(rule.get("source", "")), rule.get("port")
        if _world_open(src):
            r.fail.append(f"adapter ingress on :{port} is WORLD-OPEN ({src}) -> use an SG reference.")
        elif st != "security_group":
            r.fail.append(f"adapter ingress on :{port} is a {st} ({src}) -> use SG->SG (source = the Runtime SG).")
        elif rt_sg and src != rt_sg:
            r.warn.append(f"adapter ingress on :{port} allows SG {src}, not the Runtime SG {rt_sg} — confirm intended.")
        else:
            r.ok.append(f"adapter ingress on :{port} is SG->SG from {src} (least privilege).")
    if not cfg.get("adapter_ingress_rules"):
        r.fail.append("no adapter_ingress_rules -> the sandbox cannot reach the adapter (or you forgot the SG rule).")

    # adapter host should be a private IP
    host = cfg["adapter"].get("host", "")
    r.check(_is_private(host),
            f"adapter host {host} is a private address.",
            f"adapter host {host} is NOT a provably-private IP -> use the private VPC address.")

    # 3. required PrivateLink endpoints present
    have = set(cfg.get("vpc_endpoints", []) or [])
    missing = REQUIRED_ENDPOINTS - have
    r.check(not missing,
            f"all required PrivateLink endpoints present ({', '.join(sorted(REQUIRED_ENDPOINTS))}).",
            f"missing PrivateLink endpoints: {', '.join(sorted(missing))} "
            f"(a VPC-mode Runtime can't pull its image / phone home without these).")

    # 4. private DNS
    r.check(bool(cfg.get("private_dns_enabled")),
            "PrivateDnsEnabled=True.",
            "PrivateDnsEnabled is not True -> in-VPC name resolution to the endpoints won't work.")

    # 5. endpoint SG must allow tcp/443 from the VPC CIDR (not world)
    ep443 = [x for x in (cfg.get("endpoint_sg_ingress") or []) if x.get("port") == 443]
    if not ep443:
        r.fail.append("endpoint SG has no tcp/443 ingress -> ECR/API calls hang VPC-wide (PrivateDns override).")
    for x in ep443:
        src = str(x.get("source", ""))
        if _world_open(src):
            r.fail.append(f"endpoint SG tcp/443 is WORLD-OPEN ({src}) -> restrict to the VPC CIDR.")
        else:
            r.ok.append(f"endpoint SG allows tcp/443 from {src}.")

    # 6. one subnet per AZ + AZ covered by every endpoint
    subnets = cfg.get("subnets", []) or []
    azs = [s.get("az") for s in subnets]
    dupes = {a for a in azs if azs.count(a) > 1}
    r.check(not dupes,
            "at most one subnet per AZ.",
            f"multiple subnets in the same AZ ({', '.join(sorted(dupes))}) -> interface endpoints reject "
            "this (DuplicateSubnetsInSameZone).")

    coverage = cfg.get("endpoint_az_coverage", {}) or {}
    if coverage:
        for az in sorted(set(azs)):
            uncovered = [ep for ep, cov in coverage.items() if az not in cov]
            if uncovered:
                r.fail.append(f"AZ {az} is NOT served by: {', '.join(sorted(uncovered))} "
                              "-> pin the Runtime to AZs covered by EVERY endpoint.")
            else:
                r.ok.append(f"AZ {az} is covered by all listed endpoints.")
    else:
        r.warn.append("no endpoint_az_coverage given -> can't verify bedrock-agentcore PrivateLink is offered "
                      "in your AZs (it is missing in some, e.g. us-west-2d).")

    # blanket world-open sweep across any rule list
    for key in ("adapter_ingress_rules", "endpoint_sg_ingress"):
        for rule in cfg.get(key, []) or []:
            if _world_open(str(rule.get("source", ""))) and rule not in ep443:
                pass  # already reported above
    return r


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to a networking config JSON (default: the built-in GOOD sample)")
    ap.add_argument("--write-samples", action="store_true",
                    help="write netcheck_sample_{good,bad}.json next to this script and exit")
    args = ap.parse_args(argv)

    if args.write_samples:
        for name, cfg in (("good", _GOOD), ("bad", _BAD)):
            p = HERE / f"netcheck_sample_{name}.json"
            p.write_text(json.dumps(cfg, indent=2) + "\n")
            print(f"wrote {p}")
        return 0

    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        label = args.config
    else:
        cfg = _GOOD
        label = "<built-in GOOD sample>"

    print(f"netcheck: validating {label}\n")
    r = validate(cfg)
    for m in r.ok:
        print(f"  OK    {m}")
    for m in r.warn:
        print(f"  WARN  {m}")
    for m in r.fail:
        print(f"  FAIL  {m}")

    print(f"\n{len(r.ok)} ok, {len(r.warn)} warn, {len(r.fail)} fail")
    if r.fail:
        print("==> NOT safe to deploy — fix the FAILs above (see docs/agentcore-networking.md).")
        return 1
    print("==> networking checklist passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
