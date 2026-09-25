#!/usr/bin/env python3
"""demo_cpu_instance_pool — offline probe: why this image needs an Instance, not a microVM.

Two things, no AWS required:

  1. IMAGE-SIZE verdict. Statically estimate the installed footprint of deploy/Dockerfile.sandbox.bigcpu
     and compare it to the AgentCore Code Interpreter microVM image cap (~2 GB). The heavy stack
     (torch CPU + scientific libs) blows past it, which is the whole reason this demo deploys the
     Runtime on a capacity-provider *Instance*. (Estimate, not a build — clearly labelled. If you
     HAVE built the image, pass --built-tag acs-sandbox-bigcpu to read the real `docker images` size.)

  2. CONTRACT parity. Run the SAME trivial episode the trainer runs, against LocalSandbox, to prove
     the harness/sandbox contract (write_file/exec/read_file) is unchanged — a big image changes only
     WHERE the sandbox runs, never the training code. (Identical to compare_backends.py's local path.)

Usage:
    python3 probe_bigimage.py                       # offline: size estimate + local episode
    python3 probe_bigimage.py --built-tag acs-sandbox-bigcpu   # use the real built image size
"""
from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MICROVM_IMAGE_CAP_GB = 2.0  # AgentCore Code Interpreter serverless microVM image cap (~2 GB).

# Rough INSTALLED footprint (GB) of the heavy pieces, keyed by a token that appears in the Dockerfile.
# Deliberately conservative round numbers — this is a "does it obviously exceed the cap" check, not a
# byte-accurate accounting. Build the image + --built-tag for the real number.
_FOOTPRINT_GB = {
    "python:3.12-slim": 0.15,
    "build-essential": 0.40,
    "torch": 2.90,
    "scipy": 0.10,
    "scikit-learn": 0.13,
    "pandas": 0.08,
    "numpy": 0.05,
    "sympy": 0.04,
    "bedrock-agentcore": 0.03,
}


def estimate_from_dockerfile(dockerfile: Path) -> tuple[float, list[tuple[str, float]]]:
    text = dockerfile.read_text()
    hits: list[tuple[str, float]] = []
    for token, gb in _FOOTPRINT_GB.items():
        if re.search(re.escape(token), text):
            hits.append((token, gb))
    total = sum(gb for _, gb in hits)
    return total, hits


def built_image_size_gb(tag: str) -> float | None:
    try:
        out = subprocess.check_output(
            ["docker", "image", "inspect", tag, "--format", "{{.Size}}"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return int(out) / 1e9
    except Exception:  # noqa: BLE001 — docker not present / image not built
        return None


async def _run_local_episode(family: str, seed_idx: int):
    from agentcore_slime_demo.harness import run_episode
    from agentcore_slime_demo.policy import MockPolicy
    from agentcore_slime_demo.sandbox import make_sandbox
    from agentcore_slime_demo.task import task_from_dict
    from agentcore_slime_demo.task_families import FAMILY_CLASSES

    cls = next((c for c in FAMILY_CLASSES if c.family == family), None)
    if cls is None:
        raise SystemExit(f"unknown family {family!r}: "
                         f"{', '.join(sorted(c.family for c in FAMILY_CLASSES))}")
    seed = cls.seeds()[seed_idx % len(cls.seeds())]
    task = task_from_dict({**seed.to_row_kwargs(), "mock_attempts": seed.mock_attempts})
    async with make_sandbox("local") as sb:
        traj = await run_episode(MockPolicy(task.mock_attempts), sb, task)
    return task.task_id, traj.solved, traj.reward, traj.num_turns


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dockerfile", default=str(HERE / "deploy" / "Dockerfile.sandbox.bigcpu"))
    ap.add_argument("--built-tag", help="a locally-built image tag to read the REAL size via docker")
    ap.add_argument("--family", default="ctf")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print("== 1. Image size vs the microVM cap ==\n")
    real = built_image_size_gb(args.built_tag) if args.built_tag else None
    if real is not None:
        size_gb, source = real, f"docker image inspect {args.built_tag} (REAL)"
        detail = []
    else:
        size_gb, detail = estimate_from_dockerfile(Path(args.dockerfile))
        source = f"static estimate of {Path(args.dockerfile).name}"
    print(f"  source: {source}")
    for token, gb in detail:
        print(f"    + {token:20s} ~{gb:>4.2f} GB")
    print(f"  estimated image size : ~{size_gb:.2f} GB")
    print(f"  microVM image cap    : {MICROVM_IMAGE_CAP_GB:.2f} GB")
    over = size_gb > MICROVM_IMAGE_CAP_GB
    if over:
        print(f"  ==> EXCEEDS the microVM cap by ~{size_gb - MICROVM_IMAGE_CAP_GB:.2f} GB -> deploy on a "
              "capacity-provider INSTANCE (SANDBOX=agentcore-runtime + a CPU capacity provider), "
              "NOT the Code Interpreter microVM.\n")
    else:
        print("  ==> fits a microVM; this demo intends a heavy image — check the Dockerfile.\n")

    print("== 2. Contract parity (LocalSandbox, offline) ==\n")
    task_id, solved, reward, turns = asyncio.run(_run_local_episode(args.family, args.seed))
    print(f"  family={args.family} task={task_id}  solved={solved} reward={reward} turns={turns}")
    print("  ==> the harness/sandbox contract is unchanged; a big image changes only WHERE the "
          "sandbox runs, never the training code.\n")

    ok = over and solved and reward == 1.0
    print("PASS" if ok else "CHECK", "— big-image-needs-instance:", over,
          "| local-episode-solved:", bool(solved and reward == 1.0))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
