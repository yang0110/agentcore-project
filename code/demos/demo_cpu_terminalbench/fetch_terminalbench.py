#!/usr/bin/env python3
"""Fetch the Terminal-Bench task corpus and classify it for demo_cpu_terminalbench.

Terminal-Bench (laude-institute/terminal-bench) ships each task as a self-contained directory
under `original-tasks/<task>/`:

    task.yaml            the English instruction + metadata (difficulty, category, parser_name)
    Dockerfile           builds the task's sandbox environment (seed files baked in)
    docker-compose.yaml  wraps the Dockerfile as a `client` service (multi-service = multi-container)
    run-tests.sh         installs pytest and runs the task's hidden test suite (the reward oracle)
    tests/               the pytest suite that grades the agent's work
    solution.sh          reference ("oracle") solution — NOT copied into the sandbox

This demo builds ONE sandbox image per task (the "large image pool" the demo showcases), so it
can only use tasks whose environment is a SINGLE container — a `docker-compose.yaml` that declares
extra services (a DB, a second host, ...) cannot map to AgentCore's one-sandbox-per-episode model.
We therefore split the corpus into single- vs multi-container and pool only the single-container set.

Populates a (gitignored) `tasks/` dir with the single-container task directories and writes
`tasks/manifest.json` — the authoritative list the dataset builder and deploy script both read.

    python3 fetch_terminalbench.py                 # all single-container tasks (~229)
    TB_LIMIT=15 python3 fetch_terminalbench.py      # first 15 (cheap pipeline smoke)
    TB_REF=0.1.1 python3 fetch_terminalbench.py     # pin a tag/commit instead of main

No AWS, no Docker — pure fetch + classify. Idempotent: re-run to refresh.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "https://github.com/laude-institute/terminal-bench"
HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"
SRC_SUBDIR = "original-tasks"


def sh(cmd: str, cwd: str | None = None) -> None:
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def _count_services(compose_path: Path) -> int:
    """Count the services declared under `services:` in a docker-compose file.

    A single-service compose (just `client`) maps to one AgentCore sandbox; any extra service
    (a database, a second networked host, ...) makes the task multi-container and unsupported here.
    """
    if not compose_path.exists():
        return 0
    txt = compose_path.read_text(errors="replace")
    m = re.search(r"(?ms)^services:\s*\n(.*?)(?:\n\S|\Z)", txt)
    if not m:
        return 0
    return sum(1 for line in m.group(1).splitlines() if re.match(r"^  [A-Za-z0-9_.-]+:\s*$", line))


def _instruction(task_dir: Path) -> str:
    """Pull the `instruction:` block out of task.yaml without a YAML dep (stdlib only)."""
    y = (task_dir / "task.yaml").read_text(errors="replace")
    m = re.search(r"(?ms)^instruction:\s*\|-?\s*\n(.*?)(?:\n[A-Za-z_][\w]*:|\Z)", y)
    if m:
        # Strip the common leading indent that the block scalar carried.
        lines = m.group(1).rstrip("\n").splitlines()
        indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
        return "\n".join(l[indent:] for l in lines).strip()
    m = re.search(r'(?m)^instruction:\s*"?(.*?)"?\s*$', y)
    return (m.group(1).strip() if m else "").strip()


def _yaml_scalar(task_dir: Path, key: str, default: str = "") -> str:
    y = (task_dir / "task.yaml").read_text(errors="replace")
    m = re.search(rf'(?m)^{re.escape(key)}:\s*"?([^"\n]*)"?\s*$', y)
    return m.group(1).strip() if m else default


def main() -> int:
    ref = os.environ.get("TB_REF", "main")
    limit = int(os.environ.get("TB_LIMIT", "0"))  # 0 = all single-container tasks

    tmp = Path(tempfile.mkdtemp(prefix="tb-src-"))
    try:
        # Sparse, blobless clone of only original-tasks/ — the whole repo is large.
        sh(f"git clone --depth 1 --filter=blob:none --sparse --branch {ref} {REPO} {tmp}"
           if ref != "main" else
           f"git clone --depth 1 --filter=blob:none --sparse {REPO} {tmp}")
        sh(f"git -C {tmp} sparse-checkout set {SRC_SUBDIR}")
        src = tmp / SRC_SUBDIR

        single, multi = [], []
        for d in sorted(p for p in src.iterdir() if p.is_dir()):
            compose = d / "docker-compose.yaml"
            if not compose.exists():
                compose = d / "docker-compose.yml"
            (single if _count_services(compose) <= 1 else multi).append(d)

        if limit:
            single = single[:limit]

        # Refresh tasks/ with just the single-container set we can pool.
        if TASKS_DIR.exists():
            shutil.rmtree(TASKS_DIR)
        TASKS_DIR.mkdir(parents=True)

        entries = []
        for d in single:
            dst = TASKS_DIR / d.name
            shutil.copytree(d, dst)
            entries.append({
                "task": d.name,
                # ECR/runtime names must be lowercase alnum+hyphen — task names already are, but
                # normalize defensively so the deploy script can use this verbatim.
                "image": re.sub(r"[^a-z0-9-]", "-", d.name.lower()),
                "instruction": _instruction(d),
                "difficulty": _yaml_scalar(d, "difficulty", "unknown"),
                "category": _yaml_scalar(d, "category", "unknown"),
                "parser": _yaml_scalar(d, "parser_name", "pytest"),
            })

        manifest = {
            "source": REPO,
            "ref": ref,
            "counts": {"single_container": len(single), "multi_container_skipped": len(multi),
                       "total": len(single) + len(multi)},
            "multi_container_skipped": [d.name for d in multi],
            "tasks": entries,
        }
        (TASKS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

        print(f"fetched {len(entries)} single-container tasks -> {TASKS_DIR}")
        print(f"skipped {len(multi)} multi-container tasks: {[d.name for d in multi]}")
        print(f"manifest -> {TASKS_DIR / 'manifest.json'}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
