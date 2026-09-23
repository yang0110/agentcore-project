#!/usr/bin/env python3
"""Provision demo_cpu_terminalbench's LARGE POOL of custom sandbox Runtimes — ONE per task.

This is the demo's whole point: a *large* image pool. Terminal-Bench ships one bespoke Docker
environment per task; we build each one, wrap it so it can serve as an AgentCore sandbox, push it
to its own ECR repo, and register ONE managed serverless Runtime per task. The pool map
(`AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE`, {task -> runtime_arn}) then routes each episode to the
Runtime built from ITS task's image — the exact same data-driven routing demo_cpu_imagepool uses
for 3 images, here at ~229-image scale.

Substrate choice (differs from imagepool on purpose):
  * networkMode = PUBLIC (not VPC). Terminal-Bench's `run-tests.sh` installs uv + pytest from the
    internet at grade time, so the sandbox needs egress. Reached only via the IAM control plane
    (invoke_agent_runtime); it never dials back, so PUBLIC is safe and needs no SG rule (mirrors the
    A3 CPU sandbox in deploy_two_runtime.py).
  * arm64, <2 GB per image (the managed serverless fleet cap). Most Terminal-Bench task images are
    small and build fine for arm64; images that exceed the cap or cannot install python3/pip are
    reported and SKIPPED — the pool is "every task that built + registered", which is honest for a
    large-pool showcase.

Wrapping: each task's own Dockerfile builds its environment (seed files baked in). We append a small
layer that (a) ensures python3 + bedrock-agentcore, (b) drops in sandbox_server_agent.py (the
/invocations exec/write_file/read_file server the Sandbox client speaks to), and (c) makes it the
CMD. The task's tests/ and run-tests.sh stay in the image for the reward step.

    python3 fetch_terminalbench.py            # first: populate ../tasks/ + manifest.json
    python3 deploy/deploy_terminalbench.py    # then: build+push+register the whole pool
    # writes ../tasks/arn_by_image.json and prints:
    #   export AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE="$(cat .../tasks/arn_by_image.json)"

Knobs (all optional):
    TB_TASKS=a,b,c   only these tasks (by name)          TB_LIMIT=N   only the first N in the manifest
    SKIP_BUILD=1     reuse already-pushed :latest images  TB_WORKDIR=/app  sandbox working dir (default /app)
    CONTINUE_ON_ERROR=1 (default) keep going past a failed task; set 0 to stop on the first failure.

Real AWS mutation, at SCALE: up to ~229 ECR repos + images and ~229 managed serverless Runtimes
(billed per session). Registering hundreds of Runtimes may hit the per-account AgentCore Runtime
quota — this script registers as many as the account allows and reports the rest. Prove the pipeline
with TB_LIMIT=2 before launching the full pool.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent                                   # code/demos/demo_cpu_terminalbench
TASKS_DIR = DEMO_DIR / "tasks"
SHARED_DEPLOY = (DEMO_DIR / ".." / ".." / "deploy").resolve()
sys.path.insert(0, str(SHARED_DEPLOY))

import deploy_two_runtime as d  # noqa: E402

REGION = d.REGION
PREFIX = d.PREFIX
PLATFORM = "linux/arm64"
WORKDIR = os.environ.get("TB_WORKDIR", "/app")
CONTINUE = os.environ.get("CONTINUE_ON_ERROR", "1") == "1"

# Appended to each task's own Dockerfile. `command -v python3` avoids a reinstall on python bases;
# --break-system-packages covers Debian/Ubuntu PEP-668 images. The last CMD wins, so this overrides
# the task's `sleep infinity` command with our server. Uses `python3 -m` from /opt on PYTHONPATH so
# the module name resolves regardless of the base image's WORKDIR.
#
# Warm-shared wiring (the demo's reliable warm path — one live session per task, reused across
# episodes with per-episode workdir isolation; see terminalbench_pool.py):
#   * SANDBOX_SHARED_MODE=cpu           -> the server routes sub_session ops to its CPU warm-shared
#                                          handler (fresh workdir per episode), NOT the GPU ray farm.
#   * cp -a {workdir} /opt/acs/task_template -> a PRISTINE snapshot of the task's baked workdir,
#     taken at build time. The server reseeds a fresh copy of this per episode (the cheap reset).
#   * the hidden oracle (tests/ + run-tests.sh) is baked at /opt/acs/oracle, OUTSIDE the agent's
#     workdir, so the agent can't read the tests while solving. grade.sh injects it into the episode
#     workdir + runs it AFTER the agent's turns (mirrors Terminal-Bench's own grade-after semantics).
# {oracle_copies} is filled per task with COPY lines only for the oracle files that exist.
WRAP_APPENDIX = """
# ===== demo_cpu_terminalbench sandbox wrapper (appended by deploy_terminalbench.py) =====
USER root
RUN (command -v python3 >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates)) \\
 && (python3 -m pip install --no-cache-dir --break-system-packages bedrock-agentcore \\
     || python3 -m pip install --no-cache-dir bedrock-agentcore \\
     || pip3 install --no-cache-dir bedrock-agentcore)
COPY __acs_sandbox_server_agent.py /opt/acs/sandbox_server_agent.py
COPY __acs_grade.sh /opt/acs/oracle/grade.sh
{oracle_copies}
RUN mkdir -p {workdir} && cp -a {workdir}/. /opt/acs/task_template/ 2>/dev/null || mkdir -p /opt/acs/task_template
ENV PYTHONPATH=/opt/acs
ENV SANDBOX_WORKDIR={workdir}
ENV SANDBOX_SHARED_MODE=cpu
ENV SANDBOX_TASK_TEMPLATE=/opt/acs/task_template
ENV SANDBOX_SESSIONS_ROOT=/sessions
EXPOSE 8080
CMD ["python3", "-m", "sandbox_server_agent"]
""".lstrip()

# grade.sh — the RLVR reward oracle, baked into every task image. Runs with cwd = the episode's
# freshly-reset workdir (the server sets it). Injects the hidden tests + run-tests.sh, runs them, and
# exits with their status (0 = task solved). Kept OUT of the workdir template so the agent never sees
# it while solving.
GRADE_SH = """#!/usr/bin/env bash
set -u
ORACLE=/opt/acs/oracle
if [ -d "$ORACLE/tests" ]; then cp -a "$ORACLE/tests" ./tests; fi
if [ -f "$ORACLE/run-tests.sh" ]; then
  cp "$ORACLE/run-tests.sh" ./run-tests.sh
  bash ./run-tests.sh
  exit $?
fi
if [ -d ./tests ]; then
  python3 -m pytest -q ./tests
  exit $?
fi
echo "grade: no oracle (tests/ + run-tests.sh) baked for this task" >&2
exit 2
"""


def _load_manifest() -> list[dict]:
    mpath = TASKS_DIR / "manifest.json"
    if not mpath.exists():
        d.die(f"{mpath} not found — run `python3 fetch_terminalbench.py` first")
    tasks = json.loads(mpath.read_text())["tasks"]
    only = {t for t in os.environ.get("TB_TASKS", "").split(",") if t}
    if only:
        tasks = [t for t in tasks if t["task"] in only]
    limit = int(os.environ.get("TB_LIMIT", "0"))
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        d.die("no tasks selected (check TB_TASKS / TB_LIMIT / manifest)")
    return tasks


def _prepare_context(task_dir: Path) -> Path:
    """Write the combined Dockerfile (task's own + our wrapper) into the task dir; return its path.

    Build context = the task dir, because the task's own Dockerfile COPYs its seed files from there.
    We drop in: sandbox_server_agent.py (as __acs_sandbox_server_agent.py) and grade.sh (as
    __acs_grade.sh) so their COPYs resolve, and emit COPY lines for the hidden oracle (tests/,
    run-tests.sh) ONLY when they exist in the task dir (a bare COPY of a missing path fails the build).
    """
    server_src = DEMO_DIR / "agentcore_slime_demo" / "sandbox_server_agent.py"
    (task_dir / "__acs_sandbox_server_agent.py").write_text(server_src.read_text())
    (task_dir / "__acs_grade.sh").write_text(GRADE_SH)

    # Bake the hidden oracle OUTSIDE the agent's workdir. Only copy what the task actually ships.
    oracle_lines = []
    if (task_dir / "tests").is_dir():
        oracle_lines.append("COPY tests /opt/acs/oracle/tests")
    for cand in ("run-tests.sh", "run_tests.sh"):
        if (task_dir / cand).is_file():
            oracle_lines.append(f"COPY {cand} /opt/acs/oracle/run-tests.sh")
            break

    base = (task_dir / "Dockerfile").read_text()
    combined = task_dir / "Dockerfile.acs"
    combined.write_text(
        base.rstrip() + "\n\n"
        + WRAP_APPENDIX.format(workdir=WORKDIR, oracle_copies="\n".join(oracle_lines))
    )
    return combined


def build_push_one(repo: str, task_dir: Path) -> str:
    acct = d.account()
    registry = f"{acct}.dkr.ecr.{REGION}.amazonaws.com"
    uri = d.ensure_ecr(repo) + ":latest"
    if os.environ.get("SKIP_BUILD") == "1":
        d.err(f"SKIP_BUILD=1 — reusing already-pushed {uri}")
        return uri
    dockerfile = _prepare_context(task_dir)
    d._ensure_buildx()
    d.sh(f"aws ecr get-login-password --region {REGION} | "
         f"docker login --username AWS --password-stdin {registry}")
    d.sh(f"docker buildx build --builder {d.BUILDX_BUILDER} --platform {PLATFORM} "
         f"--provenance=false --sbom=false -f {dockerfile} -t {uri} --push {task_dir}")
    d.err(f"pushed {uri} ({PLATFORM})")
    return uri


def main() -> int:
    tasks = _load_manifest()
    xrole = d.exec_role(f"{PREFIX}-exec-role")
    public_net = {"networkMode": "PUBLIC"}  # egress for run-tests.sh; reached via IAM only
    # SANDBOX_SHARED_MODE=cpu also set as an image ENV; passed here too so the warm-shared server
    # path is active regardless of how the Runtime resolves env precedence.
    env = {"AWS_REGION": REGION, "SANDBOX_WORKDIR": WORKDIR, "SANDBOX_SHARED_MODE": "cpu",
           "SANDBOX_TASK_TEMPLATE": "/opt/acs/task_template", "SANDBOX_SESSIONS_ROOT": "/sessions"}

    d.err(f"deploying pool of {len(tasks)} Terminal-Bench task Runtimes ({PLATFORM}, PUBLIC)")
    arn_by_image, failed = {}, {}
    for i, t in enumerate(tasks, 1):
        name_tag = t["image"]                       # lowercased alnum-hyphen task name
        task_dir = TASKS_DIR / t["task"]
        d.err(f"[{i}/{len(tasks)}] {t['task']} ({t['difficulty']}/{t['category']})")
        try:
            uri = build_push_one(f"{PREFIX}-tb-{name_tag}", task_dir)
            arn = d.managed_runtime(f"{PREFIX}_tb_{name_tag}".replace("-", "_"),
                                    uri, xrole, env, public_net)
            arn_by_image[t["task"]] = arn
            d.err(f"  -> {arn}")
        except Exception as e:  # noqa: BLE001 — one bad task must not sink the whole pool
            failed[t["task"]] = f"{type(e).__name__}: {e}"
            d.err(f"  FAILED: {failed[t['task']]}")
            if not CONTINUE:
                break

    # Persist the pool map next to the tasks so run.sh can load it without a huge inline export.
    (TASKS_DIR / "arn_by_image.json").write_text(json.dumps(arn_by_image, indent=2))
    if failed:
        (TASKS_DIR / "deploy_failures.json").write_text(json.dumps(failed, indent=2))

    print(f"\n# pool: {len(arn_by_image)} Runtimes registered, {len(failed)} failed")
    print(f"export AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE=\"$(cat {TASKS_DIR / 'arn_by_image.json'})\"")
    if failed:
        print(f"# {len(failed)} tasks failed to build/register — see {TASKS_DIR / 'deploy_failures.json'}")
    return 0 if arn_by_image else 1


if __name__ == "__main__":
    sys.exit(main())
