#!/usr/bin/env python3
"""Generate the 8 standalone per-task CPU demos (demo_cpu_<task>) from the demo_cpu_mbpp lib.

Each generated folder is a fully self-contained agentic-RL recipe: its own vendored
agentcore_slime_demo/ + deploy/ + data/ (only that task's train/eval JSONL) + run.sh (FAMILY baked
in) + prepare_data.sh + verify_data.py + README.md. Re-run this after changing the shared library to
re-sync all 8 (it removes and rebuilds each demo_cpu_<task> dir):

    python3 code/demos/gen_task_demos.py

Source of truth for the vendored lib + run.sh + deploy + metrics is demo_cpu_mbpp; the data files
come from demo_cpu_mbpp (coding/sql/rag/toolcall) or demo_cpu_math (math). Build the data first
with each demo's prepare_data.sh (or the source demos' prepare_data.sh) so the JSONL exists to copy.
"""
import os, shutil

DEMOS = os.path.dirname(os.path.abspath(__file__))
SRC_SINGLE = os.path.join(DEMOS, "demo_cpu_mbpp")
SRC_MATH = os.path.join(DEMOS, "demo_cpu_math")
LIB = os.path.join(SRC_SINGLE, "agentcore_slime_demo")
DEPLOY = os.path.join(SRC_SINGLE, "deploy")
METRICS = os.path.join(SRC_SINGLE, "metrics.py")
RUNSH = os.path.join(SRC_SINGLE, "run.sh")

TASKS = [
  dict(prefix="gsm8k", key="gsm8k", data_src=SRC_MATH, family="math",
       repo="openai/gsm8k (config main)", reward="exact integer match",
       train="--split train --limit 200", ev="--split test --limit 40",
       ntr=200, nev=40, loader="load_gsm8k", verify='load_gsm8k(split="train", limit=15)',
       blurb="grade-school math word problems"),
  dict(prefix="math500", key="math500", data_src=SRC_MATH, family="math",
       repo="HuggingFaceH4/MATH-500", reward="numeric match (tol 1e-4), numeric-answer subset",
       train="--split test --limit 160 --offset 0", ev="--split test --limit 40 --offset 160",
       ntr=160, nev=40, loader="load_math500", verify='load_math500(split="test", limit=15)',
       blurb="competition math"),
  dict(prefix="deepscaler", key="deepscaler", data_src=SRC_MATH, family="math",
       repo="agentica-org/DeepScaleR-Preview-Dataset", reward="numeric match (tol 1e-4), numeric-answer subset",
       train="--limit 400 --offset 0", ev="--limit 40 --offset 400",
       ntr=400, nev=40, loader="load_deepscaler", verify='load_deepscaler(limit=15)',
       blurb="competition/olympiad math"),
  dict(prefix="humaneval", key="humaneval", data_src=SRC_SINGLE, family="coding",
       repo="openai/openai_humaneval", reward="the row's own unit-test fn — check(entry_point)",
       train="--split test --limit 140 --offset 0", ev="--split test --limit 24 --offset 140",
       ntr=140, nev=24, loader="load_humaneval", verify='load_humaneval(split="test", limit=15)',
       blurb="Python function completion"),
  dict(prefix="gretelsql", key="sql", data_src=SRC_SINGLE, family="sql",
       repo="gretelai/synthetic_text_to_sql", reward="SQL execution vs the gold result set (stdlib sqlite3)",
       train="--split train --limit 300", ev="--split test --limit 40",
       ntr=300, nev=40, loader="load_sql", verify='load_sql(split="train", limit=15)',
       blurb="text-to-SQL"),
  dict(prefix="hotpotqa", key="hotpotqa", data_src=SRC_SINGLE, family="rag",
       repo="hotpotqa/hotpot_qa (distractor)", reward="SQuAD-normalized exact match (10 passages -> ./docs corpus)",
       train="--split train --limit 300", ev="--split validation --limit 40",
       ntr=300, nev=40, loader="load_hotpotqa", verify='load_hotpotqa(split="validation", limit=15)',
       blurb="multi-hop retrieval QA"),
  dict(prefix="xlam", key="xlam", data_src=SRC_SINGLE, family="toolcall",
       repo="minpeter/xlam-function-calling-60k-parsed", reward="order-insensitive, arg-normalized gold-call match",
       train="--limit 300 --offset 0", ev="--limit 40 --offset 300",
       ntr=300, nev=40, loader="load_xlam", verify='load_xlam(limit=15)',
       blurb="function calling"),
  dict(prefix="mcp", key="mcp", data_src=SRC_SINGLE, family="toolcall",
       repo="Team-ACE/ToolACE (MCP-style JSON tool schemas)", reward="order-insensitive, arg-normalized gold-call match",
       train="--limit 300 --offset 0", ev="--limit 40 --offset 300",
       ntr=300, nev=40, loader="load_mcp", verify='load_mcp(limit=15)',
       blurb="MCP-style tool calling"),
]

RUN_BASE = open(RUNSH).read()

VERIFY_TMPL = '''#!/usr/bin/env python3
"""Offline checker proof for THIS demo's dataset: the reference solution PASSES the hidden checker,
a wrong one FAILS, and the training-row round-trip is lossless. Stdlib-only (no GPU; one-time HF pull
to load the rows). Run inside the agentcore_p4 container:
    python3 verify_data.py
"""
import os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentcore_slime_demo.real_datasets import {loader}
from agentcore_slime_demo.build_real_datasets import _row
from agentcore_slime_demo.task import task_from_dict, CodingTask


def run_checker(task, submission):
    with tempfile.TemporaryDirectory() as d:
        for name, txt in task.setup_files().items():
            fp = os.path.join(d, name)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, "w").write(txt)
        open(os.path.join(d, "run_checks.py"), "w").write(task.runner_source())
        open(os.path.join(d, task.artifact_filename), "w").write(submission)
        p = subprocess.run(["python3", "run_checks.py"], cwd=d, capture_output=True, text=True)
        return p.returncode


tasks = {verify}
good = bad = 0
for t in tasks:
    mt = _row(t, None)["metadata"]["task"]                      # round-trip through the training row
    rt = task_from_dict({{**mt}}) if "family" in mt else CodingTask(**mt)
    ok = run_checker(rt, t.mock_attempts[-1]) == 0 and run_checker(rt, t.mock_attempts[0]) != 0
    good += ok
    bad += (not ok)
print(f"{{good}}/{{good + bad}} tasks: reference PASSES and wrong FAILS (checker is correct)")
sys.exit(0 if bad == 0 else 1)
'''

PREP_TMPL = '''#!/bin/bash
# Download + prepare the demo_cpu_{prefix} dataset ({repo}) into data/. Run INSIDE agentcore_p4:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_{prefix}/prepare_data.sh
# RLVR (verifiable reward): {reward} — no LLM judge. Idempotent. Builds {prefix} ({ntr}/{nev}).
# The builder is vendored (agentcore_slime_demo/build_real_datasets.py) so this folder is self-contained.
set -euo pipefail
DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
B="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
mkdir -p "$DIR/data"
$B --dataset {key} {train} --out "$DIR/data/{prefix}_train.jsonl"
$B --dataset {key} {ev}  --out "$DIR/data/{prefix}_eval.jsonl"
echo "done -> $DIR/data : {prefix} ({ntr}/{nev})"
'''

README_TMPL = '''# demo_cpu_{prefix} — {blurb}, one CPU AgentCore sandbox (in-process harness)

Self-contained agentic-RL demo: RL-trains Qwen3-4B on **{repo}** via the multi-turn
"write `solution.py` → run the hidden checker → read the failure → fix → repeat" loop. The AgentCore
Runtime is the **RL sandbox** — each turn's `write_file` / `exec` / `read_file` runs inside a managed
serverless CPU Code-Interpreter microVM (`invoke_agent_runtime`), while the `run_episode` harness and
the slime trainer stay **in-process**. Verifiable reward (RLVR): {reward} — no LLM judge.

- **Task family:** `{family}`   ·   **Dataset:** `{repo}`   ·   **Reward:** {reward}
- **Data:** `data/{prefix}_train.jsonl` ({ntr} rows) + `data/{prefix}_eval.jsonl` ({nev} rows)

## Reproduce (run one by one)
Everything runs INSIDE the persistent container `agentcore_p4` (slimerl/slime); the repo is at
`/work/code`. `data/` is already populated — step 1 only re-pulls it from scratch.
```bash
# 1. Download + prepare this task's dataset (one-time network to pull the HF rows)
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_{prefix}/prepare_data.sh

# 2. Prove the reward is real, offline (reference solution PASSES, a wrong one FAILS)
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_{prefix}/verify_data.py

# 3. Train + eval (async/decoupled slime). Smoke uses NR=3; converge with a larger NR.
docker exec -e NR=3 -e EVAL_INTERVAL=2 agentcore_p4 bash -lc 'cd /work && bash code/demos/demo_cpu_{prefix}/run.sh'
```
`run.sh` bakes in `FAMILY={prefix}` (⇒ `data/{prefix}_{{train,eval}}.jsonl`, eval label `{prefix}`).
Real remote runs need `AGENTCORE_SANDBOX_RUNTIME_ARN(S)` + AWS creds in the container (`run.sh` ships
a pinned CPU-sandbox ARNS default). `SANDBOX=local` runs fully offline (CI/unit tests).

## Knobs (env vars; defaults in run.sh)
`NR` num-rollout ⇒ real steps `NR-1`; `EVAL_INTERVAL`; `RB`/`NS` GRPO batch/group (`GBS=RB×NS`);
`TEMP` rollout temperature; `CONC` episode concurrency (`S=CONC×2`); `SANDBOX`
(`agentcore-runtime` default | `agentcore` Code-Interpreter microVM | `local` offline).

## Metrics → CSV + PNG
```bash
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_{prefix}/metrics.py <train.log> \\
    --csv train.csv --eval-csv train.eval.csv --plot train.png
```
Watch `rollout/raw_reward` (learning signal), `eval/{prefix}` (held-out solve-rate), and the
`perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` split (remote sandbox vs trainer/SGLang bottleneck).

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (real_datasets.py loaders, in-process `generate`
  hook, the assert checker, mode_b_eval, megatron_init); this demo imports only this copy
- `deploy/Dockerfile.sandbox.cpu` — the Dockerfile the CPU sandbox image is built from
- `data/` — this task's `{prefix}_train.jsonl` / `{prefix}_eval.jsonl`
- `prepare_data.sh` — download + build the dataset (step 1)
- `verify_data.py` — offline checker proof (step 2)
- `run.sh` — the RL training run (async/decoupled slime; `FAMILY={prefix}` baked in)
- `metrics.py` — reward + latency parsing (CSVs + PNG)

Shared by reference (NOT vendored): the base Megatron checkpoint
(`code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`, override `CKPT_BASE`), the slime install, and the
deploy orchestrator (`code/deploy/deploy_two_runtime.py`).
'''


def ignore_pyc(_d, names):
    return [n for n in names if n == "__pycache__"]


for t in TASKS:
    dst = os.path.join(DEMOS, f"demo_cpu_{t['prefix']}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    shutil.copytree(LIB, os.path.join(dst, "agentcore_slime_demo"), ignore=ignore_pyc)
    shutil.copytree(DEPLOY, os.path.join(dst, "deploy"), ignore=ignore_pyc)
    shutil.copy2(METRICS, os.path.join(dst, "metrics.py"))
    os.makedirs(os.path.join(dst, "data"))
    for suf in ("train", "eval"):
        shutil.copy2(os.path.join(t["data_src"], "data", f"{t['prefix']}_{suf}.jsonl"),
                     os.path.join(dst, "data", f"{t['prefix']}_{suf}.jsonl"))
    # run.sh: retarget DEMO_DIR + bake in FAMILY default
    run = RUN_BASE.replace("DEMO_DIR=/work/code/demos/demo_cpu_mbpp",
                           f"DEMO_DIR=/work/code/demos/demo_cpu_{t['prefix']}")
    run = run.replace('FAMILY="${FAMILY:-coding}"', f'FAMILY="${{FAMILY:-{t["prefix"]}}}"')
    assert f"demo_cpu_{t['prefix']}" in run and f"FAMILY:-{t['prefix']}" in run, t["prefix"]
    open(os.path.join(dst, "run.sh"), "w").write(run)
    open(os.path.join(dst, "prepare_data.sh"), "w").write(PREP_TMPL.format(**t))
    open(os.path.join(dst, "verify_data.py"), "w").write(VERIFY_TMPL.format(**t))
    open(os.path.join(dst, "README.md"), "w").write(README_TMPL.format(**t))
    os.chmod(os.path.join(dst, "run.sh"), 0o755)
    os.chmod(os.path.join(dst, "prepare_data.sh"), 0o755)
    print(f"created demo_cpu_{t['prefix']:12s} ({t['family']}, {t['ntr']}/{t['nev']})")

print("all 8 done.")
