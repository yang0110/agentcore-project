# Demo B — KernelBench (remote GPU sandbox Runtime) + our run_episode harness

**Independently runnable.** This directory owns its `data/`, runner, `metrics.py`, and dataset
builder, and needs no sibling demo. The `agentcore_slime_demo` library is imported from the
**shared** copy at `code/agentcore_slime_demo/` (on `PYTHONPATH` as `/work/code`). It also
references only genuinely SHARED infrastructure: the persistent Docker container (`agentcore_p4`),
the base Megatron checkpoint (`/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`), and slime at
`/root/slime`.

## What it is

The real **KernelBench** (`ScalingIntelligence/KernelBench`, 270 CUDA problems across 4 levels).
Each problem gives a reference `Model` (nn.Module) + `get_inputs()`/`get_init_inputs()`; the policy
must write an equivalent **`ModelNew`** that runs on CUDA and matches the reference output. A **GPU**
checker runs it (`python3 run_checks.py` rebuilds the reference on the GPU with matched seeds,
asserts `allclose` + output on a `cuda` device, and reports a speedup), reads the failure, fixes it.
Our lightweight `run_episode` (`agentcore_slime_demo/harness.py`) drives the multi-turn loop
**in-process on this box**, but the checker's CUDA does **not** run here. Because the checker needs
a GPU, the **sandbox is its own remote GPU-backed AgentCore Runtime #2** (`sandbox_server_agent.py`
on a **g6** capacity-provider instance): `write_file` / `exec` / `read_file` — including
`python3 run_checks.py` — run **there**, reached over `invoke_agent_runtime`
(`SANDBOX=agentcore-runtime` → `AgentCoreRuntimeSandbox`). This is the decoupled-sandbox path
(harness in-process, sandbox remote); no local GPUs are reserved for the checker.

**GPU split (8×L40S): 4 train + 4 inference + 0 env.** All 8 GPUs go to slime
(`ACTOR_GPUS`=4 + `ROLLOUT_GPUS`=4); the KernelBench checker's GPU is on the **remote** sandbox
Runtime #2 (the g6), so no local GPUs are held back.

Same harness/policy/training path as Demo A — differs only in the sandbox (remote GPU Runtime) and
task (KernelBench vs MBPP). This is the OUR-harness variant; **Demo B3** swaps in the mini harness on
its own harness Runtime, and **Demo B4** packs several sandboxes onto one g6 via a Ray GPU-sharing farm.

## Build the dataset

Train on **level_1** (100 single-op problems); hold out **20 level_2 fusion problems** for eval
(disjoint — no leakage). Levels: `level_1` (100 single ops), `level_2` (100 fusions), `level_3`
(50 full architectures), `level_4` (20) — the harder levels are available but slower.

```bash
docker exec agentcore_p4 python /work/code/demos/demo_b/build_dataset.py \
    --dataset kernelbench --level level_1 --limit 100 --out /work/code/demos/demo_b/data/kernelbench_train.jsonl
docker exec agentcore_p4 python /work/code/demos/demo_b/build_dataset.py \
    --dataset kernelbench --level level_2 --limit 20 --out /work/code/demos/demo_b/data/kernelbench_eval.jsonl
```

(`--dataset kernelbench-stub` still builds the old hand-written 3-op smoke set if you want a
minimal offline check.)

## Run

First deploy the remote GPU sandbox Runtime #2 (a g6) and prepare its torch volume once
(`code/deploy/deploy_two_runtime.py all`, which runs the one-time `{"mode":"prepare"}` invoke that
installs the CUDA `torch` stack onto the persistent `models` volume — without it the checker can't
import torch and every reward is 0), then export its ARN:

```bash
docker exec agentcore_p4 bash -lc \
  'cd /work && AWS_REGION=us-west-2 \
     AGENTCORE_SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/acs_sandbox_gpu-... \
     NR=11 EVAL_INTERVAL=10 \
     bash code/demos/demo_b/run.sh'
```

`SANDBOX` defaults to `agentcore-runtime` (the remote GPU sandbox Runtime #2). `NR=11
EVAL_INTERVAL=10` is a 10-train + 1-eval smoke. The container needs `AWS_BEARER_TOKEN_BEDROCK` +
`AWS_REGION`. No local GPUs are reserved for the checker — its GPU is the remote g6's, so the only
sandbox knob is `AGENTCORE_SANDBOX_RUNTIME_ARN`.

### Size / save-only-at-end note

The train pool is now **100 prompts** (level_1), so real `RB`/`NR` work. Keep `NR × RB ≤ 100` so
one epoch spans the whole run and the **only** checkpoint is the final one: slime force-saves at
every epoch boundary (`step % (dataset_size/RB) == 0`), so with `RB=4` the epoch is `100/4 = 25`
steps and the default `NR=11` (10 real steps) never crosses it. The runner defaults to
`RB=4 NS=8 NR=11`, `EVAL_INTERVAL=10 EVAL_NS=2` — all overridable; just keep `NR × RB ≤ 100`.

## Tunables

`RB` (prompts/step), `NS` (GRPO group size), `NR` (num-rollout; real steps = NR−1; keep NR×RB ≤ 100),
`TEMP`, `CONC` (sglang concurrency), `EVAL_INTERVAL`, `EVAL_NS`, `SAVE`, `CKPT_BASE`, and
`AGENTCORE_SANDBOX_RUNTIME_ARN` (the remote GPU sandbox Runtime #2).

## Checkpoints

No mid-training snapshots (`SAVE_INTERVAL` huge). slime always force-saves the **final** rollout,
so the only checkpoint is the end one under `--save`. The end-of-training save's intermittent
**deadlock is fixed** here: `run.sh` passes both `--async-save` and
`--custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt`,
which routes the save through Megatron's spawn-based `PersistentAsyncCaller` (avoiding the
fork-of-a-CUDA-process hang). See `agentcore_slime_demo/megatron_init.py`.

## Metrics → CSV / PNG

```bash
python3 code/demos/demo_b/metrics.py <train.log> \
    --csv demoB.csv --eval-csv demoB.eval.csv --plot demoB.png
```

Writes a per-step train CSV (`rollout/raw_reward`, `train/grad_norm`, `train/kl_loss`, `perf/*`)
and a per-eval CSV (`eval/step` + held-out solve rate), and a small multi-panel PNG.
