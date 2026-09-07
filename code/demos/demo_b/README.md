# Demo B — KernelBench (GPU sandbox) + our run_episode harness

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
Our lightweight `run_episode`
(`agentcore_slime_demo/harness.py`) drives the multi-turn loop. Because the checker needs CUDA,
the sandbox exec is gated onto a process-wide **`GpuPool`** (`gpu_pool.py`) that dense-packs many
bursty env checkers onto a couple of physical GPUs.

**GPU split (8×L40S): 4 train + 2 inference + 2 env.** slime is handed all 8 GPUs for ray but
reserves only 6 (`ACTOR_GPUS`=4 + `ROLLOUT_GPUS`=2); the `GpuPool` leases the remaining physical
GPUs **6,7** to the env checkers directly.

Same harness/policy/training path as Demo A — differs only in the sandbox (CUDA inside) and task
(KernelBench vs MBPP). This is the OUR-harness variant; **Demo B2** is the Claude Code (Docker)
variant of the same task.

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

```bash
docker exec agentcore_p4 bash -lc 'cd /work && bash code/demos/demo_b/run.sh'
```

`SANDBOX=agentcore` swaps the env to a real AWS GPU Runtime instance (needs
`AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION` in the container env; the managed GPU Runtime container
is still the pending piece — today the GPU path runs on this box's L40S).

### GpuPool knobs (Demo B only)

| var | default | meaning |
|---|---|---|
| `GPU_POOL_DEVICES` | `6,7` | physical GPUs leased to the env checkers |
| `GPU_POOL_CONCURRENCY` | `16` | env checkers allowed on the GPUs at once (semaphore) |
| `GPU_POOL_PER_ENV_MIB` | `2048` | per-env memory cap; concurrency × this must fit the pool |

~100 concurrent envs can share just 2 GPUs because GPU use is *bursty* and *memory*-bound — the
pool admits a few checkers at a time and queues the rest.

### Size / save-only-at-end note

The train pool is now **100 prompts** (level_1), so real `RB`/`NR` work. Keep `NR × RB ≤ 100` so
one epoch spans the whole run and the **only** checkpoint is the final one: slime force-saves at
every epoch boundary (`step % (dataset_size/RB) == 0`), so with `RB=4` the epoch is `100/4 = 25`
steps and the default `NR=20` (19 real steps) never crosses it. The runner defaults to
`RB=4 NS=8 NR=20`, `EVAL_INTERVAL=5 EVAL_NS=2` — all overridable; just keep `NR × RB ≤ 100`.

## Tunables

`RB` (prompts/step), `NS` (GRPO group size), `NR` (num-rollout; real steps = NR−1; keep NR×RB ≤ 100),
`TEMP`, `CONC` (sglang concurrency), `EVAL_INTERVAL`, `EVAL_NS`, `SAVE`, `CKPT_BASE`, plus the GpuPool
knobs above.

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
