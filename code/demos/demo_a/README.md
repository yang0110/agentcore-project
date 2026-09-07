# Demo A — MBPP · CPU sandbox · our `run_episode` harness

**Independently runnable.** This directory owns everything specific to Demo A (its `data/`,
`ckpt/`, `run.sh`) and needs no sibling demo. It imports the `agentcore_slime_demo` library from
the **shared** copy at `code/agentcore_slime_demo/` (on `PYTHONPATH` as `/work/code`), and
references the shared *infra*: the `agentcore_p4` container, the base Megatron checkpoint, and
`/root/slime`.

```
demo_a/
  run.sh                    # standalone async training runner (Demo A baked in)
  data/
    mbpp_train.jsonl        # 500 MBPP prompts (>= NR*RB, so save-only-at-end holds)
    mbpp_eval.jsonl         # held-out eval prompts (validation split, no overlap)
  ckpt/                     # OUTPUT: the single end-of-training checkpoint lands here
  build_dataset.py          # rebuild data/ if needed
  metrics.py                # parse the training log -> CSVs + PNG
```

## What it is

The MBPP demo: the policy solves a Python problem by **writing `solution.py` → running the unit
tests in a sandbox → reading the failure → fixing → repeat** (multi-turn `run_episode` in
`agentcore_slime_demo/harness.py`). The checker is plain Python/IO, so **envs run on CPU** and all
8 GPUs serve the policy.

- **Task:** MBPP (`agentcore_slime_demo/mbpp.py`).
- **Sandbox:** `SANDBOX=local` → in-container `LocalSandbox`; `SANDBOX=agentcore` → real AWS Code
  Interpreter microVM (needs `AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION` in the container env).
- **Harness:** OUR lightweight `run_episode` via `--custom-generate-function-path
  agentcore_slime_demo.generate.generate` (NOT the Claude Code CLI — that is Demo A2).
- **GPU split (8×L40S):** **4 train + 4 inference + 0 env**.
- **Trainer:** real slime, **async/decoupled** (`train_async.py`, no `--colocate`).

## Prerequisites (shared infra)

1. The persistent container `agentcore_p4` is up (`code/p4_slime_train/00_start_container.sh`).
2. The base Megatron checkpoint exists at `code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`
   (override with `CKPT_BASE=...`). This is the only cross-demo dependency, and it is read-only.
3. The Qwen3-4B HF snapshot is in the container's HF cache (path baked into `run.sh` as `$HF`).

## Run

```bash
# (data/ is already populated; rebuild only if you want to change size/splits)
# docker exec agentcore_p4 python /work/code/demos/demo_a/build_dataset.py \
#     --dataset mbpp --split "train+test" --limit 500 --out /work/code/demos/demo_a/data/mbpp_train.jsonl
# docker exec agentcore_p4 python /work/code/demos/demo_a/build_dataset.py \
#     --dataset mbpp --split validation --limit 8 --out /work/code/demos/demo_a/data/mbpp_eval.jsonl

docker exec agentcore_p4 bash -lc \
    'cd /work && SANDBOX=local bash code/demos/demo_a/run.sh 2>&1 | tee /work/code/demos/demo_a/train.log'
```

Swap to real AWS microVMs with `SANDBOX=agentcore`.

## Tunables (env vars)

| var | default | meaning |
|---|---|---|
| `RB` | 4 | prompts per rollout step |
| `NS` | 8 | GRPO group size (samples per prompt); `GBS = RB×NS` |
| `NR` | 100 | num-rollout; **real training steps = NR−1** (fresh ckpt is iter 0) |
| `TEMP` | 1.0 | rollout sampling temperature |
| `CONC` | 4 | sglang-server-concurrency (caps concurrent episodes) |
| `EVAL_INTERVAL` | 10 | held-out eval every N steps (`0` disables) |
| `EVAL_NS` | 2 | samples per eval prompt |
| `SAVE_INTERVAL` | 1000000 | save-only-at-end (see below) |
| `SANDBOX` | local | `local` or `agentcore` |
| `CKPT_BASE` | shared torch_dist | base Megatron checkpoint to resume from |

## Checkpoints (save-only-at-end)

To save disk we do **not** snapshot mid-training. slime force-saves at the final rollout **and at
every epoch boundary** (`step % (dataset_size/RB) == 0`) regardless of `--save-interval`. So the
invariant for exactly one checkpoint is **`dataset_size ≥ NR × RB`**: the train set is 500 prompts
⇒ 125 steps/epoch ≥ 99 real steps, so the **only** checkpoint written lands in `ckpt/`
(`Qwen3-4B_mbpp_async`).

The end-of-training save routes through Megatron's **spawn-based PersistentAsyncCaller** to avoid
the fork-based checkpoint deadlock — `run.sh` sets **both** `--async-save` and
`--custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt`
(the one call slime omits; see `agentcore_slime_demo/megatron_init.py`). Keep both flags together.

## Metrics

```bash
python3 code/demos/demo_a/metrics.py code/demos/demo_a/train.log \
    --csv code/demos/demo_a/train.csv \
    --eval-csv code/demos/demo_a/train.eval.csv \
    --plot code/demos/demo_a/train.png
```

Two CSVs (per-step train, per-eval held-out solve rate) + a multi-panel PNG. The learning signal
is `rollout/raw_reward`; `train/grad_norm` shows the policy moving; GRPO advantage is 0 when every
sample in a prompt's group gets the same reward, so reward variance (harder tasks / higher `TEMP`)
is what produces gradient.
