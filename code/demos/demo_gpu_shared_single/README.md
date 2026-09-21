# demo_gpu_shared_single — KernelBench, ONE GPU shared by a Ray fractional-GPU farm (0.5 + 0.5)

The **single-GPU** half of the shared-farm split (`demo_gpu_shared` was split into
`demo_gpu_shared_single` + `demo_gpu_shared_multi`). Two concurrent KernelBench episodes **share one
GPU**: a Ray fractional-GPU farm inside a LOCAL AgentCore sandbox Runtime packs `floor(1/0.5)=2`
long-lived actors onto **GPU 6**, each capped to ~half the card's VRAM in software
(`B4_PER_SANDBOX_MEM_FRACTION=0.5`, via an injected `sitecustomize.py`, because L40S has no MIG).

This is the real GPU-**sharing** case (0.5 + 0.5 on a single card). Its sibling
`demo_gpu_shared_multi` instead gives each Ray session a whole GPU across GPUs 6,7.

## What this demo shows
- **Task:** real [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) —
  write a `ModelNew` that matches a reference `Model` numerically **and** runs on CUDA. The
  `run_episode` harness runs **in-process** (drives SGLang directly); the only "remote" piece is the
  local sandbox Runtime container.
- **Fractional GPU sharing via Ray:** `ray.remote(num_gpus=0.5)(SandboxActor)` packs 2 warm-CUDA
  actors onto GPU 6; Ray does the scheduling, each actor caps its own GPU memory in software.
- **One shared `runtimeSessionId` + per-episode `sub_session`:** AgentCore maps 1 `runtimeSessionId`
  → 1 instance, so every episode invokes the local runtime under the SAME session id and carries a
  per-episode `sub_session` that keys its fractional actor.

## How to run (local, on this box — no AWS)
1. **On the HOST** (not inside `agentcore_p4`), start the local fractional farm on GPU 6:
   ```bash
   bash code/deploy/run_local_gpu_runtime.sh shared_single
   # -> B4_FARM_GPUS=1, B4_GPU_FRACTION=0.5, B4_PER_SANDBOX_MEM_FRACTION=0.5 on host GPU 6
   #    farm served at http://127.0.0.1:8080  (first /ping waits for the torch install, ~2-3 min)
   ```
2. **Inside `agentcore_p4`**, run the trainer (slime on GPUs 0-5; GPU 7 idle in this demo):
   ```bash
   bash code/demos/demo_gpu_shared_single/run.sh
   ```
3. Tear down when done: `bash code/deploy/run_local_gpu_runtime.sh stop`.

`NR=11` ⇒ 10 train steps + 1 held-out eval (smoke). Knobs: `RB`, `NS`, `NR`, `TEMP`, `CONC`
(≤ the farm's 2 slots), `EVAL_INTERVAL`. Same `agentcore_slime_demo.generate.generate` hook,
Sample contract, and reward path as every other GPU demo — only the sandbox GPU partitioning differs.

## Self-contained
This folder owns its `data/` + `ckpt/` + `logs/` and imports its OWN vendored `agentcore_slime_demo/`
copy (via `PYTHONPATH`); it references only genuinely shared infrastructure (the `agentcore_p4`
container, the base Megatron checkpoint, `/root/slime`).
