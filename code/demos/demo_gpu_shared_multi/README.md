# demo_gpu_shared_multi — KernelBench, Ray farm across MULTIPLE GPUs, one WHOLE GPU per session

The **multi-GPU** half of the shared-farm split (`demo_gpu_shared` was split into
`demo_gpu_shared_single` + `demo_gpu_shared_multi`). A Ray farm inside a LOCAL AgentCore sandbox
Runtime spans **GPUs 6,7** and leases `floor(2/1.0)=2` slots — **one whole GPU per Ray session**.
No fractional sharing ⇒ no software memory guard, which sidesteps the non-MIG nvcc-wedge that
fractional sharing can hit on L40S.

Its sibling `demo_gpu_shared_single` instead packs 2 fractional actors onto ONE GPU (0.5 + 0.5).

## What this demo shows
- **Task:** real [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) —
  write a `ModelNew` that matches a reference `Model` numerically **and** runs on CUDA. The
  `run_episode` harness runs **in-process** (drives SGLang directly); the only "remote" piece is the
  local sandbox Runtime container.
- **Whole-GPU farm across multiple GPUs:** `ray.remote(num_gpus=1.0)(SandboxActor)` gives each of 2
  concurrent episodes its own GPU (6 and 7); Ray does the placement, no memory guard needed.
- **One shared `runtimeSessionId` + per-episode `sub_session`:** AgentCore maps 1 `runtimeSessionId`
  → 1 instance, so every episode invokes the local runtime under the SAME session id and carries a
  per-episode `sub_session` that keys its whole-GPU actor.

## How to run (local, on this box — no AWS)
1. **On the HOST** (not inside `agentcore_p4`), start the local whole-GPU farm on GPUs 6,7:
   ```bash
   bash code/deploy/run_local_gpu_runtime.sh shared_multi
   # -> B4_FARM_GPUS=2, B4_GPU_FRACTION=1.0, B4_PER_SANDBOX_MEM_FRACTION=0.95 on host GPUs 6,7
   #    farm served at http://127.0.0.1:8080  (first /ping waits for the torch install, ~2-3 min)
   ```
2. **Inside `agentcore_p4`**, run the trainer (slime on GPUs 0-5):
   ```bash
   bash code/demos/demo_gpu_shared_multi/run.sh
   ```
3. Tear down when done: `bash code/deploy/run_local_gpu_runtime.sh stop`.

`NR=11` ⇒ 10 train steps + 1 held-out eval (smoke). Knobs: `RB`, `NS`, `NR`, `TEMP`, `CONC`
(≤ the farm's 2 slots), `EVAL_INTERVAL`. Same `agentcore_slime_demo.generate.generate` hook,
Sample contract, and reward path as every other GPU demo — only the sandbox GPU partitioning differs.
This is the local, multi-GPU sibling of `demo_gpu_local` (which frames the same whole-GPU farm as the
standalone on-box GPU demo).

## Self-contained
This folder owns its `data/` + `ckpt/` + `logs/` and imports its OWN vendored `agentcore_slime_demo/`
copy (via `PYTHONPATH`); it references only genuinely shared infrastructure (the `agentcore_p4`
container, the base Megatron checkpoint, `/root/slime`).
