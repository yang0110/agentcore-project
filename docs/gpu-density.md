# Dense GPU packing: 100 envs on one GPU EC2 instance

**Goal:** hold ~100 agentic-RL environments on a single GPU EC2 instance, where the envs'
workloads use the GPU. This is **Path A — DIY dense packing**: you run your own containers
on a GPU box and multiplex the GPU yourself, because AgentCore's managed model won't do
per-instance GPU sharing for you (see [Relationship to AgentCore](#relationship-to-agentcore)).

Verified on this box (**8× NVIDIA L40S, 46 GB each**, torch 2.8 / cu128): the demo runs
100 concurrent envs through real CUDA, gated and spread across all 8 GPUs. See
[Runnable demo](#runnable-demo).

---

## 1. The real constraint is GPU *memory*, not env count

An L40S is 46 GB. 100 envs with resident CUDA contexts ⇒ ~460 MB each — you exhaust
**memory** long before SM compute. So the density question is really: *how many envs need
the GPU resident at the same time?*

Most RL env GPU use is **bursty**: the env sets up a workspace, waits on the policy
(inference happens on slime's cluster, not here), then briefly touches the GPU to run a
kernel / score a reward. If work is bursty, you don't need 100 resident contexts — you need
a **queue** in front of the GPU. That queue is `agentcore_slime_demo/gpu_pool.py`.

> For MIXED / unprofiled workloads (the assumption here): gate concurrent GPU work with a
> small semaphore, cap per-env memory, and **measure** — then tune the gate to observed use.

## 2. GPU sharing mechanics (what the *host* provides)

The scheduler decides *who* runs; the **host** decides *how* the GPU is shared. On an L40S
(Ada Lovelace) the options are:

| Mechanism | What it does | L40S | Memory isolation | Use for |
|---|---|---|---|---|
| **Time-slicing** | GPU round-robins between contexts; oversubscribe freely | ✅ | ❌ none | bursty envs — pack many, share time |
| **CUDA MPS** | processes share one context concurrently (better small-kernel util) | ✅ | ❌ none | many small concurrent kernels |
| **MIG** | hard-partition GPU into isolated slices | ❌ **not on Ada** (A100/A30/H100 only), max 7 | ✅ | strong isolation on Ampere/Hopper |

**L40S has no MIG**, so there is no hardware memory isolation — a runaway env can OOM its
neighbours. We therefore enforce a **software** per-env memory cap
(`torch.cuda.set_per_process_memory_fraction`, driven by the lease's `ENV_GPU_MEM_FRACTION`).

Host setup (the ops half, outside this repo):
- **Time-slicing:** NVIDIA k8s device plugin `time-slicing` replicas, or just run N
  containers with `--gpus all` and let them share.
- **MPS:** run the MPS control daemon on the host; optionally cap with
  `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`.

## 3. The software scheduler — `gpu_pool.py`

`GpuPool` is backend-agnostic (Agent Lightning decoupling): it **never runs CUDA**. It only
does admission control + device assignment, and hands each env a **lease** of env vars:

```python
from agentcore_slime_demo.gpu_pool import GpuPool

pool = GpuPool(max_concurrent_gpu=16, per_env_mem_mib=2048)   # probes GPUs via nvidia-smi
async with pool.slot() as lease:                              # blocks past the gate
    await sandbox.exec("python3 work.py", env=lease.env)      # pinned + capped GPU work
```

- `max_concurrent_gpu` — **the knob**: how many envs touch a GPU at once. Bursty ⇒ keep it
  small; the other 80–90 envs sit in their CPU phase or queue. This is why 100 fit on 8.
- devices assigned **least-loaded-first**, so work spreads evenly across all GPUs.
- `lease.env` = `CUDA_VISIBLE_DEVICES` (pin to one device) + `ENV_GPU_MEM_FRACTION`
  (software memory cap the workload applies).
- **Static safety check** (`pool.memory_safety()`): worst-case co-residency per GPU =
  `ceil(max_concurrent_gpu / num_gpus) × per_env_mem_mib` must be ≤ GPU memory. The demo
  refuses to rely on bursty timing to dodge OOM — it warns on overcommit.

The pool records `PoolStats` — `peak_concurrent_gpu`, per-device admits, admission waits —
so you tune the gate against **real** numbers.

## 4. Runnable demo

```bash
python3 code/scripts/gpu_density.py --envs 100 --gpu-concurrency 8
python3 code/scripts/gpu_density.py --envs 100 --gpu-concurrency 32 --per-env-mem-mib 2048
```

Each env is a `LocalSandbox` that runs a CPU phase then a real CUDA matmul (or CPU fallback
with no GPU). Measured on 8× L40S:

| gate | per-env cap | wall | peak concurrent | avg admission wait | per-device spread |
|---|---|---|---|---|---|
| 8  | 1024 MiB | 38.7 s | 8  | 18.1 s | 12–13 each ×8 |
| 32 | 2048 MiB | 26.9 s | 32 | 9.1 s  | 12–14 each ×8 |

All 100 envs ran on `cuda`; observed peak per env was **12 MiB** (vs the 1–2 GiB cap) — this
workload is tiny, so the box has enormous headroom and the gate can go far higher. The point
stands: **100 envs coexisted; only `gate` touched a GPU at once, spread over all 8.**

## 5. Tuning recipe (mixed / unprofiled workloads)

1. Start with `max_concurrent_gpu ≈ 2–4 × num_gpus`, `per_env_mem_mib` = a generous cap.
2. Run the demo (or your rollouts) and read `PoolStats` + `nvidia-smi dmon`.
3. If `avg_admission_wait` is high but GPU memory/util is low → **raise the gate**.
4. If you see OOM or memory pressure → **lower the gate or the per-env cap**; confirm
   `memory_safety()["fits"]`.
5. For steady (non-bursty) resident workloads, the gate ≈ `num_gpus × (GPU_mem / per_env_mem)`
   and there's little to hide behind — you're memory-bound; add GPUs or shrink the model.

## 6. Relationship to AgentCore

AgentCore maps **1 session → 1 isolated microVM** and does **not** expose GPU-sharing config
(time-slicing / MPS / MIG) across sessions on one instance
(see [`agentcore-slime-integration.md`](agentcore-slime-integration.md) §2.3–2.4,
[`sandbox-comparison.md`](sandbox-comparison.md)). Its GPU story is Runtime **instances** +
a **capacity provider** (`agentcore add capacity-provider --instance-types g6... --name
gpu-pool`), where AWS bin-packs sessions across EC2 — you get **isolation and zero ops**, but
not "100 envs on *one* GPU with *my* sharing policy."

So the two paths are complementary:
- **This density (100/instance, controlled sharing)** → **Path A**, this doc: your containers
  on a GPU EC2 box + NVIDIA time-slicing/MPS + `GpuPool`.
- **Managed isolation, scale across instances** → AgentCore instances + capacity provider
  (`profiles.py:GPU_SANDBOX`, [`demo-cases.md`](demo-cases.md)).

Either way slime is unchanged: each env still satisfies slime's `Sandbox` Protocol, and the
pool just supplies the `env=` for GPU-touching `exec` calls.

## 7. Honest caveats

- **No memory isolation on L40S** (no MIG). The per-env cap is software (torch memory
  fraction / MPS limit); a process that ignores it can still OOM neighbours. For hard
  isolation you need Ampere/Hopper MIG, or one env per GPU.
- **The demo workload is tiny** (512×512 matmul, ~12 MiB). It proves scheduling + real CUDA
  across 100 envs on 8 GPUs; it does **not** prove your specific model fits at a given gate —
  measure with your real workload.
- **`GpuPool` gates GPU work, not total envs.** Holding 100 envs also costs CPU/RAM/file
  descriptors on the host — size the instance for the CPU phase too.
- **Placement across GPUs is per-lease, not sticky.** An env that takes several GPU turns may
  land on different devices across turns; fine for stateless kernels, not for a persistently
  resident per-device cache (add device affinity if you need that).
