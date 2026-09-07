# Demos Guide — build, run, debug, pitfalls & solutions

The operator's manual for the two end-to-end agentic-RL demos in this repo. It is the
companion to the client deck: the deck tells the *story*, this tells you how to *run it*.

Everything below has been executed live on 2026-09-05 (real AWS + real GPU + real Qwen3-4B).

---

## 1. What we build — two demos, one harness

Both demos are the **same** multi-turn agentic-RL loop (write `solution.py` → run a checker
in the sandbox → read the failure → fix → repeat), the **same** policy contract, and the
**same** slime training-data path. They differ in exactly two dimensions: the **sandbox**
(where the agent's tool actions run) and the **task** (what the checker verifies).

| | **Demo A — CPU microVM** | **Demo B — GPU instance** |
|---|---|---|
| AgentCore mode | **Code Interpreter** (serverless microVM) | **Runtime instance** + GPU capacity provider |
| Compute | CPU, 8-hour sessions | Customer EC2 GPU (e.g. `g6.xlarge` L40S), 14-day sessions |
| Dataset | **MBPP** (`mbpp.py`, full config via HF `datasets`) | **KernelBench** (`kernelbench.py`, real `ScalingIntelligence/KernelBench`: 270 CUDA problems; train level_1=100, eval 20 level_2) |
| Checker verifies | assert-style unit tests pass | output matches a torch reference **and** ran on `cuda` |
| Why this compute | RL default: tool exec is cheap shell/IO; policy GPUs stay on slime | task itself needs CUDA inside the sandbox |
| Scaling story | many cheap microVMs | **GpuPool** dense-packs many envs on few GPUs (§6) |

> **Key architectural point (Agent Lightning decoupling):** the policy's training GPUs
> always live on slime's cluster. The *sandbox* only needs a GPU when the *task* does
> (Demo B). Demo A's sandbox is pure CPU even though the policy is a 4B model on a GPU.

### Both demos run slime in ASYNC (decoupled) mode — the default

Agentic rollouts have long, variable latency (multi-turn code repair in a sandbox), so a
colocated *synchronous* loop wastes wall-clock: the trainer idles while rollouts run and
vice-versa. The canonical run path is therefore slime's **async driver `train_async.py`** on
the **decoupled placement** (trainer GPUs and SGLang inference GPUs are **disjoint**, no
`--colocate`). Async generates rollout *N+1* while training on rollout *N* (a one-step
off-policy pipeline); it hard-requires the decoupled layout (`train_async.py` asserts
`not args.colocate`). Everything runs inside the `slimerl/slime` docker container (trainer +
SGLang), driven by `code/p4_slime_train/run_async.sh` (`DEMO=a|b`).

> The earlier `run_agentcore_mbpp_qwen3_4b.sh` is the **legacy sync/colocate** script (kept for
> reference — it was the first proven P4 run). New runs use `run_async.sh`.

### GPU allocation on this 8×L40S box (async, decoupled)

slime manages only **train + inference** GPUs; the **env** GPUs (Demo B) are deliberately left
out of slime's placement and leased to the sandboxes by `GpuPool`.

| | Train (Megatron actor) | Inference (SGLang) | Envs (sandbox checkers) | Total |
|---|---|---|---|---|
| **Demo A (MBPP, CPU envs)** | 4 GPUs (TP=2 × DP=2) | 4 GPUs (2 engines × TP=2) | **0** — envs are CPU shell/IO | 8 |
| **Demo B (KernelBench, GPU envs)** | 4 GPUs (TP=2 × DP=2) | 2 GPUs (1 engine × TP=2) | **2** — physical GPUs 6,7 via GpuPool | 8 |

- **Demo A (4 + 4 + 0):** the MBPP checker is plain Python/IO, so envs use no GPU; all 8 GPUs
  serve the policy (train + infer).
- **Demo B (4 + 2 + 2):** the KernelBench checker runs torch on CUDA *inside the sandbox*, so
  2 GPUs are reserved for envs. slime is told `--actor-num-gpus-per-node 4 --rollout-num-gpus 2`
  (= 6 GPUs); it never touches GPUs 6,7, which `GpuPool` hands to the env checkers (§6).

### Checkpoints & eval cadence (disk-saving policy)

- **No mid-training checkpoints.** slime always force-saves at the final rollout
  (`rollout_id == num_rollout-1`), so `run_async.sh` sets `--save-interval` huge (`SAVE_INTERVAL`,
  default 1,000,000) and keeps the run under one epoch. Net effect: the **only** checkpoint on disk
  is the one written at the **end** of training (under `--save`). This is deliberate — checkpoints
  of a 4B model are large and we don't need intermediates.
- **Eval every 10 steps.** `--eval-interval` (`EVAL_INTERVAL`, default 10) runs a held-out eval on
  a separate jsonl (`--eval-prompt-data <name> <path>`, built by `02_build_dataset.py`: MBPP
  `--split validation` for Demo A; KernelBench `--level level_2 --limit 20` held-out fusions for
  Demo B, disjoint from the level_1 train pool). slime routes each eval prompt
  through our **same** custom generate with `evaluation=True` — a full multi-turn agentic episode,
  **no gradient** — and logs the mean reward as `eval/<name>` tagged with `eval/step`.
- **Metrics → CSV.** `metrics.py` writes a **per-step train CSV** (`--csv`) and a **per-eval CSV**
  (`--eval-csv`, keyed by `eval/step`), and can plot both (incl. the held-out eval curve). So the
  record is: key training metrics every step, held-out eval every 10 steps, checkpoint only at end.

### The pieces (all under `code/agentcore_slime_demo/`)

- `sandbox.py` — `Sandbox` Protocol (drop-in for slime's `E2BSandbox`); `LocalSandbox`
  (offline) + `AgentCoreSandbox` (real AWS Code Interpreter).
- `task.py` — `CodingTask`, multi-turn, exit-code reward. `check_mode="expr"|"assert"`.
- `policy.py` — `MockPolicy` (model-free), **`LocalHFPolicy`** (real model, local
  transformers), `SGLangPolicy` (production router path).
- `harness.py` — `run_episode()`: the multi-turn loop; optional `gpu_pool=` gating.
- `gpu_pool.py` — `GpuPool` admission control + `GpuGatedSandbox` decorator (Demo B scaling).
- `generate.py` — `build_token_stream()`: trajectory → loss-masked `(tokens, mask)` for slime.
- `mbpp.py` / `kernelbench.py` — the two datasets.
- `profiles.py` — `CPU_SANDBOX` / `GPU_SANDBOX` (compute + provisioning metadata).
- `megatron_init.py` — slime `--custom-megatron-init-path` hook; enables Megatron's spawn-based
  persistent async checkpoint worker to avoid the fork-of-CUDA-process save deadlock (§7).

---

## 2. Quick start (TL;DR)

```bash
cd code

# --- offline dry runs (no AWS, no model): prove the loop end-to-end ---
python3 scripts/e2e_rl.py --profile cpu --policy mock --limit 3      # MBPP, LocalSandbox
python3 scripts/e2e_rl.py --profile gpu --policy mock                # KernelBench, LocalSandbox

# --- REAL policy (Qwen3-4B), local sandbox ---
HF_HOME=./.hf_cache python3 scripts/e2e_rl.py --profile gpu --policy real --gpu-concurrency 8

# --- REAL policy on REAL AWS AgentCore microVM (Demo A live) ---
HF_HOME=./.hf_cache SANDBOX=agentcore AWS_REGION=us-west-2 \
    python3 scripts/e2e_rl.py --profile cpu --policy real --limit 3

# --- tests (19, all offline; GPU/network ones skip cleanly) ---
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q
```

**Full slime training (async, decoupled) — the canonical run path**, inside the
`slimerl/slime` container (`agentcore_p4`):

```bash
# Demo A — MBPP, 4 train + 4 infer + 0 env GPUs
docker exec agentcore_p4 bash -lc \
  'cd /work && DEMO=a SANDBOX=local bash code/p4_slime_train/run_async.sh'

# Demo B — KernelBench, 4 train + 2 infer + 2 env GPUs (GpuPool on physical 6,7)
docker exec agentcore_p4 bash -lc \
  'cd /work && DEMO=b SANDBOX=local GPU_POOL_CONCURRENCY=16 bash code/p4_slime_train/run_async.sh'
```

Build the datasets first with `02_build_dataset.py --dataset mbpp|kernelbench`.

Expected: **solved 2/3** on both live profiles (a healthy mixed reward — the RL signal),
each solved episode printing `training tokens=N (trainable=M)`.

---

## 3. The unified driver — `scripts/e2e_rl.py`

One entry point behind both demos.

| Flag / env | Meaning | Default |
|---|---|---|
| `--profile cpu\|gpu` | Demo A (MBPP) or Demo B (KernelBench) | *required* |
| `--policy real\|mock` | `LocalHFPolicy` (Qwen3-4B) or model-free `MockPolicy` | `real` |
| `--model` / `POLICY_MODEL_PATH` | HF model id/path | `Qwen/Qwen3-4B` |
| `--device` | policy device | `cuda:0` |
| `--limit` | number of tasks | `3` |
| `--gpu-concurrency` | `GpuPool` admission gate (Demo B) | `8` |
| `--per-env-mem-mib` | per-env soft memory cap (Demo B) | `2048` |
| `SANDBOX=local\|agentcore` (env) | sandbox backend | `local` |
| `AWS_REGION` (env) | AWS region for AgentCore | — |

The driver loads the policy **once** (model weights stay resident), runs each task through
`run_episode`, and for every real-policy episode builds and reports the loss-masked token
stream — proving each rollout is genuinely trainable data, not just a pass/fail.

---

## 4. How to build / prerequisites

- **Python deps** (this box is PEP 668 externally-managed → use `--user --break-system-packages`):
  ```bash
  pip install --user --break-system-packages bedrock-agentcore datasets transformers torch
  ```
  Installed pkgs land in `~/.local`. `torch` here is 2.8 + cu128.
- **slime** is a cloned repo (`repos/slime`), **not** pip-installed. Code adds it to
  `sys.path` when needed; the protocol-conformance test asserts our sandboxes are real
  `slime.agent.sandbox.Sandbox` instances.
- **AWS / AgentCore:** works from this box's EC2 instance role
  (`SSMDefaultRoleForOneClickPvreReporting`, acct `<ACCOUNT_ID>`) with **no extra AgentCore
  policy attached** — confirmed by a live session. Region `us-west-2`. A bearer token is
  present in the env (`AWS_BEARER_TOKEN_BEDROCK`); **never echo it**.
- **HF cache:** `~/.cache/huggingface` is root-owned here → always set `HF_HOME=./.hf_cache`
  (writable project dir). First model/dataset load needs network.
- **GPU:** this box has 8× NVIDIA L40S (46 GB each). Demo B runs on real CUDA.

---

## 5. How to debug

- **Start offline, model-free.** `--policy mock` isolates the loop from model/AWS issues.
  If mock fails, the bug is in the harness/task/sandbox, not the policy or AWS.
- **Peel one layer at a time:** `local` sandbox before `agentcore`; `mock` before `real`.
- **Read the full observation.** The driver prints only the first line per turn. To see the
  whole checker output (which assertion failed, stderr, the generated code), run a single
  task inline and print `traj.turns[i].observation` + `CodingTask.extract_code(...)` — see
  the failure-triage snippets used for kb_fused_bias_gelu / mbpp_603.
- **Sandbox identity:** printed per episode. `ac-...` = a **real** AgentCore session;
  `local-...` = LocalSandbox. If you expected AWS and see `local-`, `SANDBOX` wasn't set.
- **Is the failure legit?** A model producing runnable code that fails an *assertion*
  (exit 1, not `NameError`/`SyntaxError`) is a real RL signal, not a bug. Verify by reading
  the diff between generated output and the reference (e.g. wrong broadcast axis, off-by-one).
- **Tests:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is **required** here (a broken libtmux pytest
  plugin otherwise errors on collection). 19 tests; GPU/network-dependent ones `skip` cleanly.

---

## 6. Scaling Demo B — ~100 envs on just **2** GPUs (GpuPool)

In Demo B only **2** of the 8 GPUs (physical **6,7**) are reserved for env checkers — the
other 6 run the policy (4 train + 2 infer). Yet a rollout step fans out to *many* concurrent
episodes (RB × NS can be ~64–100). How do ~100 envs share 2 GPUs?

**Because the constraint is GPU *memory*, not env *count*, and RL GPU use is *bursty*.** An
env is a multi-turn code-repair loop; it only touches CUDA for the brief moment its checker
runs (`python3 run_checks.py`). The rest of the episode is the policy generating and the
sandbox doing IO. So instead of 100 resident CUDA contexts, we put a **queue in front of the
2 GPUs**:

- **`GpuPool`** (`gpu_pool.py`) is backend-agnostic admission control:
  - a **semaphore gate** `max_concurrent_gpu` (env `GPU_POOL_CONCURRENCY`, default 16) — at
    most that many checkers run CUDA *at once*; the other ~84 queue on the semaphore, holding
    **no** GPU memory while they wait.
  - **least-loaded assignment** across the reserved devices, so admitted work spreads evenly
    over GPUs 6 and 7 (≈8 co-resident per GPU at gate 16).
  - a **per-env soft memory cap** via lease env vars `CUDA_VISIBLE_DEVICES` (pins to physical
    6 or 7) + `ENV_GPU_MEM_FRACTION` (`GPU_POOL_PER_ENV_MIB`, default 2048 → the task calls
    `torch.cuda.set_per_process_memory_fraction`). Worst-case co-residency × per-env MiB must
    fit one GPU — `GpuPool.memory_safety()` checks this statically (8 × 2048 MiB = 16 GB ≪
    46 GB L40S → fits with room to spare).
- **`GpuGatedSandbox`** transparently wraps the sandbox and routes GPU-touching exec through
  the pool. For KernelBench the checker imports torch *internally*, so the command text can't
  reveal GPU use → we gate **every** exec (`AGENTCORE_GPU_GATE=all`, one checker exec per turn).
- **Wiring:** `generate.py` builds **one `GpuPool` per rollout-worker process** (a module
  singleton) so all concurrent episodes in that worker share the *same* semaphore — a
  per-episode pool would defeat the gate. It's created only when `AGENTCORE_GPU_GATE` is set
  (Demo B); Demo A leaves it `None`. Devices come from `GPU_POOL_DEVICES=6,7`.

So the split is really **6 GPUs owned by slime + 2 GPUs owned by GpuPool**, and the "100 envs
on 2 GPUs" story is: 100 logical envs, at most 16 on the GPUs at any instant, each capped to a
memory slice, the rest queued — CUDA memory, not env count, is what's rationed.

- Verified live (the mechanism, at larger scale): 100 envs on 8× L40S — gate 8 → 38.7 s,
  gate 32 → 26.9 s, spread evenly, peak ~12 MiB/env. See `scripts/gpu_density.py` and
  `docs/gpu-density.md`. The Demo-B run pins the same pool to just devices 6,7.
- L40S (Ada) has **no MIG** (that's Ampere/Hopper) → sharing is time-slicing / CUDA MPS with
  **software** memory caps, no HW isolation. The semaphore gate is what keeps time-slicing
  contention and memory pressure bounded.
- AgentCore does **not** expose per-instance GPU sharing across sessions → this is a DIY
  container concern, complementary to the managed GPU capacity-provider path.

---

## 7. Pitfalls & solutions (all hit live, all fixed)

| Pitfall | Symptom | Solution |
|---|---|---|
| SGLang server won't start | `sgl_kernel/common_ops...: undefined symbol _ZN3c104cuda9SetDeviceEa` | sgl_kernel↔torch **ABI mismatch**. Use `LocalHFPolicy` (direct transformers) for the demo; production rebuilds a matched sgl_kernel/torch. |
| `apply_chat_template(tokenize=True)` | `RuntimeError: Could not infer dtype of tokenizers.Encoding` in `.generate()` | Newer `tokenizers` returns an `Encoding` object. Use `tokenize=False` then `tokenizer(text, return_tensors="pt")`. (Done in `LocalHFPolicy`.) |
| `/workspace` not writable on AgentCore | `Permission denied` on every write | Code Interpreter runs as **unprivileged** `genesis1ptools` (uid 991), **not root**. Native CWD (`/opt/amazon/genesis1p-tools/var`) *is* writable & persists. Fix: `AgentCoreSandbox` `workdir=""` (native CWD, no `cd`, relative paths). |
| `read_file` returns junk | got a dict, not text | Code Interpreter `readFiles` returns `{"content":[{"type":"resource","resource":{"text":...}}]}`. Fix: extract `resource["text"]`. |
| HF cache PermissionError | can't write `~/.cache/huggingface` | root-owned here. Set `HF_HOME=./.hf_cache`. |
| MBPP repo id fails | `HfUriError` on `load_dataset("mbpp")` | Use namespaced id `google-research-datasets/mbpp`, config `"full"`. |
| pip refuses install | `externally-managed-environment` (PEP 668) | `pip install --user --break-system-packages`. |
| pytest collection error | libtmux plugin crash | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. |
| Qwen3 emits `<think>` blocks | slow, bloated trainable span | Pass `enable_thinking=False` to the chat template (`LocalHFPolicy(enable_thinking=False)`). |
| GPU test uses ambient env | `CUDA_VISIBLE_DEVICES` leaks from host | Assert on `pool.stats.admitted`, not the echoed device value. |
| Speedup metric not in trajectory | `SPEEDUP` line missing on pass | Known minor: `task.observation` drops stdout on pass; the metric is still printed by the checker. |
| Checkpoint save deadlocks | job stuck in `save_model` → `…join()`, `.distcp` files already fully written, GPUs 0% (intermittent) | **Root cause:** Megatron's checkpoint writer forks a writer process off the MAIN training process (`mp.get_context("fork")`); forking a process holding CUDA/NCCL/inductor threads can inherit a locked mutex → child hangs forever. **Fix (solved):** route saves through Megatron's SPAWN-based `PersistentAsyncCaller` (clean interpreter, safe fork). It needs `--async-save` **and** `init_persistent_async_worker()` — slime forces `use_persistent_ckpt_worker=True` but its own `init()` bypasses `megatron.training.initialize`, so that call is never made and `--async-save` alone falls back to the fork-based `TemporalAsyncCaller`. `run_async.sh` supplies the missing call via `--custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt` (slime's supported init hook). slime still finalizes synchronously at the final rollout, so the save completes before exit. |
| Eval dict has no `eval/step` in stdout | `eval N: {...}` printed without an `eval/step` key | slime prints the dict *before* adding `eval/step`. `metrics.py` falls back to the logged id `N` (which is the training step) as the eval CSV key — no fix needed. |

---

## 8. What's real vs. still simulated

**Real:** the multi-turn loop; the policy (Qwen3-4B generating code + returning
token_ids/logprobs); AgentCore Code Interpreter microVMs (Demo A, live AWS); CUDA execution
+ correctness checking on real L40S (Demo B); GpuPool dense-packing; loss-masked training
tokens. Both demos ran live 2026-09-05.

**Real (P4 done):** the **slime weight-update step** runs end-to-end. Real GRPO training steps
executed inside the `slimerl/slime` container — real 4-engine SGLang rollout → our custom
generate → Megatron actor (ref logprobs + GRPO advantage + policy loss + optimizer + weight
sync) → torch_dist checkpoint. Proven on **both** `SANDBOX=local` and **real AgentCore
microVMs**, including a 5-step learning run with **nonzero gradients** (grad_norm 1.5→1.05,
kl_loss monotonically rising). This was first done in the sync/colocate script; `run_async.sh`
now makes **async/decoupled** the default (§1).

**Real (eval + metrics, async):** the **held-out eval path** runs end-to-end in async mode — slime
routes each eval prompt through our *same* custom generate (`evaluation=True`); a live MBPP eval
logged `eval/mbpp = 0.625` (10/16 solved) mid-run. `metrics.py` splits the log into a per-step
train CSV and a per-eval CSV (keyed by `eval/step`). Checkpoints are written **only at the end**
(large `--save-interval`) — verified by a run that saved exactly one `iter_*` at the final step,
and the end-of-training save now completes reliably via the spawn-based persistent checkpoint
worker (the fork-of-CUDA-process deadlock is **fixed** — §7, `megatron_init.py`).

**Not yet:** (a) a logged end-to-end run of the **async Demo B** combination specifically —
async trainer + SGLang + the GpuPool-gated KernelBench envs on the 4+2+2 split all at once
(each piece is proven; the combined async run is the next milestone); (b) **Demo B on a real
AgentCore GPU Runtime instance** — the GPU path runs on this box's L40S today (a custom Runtime
container, demo-plan Option B, is the remaining piece; the *managed* Code Interpreter can't run
the full multi-user Claude Code harness); (c) **SGLang throughput path for the standalone
`e2e_rl.py` driver** — blocked on the local ABI break, so that driver uses `LocalHFPolicy`
(the in-container slime run uses real SGLang engines).

---

## 9. Related docs

- `docs/demo-cases.md` — the CPU vs GPU case framing.
- `docs/gpu-density.md` — the "100 envs on one GPU" deep dive.
- `docs/fault-recovery.md` — session death / retry / self-heal.
- `docs/observability.md` — the 5-stage metrics map (trainer → sandbox → inference → sync).
- `docs/agentcore-slime-integration.md` — full cited integration reference.
- `docs/agent-lightning-v1-harnessed-agentic-rl.md` — the design-principle source (arXiv 2608.17528).
