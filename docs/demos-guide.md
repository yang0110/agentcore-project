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
| Scaling story | many cheap microVMs | checker CUDA runs on a **remote GPU sandbox Runtime** (a disposable g6), off the trainer box (§6) |

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

slime manages all **train + inference** GPUs on this box; the KernelBench checker's GPU is
**not** on this box at all — it lives on the remote GPU sandbox Runtime (§6).

| | Train (Megatron actor) | Inference (SGLang) | Envs (sandbox checkers) | Total |
|---|---|---|---|---|
| **Demo A (MBPP, CPU envs)** | 4 GPUs (TP=2 × DP=2) | 4 GPUs (2 engines × TP=2) | **0** — envs are CPU shell/IO | 8 |
| **Demo B (KernelBench, GPU envs)** | 4 GPUs (TP=2 × DP=2) | 4 GPUs (2 engines × TP=2) | **0** — checker GPU is on the remote sandbox Runtime | 8 |

- **Demo A (4 + 4 + 0):** the MBPP checker is plain Python/IO, so envs use no GPU; all 8 GPUs
  serve the policy (train + infer).
- **Demo B (4 + 4 + 0):** the KernelBench checker runs torch on CUDA, but *inside the remote
  GPU sandbox Runtime #2* (a g6 capacity-provider instance), not on this box. So all 8 local
  GPUs go to slime and none are held back for envs (§6).

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
- `harness.py` — `run_episode()`: the multi-turn loop.
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
HF_HOME=./.hf_cache python3 scripts/e2e_rl.py --profile gpu --policy real

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

# Demo B — KernelBench, 4 train + 4 infer + 0 env GPUs (checker GPU on the remote sandbox Runtime)
docker exec agentcore_p4 bash -lc \
  'cd /work && DEMO=b SANDBOX=agentcore-runtime \
     AGENTCORE_SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/acs_sandbox_gpu-... \
     bash code/p4_slime_train/run_async.sh'
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

## 6. Demo B's checker GPU — a remote GPU sandbox Runtime, not a local pool

In Demo B **none** of the 8 local GPUs are held back for env checkers — all 8 go to slime
(4 train + 4 infer). The KernelBench checker still needs CUDA (it imports torch and asserts
the candidate `ModelNew` ran on a `cuda` device), but that CUDA runs on a **separate remote
AgentCore GPU Runtime** — the sandbox as its own Runtime #2 (`sandbox_server_agent.py` on a
**g6** capacity-provider EC2 GPU instance).

- The harness drives the multi-turn loop **in-process on this box**, but each `write_file` /
  `exec` / `read_file` — including `python3 run_checks.py` — is dispatched to the remote GPU
  Runtime over `invoke_agent_runtime`. Selected with `SANDBOX=agentcore-runtime`
  (→ `AgentCoreRuntimeSandbox`) + `AGENTCORE_SANDBOX_RUNTIME_ARN` pointing at that Runtime.
- This is the **③ harness ⟂ sandbox** decoupling made real: the trainer box owns only the
  policy GPUs; the checker's CUDA is isolated on a disposable g6 that AgentCore provisions and
  tears down. Deploy it with `code/deploy/deploy_two_runtime.py` (one-time `{"mode":"prepare"}`
  invoke installs the CUDA torch stack onto the persistent `models` volume).
- **Demo B4** shows how to pack *several* sandboxes onto that single g6 via a Ray GPU-sharing
  farm running *inside* the sandbox Runtime — sharing is a concern of the remote Runtime image,
  not of the trainer box.

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
+ correctness checking on real L40S (Demo B); loss-masked training
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
async trainer + SGLang + KernelBench envs whose checker runs on the remote GPU sandbox Runtime,
all at once (each piece is proven; the combined async run is the next milestone); (b) **Demo B on a real
AgentCore GPU Runtime instance** — the sandbox Runtime #2 image is built and the GPU Runtime
deploys, but a full logged training run against it is still pending (the *managed* Code
Interpreter can't run the full multi-user Claude Code harness); (c) **SGLang throughput path for the standalone
`e2e_rl.py` driver** — blocked on the local ABI break, so that driver uses `LocalHFPolicy`
(the in-container slime run uses real SGLang engines).

---

## 9. Related docs

- `docs/demo-cases.md` — the CPU vs GPU case framing.
- `docs/fault-recovery.md` — session death / retry / self-heal.
- `docs/observability.md` — the 5-stage metrics map (trainer → sandbox → inference → sync).
- `docs/agentcore-slime-integration.md` — full cited integration reference.
- `docs/agent-lightning-v1-harnessed-agentic-rl.md` — the design-principle source (arXiv 2608.17528).
