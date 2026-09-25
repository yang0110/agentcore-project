# demo_cpu_branching — one episode → many training samples ("fan-out")

> **EXPERIMENTAL — not part of the main 7-demo set; fixes pending a live re-run.** History of the blockers:
> 1. *CLI exits 1 before any model call* (`fan_out=0`): the CLI refuses `--permission-mode
>    bypassPermissions` as root. Fixed by `IS_SANDBOX=1` (confirmed on the sandbox 2026-09-19; the empty-group
>    crash it exposed is fixed in 18ccb91 / 3ecddc9).
> 2. *One turn, reward 0* (`exit=0 fan_out=1`, no `solution.py`, the 2026-09-19 smoke): `run.sh` had no
>    `--sglang-tool-call-parser`, so the adapter returned Qwen3's `<tool_call>` text as a plain `end_turn`
>    reply and the CLI stopped. Fixed 2026-09-25: `--sglang-tool-call-parser qwen25 --sglang-reasoning-parser qwen3`
>    (the A3/B3 fix terminalbench/scale_lanes already carry).
> 3. *CLI `exit=1` "response exceeded the output token maximum"*: a turn that hit the 1024-token per-turn cap
>    came back as `stop_reason=max_tokens`, which the CLI treats as fatal. Now `MAX_RESPONSE_LEN` (default 2048).
>
> Checked 2026-09-25 without a trainer: the Runtime starts a session, the CLI (2.1.278) runs as root with
> `IS_SANDBOX=1`, the sandbox dials back to `PRIVATE-IP-f99864bc:18021`, the qwen25 parser turns a Qwen3
> `<tool_call>` into a tool call, and the checker scores no/buggy/fixed `solution.py` 0 / 0 / 1. Whether the
> policy now reaches sub-agents / compaction (`fan_out>1`) needs the RL smoke.

Every other demo treats one episode as one training sample. This one does not. It runs the **real
Claude Code CLI** as the harness and deliberately provokes the two CLI behaviors that split a single
episode into several trainable branches — **sub-agents** (the `Task` tool) and **context
summarization / compaction** — so a single rollout fans out into a **list of `Sample`s** (one per
leaf of the trajectory tree). The interesting variable here is the *fan-out*, not the task; the
dataset is a single family (SWE bug-fix).

The policy is Qwen3-4B trained by slime. Unlike the other CPU demos, the harness is **not** the
in-process mini-agent — it is the genuine Claude Code CLI, co-located with slime and wired to slime's
`AnthropicAdapter`, whose **tools execute in a remote CPU AgentCore Runtime microVM**. Because the
sandbox has to host the CLI itself, its image is **not** the plain CPU checker image the other CPU
demos use — it **bakes node 22 + the `claude` CLI** (`deploy/Dockerfile.sandbox.cli`).

This demo is **REMOTE-ONLY** (see [Remote-only](#remote-only-no-local-sandbox)) and this folder is
**self-contained**: it owns its `data/`, `ckpt/`, `run.sh`, a vendored `agentcore_slime_demo/`, and
its own `deploy/` Dockerfiles. The only outside prerequisites are those already in the `agentcore_p4`
container: the slime install (`/root/slime`), the base Qwen3-4B Megatron checkpoint (`CKPT_BASE`,
default `/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`), and AWS credentials.

---

## How the fan-out is wired end-to-end

One rollout is one episode of the real CLI. Two things make an episode produce **more than one**
`Sample`:

- **Sub-agents.** When the CLI spawns a sub-agent (`Task` tool), that sub-agent runs its own call
  chain with its own token context. That chain is a separate branch of the trajectory tree.
- **Summarization / compaction.** When the context is summarized, the token prefix is reset: the
  calls before the summary close as one branch and the calls after it open a new one.

Two source files turn that tree into correct training data:

- **`agentcore_slime_demo/claude_code_generate.py`** — the custom generate hook
  (`--custom-generate-function-path`). It runs the CLI in the sandbox against the slime-served
  Anthropic endpoint, then `finish_session` **linearizes the per-session trajectory tree into one
  `Sample` per leaf**, every leaf sharing the same `rollout_id` and the episode's single **outcome
  reward**. It returns `list[Sample]`.
- **`agentcore_slime_demo/rollout_advantage.py`** — the custom advantage
  (`rollout_level_grpo_advantage`). It computes the GRPO baseline over **distinct rollouts**
  (episodes), never over the fanned-out leaves, then broadcasts each episode's advantage to all of
  its leaves.

### Three flags make the reconciliation correct

An incidental fan-out must not silently reweight the update. Three settings in `run.sh` guarantee
each **episode** counts once, regardless of how many leaves it produced:

1. **`--custom-advantage-function-path …rollout_advantage.rollout_level_grpo_advantage`** — the
   built-in GRPO path baselines over *samples*; under fan-out its `_post_process_rewards` guard
   (`rewards.shape[-1] == n_samples_per_prompt * rollout_batch_size`) fails and it mis-groups the
   inflated sample count. So the baseline is computed here, per episode.
2. **`--disable-rewards-normalization`** — keep **raw** rewards flowing to the actor; the custom
   advantage does the only normalization.
3. **No `--calculate-per-token-loss`** (left OFF on purpose) — the default `sum_of_sample_mean`
   reducer plus the shared `rollout_id` makes slime key the loss denominator on `rollout_mask_sums`
   (a per-**episode** token-weighted mean), so each episode counts once in the loss too. Passing the
   flag would switch to a global per-token mean and let prolific (many-leaf) episodes dominate.

This is the concrete implementation of the report's §"When one episode becomes many samples".

---

## Constraints (why MODE=A, dp=1, and sync save)

- **MODE=A only** (enforced in `run.sh`). Returning `list[Sample]` puts slime on the nested
  `list[list[Sample]]` shape. `train_async.py`'s stock 1-step-async driver tolerates it
  (`isinstance(group[0], list)` guards everywhere); the **fully-async** driver
  (`generate_rollout_fully_async`) crashes on it (its `_key` sort). `MODE=B` is rejected.
- **`dp_size == 1` required.** The custom advantage runs in the Megatron actor on each rank's
  partition and assumes a whole prompt-group (all its rollouts + their leaves) is co-resident on the
  rank — true only at data-parallel size 1. With 4 actor GPUs that is `TP=2 × PP=2 = 4`.
- **Sync checkpoint save.** `PP=2` breaks **both** of Megatron's async-checkpoint backends (the
  `pidfd_getfd` `EPERM` also seen in the earlier real-CLI runs), so this demo saves synchronously
  (no `--async-save`, no persistent-async-ckpt init hook).

## Remote-only (no local sandbox)

`SANDBOX=local` is **rejected**: the sandbox must host the real Claude Code CLI, so it needs the
CLI-capable image, and a CPU training run must never silently fall off the AgentCore substrate it is
meant to exercise. Use `SANDBOX=agentcore-runtime` (default). It **requires**:

- **`AGENTCORE_SANDBOX_RUNTIME_ARN`** — a Runtime built from `deploy/Dockerfile.sandbox.cli` (node 22
  + `claude` baked in; the plain CPU checker image will not work — streaming those binaries through
  `write_file` corrupts them).
- **`ADAPTER_PUBLIC_HOST`** — a private VPC IP of this training box that the sandbox microVM can dial;
  the CLI inside the microVM reaches slime's `AnthropicAdapter` there.
- **`ADAPTER_PORT`** (default `18021`) — a pinned port so one security-group ingress rule stays
  stable across the whole run.

---

## Step by step (numbered scripts; run from this dir on the HOST — container steps `docker exec` themselves)

```bash
bash 1_prepare_data.sh     # 1. container: single-family SWE bug-fix set -> data/swe_{train,eval}.jsonl (REPEAT, default 8)
bash 2_build_image.sh      # 2. arm64 image from deploy/Dockerfile.sandbox.cli (node 22 + claude CLI)
bash 3_push_ecr.sh         # 3. -> ECR acs-sandbox-cli (deploy/images.env)
bash 4_deploy_runtime.sh   # 4. deploy/deploy_branching.py: VPC-mode Runtime acs_sandbox_cli + one SG->SG ingress rule
                           #    (TCP ADAPTER_PORT on the host SG, source = the runtime SG)
                           #    -> deploy/runtime.env: AGENTCORE_SANDBOX_RUNTIME_ARN + ADAPTER_PUBLIC_HOST
bash 5_run.sh              # 5. container: MODE=A train + eval (see the banner for the fixes)
bash 6_cleanup.sh          # 6. delete the Runtime and revoke the SG ingress rule (deploy_branching.py --cleanup)
```

`HOST_SG` overrides the auto-detected (IMDS) host security group. Step 4 is idempotent: the existing
repo, Runtime and SG rule are reused (`FORCE_RECREATE=1` recreates the Runtime to pick up a new image).

The reward is the episode's SWE outcome (bug-fix accepted or not); every leaf of the fanned-out
episode inherits it.

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `agentcore-runtime` | remote CLI sandbox. **`local` is rejected.** |
| `AGENTCORE_SANDBOX_RUNTIME_ARN` | — (required) | CLI-capable sandbox Runtime (node 22 + `claude`) |
| `ADAPTER_PUBLIC_HOST` | — (required) | private VPC IP the microVM dials for slime's adapter |
| `ADAPTER_PORT` | `18021` | pinned adapter port (stable security-group ingress) |
| `RB` / `NS` | 4 / 8 | prompts-per-step / **distinct rollouts** per prompt (leaves inflate the sample count) |
| `NR` | 11 | num-rollout ⇒ real steps `NR-1` |
| `CONC` | 4 | episode concurrency; episodes in flight `S = CONC × num_engines` |
| `MODE` | `A` | **A only** — fully-async (B) crashes on the nested fan-out shape |
| `TEMP` | 1.0 | rollout sampling temperature |
| `BRANCHING_FORK_THRESHOLD` | 512 | lower ⇒ record more compactions as forks (more fan-out) |
| `AGENT_TIME_BUDGET_SEC` | 900 | per-episode wall-clock budget for the CLI |
| `MAX_RESPONSE_LEN` | 2048 | per-turn token cap; a turn that hits it ends the CLI with `exit=1` (keep it above Qwen3's `<think>`) |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |
| `SAVE_INTERVAL` | 1000000 | save-only-at-end (final rollout is force-saved) |
| `SLIME_CUDA_VISIBLE` / `SLIME_NUM_GPUS` | unset / 8 | pin ray to a GPU subset on a shared box (e.g. `2,3,4,5,6,7` / `6` with `ROLLOUT_GPUS=2`) |

## What to watch
`rollout/raw_reward` (per-episode learning signal), `eval/swe` (held-out solve rate), and the
per-episode leaf count (how much each episode actually fanned out). The point of the demo is that the
advantage and loss stay **per-episode** no matter how the leaf count moves.

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored library: `claude_code_generate.py` (the CLI generate hook +
  `finish_session` tree→leaf linearization), `rollout_advantage.py` (rollout-level GRPO advantage),
  the remote sandbox client, and the task registry. Imported only from here.
- `build_swe_dataset.py` — builds the single-family SWE bug-fix dataset (offline).
- `1_…6_*.sh` + `deploy/` (`demo.env`, `steps_common.sh`, `agentcore_deploy.py`, `deploy_branching.py`) — the numbered steps.
- `prepare_data.sh` — step 1 wrapper · `run.sh` — the remote RL run (step 5).
- `deploy/Dockerfile.sandbox.cli` — the CLI-capable CPU sandbox image (node 22 + `claude`).
  `deploy/Dockerfile.sandbox.cpu` — the plain checker image (reference).
- `data/` — `swe_train.jsonl` / `swe_eval.jsonl` (built by step 1; currently empty until you run it).
