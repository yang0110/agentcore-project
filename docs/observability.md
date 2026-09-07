# Observability: what to collect across the whole agentic-RL workflow

Training an agent with slime × AgentCore × Claude Code is a **five-stage pipeline**, and each
stage fails in its own way. If you only watch reward, you can't tell a *learning* problem (flat
advantage, KL blow-up) from a *plumbing* problem (sandboxes dying, router queueing, weight sync
stalling). This doc says exactly **which metrics to collect at each stage, where they come from,
and what they tell you** — grounded in real code and primary docs.

```
   ┌─────────┐  weights   ┌──────────────┐  tokens+logprobs  ┌───────────┐  exec/files  ┌──────────────┐
   │ TRAINER │──────────▶ │  INFERENCE   │◀─────────────────│  HARNESS  │────────────▶│  SANDBOX     │
   │ (slime/ │  (weight   │  (SGLang)    │   (Anthropic       │ (Claude   │   (checker,  │ (AgentCore   │
   │ Megatron)│  sync)    │              │    adapter)        │  Code)    │    files)    │  microVM)    │
   └─────────┘            └──────────────┘                    └───────────┘              └──────────────┘
        ▲                        │  rollout samples (reward, tokens, status)                    │
        └────────────────────────┴──────────────────────────────────────────────────────────────┘
```

**Grounding key.** slime facts cite `repo path:line` under `repos/slime` (verified 2026-09-03).
External facts carry source URLs. Anything not verified in code is flagged **[unverified]**.

slime centralizes its sinks in `slime/observability/`: the generic emitter is
`logging_utils.log(args, metrics, step_key)` (`slime/observability/logging_utils.py:45`), which
fans out to **Weights & Biases** `wandb.log(metrics)` (`:47`) and **TensorBoard**
`_TensorboardAdapter(args).log(...)` (`:51`). wandb metric-axis binding lives in
`_init_wandb_common` (`slime/observability/wandb_utils.py:167-176`): `train/*`→`train/step`,
`rollout/*`+`perf/*`→`rollout/step`, `eval/*`→`eval/step`, plus `multi_turn/*`, `passrate/*`.

---

## 1. TRAINER metrics — is the policy actually learning? (`train/*`)

Emitted per train step; the `log_dict` is assembled in
`slime/backends/megatron_utils/model.py:882-898`. `{role_tag}` is `""` for the actor, or e.g.
`critic-` for the value net.

| Metric key | Source (file:line) | What it tells you / why watch it |
|---|---|---|
| `train/{role_tag}loss` | `loss.py:1142` | Total policy-loss scalar. |
| `train/{role_tag}pg_loss` | `loss.py:1143` | Policy-gradient term — the actual RL signal. |
| `train/{role_tag}pg_clipfrac` | `loss.py:1145` | Fraction of tokens hitting the PPO clip. Persistently high → LR/step too large. |
| `train/{role_tag}ppo_kl` | `loss.py:1146` | KL(new‖old) per step. **The single most important stability gauge** — spikes precede divergence. |
| `train/{role_tag}entropy_loss` | `loss.py:1144` | Policy entropy. Collapsing → premature determinism / mode collapse. |
| `train/{role_tag}kl_loss` | `loss.py:1153` | KL-to-reference penalty (only if `use_kl_loss`). Guards against drifting off the base model. |
| `train/{role_tag}grad_norm` | `model.py:886` | Pre-clip gradient norm. NaN/Inf or huge spikes = the canonical "training just broke" alarm. |
| `train/{role_tag}lr-pg_{id}` | `model.py:893` | LR per optimizer param group — confirms the schedule/warmup is doing what you set. |
| `train/{role_tag}global_batch_size` | `model.py:896` | Effective batch size *after aborted samples are dropped* — see §2; a shrinking batch means the sandbox layer is bleeding rollouts. |
| `train/{role_tag}mtp_{i}_loss`, `…mtp_loss` | `model.py:889-890` | Multi-token-prediction losses (only if `enable_mtp_training`). |
| `train/{role_tag}ois`, `tis`, `tis_clipfrac` | `loss.py:1158-1162` | (Truncated) importance-sampling correction for off-policy/stale rollouts — TIS clipfrac tells you how off-policy your data is. |
| `train/{role_tag}opsm_clipfrac`, `opd_reverse_kl` | `loss.py:1165,1170` | Optional advanced-objective diagnostics. |
| **Critic:** `train/critic-value_loss`, `value_clipfrac` | `loss.py:1224-1227` | Value-head fit quality (PPO/GAE runs only). |

**Rollout-field summaries** (per-token means, reported under `rollout/` from the Megatron side,
`slime/observability/train_metric_utils.py:178-187`): `log_probs`, `ref_log_probs`,
`rollout_log_probs`, `returns`, `advantages`, `values`, `teacher_log_probs`, `raw_reward`.
Watch **`train/rollout_logprob_abs_diff`** (`loss.py:1150`) and the gap between
`rollout_log_probs` (what SGLang sampled) and `log_probs` (what Megatron recomputes): a large gap
is the classic **train/inference numerical-mismatch** bug — the reason Agent Lightning's gateway
returns token ids+logprobs instead of retokenizing.

---

## 2. HARNESS / ROLLOUT metrics — are episodes healthy and diverse? (`rollout/*`, `eval/*`)

Entry point `log_rollout_data(...)` (`slime/observability/rollout_metrics.py:256-271`):
`rollout/*` ← `compute_metrics_from_samples` (`:266`), and it merges `rollout_extra_metrics`
returned from your generate function (`slime/ray/rollout.py:170,177`).

| Metric key (under `rollout/`) | Source | What it tells you |
|---|---|---|
| `response_len/{mean,median,max,min}` | `rollout_metrics.py:48` | Trajectory length. Creeping toward the cap → truncation, reward-hacking via verbosity. |
| `truncated_ratio` | `rollout_metrics.py:55` | Fraction hitting the token cap before finishing — these have degraded/zero learning signal. |
| `zero_std/count_{reward}` | `rollout_metrics.py:49` | Prompts where **all** samples got equal reward → **zero GRPO advantage** = wasted compute. The KPI for whether your task difficulty is calibrated. |
| `error_cat/{reward_cat}` | `rollout_metrics.py:52` | Histogram of reward/error categories — how rollouts are failing (agent error vs. checker fail vs. abort). |
| `repetition_frac` | `rollout_metrics.py:54` | Degenerate repetition — early reward-hacking / collapse signal. |
| `prefix_cache_hit_rate`, `avg_cached_tokens_per_sample` | `rollout_metrics.py:51` | KV-cache reuse across samples of a prompt — throughput lever. |
| `spec_accept_rate`, `spec_accept_length` | `rollout_metrics.py:50` | Speculative-decoding acceptance (if enabled). |
| `top_p_kept_vocab_per_token` | `rollout_metrics.py:53` | Sampling-distribution sharpness. |
| **Multi-turn:** `multi_turn_metric/round_number_{mean,max,min}`, `raw_response_length/*`, `wo_obs_response_length/*` | `train_metric_utils.py:317-336` | Turns per episode and length with/without observations — the core "is it really multi-turn?" signals for a coding agent. |
| **Pass-rate:** `passrate/pass@{k}` | `metric_utils.py:38`; `train_metric_utils.py:340-355` | pass@k — the headline capability curve. |
| **Eval:** `eval/{key}`, `eval/{key}-pass@k`, `eval/{key}-truncated_ratio` | `rollout_metrics.py:223-253` | Held-out generalization (bound to `eval/step`). |

**Aborted / dropped rollouts** are the harness↔sandbox health signal. Status comes from
`Sample.Status.{ABORTED,TRUNCATED,COMPLETED,PENDING}` (`slime/rollout/sglang_rollout.py:163-304`;
group flag `state.aborted` `:134,376`). This is exactly what our fault-recovery layer produces on
an unrecoverable sandbox death (see [`fault-recovery.md`](fault-recovery.md)) — a clean ABORTED
sample the trainer drops. **Track the abort rate**; a rising abort rate with steady reward means
sandboxes are dying and your effective batch (`train/global_batch_size`, §1) is silently shrinking.

**Claude Code harness fields.** In `examples/coding_agent_rl/generate.py` these are
**console-logged / stored on `sample.metadata`, not emitted as their own wandb keys** — they
surface only via reward/`error_cat` aggregation (verified — no dedicated wandb key):
`reward`, `applied` (patch applied cleanly), `agent_exit_code`, `elapsed`, `segments`
(=turns) at `generate.py:259-267`; `sample.metadata["agent_exit_code"]` (`:250`);
`abort_reason` via `_abort_result` (`:329`); eval `grading_solved`, `applied_cleanly` (`:357-359`).
**Recommendation:** if you want the Claude Code exit code / wall-clock / patch-apply rate on your
dashboards, add them to the `metrics` dict your generate function returns (they then ride the
`rollout_extra_metrics` merge at `rollout/rollout.py:170`) — don't rely on the console line.

---

## 3. SANDBOX (AgentCore) metrics — are microVMs alive, fast, and in-quota?

slime has no knowledge of AgentCore internals — you collect these **from AWS**, correlating on
`session.id` (the slime rollout's session ID passed as the AgentCore runtime session ID).

### 3a. CloudWatch runtime metrics (namespace `AWS/Bedrock-AgentCore`, 1-min batches)
Source: [AgentCore generated runtime observability data](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-runtime-metrics.html)

| Metric | What it tells you for RL |
|---|---|
| `Invocations` / `Invocations (aggregated)` | Tool-call volume = rollout activity. Drops = the rollout loop stalled. |
| `Session Count` (cumulative) vs. `ActiveSessionCount` (real-time gauge, `Service` dim ∈ `AgentCore.Runtime`/`AgentCore.CodeInterpreter`/`AgentCore.Browser`) | New sessions per minute vs. concurrently-live sessions. `ActiveSessionCount` is your **concurrency-vs-quota** headroom — the metric that catches leaked/`HealthyBusy`-stuck sessions before they exhaust quota. |
| `Throttles` (429 `ThrottlingException`) | You're exceeding TPS/quota → back off boots or raise quota. Feeds the retry logic in `recovery.py`. |
| `System Errors` (500 `InvocationError.Internal`) | Service/infra faults → the recoverable class in `is_recoverable`. |
| `User Errors` (400/403/404/409: `Validation`/`AccessDenied`/`ResourceNotFound`/`Conflict`) | Client-side bugs. **`AccessDenied` is fatal** — our classifier stops retrying it (`recovery.py`). |
| `Total Errors` | Errors as % of invocations — the top-line sandbox-health SLO. |
| `Latency` | End-to-end request time (receive → final response token). Rising tail latency = slow rollouts throttling the whole trainer. |
| `CPUUsed-vCPUHours`, `MemoryUsed-GBHours` (dims `Service, Resource, Name`) | Resource + cost tracking per agent endpoint (may lag up to 60 min). |
| WebSocket: `ActiveStreamingConnections`, `In/OutboundStreamingBytesProcessed` | Only if you stream; connection-drop detection. |

**Error codes** map straight onto recovery: `InvocationError.Throttling`/`.Internal` →
retry-with-backoff; `.AccessDenied` → fatal abort; `.Conflict` (409, `RetryableConflictException`)
→ SDK auto-retries. See [`fault-recovery.md §1`](fault-recovery.md) for the recovery semantics.

### 3b. Spans, logs, health
- **Span** `InvokeAgentRuntime` (enable observability first): attributes `session.id`, `latency_ms`,
  `error_type` ∈ {throttle, system, user}, `aws.request_id`, `aws.agent.id`, `trace_id` — the
  per-invocation trace to correlate a bad rollout with its exact tool call.
- **`USAGE_LOGS`** (1-s granularity): `agent.runtime.vcpu.hours.used`,
  `agent.runtime.memory.gb_hours.used` per `session.id` — session-level cost attribution.
- **`APPLICATION_LOGS`**: `request_payload`/`response_payload` with `trace_id`/`span_id` for debugging.
- **Health probe:** `GET /ping` → `Healthy`/`HealthyBusy`
  ([http-protocol-contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html)),
  and Code Interpreter `get_session().status ∈ {READY, TERMINATED}`. Our
  `AgentCoreSandbox.healthy()` polls exactly this. All metrics/spans require the one-time
  **CloudWatch Transaction Search** enablement
  ([observability](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html)).

> Emitted in **OpenTelemetry (OTEL)-compatible** format, so you can fan the same telemetry into
> your existing stack instead of (or alongside) CloudWatch.

---

## 4. INFERENCE ENGINE (SGLang) metrics — is generation the bottleneck?

**Important, verified:** slime does **not** scrape SGLang's Prometheus `/metrics` — a repo-wide grep
for `/metrics` finds nothing. slime reads SGLang two other ways, and *also* re-derives per-request
perf from generation trace spans. So you have **two complementary sources**:

### 4a. What slime already collects from SGLang
- **Server topology:** `get_server_info(url)` (`slime/backends/sglang_utils/external.py:74-83`,
  used at `sglang_engine.py:168`) → `tp_size`/`pp_size`/`ep_size`/`num_gpus`/`disaggregation_mode`.
- **Live load / queue depth:** `num_requests_from_load` (`server_control.py:12-29`) parses
  `num_running_reqs` / `num_waiting_reqs` from `GET {url}/v1/loads?include=core` (`:40`). Used by the
  drain loop `abort_server_until_idle` (`:43-63`). **Watch `num_waiting_reqs`** — a growing wait
  queue is the definitive "inference is the bottleneck, trainer is starved" signal.
- **Per-request perf, from `sglang_generate` trace spans** (NOT the metrics endpoint), mapped in
  `slime/observability/rollout_metrics.py:20-41` and reported under `perf/` via `compute_statistics`
  (`_compute_sglang_request_perf_metrics` `:93-129`):
  - `request/e2e_latency`, `request/queue_time`, `decode/throughput` (`:20-24`)
  - Prefill: `prefill/{bootstrap_queue_duration,bootstrap_duration,alloc_wait_duration,forward_duration,transfer_queue_duration,transfer_speed_gb_s,transfer_total_mb,retry_count}` (`:25-34`)
  - Decode: `decode/{prealloc_duration,bootstrap_duration,alloc_wait_duration,transfer_duration,forward_duration}` (`:35-41`)

  (The prefill/decode-transfer split is meaningful under **disaggregated** serving.)

### 4b. What to add: scrape SGLang's Prometheus `/metrics` yourself
For engine-level health (KV pressure, throughput, TTFT) that slime doesn't pull, scrape the server's
`/metrics` endpoint. Exact names, from
[SGLang production metrics](https://docs.sglang.io/references/production_metrics.html):

| Metric | Type | What it tells you |
|---|---|---|
| `sglang:num_running_reqs` | gauge | In-flight requests = engine concurrency. |
| `sglang:num_queue_reqs` | gauge | Waiting queue depth — corroborates §4a's `num_waiting_reqs`; sustained >0 = under-provisioned inference. |
| `sglang:token_usage` | gauge | KV-cache/memory utilization; near 1.0 → preemption/OOM risk. |
| `sglang:num_used_tokens` | gauge | Currently allocated tokens. |
| `sglang:cache_hit_rate` | gauge | Radix/prefix cache effectiveness. |
| `sglang:gen_throughput` | gauge | Generation tokens/s — top-line inference speed. |
| `sglang:time_to_first_token_seconds` | histogram | TTFT — prefill/scheduling latency; spikes right after a weight sync. |
| `sglang:time_per_output_token_seconds` | histogram | Inter-token latency — decode speed. |
| `sglang:e2e_request_latency_seconds` | histogram | End-to-end per request. |
| `sglang:prompt_tokens_total`, `sglang:generation_tokens_total` | counter | Prefill vs. generation token volume. |
| `sglang:spec_num_steps`, `sglang:spec_num_draft_tokens` | gauge | Speculative-decoding activity. |
| `sglang:estimated_flops_per_gpu_total` (+ read/write bytes) | counter | MFU accounting — requires `--enable-mfu-metrics`. |

---

## 5. WEIGHT SYNC (Megatron → SGLang) — the correctness-critical seam

After each train step the new weights must reach the inference engine, or SGLang keeps sampling
with a **stale policy** — silently poisoning the next batch with off-policy data. This is the seam
most likely to fail quietly, so instrument it.

**Path (verified).** `TrainRayActor.update_weights` (`slime/backends/megatron_utils/actor.py:562`)
is decorated `@timer`, calls `self.weight_updater.update_weights()` (`:607`), bracketed by
`print_memory("before/after update_weights")` (`:606,608`). Transport chosen by
`create_weight_updater` (`.../update_weight/__init__.py:42`, NCCL path). The NCCL updater
`UpdateWeightFromDistributed.update_weights` (`update_weight_from_distributed.py:102`) does:
`pause_generation` → `flush_cache` (`:109-110`) → `_send_weights` (`:122`) → `continue_generation`
(`:132`), all gated by `dist.barrier(get_gloo_group())`; broadcast via `dist.broadcast(param.data,
0, group, async_op=True)` (`:350`). SGLang-side receivers in
`slime/backends/sglang_utils/sglang_engine.py`: `update_weights_from_distributed:397`,
`update_weights_from_tensor:245`, `flush_cache:270` (HTTP `GET /flush_cache` `:277`),
`init_weights_update_group:372`.

| What to collect | Source | Why |
|---|---|---|
| **`perf/update_weights_time`** | `@timer` on `actor.py:562` → `perf/{key}_time` (`train_metric_utils.py:376`) | Wall-clock of the whole sync. A rising trend or a step that never emits it = a stalled/hung broadcast (all ranks block on the `gloo` barrier). **The primary weight-sync alarm.** |
| Console `Timer update_weights end (elapsed: Xs)` | `slime/observability/timer.py:32` | Human-readable confirmation each step actually synced. |
| `print_memory` before/after | `actor.py:606,608` | Catches memory blow-ups / leaks around the transfer. |
| `flush_cache` success | `sglang_engine.py:270-277` | The KV cache **must** be flushed post-sync, else the engine serves cached activations from stale weights. A failed flush = subtle correctness bug. |
| `train/rollout_logprob_abs_diff` (§1) | `loss.py:1150` | The **end-to-end correctness check on weight sync**: if synced weights ≠ sampling weights, recomputed logprobs diverge from `rollout_log_probs`. Rising diff = sync is broken even if timing looks fine. |

> **Verified negative:** there is **no bespoke weight-sync timer** beyond the generic
> `@timer("update_weights")` → `perf/update_weights_time`, and **no separate broadcast-bandwidth
> metric**. If you need per-tensor transfer bandwidth, add it in `_send_weights`; slime doesn't emit it.

---

## 6. Timing / perf infrastructure (how `perf/*` is produced)

The `Timer` singleton (`slime/observability/timer.py:15-52`) accumulates named stage durations;
`log_perf_data` (`slime/observability/train_metric_utils.py:358-405`) reads `Timer().log_dict()`
(`:366`), resets (`:367`), and emits **`perf/{stage}_time`** for every stage (`:376`) plus derived:
`perf/step_time` (`:398`), `perf/wait_time_ratio` (`:399`), `perf/actor_train_tflops` (`:392`),
`perf/actor_train_tok_per_s` (`:393`), `perf/log_probs_tflops` (`:386`). Stages actually timed
(`slime/backends/megatron_utils/actor.py`): `train_wait` (`:57,400`), `train` (`:400`),
`actor_train` (`:491`), `ref_model_update` (`:526`), `log_probs`/`ref_log_probs` (`:330`),
`data_preprocess` (`:348`), `sleep` (`:173`), `wake_up` (`:192`), `save_model` (`:537`),
`update_weights` (`:562`).

**`perf/wait_time_ratio` is the pipeline-balance KPI:** it's the share of step time the trainer
spent *waiting on rollouts* (`train_wait`). High ratio → the trainer is starved and the bottleneck
is upstream — go read §2 (aborts?), §3 (sandbox latency/throttles?) and §4 (`num_waiting_reqs`,
`sglang:gen_throughput`) to find which stage is the culprit. Low ratio with slow steps → the GPU
compute in §1/§5 is the limiter.

---

## 7. Minimal dashboard (start here)

One row per stage, chosen to *localize* a regression fast:

1. **Learning** — `passrate/pass@k`, `rollout/response_len/mean`, `train/ppo_kl`, `train/grad_norm`, `train/entropy_loss`
2. **Rollout health** — abort rate (`Sample.Status.ABORTED`), `rollout/zero_std/count_*`, `rollout/truncated_ratio`, `train/global_batch_size`
3. **Sandbox (AWS)** — `Total Errors` %, `Throttles`, `ActiveSessionCount` (vs. quota), `Latency`
4. **Inference** — `sglang:num_queue_reqs`, `sglang:gen_throughput`, `sglang:token_usage`, `sglang:time_to_first_token_seconds` p99
5. **Sync & balance** — `perf/update_weights_time`, `train/rollout_logprob_abs_diff`, `perf/wait_time_ratio`, `perf/step_time`

If reward stalls: `perf/wait_time_ratio` splits it into a *compute* problem (rows 1/5) vs. a
*pipeline* problem (rows 2/3/4) in one glance.

---

## Sources
- slime: cloned repo `repos/slime` (THUDM/slime), paths/lines verified 2026-09-03.
- AgentCore runtime metrics: <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-runtime-metrics.html>
- AgentCore observability overview: <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html>
- AgentCore HTTP contract (`/ping`, errors): <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html>
- SGLang production metrics: <https://docs.sglang.io/references/production_metrics.html>
- Fault-recovery companion: [`fault-recovery.md`](fault-recovery.md)
