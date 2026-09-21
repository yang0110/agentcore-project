#!/usr/bin/env python3
"""Parse a slime training log into a per-step CSV and (optionally) plot the key curves.

slime logs metrics to stdout (and to wandb/tensorboard if configured), but for the demos we
want a self-contained, dependency-light record: a CSV you can open anywhere + a PNG of the
curves that matter. This script reads the raw ray-job log that a demo's `run.sh` tees to a
file and extracts the three metric families slime prints, each tagged with the rollout/step id:

    model.py:918            - step N:    {'train/pg_loss': ..., 'train/grad_norm': ..., ...}
    train_metric_utils.py   - rollout N: {'rollout/raw_reward': ..., 'rollout/advantages': ...}
    *                       - perf N:    {'perf/step_time': ..., 'perf/wait_time_ratio': ...}
    rollout_metrics.py:247  - eval N:    {'eval/<name>': <solve rate>, ..., 'eval/step': S}

The train families are merged by id (rollout N is trained as step N) and written to a per-step
CSV. The `eval` family is a DIFFERENT cadence (held-out eval runs every EVAL_INTERVAL steps, no
gradient) so it is written to a SEPARATE eval CSV keyed by `eval/step`. Both CSVs, and a PNG of the
key curves, come from the one raw log. Pure stdlib for parsing; matplotlib only for --plot.

We ALSO parse three raw families our own code prints (not slime's `- perf N:` dicts):

    agentcore_runtime_sandbox._timed_invoke   [sbx-op] op=… ms=…       (one per sandbox round-trip)
    policy.SGLangPolicy.complete              [sglang-gen] ms=…        (one per pure SGLang decode)
    <per-demo harness>                        [demo] <k=v …>           (this demo's OWN metric)

`[sbx-op]`/`[sglang-gen]` are written to raw per-op CSVs (`<log>.sandbox_ops.csv`,
`<log>.sglang_gen.csv`), summarized per op at the run level, AND attributed per-step into
`perf/sandbox_ms_*` / `perf/sandbox_open_ms_*` / `perf/sglang_gen_ms_*` columns on the train CSV —
splitting sandbox time and decode time out of slime's aggregate `perf/rollout_time`. Sandbox time
is itself split into WARM data ops and the COLD session-open op that blocks on the instance boot.

`[demo]` is the GENERIC per-demo channel: whatever demo-specific signal a demo cares about (a
code demo's compile-vs-assert split, a SQL demo's exec-match rate, a sharing demo's #farm-actors,
the big-image demo's cold-boot count, …) is emitted by that demo's harness as one `[demo] k=v …`
line and lands here with NO change to this shared parser. Numeric keys are aggregated per step into
`demo/<key>_*` columns and summarized at the run level; every record is also dumped raw to
`<log>.demo.csv`. Non-numeric keys get a run-level value-count. See each demo's README for what it
emits. (This file is identical across all demos; only what a demo's harness emits differs.)

    python3 metrics.py train.log --csv metrics.csv --eval-csv eval.csv --plot fig.png
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
import re
from pathlib import Path

# " - step 5: {…}" / " - rollout 5: {…}" / " - perf 5: {…}" / " - eval 5: {…}"  (dict to EOL)
_LINE = re.compile(r"-\s+(step|rollout|perf|eval)\s+(\d+):\s+(\{.*\})\s*$")

# --- Sandbox response-time + SGLang decode-time records (raw per-op lines) ----------------------
# Emitted by agentcore_slime_demo.agentcore_runtime_sandbox._timed_invoke ("[sbx-op] …", one per
# invoke_agent_runtime round-trip) and policy.SGLangPolicy.complete ("[sglang-gen] …", one per pure
# SGLang decode POST). Both are plain `key=value` lines (NOT slime's `- perf N:` dicts) so they get
# their own regexes. Splitting them lets us separate sandbox latency AND SGLang generation latency
# out of slime's aggregate perf/rollout_time, which lumps decode + sandbox + harness together.
_SBX_OP = re.compile(r"\[sbx-op\]\s+(.*)$")
_SGLANG_GEN = re.compile(r"\[sglang-gen\]\s+(.*)$")

# One per episode, emitted by agentcore_slime_demo.router_trace.emit — the per-episode routing path
# (pinned SGLang engine / sandbox session / runtime ARN / image / family). Parsed into
# <log>.router.csv and checked for per-episode stickiness (see check_stickiness).
_ROUTER = re.compile(r"\[router\]\s+(.*)$")

# The GENERIC per-demo metric channel: `[demo] k=v k=v …`, emitted by a demo's own harness for the
# signal THAT demo demonstrates (see module docstring). Parsed into <log>.demo.csv, aggregated per
# step (numeric keys) into demo/<key>_* columns, and summarized at the run level. A demo that emits
# nothing here simply has no demo/* columns — this shared parser is unchanged either way.
_DEMO = re.compile(r"\[demo\]\s+(.*)$")

# Sandbox ops that ESTABLISH a session (open / ping / start_session) vs teardown ops (~0 ms,
# excluded from the warm p50/p95 so they don't skew it). Everything else is a WARM data round-trip
# (exec / write_file / read_file). A session-open is NOT automatically a cold boot: on the warm-reuse
# farm, `op=open` just attaches a fresh sub_session to an ALREADY-running instance (~150 ms), while
# the FIRST open on a cold instance blocks on the capacity-provider launch (~60–80 s). So we bucket
# all session-opens into perf/sandbox_open_*, and separately COUNT the ones slow enough to be a real
# instance boot (ms >= COLD_BOOT_MS) — that count is the warm-reuse headline (N boots -> 1).
_OPEN_OPS = {"ping", "open", "start_session"}
_SKIP_OPS = {"stop_session", "close"}
# A session-open this slow (ms) must have blocked on an instance cold-boot rather than re-attached to
# a warm one. 5 s cleanly separates a warm sub_session attach (~0.1-1 s) from an Instance launch
# (~60-80 s); microVM opens (~1-2 s) also fall below it. Override with env COLD_BOOT_MS.
COLD_BOOT_MS = float(os.environ.get("COLD_BOOT_MS", "5000"))


def _kv(s: str) -> dict[str, str]:
    """Parse a space-separated `k=v k=v` tail into a dict (values stay strings)."""
    out: dict[str, str] = {}
    for tok in s.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k] = v
    return out


def _pct(xs_sorted: list[float], q: float) -> float:
    """Linear-interpolation percentile of an ALREADY-SORTED list (stdlib only, no numpy)."""
    if len(xs_sorted) == 1:
        return xs_sorted[0]
    pos = (len(xs_sorted) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs_sorted) - 1)
    return xs_sorted[lo] + (xs_sorted[hi] - xs_sorted[lo]) * (pos - lo)


def _agg(ms_list: list[float], prefix: str) -> dict[str, float]:
    """count + mean/p50/p95 of a latency list, as `{prefix}_count`, `{prefix}_ms_{mean,p50,p95}`."""
    if not ms_list:
        return {}
    s = sorted(ms_list)
    n = len(s)
    return {
        f"{prefix}_count": n,
        f"{prefix}_ms_mean": round(sum(s) / n, 1),
        f"{prefix}_ms_p50": round(_pct(s, 0.50), 1),
        f"{prefix}_ms_p95": round(_pct(s, 0.95), 1),
    }


def _agg_plain(vals: list[float], prefix: str) -> dict[str, float]:
    """count + mean/p50/p95 of a plain numeric list (no `_ms_` infix), as `{prefix}_{count,mean,p50,p95}`.
    Used for the generic [demo] channel, whose numbers are usually NOT latencies (counts, ratios…)."""
    if not vals:
        return {}
    s = sorted(vals)
    n = len(s)
    return {
        f"{prefix}_count": n,
        f"{prefix}_mean": round(sum(s) / n, 4),
        f"{prefix}_p50": round(_pct(s, 0.50), 4),
        f"{prefix}_p95": round(_pct(s, 0.95), 4),
    }

# --- Goodput & MFU (see docs/sandbox-concurrency-ablation.md "Metrics" table) ------------------
# MFU = achieved trainer FLOPs / peak. slime already logs per-GPU training TFLOPS as
# `perf/actor_train_tflops` (= 3 * fwd_flops / actor_train_time / world_size, train_metric_utils.py),
# so MFU is just that / the device's dense BF16 peak. This box is 8x NVIDIA L40S; L40S dense
# BF16 tensor-core peak is 362 TFLOPS (733 only with 2:4 sparsity, which we don't use).
# Override for other hardware with env GPU_PEAK_TFLOPS.
PEAK_TFLOPS = float(os.environ.get("GPU_PEAK_TFLOPS", "362.0"))

# Goodput = *useful* trainer throughput = actor_train_tok_per_s * (1 - discard_ratio): tokens/sec
# that actually land in a gradient step, net of samples the buffer filter discards for staleness.
# We read discard_ratio from any parsed rollout key that reports it (fully-async buffer filter);
# absent that (e.g. Mode A / stock rollout, no discards), discard_ratio = 0 so goodput == tok/s.
_DISCARD_KEYS = ("rollout/discard_ratio", "rollout/staleness_discard_ratio", "perf/discard_ratio")


def _discard_ratio(row: dict) -> float:
    for k in _DISCARD_KEYS:
        v = row.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    # tolerate any other key that plainly names a discard/drop ratio
    for k, v in row.items():
        if isinstance(v, (int, float)) and "discard" in k and "ratio" in k:
            return float(v)
    return 0.0


def derive_perf_metrics(rows: dict[int, dict]) -> None:
    """Add derived perf/mfu + perf/goodput_tok_per_s columns to each train row in place."""
    for row in rows.values():
        tflops = row.get("perf/actor_train_tflops")
        if isinstance(tflops, (int, float)) and PEAK_TFLOPS > 0:
            row["perf/mfu"] = round(float(tflops) / PEAK_TFLOPS, 4)
        tok_s = row.get("perf/actor_train_tok_per_s")
        if isinstance(tok_s, (int, float)):
            row["perf/goodput_tok_per_s"] = round(float(tok_s) * (1.0 - _discard_ratio(row)), 2)

# The columns we care about most, in display order; any other parsed keys are appended after.
# (Demo-specific `demo/*` columns are discovered at runtime and appended by write_csv.)
KEY_COLUMNS = [
    "rollout_id",
    "rollout/raw_reward",          # solve rate this step (the RL signal)
    "train/grad_norm",             # is the policy actually moving?
    "train/pg_loss",
    "train/kl_loss",               # drift from the frozen reference
    "train/train_rollout_logprob_abs_diff",  # weight-sync correctness (want ~small/stable)
    "train/global_batch_size",
    "rollout/response_lengths",
    "perf/step_time",
    "perf/actor_train_tok_per_s",
    "perf/goodput_tok_per_s",      # useful tok/s = tok/s * (1 - discard_ratio) — the optimization target
    "perf/actor_train_tflops",     # per-GPU achieved trainer TFLOPS (slime-logged)
    "perf/mfu",                    # actor_train_tflops / device peak (L40S 362 TFLOPS BF16)
    "perf/wait_time_ratio",        # pipeline balance (train vs rollout idle); 1-this = starvation-free frac
    # Sandbox response time — per-step aggregate of the invoke_agent_runtime round-trips this step's
    # rollout paid (from [sbx-op] records; count = #ops attributed to the step). Split by cost class:
    # `perf/sandbox_*` = WARM data ops (exec/write_file); `perf/sandbox_open_*` = the COLD session-open
    # op that blocks on the instance boot (~60-80 s on a capacity-provider Instance, ~1-2 s on a
    # microVM) — kept separate so the warm p50/p95 isn't swamped by the one-time cold boot.
    "perf/sandbox_count",
    "perf/sandbox_ms_mean",
    "perf/sandbox_ms_p50",
    "perf/sandbox_ms_p95",
    "perf/sandbox_open_count",
    "perf/sandbox_open_ms_mean",
    "perf/sandbox_open_ms_p50",
    "perf/sandbox_open_ms_p95",
    "perf/sandbox_coldboot_count",  # session-opens slow enough to be a real instance boot (>=COLD_BOOT_MS)
    # SGLang decode time — pure /generate round-trips, split OUT of sandbox+harness time
    # (from [sglang-gen] records; count = #decode turns attributed to the step).
    "perf/sglang_gen_count",
    "perf/sglang_gen_ms_mean",
    "perf/sglang_gen_ms_p50",
    "perf/sglang_gen_ms_p95",
]

# Eval CSV columns: the held-out solve rate(s) keyed by the training step eval ran at.
EVAL_KEY_COLUMNS = ["eval/step"]

# Which columns to plot, and a human title for each panel. `eval/*` reward columns and `demo/*`
# demo-specific columns are discovered at runtime (their names depend on the eval dataset / demo)
# and appended.
PLOT_COLUMNS = [
    ("rollout/raw_reward", "solve rate (raw_reward)"),
    ("train/grad_norm", "grad norm"),
    ("train/kl_loss", "KL(policy‖ref)"),
    ("train/train_rollout_logprob_abs_diff", "logprob abs diff (weight-sync)"),
    ("perf/wait_time_ratio", "wait-time ratio"),
    ("perf/actor_train_tok_per_s", "actor train tok/s"),
    ("perf/goodput_tok_per_s", "goodput (useful tok/s)"),
    ("perf/mfu", "MFU (frac of L40S peak)"),
    ("perf/sandbox_ms_p50", "sandbox warm-op p50 (ms)"),
    ("perf/sglang_gen_ms_p50", "SGLang decode p50 (ms)"),
]


def parse_log(path: Path):
    """Parse the log into (train_rows, eval_rows, op_records, gen_records, router_records, demo_records).

    train_rows: {rollout_id: {metric: value}} merged across step/rollout/perf, PLUS the per-step
                sandbox/decode aggregates (perf/sandbox_*, perf/sandbox_open_*, perf/sglang_gen_*) and
                the per-step demo aggregates (demo/<key>_*) derived below.
    eval_rows:  {eval_step: {eval/*: value}} — a separate cadence (eval every N steps), keyed by
                the `eval/step` value inside each eval dict (fallback: the logged id).
    op_records: raw [sbx-op] round-trips {ts, step, op, ms, arn, session, status}.
    gen_records: raw [sglang-gen] decodes {ts, step, ms, prompt_toks, gen_toks}.
    router_records: per-episode [router] routing paths.
    demo_records: raw [demo] per-demo metric records (arbitrary k=v, this demo's own signal).

    Per-step attribution: every raw record seen SINCE the previous train marker is attributed to
    the NEXT step's marker (that step's rollout finished just before slime printed its metrics).
    Exact for the coupled / 1-step-async (Mode A) path; approximate under fully-async Mode B, where
    the background buffer streams ops across step boundaries — read the per-step split as indicative
    there, and the run-level aggregate (main()) as the ground truth.
    """
    train_rows: dict[int, dict] = {}
    eval_rows: dict[int, dict] = {}
    op_records: list[dict] = []
    gen_records: list[dict] = []
    router_records: list[dict] = []
    demo_records: list[dict] = []
    pending_ops: list[dict] = []   # raw records awaiting the next marker (their attributed step)
    pending_gens: list[dict] = []
    pending_router: list[dict] = []
    pending_demo: list[dict] = []
    op_step_records: dict[int, list[dict]] = {}   # full records (need op name for cold/warm split)
    gen_buckets: dict[int, list[float]] = {}
    demo_step_vals: dict[int, dict[str, list[float]]] = {}   # step -> {demo key -> [numeric vals]}

    for line in path.read_text(errors="replace").splitlines():
        m = _LINE.search(line)
        if m:
            kind, sid, payload = m.group(1), int(m.group(2)), m.group(3)
            try:
                d = ast.literal_eval(payload)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(d, dict):
                continue
            nums = {k: v for k, v in d.items() if isinstance(v, (int, float))}
            if kind == "eval":
                step = int(d.get("eval/step", sid))
                row = eval_rows.setdefault(step, {"eval/step": step})
                row.update(nums)
            else:
                row = train_rows.setdefault(sid, {"rollout_id": sid})
                row.update(nums)
                for r in pending_ops:
                    r["step"] = sid
                    op_step_records.setdefault(sid, []).append(r)
                for r in pending_gens:
                    r["step"] = sid
                    gen_buckets.setdefault(sid, []).append(r["ms"])
                for r in pending_router:
                    r["step"] = sid
                for r in pending_demo:
                    r["step"] = sid
                    for k, v in r.items():
                        if k in ("ts", "step"):
                            continue
                        try:
                            fv = float(v)
                        except (TypeError, ValueError):
                            continue
                        demo_step_vals.setdefault(sid, {}).setdefault(k, []).append(fv)
                pending_ops = []
                pending_gens = []
                pending_router = []
                pending_demo = []
            continue
        mo = _SBX_OP.search(line)
        if mo:
            kv = _kv(mo.group(1))
            try:
                rec = {"ts": float(kv.get("ts") or 0), "step": None, "op": kv.get("op", "?"),
                       "ms": float(kv["ms"]), "arn": kv.get("arn", "?"),
                       "session": kv.get("session", "?"), "status": kv.get("status", "?")}
            except (KeyError, ValueError):
                continue
            op_records.append(rec)
            pending_ops.append(rec)
            continue
        mg = _SGLANG_GEN.search(line)
        if mg:
            kv = _kv(mg.group(1))
            try:
                rec = {"ts": float(kv.get("ts") or 0), "step": None, "ms": float(kv["ms"]),
                       "prompt_toks": int(kv.get("prompt_toks") or 0),
                       "gen_toks": int(kv.get("gen_toks") or 0)}
            except (KeyError, ValueError):
                continue
            gen_records.append(rec)
            pending_gens.append(rec)
            continue
        mr = _ROUTER.search(line)
        if mr:
            kv = _kv(mr.group(1))
            rec = {"ts": float(kv.get("ts") or 0), "step": None,
                   "task_id": kv.get("task_id", "?"),
                   "rollout_id": kv.get("rollout_id", "?"),
                   "harness_worker": kv.get("harness_worker", "?"),
                   "engine": kv.get("engine", "?"),
                   "engine_pinned": kv.get("engine_pinned", "?"),
                   "adapter_session": kv.get("adapter_session", "?"),
                   "sandbox_session": kv.get("sandbox_session", "?"),
                   "runtime_arn": kv.get("runtime_arn", "?"),
                   "sandbox_image": kv.get("sandbox_image", ""),
                   "family": kv.get("family", "")}
            router_records.append(rec)
            pending_router.append(rec)
            continue
        md = _DEMO.search(line)
        if md:
            kv = _kv(md.group(1))
            if not kv:
                continue
            rec = {"ts": float(kv.get("ts") or 0), "step": None}
            for k, v in kv.items():
                if k != "ts":
                    rec[k] = v
            demo_records.append(rec)
            pending_demo.append(rec)

    # Split remote-sandbox latency by op class: a fresh session's first op establishes the session
    # and blocks on the instance cold-boot, which is a totally different cost from a WARM data
    # round-trip (exec/write_file). Lumping them makes p50/p95 meaningless, so bucket separately:
    # `perf/sandbox` = warm ops only, `perf/sandbox_open` = the cold session-open. Teardown dropped.
    for sid, recs in op_step_records.items():
        warm = [r["ms"] for r in recs if r["op"] not in _OPEN_OPS and r["op"] not in _SKIP_OPS]
        opens = [r["ms"] for r in recs if r["op"] in _OPEN_OPS]
        row = train_rows.setdefault(sid, {"rollout_id": sid})
        row.update(_agg(warm, "perf/sandbox"))
        row.update(_agg(opens, "perf/sandbox_open"))
        # A session-open is a real instance COLD boot only if it was slow enough to have blocked on
        # the launch (ms >= COLD_BOOT_MS); a warm-farm sub_session attach is a fast "open" and does
        # NOT count. This per-step count is ~0 on warm steps and jumps only when an instance booted.
        if opens:
            row["perf/sandbox_coldboot_count"] = sum(1 for ms in opens if ms >= COLD_BOOT_MS)
    for sid, ms_list in gen_buckets.items():
        train_rows.setdefault(sid, {"rollout_id": sid}).update(_agg(ms_list, "perf/sglang_gen"))
    for sid, keyvals in demo_step_vals.items():
        row = train_rows.setdefault(sid, {"rollout_id": sid})
        for k, vals in keyvals.items():
            row.update(_agg_plain(vals, f"demo/{k}"))
    return train_rows, eval_rows, op_records, gen_records, router_records, demo_records


def write_csv(rows: dict[int, dict], out: Path, key_columns: list[str]) -> list[str]:
    ids = sorted(rows)
    extra = sorted({k for r in rows.values() for k in r} - set(key_columns))
    columns = key_columns + extra
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for i in ids:
            w.writerow(rows[i])
    return columns


def write_raw_csv(records: list[dict], out: Path, columns: list[str]) -> None:
    """Dump the raw per-op / per-decode records (one row each), in log order."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def write_demo_csv(records: list[dict], out: Path) -> list[str]:
    """Dump the raw [demo] records; columns are the union of keys seen (ts, step first)."""
    head = ["ts", "step"]
    cols = head + sorted({k for r in records for k in r} - set(head))
    write_raw_csv(records, out, cols)
    return cols


def check_stickiness(router_records: list[dict], op_records: list[dict]) -> bool:
    """Verify each episode stayed on ONE path through the replicated tiers, and print a report.

    Episode key = the sandbox session (also stamped on every [sbx-op]). The invariants:
      * inference — each episode's turns were served by ONE SGLang engine. This demo makes a fresh
        policy per episode PINNED to a single engine (router_trace.pin_engine), so there is one
        [router] line — hence one engine — per episode by construction; we surface the spread and
        flag any episode recorded unpinned (engine_pinned=0, discovery fell back to the router).
      * sandbox/runtime — EVERY [sbx-op] for the episode's sandbox session shares ONE runtime ARN
        (the sandbox is stateful/sticky for the episode's life). This is the real cross-check: it
        JOINS the raw per-op records to the episode via `session`.

    A harness-scale-style run (WARM workers each pinned to one engine) is also handled: when the
    [router] records carry a harness_worker, we additionally assert no worker used >1 engine.

    Returns True if all invariants hold (or there is nothing to check)."""
    if not router_records:
        return True
    from collections import Counter

    # sandbox_session -> set of runtime ARNs seen across its raw ops (from [sbx-op] session=/arn=).
    arns_by_session: dict[str, set[str]] = {}
    for r in op_records:
        arns_by_session.setdefault(r.get("session", "?"), set()).add(r.get("arn", "?"))

    violations: list[str] = []
    for rec in router_records:
        sess = rec["sandbox_session"]
        seen = arns_by_session.get(sess)
        if seen and len(seen) > 1:
            violations.append(
                f"episode {rec['rollout_id']} (task {rec['task_id']}): sandbox session {sess} "
                f"spanned {len(seen)} runtimes {sorted(seen)} — NOT sticky"
            )

    has_workers = any(str(r.get("harness_worker", "?")) not in ("?", "") for r in router_records)
    engines = sorted({r["engine"] for r in router_records})
    unpinned = sum(1 for r in router_records if str(r.get("engine_pinned")) in ("0", "?", ""))
    print("\nrouter stickiness ([router] episodes joined to [sbx-op] by sandbox session):")

    if has_workers:
        engines_by_worker: dict[str, set[str]] = {}
        for rec in router_records:
            engines_by_worker.setdefault(rec["harness_worker"], set()).add(rec["engine"])
        for w, engs in engines_by_worker.items():
            if len(engs) > 1:
                violations.append(f"worker {w} used {len(engs)} engines {sorted(engs)} — inference NOT pinned")
        workers = sorted(engines_by_worker)
        print(f"  episodes={len(router_records)}  workers={len(workers)}  engines={len(engines)}")
        for w in workers:
            print(f"    worker {w} -> engine(s) {sorted(engines_by_worker.get(w, set()))}")
        ok_msg = "  ✓ every episode sticky to one worker + one engine + one runtime"
    else:
        dist = Counter(r["engine"] for r in router_records)
        print(f"  episodes={len(router_records)}  engines={len(engines)}  unpinned={unpinned}")
        for e in engines:
            print(f"    engine {e} <- {dist[e]} episode(s)")
        ok_msg = "  ✓ every episode pinned to one engine + one sandbox session + one runtime"

    if violations:
        print("  STICKINESS VIOLATIONS:")
        for v in violations:
            print(f"    ✗ {v}")
    else:
        print(ok_msg)
    return not violations


def summarize_latency(op_records: list[dict], gen_records: list[dict]) -> None:
    """Print the RUN-LEVEL sandbox/decode latency aggregate (the ground truth across the whole run,
    independent of the approximate per-step attribution). Sandbox is broken out per op type AND by
    cost class (WARM data ops vs the COLD session-open that blocks on the instance boot)."""
    def agg_line(label: str, ms: list[float]) -> str:
        s = sorted(ms)
        return (f"    {label:<22} n={len(s):<5} mean={sum(s)/len(s):8.1f}  "
                f"p50={_pct(s, 0.50):8.1f}  p95={_pct(s, 0.95):8.1f}  max={s[-1]:8.1f}  (ms)")

    if op_records:
        print("sandbox response time ([sbx-op], invoke_agent_runtime round-trips):")
        by_op: dict[str, list[float]] = {}
        for r in op_records:
            by_op.setdefault(r["op"], []).append(r["ms"])
        for op in sorted(by_op):
            print(agg_line(op, by_op[op]))
        # Roll up by cost CLASS: WARM data round-trips (exec/write_file) vs session-open ops (open/
        # ping/start_session). Teardown ops (_SKIP_OPS, ~0 ms) excluded. A session-open is only a real
        # instance COLD boot if it was slow (ms >= COLD_BOOT_MS) — on the warm-reuse farm most opens
        # are fast sub_session attaches, so we report cold-boots vs warm-reopens separately: that
        # count is the warm-reuse headline (e.g. one shared instance -> cold-boots collapses to ~1).
        warm = [r["ms"] for r in op_records if r["op"] not in _OPEN_OPS and r["op"] not in _SKIP_OPS]
        opens = [r["ms"] for r in op_records if r["op"] in _OPEN_OPS]
        if warm:
            print(agg_line("WARM data ops", warm))
        if opens:
            print(agg_line("session-open ops", opens))
            coldboots = sum(1 for ms in opens if ms >= COLD_BOOT_MS)
            print(f"    -> instance cold-boots (open ms>={COLD_BOOT_MS:.0f}) = {coldboots}; "
                  f"warm session-reopens = {len(opens) - coldboots}; warm data ops = {len(warm)}")
        errs = sum(1 for r in op_records if r.get("status") not in ("ok", None))
        if errs:
            print(f"    ({errs} op(s) ended in an error status — see the ops CSV)")
    if gen_records:
        print("SGLang decode time ([sglang-gen], pure /generate round-trips):")
        print(agg_line("decode", [r["ms"] for r in gen_records]))


def summarize_demo(demo_records: list[dict]) -> None:
    """Print the RUN-LEVEL summary of this demo's OWN [demo] metric channel: numeric keys get
    count/mean/p50/p95/max; non-numeric keys get a value-count. Nothing prints if the demo emitted
    no [demo] records (most single-purpose dataset demos rely on rollout/eval + [demo] alike)."""
    if not demo_records:
        return
    from collections import Counter

    numeric: dict[str, list[float]] = {}
    categorical: dict[str, Counter] = {}
    for r in demo_records:
        for k, v in r.items():
            if k in ("ts", "step"):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                categorical.setdefault(k, Counter())[v] += 1
                continue
            numeric.setdefault(k, []).append(fv)
    print(f"\ndemo-specific metrics ([demo] channel, {len(demo_records)} record(s)):")
    for k in sorted(numeric):
        s = sorted(numeric[k])
        print(f"    {k:<22} n={len(s):<5} mean={sum(s)/len(s):10.4f}  "
              f"p50={_pct(s,0.50):10.4f}  p95={_pct(s,0.95):10.4f}  max={s[-1]:10.4f}")
    for k in sorted(categorical):
        # categorical keys the numeric loop never touched (a key is one or the other per run)
        if k in numeric:
            continue
        items = ", ".join(f"{val}={cnt}" for val, cnt in categorical[k].most_common(10))
        print(f"    {k:<22} {items}")

    # Mean reward grouped by each categorical key — the generic way a demo whose batch spans
    # categories reads out per-category learnability from ONE channel: `family` -> per-family reward
    # (mixtask), `image` -> per-image correctness (imagepool), `checker` -> reward per outcome class.
    if "reward" in numeric:
        for k in sorted(categorical):
            if k in numeric or k == "task":
                continue
            groups: dict[str, list[float]] = {}
            for r in demo_records:
                if k in r and "reward" in r:
                    try:
                        groups.setdefault(str(r[k]), []).append(float(r["reward"]))
                    except (TypeError, ValueError):
                        pass
            if len(groups) > 1:
                print(f"    reward by {k}:")
                for val in sorted(groups):
                    g = groups[val]
                    print(f"        {val:<20} n={len(g):<5} mean_reward={sum(g)/len(g):.4f}")


def plot(rows: dict[int, dict], eval_rows: dict[int, dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    ids = sorted(rows)
    panels = [(c, t) for c, t in PLOT_COLUMNS if any(c in rows[i] for i in ids)]
    # add one panel per demo/*_mean column (this demo's own signal), discovered at runtime
    demo_cols = sorted({k for i in ids for k in rows[i] if k.startswith("demo/") and k.endswith("_mean")})
    for c in demo_cols:
        panels.append((c, c.replace("demo/", "demo: ").replace("_mean", " (mean)")))
    # add one panel per eval/* reward column (skip the eval/step key itself)
    eval_cols = sorted({k for r in eval_rows.values() for k in r} - {"eval/step"})
    for c in eval_cols:
        panels.append((c, f"held-out {c}"))
    if not panels:
        print("no plottable columns found; skipping figure")
        return
    ncol = 2
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3 * nrow), squeeze=False)
    for ax, (col, title) in zip(axes.flat, panels):
        src = eval_rows if col.startswith("eval/") else rows
        xs = [i for i in sorted(src) if col in src[i]]
        ys = [src[i][col] for i in xs]
        ax.plot(xs, ys, marker="o")
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote figure -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="path to the training log (ray job output tee'd by run_*.sh)")
    ap.add_argument("--csv", default=None, help="per-step train CSV path (default: <log>.csv)")
    ap.add_argument("--eval-csv", default=None,
                    help="per-eval CSV path (default: <log>.eval.csv); written only if eval ran")
    ap.add_argument("--plot", nargs="?", const="", default=None,
                    help="also write a PNG of the key curves (default: <log>.png)")
    ap.add_argument("--ops-csv", default=None,
                    help="raw per-op sandbox-latency CSV (default: <log>.sandbox_ops.csv); "
                         "written only if [sbx-op] records are present")
    ap.add_argument("--gen-csv", default=None,
                    help="raw per-decode SGLang-latency CSV (default: <log>.sglang_gen.csv); "
                         "written only if [sglang-gen] records are present")
    ap.add_argument("--router-csv", default=None,
                    help="per-episode router CSV (default: <log>.router.csv); written only if "
                         "[router] records are present")
    ap.add_argument("--demo-csv", default=None,
                    help="raw per-demo metric CSV (default: <log>.demo.csv); written only if "
                         "[demo] records are present")
    args = ap.parse_args()

    log = Path(args.log)
    rows, eval_rows, op_records, gen_records, router_records, demo_records = parse_log(log)
    derive_perf_metrics(rows)
    if not rows and not eval_rows:
        print(f"no metric rows found in {log} — is the run past step 1?")
        return 1

    if rows:
        csv_out = Path(args.csv) if args.csv else log.with_suffix(".csv")
        cols = write_csv(rows, csv_out, KEY_COLUMNS)
        print(f"wrote {len(rows)} step rows x {len(cols)} cols -> {csv_out}")

    if op_records:
        ops_out = Path(args.ops_csv) if args.ops_csv else log.with_suffix(".sandbox_ops.csv")
        write_raw_csv(op_records, ops_out,
                      ["ts", "step", "op", "ms", "arn", "session", "status"])
        print(f"wrote {len(op_records)} raw sandbox ops -> {ops_out}")
    if gen_records:
        gen_out = Path(args.gen_csv) if args.gen_csv else log.with_suffix(".sglang_gen.csv")
        write_raw_csv(gen_records, gen_out, ["ts", "step", "ms", "prompt_toks", "gen_toks"])
        print(f"wrote {len(gen_records)} raw SGLang decodes -> {gen_out}")
    if op_records or gen_records:
        summarize_latency(op_records, gen_records)

    if router_records:
        router_out = Path(args.router_csv) if args.router_csv else log.with_suffix(".router.csv")
        write_raw_csv(router_records, router_out,
                      ["ts", "step", "rollout_id", "task_id", "harness_worker", "engine",
                       "engine_pinned", "adapter_session", "sandbox_session", "runtime_arn",
                       "sandbox_image", "family"])
        print(f"wrote {len(router_records)} per-episode router records -> {router_out}")
        check_stickiness(router_records, op_records)

    if demo_records:
        demo_out = Path(args.demo_csv) if args.demo_csv else log.with_suffix(".demo.csv")
        dcols = write_demo_csv(demo_records, demo_out)
        print(f"wrote {len(demo_records)} raw [demo] records x {len(dcols)} cols -> {demo_out}")
        summarize_demo(demo_records)

    if eval_rows:
        eval_out = Path(args.eval_csv) if args.eval_csv else log.with_suffix(".eval.csv")
        ecols = write_csv(eval_rows, eval_out, EVAL_KEY_COLUMNS)
        print(f"wrote {len(eval_rows)} eval rows x {len(ecols)} cols -> {eval_out}")

    if args.plot is not None:
        png = Path(args.plot) if args.plot else log.with_suffix(".png")
        plot(rows, eval_rows, png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
