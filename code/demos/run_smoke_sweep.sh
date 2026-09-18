#!/bin/bash
# Smoke-sweep driver — Phase 0 (offline PROBES) + Phase 1 (CPU RL trainers). See SMOKE_SWEEP.md.
# NOTE: the five property demos are now REAL trainers (their run.sh launches slime). Phase 0 runs
# each one's RETAINED offline probe SCRIPT directly (quiesce_demo.py / snapshot_demo.py /
# harness_scaling_demo.py) — NOT run.sh — and the three CPU property trainers join Phase 1.
# Runs INSIDE the agentcore_p4 container (as root => can delete its own root-owned checkpoints).
# Serial by necessity (each trainer claims the GPUs). Deletes each trainer's checkpoint right after
# verifying it, so disk never accumulates. Appends a live results row per demo.
set -uo pipefail

BASE="${BASE:-/work/code/demos}"   # container path by default; override for a host-only offline run
RESULTS="${RESULTS:-$BASE/smoke_sweep_results.tsv}"
OFFLINE_TIMEOUT="${OFFLINE_TIMEOUT:-420}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-1800}"

# Phase lists (override via env to resume a subset).
OFFLINE_DEMOS="${OFFLINE_DEMOS:-demo_cpu_quiesce demo_gpu_snapshot demo_cpu_harness_scale}"
CPU_TRAINERS="${CPU_TRAINERS:-demo_cpu_single demo_cpu_math demo_cpu_mixtask demo_cpu_gsm8k demo_cpu_math500 demo_cpu_deepscaler demo_cpu_humaneval demo_cpu_gretelsql demo_cpu_hotpotqa demo_cpu_xlam demo_cpu_mcp demo_cpu_quiesce demo_cpu_harness_scale demo_cpu_customruntime}"

# Each offline demo's RETAINED probe script (run directly; run.sh is now the trainer). Args match the
# pre-conversion run.sh so the behavior markers below still appear.
offline_cmd() {
  case "$1" in
    demo_cpu_quiesce)       echo "python3 quiesce_demo.py" ;;
    demo_gpu_snapshot)      echo "python3 snapshot_demo.py" ;;
    demo_cpu_harness_scale) echo "python3 harness_scaling_demo.py --episodes 48 --workers 1,2,4,6,8,12 --cinf 4 --csbx 8 --tinf 0.10 --tsbx 0.05" ;;
    *) echo "" ;;
  esac
}

printf 'demo\tstatus\tsecs\tnote\n' > "$RESULTS"
echo "[sweep] start $(date -u +%FT%TZ)"

row() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$RESULTS"; }

# Per-offline-demo BEHAVIOR assertions: the exact stepwise markers that prove the demo did what it
# claims, not merely that it exited 0. A truncated/half-run log (e.g. a quiesce that dies before ACT3
# recovers) exits 0-or-killed but is MISSING these — so PASS now means "behaved as expected". Emitted
# one literal substring per line; ALL must appear in the log (grep -F, so brackets/parens are literal).
offline_markers() {
  case "$1" in
    demo_cpu_quiesce)      # 3-act arc: naive loss -> heartbeat survives (gate order) -> recover
      printf '%s\n' \
        "LOST THE SESSION" \
        "SESSION SURVIVED" \
        "['sync-ended', 'gated-invoke-ran']" \
        ">> RECOVERED" \
        "takeaway:" ;;
    demo_gpu_snapshot)     # no-snapshot reworks 2 steps; app-snapshot resumes (6 vs 4 executed)
      printf '%s\n' \
        "RUN A" \
        "RUN B" \
        "no-snapshot executed 6 steps; app-snapshot executed 4 steps" \
        "rework step" ;;
    demo_cpu_harness_scale) # 48/48 solved every W; throughput is the min of the three tiers
      printf '%s\n' \
        "48/48" \
        "throughput = min(inference, sandbox, harness)" \
        "plateaus" ;;
    *) : ;;  # no assertions defined -> exit 0 alone (should not happen for known offline demos)
  esac
}

# Return 0 iff every marker for $1 is present in log $2; else print the first missing marker to stderr.
check_markers() {
  local d="$1" log="$2" m
  while IFS= read -r m; do
    [ -z "$m" ] && continue
    grep -qaF -- "$m" "$log" || { echo "MISSING: $m" >&2; return 1; }
  done < <(offline_markers "$d")
  return 0
}

free_gpus() {
  pkill -9 sglang 2>/dev/null; ray stop --force >/dev/null 2>&1; pkill -9 ray 2>/dev/null; sleep 4
}

# ---- Phase 0: offline PROBES (probe script + behavior markers) ----------------------------------
# Runs the demo's retained offline probe SCRIPT directly (NOT run.sh — that's now the trainer). Logs
# to offline_probe.log so it never clobbers the trainer's smoke_run.log written in Phase 1.
for d in $OFFLINE_DEMOS; do
  echo "[sweep] OFFLINE-PROBE $d"
  t0=$SECONDS
  log="$BASE/$d/offline_probe.log"
  ( cd "$BASE/$d" && PYTHONPATH="$BASE/$d:${PYTHONPATH:-}" \
      timeout "$OFFLINE_TIMEOUT" $(offline_cmd "$d") ) > "$log" 2>&1
  rc=$?
  if [ "$rc" != "0" ]; then
    [ "$rc" = "124" ] && row "$d" TIMEOUT $((SECONDS-t0)) "exceeded ${OFFLINE_TIMEOUT}s" \
                       || row "$d" FAIL $((SECONDS-t0)) "exit $rc (see $d/offline_probe.log)"
  elif miss=$(check_markers "$d" "$log" 2>&1 >/dev/null); then
    # exit 0 AND all behavior markers present
    row "$d" PASS $((SECONDS-t0)) "behavior verified (all markers)"
  else
    # exited 0 but the expected stepwise behavior is absent (e.g. truncated / half-run log)
    row "$d" FAIL $((SECONDS-t0)) "exit 0 but $miss (see $d/offline_probe.log)"
  fi
done

# ---- Phase 1: CPU RL trainers (NR=3), delete checkpoint after each ------------------------------
for d in $CPU_TRAINERS; do
  echo "[sweep] TRAINER $d"
  free_gpus
  t0=$SECONDS
  NR=3 timeout "$TRAIN_TIMEOUT" bash "$BASE/$d/run.sh" > "$BASE/$d/smoke_run.log" 2>&1
  rc=$?
  log="$BASE/$d/smoke_run.log"
  # This run's ACTUAL save dir, parsed from the launched --save arg (NOT a newest-mtime guess:
  # a demo dir can hold stale sibling ckpts, and an overwritten iter_* subdir does not bump the
  # parent dir mtime, so "newest" is unreliable — that gave demo_cpu_single a false FAIL).
  save=$(grep -aoE -- "--save /[^ ]+" "$log" | head -1 | awk '{print $2}')
  has_final=0
  [ -n "$save" ] && [ -f "$save/latest_checkpointed_iteration.txt" ] && has_final=1
  if grep -q "succeeded" "$log" \
     && ! grep -qE "No space|CheckpointException|unexpected pos" "$log" \
     && [ "$has_final" = "1" ]; then
    row "$d" PASS $((SECONDS-t0)) "job succeeded; final ckpt $(cat "$save/latest_checkpointed_iteration.txt" 2>/dev/null)"
  elif [ "$rc" = "124" ]; then
    row "$d" TIMEOUT $((SECONDS-t0)) "exceeded ${TRAIN_TIMEOUT}s"
  else
    reason="rc=$rc"
    grep -qE "No space" "$log" && reason="$reason ENOSPC"
    grep -qE "CheckpointException|unexpected pos" "$log" && reason="$reason ckpt-corrupt"
    [ "$has_final" = "0" ] && reason="$reason no-final-ckpt"
    grep -q "succeeded" "$log" || reason="$reason no-success-marker"
    row "$d" FAIL $((SECONDS-t0)) "$reason"
  fi
  # reclaim disk: delete THIS run's checkpoint (its own --save dir), keep the smoke_run.log
  [ -n "$save" ] && rm -rf "$save" 2>/dev/null
done

free_gpus
echo "[sweep] done $(date -u +%FT%TZ)"
echo "[sweep] results:"; cat "$RESULTS"
