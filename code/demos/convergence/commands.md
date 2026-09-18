# commands.md — convergence cheat-sheet

Run **one at a time**, INSIDE `agentcore_p4` (repo at `/work/code`). Each line trains
that demo to convergence with the defaults in its script (CPU `NR=201`, GPU `NR=101`,
`RB=8 NS=8 EVAL_INTERVAL=20 EVAL_NS=4 SAVE_INTERVAL=100`; CPU adds `MODE=B`). Logs land
next to each demo's `ckpt/` as `train.convergence_<timestamp>.log`. See `README.md`.

```bash
# ---- CPU demos (remote managed AgentCore sandbox, SANDBOX=agentcore-runtime) ----
bash code/demos/convergence/01_cpu_single.sh      # MBPP coding
bash code/demos/convergence/02_cpu_math.sh        # Hendrycks MATH
bash code/demos/convergence/03_cpu_gsm8k.sh       # GSM8K (easiest — good first run)
bash code/demos/convergence/04_cpu_math500.sh     # MATH-500
bash code/demos/convergence/05_cpu_deepscaler.sh  # DeepScaleR (hardest math)
bash code/demos/convergence/06_cpu_humaneval.sh   # HumanEval coding
bash code/demos/convergence/07_cpu_gretelsql.sh   # text-to-SQL
bash code/demos/convergence/08_cpu_hotpotqa.sh    # multi-hop RAG QA
bash code/demos/convergence/09_cpu_xlam.sh        # function-calling
bash code/demos/convergence/10_cpu_mcp.sh         # MCP tool-use
bash code/demos/convergence/11_cpu_mixtask.sh     # mixed families

# ---- GPU demos (KernelBench; CUDA sandbox) ----
# 12/13 are REMOTE-ONLY: export a live g6 sandbox ARN first.
AGENTCORE_SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/<g6> \
  bash code/demos/convergence/12_gpu_single.sh    # one dedicated g6 GPU per session
AGENTCORE_SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/<g6-farm> \
  bash code/demos/convergence/13_gpu_shared.sh    # fractional Ray farm (B4_GPU_FRACTION=0.5)

# 14 is LOCAL: start the farm on the HOST first, then run inside the container.
#   (host)      bash code/deploy/run_local_gpu_runtime.sh local
bash code/demos/convergence/14_gpu_local.sh        # whole-GPU farm on GPUs 6,7
#   (host)      bash code/deploy/run_local_gpu_runtime.sh stop   # teardown

# ---- Property demos, now real trainers (headline knob default-ON) ----
bash code/demos/convergence/15_cpu_quiesce.sh        # keep-warm across the barrier (QUIESCE_KEEP_WARM=1)
bash code/demos/convergence/16_cpu_harness_scale.sh  # warm run_episode worker pool (HARNESS_WORKERS=4, MODE=A)
bash code/demos/convergence/17_cpu_customruntime.sh  # deepscaler on your custom CPU Runtime
# 18 is REMOTE-ONLY (no LOCAL_GPU path): export a live g6 ARN.
AGENTCORE_SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/<g6> \
  bash code/demos/convergence/18_gpu_snapshot.sh     # app-level snapshot/resume (SNAPSHOT_EVERY=1)
# 19 runs REMOTE (g6 ARN) OR on-box (LOCAL_GPU=1, farm up on 6,7):
LOCAL_GPU=1 bash code/demos/convergence/19_gpu_customruntime.sh   # KernelBench on custom GPU Runtime
```

## Common overrides

```bash
# Train longer:
NR=401 bash code/demos/convergence/03_cpu_gsm8k.sh          # 400 steps

# Save only at the end (no ~56 GB mid-run checkpoints — judge convergence by the eval curve):
SAVE_INTERVAL=1000000 bash code/demos/convergence/02_cpu_math.sh

# Bigger batch / more exploration:
RB=16 TEMP=1.1 bash code/demos/convergence/05_cpu_deepscaler.sh

# GPU-shared: push fraction up if the non-MIG GPU wedges:
B4_GPU_FRACTION=1.0 bash code/demos/convergence/13_gpu_shared.sh
```

## Watch convergence live

```bash
tail -f code/demos/demo_cpu_gsm8k/train.convergence_*.log | grep -E 'eval/|reward'
```

Converged ⇒ `eval/<name>` has risen and plateaued for ~3+ consecutive eval points.
