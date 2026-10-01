# Convergence runs

The long (~200-step) training run for the Terminal-Bench demo. The wrapper sets the long-run knobs and calls
the demo's own `run.sh`.

| Script | Demo | Dataset / reward |
|---|---|---|
| `20_cpu_terminalbench.sh` | [`demo_cpu_terminalbench`](../demo_cpu_terminalbench/README.md) | Terminal-Bench; the task's hidden pytest oracle |

Wrappers 01-19 and `21_cpu_swebench.sh` were removed on 2026-09-30 together with their demos.

## Prerequisites

- Run the wrapper **inside the `agentcore_p4` container** (repo at `/work`) with AWS credentials.
- Deploy the demo's pool first (steps 2-4), then run step 1 to rebuild the data from that pool.
- You need the base checkpoint `/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`.
- Budget about 56 GB of disk per saved checkpoint (`SAVE_INTERVAL=100`).
- **Run one `run.sh` at a time** (this wrapper or `5_run.sh`): a second Ray start kills the first job.

## Knobs

`NR` (default 201, so 200 train steps), `EVAL_INTERVAL` (20), `EVAL_NS` (4), `SAVE_INTERVAL` (100).
Every other `run.sh` knob can be overridden too, e.g. `NR=401 bash 20_cpu_terminalbench.sh`.

Results: `../demo_cpu_terminalbench/train.convergence_terminalbench_*.log`.
