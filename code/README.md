# code/

The code for **Agentic RL on Terminal-Bench + KernelBench with slime and AgentCore**.

- [`demos/demo_terminalbench_kernelbench/`](demos/demo_terminalbench_kernelbench/README.md): the demo.
  It is self-contained: the harness lib is vendored in `agentcore_slime_demo/`, the deploy config is
  in `deploy/`, and the numbered step scripts `1_prepare_data.sh` … `6_cleanup.sh` run it end to end.
- `p4_slime_train/`: `00_start_container.sh` starts the `slime_trainer` slime container, and
  `ckpt/Qwen3-4B_torch_dist` is the base Megatron checkpoint.

The microVM-only `demos/demo_cpu_terminalbench` and `demos/convergence/` were removed on 2026-10-06.
The original offline library (`code/agentcore_slime_demo`), `code/scripts`, `code/tests`,
`code/deploy` and all other demos were removed on 2026-09-30. They are still in git history.
