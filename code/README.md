# code/

The code for **Agentic RL on Terminal-Bench with slime and AgentCore Runtime**.

- [`demos/demo_cpu_terminalbench/`](demos/demo_cpu_terminalbench/README.md): the demo. It is
  self-contained: the harness lib is vendored in `agentcore_slime_demo/`, the deploy config is in
  `deploy/`, and the numbered step scripts `1_prepare_data.sh` … `6_cleanup.sh` run it end to end.
- [`demos/convergence/`](demos/convergence/README.md): `20_cpu_terminalbench.sh`, the ~200-step run.
- `p4_slime_train/`: `00_start_container.sh` starts the `agentcore_p4` slime container, and
  `ckpt/Qwen3-4B_torch_dist` is the base Megatron checkpoint.

The original offline library (`code/agentcore_slime_demo`), `code/scripts`, `code/tests`,
`code/deploy` and all other demos were removed on 2026-09-30. They are still in git history.
