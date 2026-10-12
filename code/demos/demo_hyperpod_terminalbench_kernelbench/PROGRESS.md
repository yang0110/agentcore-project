# PROGRESS — demo_hyperpod_terminalbench_kernelbench (pick up here)

Last updated 2026-10-12. Design and how-to are in [README.md](README.md); the original plan is
[docs/hyperpod-agentcore-build-plan.md](../../../docs/hyperpod-agentcore-build-plan.md) (written when HyperPod was
going to be a mode of the first demo; the user then asked for a separate, standalone demo).

## Goal

demo_terminalbench_kernelbench with slime moved onto **SageMaker HyperPod (Slurm), 2 x ml.g6e.48xlarge**:
the lower-IP node trains (Megatron, 8 x L40S), the other serves inference (SGLang, 8 x L40S) + the
rollout/harness tier. Same Terminal-Bench + KernelBench data, same agent and reward, same hybrid sandbox tier
(AgentCore microVM, AgentCore g6 Instance, m5 EC2, plain-EC2 g6). **Model: Qwen3.5-9B only** (user choice).
**Orchestrator: Slurm** (user choice). NCCL between the nodes over **EFA** (user asked).

## State

| Item | State |
|---|---|
| Code (this folder) | **written, committed, never run on AWS** |
| HyperPod cluster, FSx, cluster SG, node role, S3 bucket, ECR repo | **none exist**. Do NOT create without the user's go (~$60+/h while up) |
| Quota "ml.g6e.48xlarge for cluster usage" (us-west-2) | 20, enough |
| Sandbox tiers | already deployed by the first demo and **shared** (same AWS resources; route maps + m5/g6 state copied into this folder, gitignored) |
| EFA image | built locally only: `acs-tbhp-slime:agentcore_p4-20261006-efa1.48.0` (not pushed) |
| Data | `data/` built 2026-10-12: 394 train (151 TB tasks + 243 KernelBench, all 4 levels), eval 31 + 4 TB + 27 KB; every row routes (`image_routing.py`: 162 microVM, 24 m5, KB L1 -> AgentCore g6, L2-4 -> EC2 g6) |

## Verified (offline / on this box only)

- `bash -n` on every script, `py_compile` on the Python.
- botocore parameter validation of `CreateCluster`, `CreateFileSystem` and `ListClusterNodes` requests; `ml.g6e.48xlarge` is a valid `ClusterInstanceType`.
- Run-knob quoting: 5_run.sh -> SSM `env K=V` -> sbatch keeps JSON, commas and spaces intact.
- EFA layer: builds; torch NCCL 2.28.9 loads aws-ofi-nccl 1.19.0 (libfabric 2.4, platform AWS, provider efa); with no EFA device it falls back; slime / sglang / megatron import.
- `net_env.sh`: `HP_EFA=1` refuses without an EFA device; the interface lookup works (enp135s0 here).
- `nccl_bw.py` mechanics on 2 local GPUs over sockets: ~4-5 GB/s.

**Not verified** (needs the cluster): the lifecycle scripts / provisioning parameters, SSM exec as `ubuntu`,
enroot import of the ~68 GB image, pyxis named-container reuse across srun steps, Ray across 2 nodes,
Ray workers inheriting `ADAPTER_PUBLIC_HOST` from `ray start`, EFA on g6e inside the container, the 9B
memory fit at TP=4 x DP=2, the sandbox dial-back to the cluster SG, FSx checkpoint save time.

## Next steps (in order)

1. **Install the AWS session-manager-plugin** on this box (HyperPod nodes are reached over SSM):
   https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html
2. **Get the user's go** for the spend.
3. Start the m5 host (EC2-stopped between runs): `python3 deploy/m5_host.py status` and start it.
4. `bash 4b_hyperpod_up.sh` (image + EFA -> ECR, ~68 GB push; bucket/role/SG/FSx/cluster ~25 min; SG wiring;
   enroot import; `nccl-test`). **Record the EFA vs sockets numbers** from `logs/nccl_bw.out` in the README.
   - If create fails: `aws sagemaker describe-cluster --cluster-name acs-tbhp` (FailureMessage), and the
     lifecycle log on a node: `/var/log/provision/provisioning.log`.
   - If EFA is refused in a node: check `ls /dev/infiniband` on the host vs in the container; the escape hatch
     is `HP_EFA=0` (sockets), which must be passed to `hyperpod.py image` AND the run.
5. `NR=11 bash 5_run.sh` (10-step smoke). Check in the log: the `>>> [..] job N: trainer=… inference+head=…`
   line, both `>>> ray … nccl=EFA …` lines, slime's bundle lines (all 8 actor bundles on the trainer node),
   0 infra errors per substrate in `metrics.py`'s "by substrate" table, the weight-update time per step.
6. `bash 6_cleanup.sh` right after (cluster + FSx + SG; `KEEP_FSX=1` keeps checkpoints). The sandbox tiers
   stay (shared); `CLEAN_SANDBOXES=1` would remove them for BOTH demos.
7. Write the measured numbers into README.md (replace the "code only" status), then the deck/report
   (use case / why / design per demo, measured numbers only).

## Open items / decisions for later

- **Data balancing:** the first demo has an UNCOMMITTED builder change by another session (`TB_KB_SHARE=0.4`
  cycles Terminal-Bench rows so KernelBench is ~40% of train: 607 rows). This copy predates it (natural mix,
  394 rows). Copy `terminalbench/build_terminalbench_dataset.py` over once that is committed, if wanted.
- `OPT_OFFLOAD=0` with DP=2 may fit 9B (untested); `GPUS_PER_ENGINE=1` (8 engines) untested for 9B.
- No `perf/weight_sync_s` metric yet: read slime's own update-weights timing from the log for now.
- Resilience (plan P5): kill mid-run + resume from the FSx ckpt; HyperPod auto-resume (`--auto-resume=1`).
- The first demo's metrics/deck are separate; this demo has no logs yet.

## Rules that apply (from memory)

- Never two `run.sh` at once; never edit run.sh / step scripts while a run is going.
- Never rebuild the trainer image from `:latest`; only the EFA layer goes on top of `agentcore_p4-20261006`.
- Never put `AWS_BEARER_TOKEN_BEDROCK` (or any secret) in the SSM command; 5_run.sh already skips it.
- Single branch `master`; commit there.
