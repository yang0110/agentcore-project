# PROGRESS — demo_hyperpod_terminalbench_kernelbench (pick up here)

Last updated 2026-10-13. Design and how-to are in [README.md](README.md); the original plan is
[docs/hyperpod-agentcore-build-plan.md](../../../docs/hyperpod-agentcore-build-plan.md) (written when HyperPod was
going to be a mode of the first demo; the user then asked for a separate, standalone demo).

## Goal

demo_terminalbench_kernelbench with slime moved onto **SageMaker HyperPod (Slurm), 2 x ml.g6e.48xlarge**:
the lower-IP node trains (Megatron, 8 x L40S), the other serves inference (SGLang, 8 x L40S) + the
rollout/harness tier. Same Terminal-Bench + KernelBench data, same agent and reward, same hybrid sandbox tier
(AgentCore microVM, AgentCore g6 Instance, m5 EC2, plain-EC2 g6). **Model: Qwen3.5-9B only** (user choice).
**Orchestrator: Slurm** (user choice). NCCL between the nodes over **EFA** (user asked).

## State (2026-10-13 ~02:50 UTC)

| Item | State |
|---|---|
| HyperPod cluster `acs-tbhp` (us-west-2) | **UP, kept up by the user's decision**: controller (ml.m5.xlarge) Running, Slurm + enroot + pyxis OK, FSx mounted at /fsx |
| GPU nodes | **0 of 2: no on-demand `ml.g6e.48xlarge` capacity in ANY us-west-2 AZ** (2a/2b/2c/2d all "We currently do not have sufficient capacity"). Active group `gpu-d` (us-west-2d, count 2) keeps retrying automatically; groups `gpu`, `gpu-c`, `gpu-a` are at 0 |
| p-family (p4d/p4de/p5/p5e/p5en/p6-b200) | **HyperPod cluster quota = 0**. The user applies for an increase themselves (codes: p5 L-8762A75F, p5en L-6EF3AE05, p5e L-FD431820, p4de L-BEF3120E, p4d L-47E0BC8F, p6-b200 L-44097F05; region us-west-2; value 2). This box's role cannot request quotas |
| FSx | `fs-081d25e0eab944907` 2.4 TiB PERSISTENT_2, us-west-2b, AVAILABLE |
| Cluster SG / role / bucket | `sg-REDACTEDaeb093e7` (wired to the runtime, m5 and g6 SGs) / `acs-tbhp-cluster` / `sagemaker-acs-tbhp-<ACCOUNT_ID>-us-west-2` |
| Trainer image | pushed: `acs-tbhp-slime:agentcore_p4-20261006-efa1.48.0` (ECR). NOT yet enroot-imported (needs a GPU node: the controller's disk is too small) |
| Data on S3 | staged (3933 files: this demo, 9B HF dirs, base ckpt); `/fsx/work` sync was running on the controller (24 GB at 02:50) |
| Billing now | controller + FSx, < $1/h. GPU nodes bill as soon as HyperPod manages to launch them |
| Data | 394 train (151 TB tasks + 243 KernelBench, all 4 levels), eval 31 + 4 TB + 27 KB; every row routes |

## Findings from the first AWS attempt (all fixed in the code)

- **CreateCluster API changed:** the account/API defaults to "Continuous" node provisioning, which REQUIRES
  `SlurmConfig` (NodeType Controller/Compute + PartitionNames) on each instance group, REJECTS
  `Orchestrator.Slurm.SlurmConfigStrategy`, and REJECTS a `provisioning_parameters.json` in the lifecycle
  prefix. HyperPod now sets up Slurm itself; FSx is mounted via `InstanceStorageConfigs.FsxLustreConfig`;
  our hook is `lifecycle/on_init_complete.sh` (`OnInitComplete`, enroot/pyxis via the upstream script).
- **The cluster reports InService before its GPU nodes exist** in Continuous mode: check the groups'
  `CurrentCount` and `list-cluster-events` / `describe-cluster-event` (the FailureMessage is only there).
- `OverrideVpcConfig` cannot be changed on an existing group: `deploy/hyperpod/move_gpu_az.py <az> [type]` adds
  a new group `gpu-<az letter>` (same partition) and scales the other GPU groups to 0.
- SSM exec: the document runs its command without a shell and mangles nested quotes (now base64 in one
  `bash -c`), the plugin always exits 0 (now an `@@RC n` line), and the plugin ENDS the session when its
  stdin closes (now stdin is held open until `@@RC`). The `pull` survived that because its process kept running.
- enroot reads `ENROOT_CONFIG_PATH` (`$HOME/enroot` by the upstream config), and a compute node's home is not
  the controller's: the ECR credentials now live in `/fsx/enroot/config` with `ENROOT_CONFIG_PATH` set.

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

1. **Get GPU nodes.** Either HyperPod's retry in `gpu-d` succeeds (watch
   `aws sagemaker describe-cluster --cluster-name acs-tbhp --query 'InstanceGroups[].[InstanceGroupName,CurrentCount,TargetCount]'`),
   try other AZs again later (`python3 deploy/hyperpod/move_gpu_az.py us-west-2b`), or, once a p-family quota is
   approved: `python3 deploy/hyperpod/move_gpu_az.py <az> ml.p5.48xlarge` (or p4de/p5en/p6-b200).
   - **For a p-family type, adjust run.sh first:** sm80/sm90 run FlashAttention for the 9B's head_dim 256
     (L40S needed `ATTN_BACKEND=unfused` + `MAX_TOKENS_PER_GPU=4096`), 80-141 GB GPUs allow a larger
     `MAX_TOKENS_PER_GPU` and likely `OPT_OFFLOAD=0`; EFA with GPUDirect RDMA (`FI_EFA_USE_DEVICE_RDMA=1`
     in net_env.sh for p4d/p5). If the cluster were created anew, set `HP_GPU_TYPE`.
2. Once both nodes are Running: `python3 deploy/hyperpod/hyperpod.py import-image` (once), then
   `python3 deploy/hyperpod/hyperpod.py nccl-test`. **Record the EFA vs sockets numbers** in the README.
   - If EFA is refused on a node: `ls /dev/infiniband` on the host vs in the container; escape hatch `HP_EFA=0`.
3. Start the m5 host if stopped (`python3 deploy/m5_host.py status`).
4. `NR=11 bash 5_run.sh` (10-step smoke). Check: the `job N: trainer=… inference+head=…` line, both
   `ray … nccl=EFA` lines, slime's bundle lines (8 actor bundles on the trainer node), 0 infra errors per
   substrate in `metrics.py`'s "by substrate" table, the weight-update time per step.
5. `bash 6_cleanup.sh` right after (cluster + FSx + SG; `KEEP_FSX=1` keeps checkpoints and staged data).
   The sandbox tiers stay (shared); `CLEAN_SANDBOXES=1` would remove them for BOTH demos.
6. Measured numbers into README.md (replace the "code only" status), then the deck/report.

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
