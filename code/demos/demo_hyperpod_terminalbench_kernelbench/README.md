# demo_hyperpod_terminalbench_kernelbench — demo_terminalbench_kernelbench + SageMaker HyperPod: the model on a 2-node cluster, the environments on AgentCore + EC2

**Status (2026-10-13): cluster up, no GPU nodes yet, nothing measured.** The cluster `acs-tbhp` (controller +
FSx) is InService, but no us-west-2 AZ has on-demand `ml.g6e.48xlarge` capacity and the p-family HyperPod
quotas are 0. See [PROGRESS.md](PROGRESS.md) for the live state and next steps.

This is [`demo_terminalbench_kernelbench`](../demo_terminalbench_kernelbench/README.md) with **slime moved
off the single box onto a SageMaker HyperPod Slurm cluster**. The data, the agent, the reward and every
sandbox are unchanged. It is a standalone folder (its own vendored lib, deploy scripts, data and step
scripts), and it uses the **same deployed sandbox tiers** (same AWS resources) as the first demo.

```
 SageMaker HyperPod (Slurm, default VPC, acs-egress-us-west-2b, FSx for Lustre at /fsx)
 ┌─────────────────────────────────┐    ┌────────────────────────────────────┐
 │ gpu node 1  ml.g6e.48xlarge     │    │ gpu node 2  ml.g6e.48xlarge        │
 │ (lower IP)  TRAINER             │    │ INFERENCE + rollout tier + Ray head │
 │ Megatron, 8 x L40S, TP=4 x DP=2 │───►│ SGLang, 4 engines x TP=2           │
 │ ckpt -> /fsx (FSx for Lustre)   │NCCL│ router, fully-async buffer,         │
 │                                 │    │ harness pool, adapters :18100-18163 │
 └─────────────────────────────────┘    └──────────┬──────────────▲───────────┘
   controller ml.m5.xlarge (slurmctld)             │ open/exec/   │ agent loop
                                                   │ grade        │ dials back
                                                   ▼              │
 ┌──────────────────┬───────────────────┬──────────────┬─────────┴────────┐
 │ AgentCore microVM│ AgentCore g6       │ m5 EC2       │ plain-EC2 g6     │
 │ Terminal-Bench   │ KernelBench L1     │ TB compose / │ KernelBench L2-4 │
 │ (162 Runtimes)   │ (capacity provider)│ x86 / >2 GB  │ (warm pool)      │
 └──────────────────┴───────────────────┴──────────────┴──────────────────┘
```

| Piece | Where | Change vs demo_terminalbench_kernelbench |
|---|---|---|
| Policy | Qwen3.5-9B (the only model here) | was Qwen3-4B by default |
| Trainer | HyperPod node 1: all 8 L40S (TP=4 x DP=2, Adam offloaded to host RAM) | was 4 GPUs of the box, TP=4 x DP=1 |
| Inference | HyperPod node 2: all 8 L40S (4 SGLang engines x TP=2) | was 4 GPUs (2 engines) |
| Rollout buffer, harness pool, adapters | RolloutManager on node 2 (wherever Ray places it, the adapter host follows) | was the box |
| Checkpoints, model, logs | FSx for Lustre (`/fsx/work` = `/work` in the container) | was the box's EBS |
| Weight sync | Megatron (node 1) → SGLang (node 2), NCCL over **EFA** (aws-ofi-nccl; `HP_EFA=0`: TCP sockets) | was inside one box |
| Sandboxes | unchanged: microVM / AgentCore g6 / m5 / plain-EC2 g6, routed per task | — |
| Data | 394 train rows (151 Terminal-Bench tasks + 243 KernelBench, all 4 levels), eval 31 + 4 TB + 27 KB | the first demo's built `data/` still had the old 100 KB rows |

## Use case covered

Agentic RL on a model too big for one box's split, where the trainer, the inference engines and
hundreds of heterogeneous sandboxes each need to scale on their own: the GPUs that hold the policy on
a managed training cluster, and the environments on the services built for them.

## Why this demo

- **The first demo splits ONE 8-GPU box** 4 train + 4 inference. The 9B ran there at TP=4 with offload and
  2x the 4B step time. Here each role gets a whole node: 2x the trainer GPUs and 2x the engines.
- **It shows the division of labour a production setup would use:** HyperPod scales the model
  (Slurm, node health checks and auto-replacement, FSx checkpoints), AgentCore scales the environments,
  and our own EC2 fills the gaps (compose tasks, warm GPU slots). The rollout code is the same whichever
  substrate an episode lands on.
- **Checkpoints on FSx, not EBS:** a 9B save is ~200 GB, which was slow on the box's EBS.

## Design considerations

- **Roles by IP, no slime patch.** slime puts the actor on the first `ACTOR_GPUS` bundles of one placement
  group sorted by node IP, so the lower-IP node trains. `train.sbatch` sorts the 2 nodes the same way,
  logs the roles, and starts the Ray head (and `run.sh`) on the inference node. Both nodes are one
  instance group (`gpu` x2, partition `rl`): a Slurm job runs in one partition, and the roles are a
  per-job decision anyway.
- **The adapter host follows the RolloutManager.** slime gives the manager no node pin. `ray_node.sh`
  exports each node's own VPC IP as `ADAPTER_PUBLIC_HOST` (and its interface as `NCCL_SOCKET_IFNAME`)
  before `ray start`, Ray workers inherit it, and `run.sh` leaves it out of the job's runtime_env. The
  cluster SG admits the dial-back on both nodes, so either placement works.
- **Network: SG to SG only**, in the same default VPC and AZ (us-west-2b) as the m5. `hyperpod.py wire`
  reuses the vendored helpers with `AGENTCORE_TRAINER_SG=<cluster SG>`, so every rule the sandbox tiers
  wrote for the box is written for the cluster: runtime / m5 / g6 SGs → cluster 18100-18163 (+18021);
  cluster → m5 / g6 18300 and 32768-60999. The nodes reach AWS APIs, ECR and pip through the NAT subnet.
- **One image, never rebuilt.** The trainer container is the committed snapshot
  `slimerl/slime:agentcore_p4-20261006` (in-place slime edits), pushed to ECR once and squashed to
  `/fsx/enroot/slime.sqsh`; pyxis runs it on both nodes with host networking.
- **Same paths as the box.** `stage_s3.py` (in `slime_trainer`, where the HF symlinks resolve) uploads this
  demo, the 9B HF dirs and the base Megatron ckpt to S3. `hyperpod.py pull` syncs them to `/fsx/work`,
  which the container mounts at `/work`, so `run.sh`'s paths are unchanged.
- **EFA for the cross-node link** (`HP_EFA=1`, default). Training collectives stay inside node 1; what
  crosses nodes is the per-step weight push, Megatron → the SGLang engines (9B bf16 ≈ 18 GB). The snapshot
  image has no EFA libraries, so `deploy/hyperpod/Dockerfile.efa` adds ONE layer on top of it: EFA installer
  1.48.0 with `--skip-kmod` (libfabric 2.4 with the efa provider, aws-ofi-nccl 1.19.0, rdma-core; the
  driver comes from the node), the same recipe as aws-samples' `slime.Dockerfile`. `slurm/net_env.sh` sets
  `FI_PROVIDER=efa NCCL_NET_PLUGIN=ofi FI_EFA_USE_HUGE_PAGE=0` and **refuses to start** when libfabric finds
  no EFA device: with none, NCCL would silently fall back to sockets. g6e (L40S) has no GPUDirect RDMA, so
  aws-ofi-nccl stages through host memory. The cluster SG has the self-referencing ingress AND egress EFA
  needs. `HP_EFA=0` = TCP sockets (the A/B baseline; tags the image without the EFA layer).
  **Verified on this box (2026-10-12, no EFA device):** the layer builds; torch NCCL 2.28.9 loads
  `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so` (platform AWS, provider filter efa) and, with no device,
  falls back; slime / sglang / megatron still import; `nccl_bw.py` runs (2 local GPUs, sockets ~4-5 GB/s).
  Not yet run on EFA hardware: `hyperpod.py nccl-test` (step 4b.5) measures EFA vs sockets across the 2 nodes.
- **Secrets stay off the command line.** The run knobs travel to the controller as `env K=V` words in an
  SSM command; `AWS_BEARER_TOKEN_BEDROCK` is never forwarded (the nodes use their own IAM role).
- **Node IAM role** (`acs-tbhp-cluster`): HyperPod's managed instance policy, plus what the rollout process
  calls from the cluster: `InvokeAgentRuntime` / `StopRuntimeSession` on `acs_*` Runtimes, EC2
  `RunInstances` / `TerminateInstances` / `Describe*` and `PassRole` on `acs-tbh-m5-episode` (the m5 and
  plain-EC2 g6 tiers), ECR pull, and the staging bucket.
- **HyperPod sets up Slurm itself** (the API's Continuous node-provisioning mode): each instance group carries
  `SlurmConfig` (controller / compute + partition `rl`), FSx is mounted at `/fsx` through
  `InstanceStorageConfigs.FsxLustreConfig`, and our only hook is `lifecycle/on_init_complete.sh` (enroot +
  pyxis through the vendored aws-samples `base-config` script, `7bca2cf`, MIT-0; it skips what the AMI has).
  No `provisioning_parameters.json` (this mode rejects one).
- **No capacity in one AZ:** `deploy/hyperpod/move_gpu_az.py <az> [type]` adds a GPU group in another AZ (an
  existing group's subnet can't change) and scales the others to 0; the controller and FSx stay put.

## Cost (list prices, check before a run)

2 x ml.g6e.48xlarge (the SageMaker `ml.` rate is above EC2's ~$30/h each), 1 x ml.m5.xlarge, FSx
PERSISTENT_2 2.4 TiB at 250 MB/s/TiB, plus the sandbox tiers per episode as in the first demo. **Keep the
cluster up only for runs: `bash 6_cleanup.sh` deletes it** (FSx too unless `KEEP_FSX=1`).

## Step by step (run from this dir on the HOST)

Steps 1-4 are the first demo's: they deploy (or reuse; all idempotent) the sandbox tiers. They act on
the same AWS resources, which are already deployed, so on this account steps 2-4 only re-check them.

```bash
python3 deploy/m5_host.py status   # the m5 is EC2-stopped between runs: start it first
bash 2_build_image.sh      # sandbox images (reused)
bash 3_push_ecr.sh         # report only
bash 4_deploy_runtime.sh   # sandbox Runtimes + oracle validation (reused)
bash 4b_hyperpod_up.sh     # NEW: trainer image (+EFA) -> ECR, bucket/role/SG/FSx/cluster (~25 min), SG wiring, image -> /fsx, EFA-vs-socket NCCL test
bash 1_prepare_data.sh     # rows for every routed + validated task (394 train / 62 eval as of 2026-10-12)
NR=11 bash 5_run.sh        # stage -> /fsx, sbatch on both nodes, follow the log; logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # delete the cluster + FSx + its SG; CLEAN_SANDBOXES=1 also removes the shared sandbox tiers
```

Needs, on this box: the AWS **session-manager-plugin** (HyperPod nodes are reached over SSM), docker
(the image push), and the quota "ml.g6e.48xlarge for cluster usage" ≥ 2 (20 on this account).
Drive the cluster directly with `python3 deploy/hyperpod/hyperpod.py status|exec '<cmd>'|submit|follow|cancel|down`.

**What is left after a run.** The Slurm job's end stops Ray and SGLang on both nodes; `5_run.sh` then
sweeps the m5 / g6 EC2 instances (`gc-episodes`, `gc`), as in the first demo. A Ctrl-C on `5_run.sh`
`scancel`s the job (run.sh stops its ray job on the signal). Checkpoints stay on FSx until `6_cleanup.sh`.

## HyperPod knobs

| Knob | Default | What |
|---|---|---|
| `HP_CLUSTER` | `acs-tbhp` | cluster + FSx name |
| `HP_AZ` | `us-west-2b` | the one AZ of the cluster, FSx and its egress subnet (the m5's AZ) |
| `HP_GPU_TYPE` / `HP_GPU_COUNT` | `ml.g6e.48xlarge` / `2` | the `gpu` instance group; the job uses 2 nodes |
| `HP_CONTROLLER_TYPE` | `ml.m5.xlarge` | Slurm controller |
| `HP_EBS_GB` | `500` | extra EBS per GPU node (enroot also uses the local NVMe) |
| `HP_FSX_GIB` | `2400` | FSx for Lustre size (1200 GiB steps) |
| `KEEP_FSX` | `0` | `6_cleanup.sh` keeps the FSx (and its checkpoints) |
| `HP_EFA` | `1` | NCCL between the nodes over EFA (image + Dockerfile.efa; fails without an EFA device); `0` = TCP sockets |
| `HP_EFA_VERSION` | `1.48.0` | EFA installer baked into the image (tag `agentcore_p4-20261006-efa<ver>`) |
| `ACTOR_GPUS` / `ROLLOUT_GPUS` | `8` / `8` | slime's split over the 16 GPUs |
| `GPUS_PER_ENGINE` | `2` | SGLang TP per engine (4 engines) |
| `OPT_OFFLOAD` | `1` | Adam state in host RAM; `0` may fit with DP=2 (untested) |

## Folder layout (self-contained)

- `1_…6_*.sh` + `4b_hyperpod_up.sh`: the step scripts. `run.sh` is the slime launch, run by
  `train.sbatch` inside the container on the inference node (it no longer starts Ray).
- `deploy/hyperpod/`: `hyperpod.py` (image, cluster, FSx, SG wiring, SSM exec, submit/follow, nccl-test),
  `stage_s3.py`, `Dockerfile.efa`, `slurm/` (`train.sbatch`, `ray_node.sh`, `net_env.sh`, `nccl_bw.sbatch` +
  `nccl_bw.py`), `lifecycle/` (vendored), `hyperpod.json` (state, not in git).
- Everything else is the first demo's, vendored: `agentcore_slime_demo/`, `m5_agent/`, `terminalbench/`,
  `kernelbench/`, `skills/`, `deploy/` (sandbox tiers), `image_routing.py`, `metrics.py`, `prepare_data.sh`.
  `terminalbench/tasks/` (corpus + route maps) and `deploy/{m5,g6}_host.json` were copied from the first demo
  (not in git) so both demos route to the same deployed sandboxes.

# Sandbox side (unchanged from demo_terminalbench_kernelbench)

The sections below are the first demo's, kept because they still describe how every episode runs.
Where they say "the box" or "the trainer", read: the HyperPod inference node, where the rollout tier runs.

## How the m5 tier runs (who launches what)

AgentCore doesn't manage the m5s; the trainer does. `TerminalBenchPool` (in slime's rollout process) calls
the EC2 API with boto3, using the g6e's role. See [`docs/m5-per-episode-isolation.md`](../../../docs/m5-per-episode-isolation.md).
- **Builder** (`acs-tbh-m5`, one `m5.8xlarge`, `deploy/m5_host.py up`): builds every m5 task's images and
  pushes them, sidecars included, to ECR `acs-tbhdk-<task>-<service>`. It is also where oracle validation
  runs (`TB_M5_MODE=shared`). Training doesn't need it.
- **Per episode** (`Ec2EpisodeSandbox`):
  1. `run_instances` from the baked AMI (`deploy/m5_host.py bake-ami`: Docker + compose agent, no task
     images), with a fresh agent secret.
  2. Upload the task folder (no tests or solution) and its ECR image list.
  3. The agent runs `docker login` + pull, then `compose up`.
  4. The episode and its grading run.
  5. `terminate_instances`.
- **Pre-open:** slow substrates are opened `TB_PREOPEN_BATCHES_SLOW=3` batches ahead (microVMs 2); at most
  `TB_M5_MAX_INSTANCES=32` instances are live at once.
- **Leftovers** (tag `acs-tbh-m5-ep`) are terminated at the next pool start, by `6_cleanup.sh`, or by
  `deploy/m5_host.py gc-episodes`.

## KernelBench: all 270 problems, routed per level

`kernelbench/build_kernelbench.py` (run in slime_trainer; needs `datasets`) builds the rows from
`ScalingIntelligence/KernelBench`, all four levels: 100 single ops (L1), 100 fusions (L2), 50 architectures
(L3), 20 HF models (L4). It is the old `agentcore_slime_demo.kernelbench` generator, and it reproduces the
120 rows earlier runs used byte for byte.
- **Held-out eval, 27 rows:** the same 20 L2 problems as before (so eval/kernelbench stays comparable),
  plus L3 ids 10/20/30/40/50 and L4 ids 10/20. **Train:** the other 243.
- **One image key per level**, so `image_routing.py` can route levels separately:
  `kernelbench-gpu` (L1), `kernelbench-gpu-l2`, `-l3`, `-l4`. `TB_KB_SUBSTRATE`:

| `TB_KB_SUBSTRATE` | L1 | L2, L3 | L4 |
|---|---|---|---|
| `split` (default) | AgentCore g6 Instance | EC2 GPU pool (`TB_KB_EC2_LEVELS`, default 2,3,4) | EC2 GPU pool |
| `agentcore` | AgentCore g6 Instance | AgentCore g6 Instance | not routable: build without it (`TB_KB_LEVELS=1,2,3`) |
| `ec2` | EC2 GPU pool | EC2 GPU pool | EC2 GPU pool |

- **L4 runs only on the EC2 pool.** Each problem calls `AutoModelForCausalLM.from_pretrained`: 7 distinct HF
  models, ~16 GB, the largest gpt-neo-2.7B at 10.7 GB. They are baked into the EC2 image
  (`HF_HUB_OFFLINE=1`; the image is 17 GB compressed), which AgentCore's 2 GB image cap rules out.
  Grading must not download them per episode.
- **L3** needs `einops` (in both images; the AgentCore one installs it at start, so redeploy step 4g for
  `agentcore` mode).
- **Oracle check:** `kernelbench/validate_kernelbench.py` submits each problem's reference as `ModelNew`
  through the real hidden grader on the EC2 pool and writes `kernelbench/oracle.csv`. `build_kernelbench.py`
  then drops the rows whose oracle failed (`fail` / `timeout`, never `infra`) and prints each with its reason.

## KernelBench on the g6 tier

The level-1 rows (and every level in `agentcore` mode) name the image `kernelbench-gpu`, which routes to
`gpu:<acs_tbhgpu_kernelbench ARN>`. `TerminalBenchPool` runs them with the same fresh, pre-opened,
gated session as Terminal-Bench rows, but grades them with the coding episode (`mini_claude_episode`):
- the agent writes `solution.py` with `ModelNew`;
- the hidden checker (random name, nonce'd PASSED line) builds it next to the reference `Model` on the
  L4 and checks shape, values, and that it ran on CUDA;
- it prints `speedup=`;
- the reward is 1 if the kernel is correct and ≥ `TB_KB_SPEEDUP_MIN` (1.10×) faster, 0.5 if only correct,
  0 otherwise. Copies of the reference measured 1.00–1.03×, so the margin is above timing noise.

The image (`deploy/Dockerfile.kernelbench_gpu`, 145 MB) installs CUDA torch at container start, because
AgentCore's 2 GB image cap applies to capacity-provider Runtimes too. Its server is
`deploy/acs_http_server_chunked.py`: Instance Runtimes forward request bodies chunked. `g5.xlarge` (A10G) is
accepted by AgentCore as an alternative (`TB_GPU_INSTANCE`), but g6 is cheaper (~$0.80/h vs ~$1.01/h)
and the verified default.

## Where the g6 instances are

AgentCore runs capacity-provider instances in a service-managed account, so they do **not** appear in
your EC2 console. A g6 is launched when a session's first call arrives (the pool's pre-open ping), serves
that one session, and is released after `StopRuntimeSession` at the episode's end (any instance with no
session goes after the capacity provider's 600 s idle timeout).

To see them: CloudWatch log group `/aws/bedrock-agentcore/runtimes/<acs_tbhgpu_kernelbench id>-DEFAULT`
has one log stream per session (= per g6 instance), named by the session id; the run log has each
session's `[sbx-op]` open (first `ping`) and `stop_session` lines.

## KernelBench on the plain-EC2 GPU pool (levels 2-4 by default)

KernelBench levels in `TB_KB_EC2_LEVELS` (`gpuec2:<image>` routes) run on a **warm pool of plain EC2 GPU
instances in this account**: the warm m5 pool carried over to GPUs (`agentcore_slime_demo/ec2_gpu_sandbox.py`, `deploy/g6_host.py`):

- **AMI** (`g6_host.py bake-ami`): Deep Learning Base OSS Nvidia AMI (AL2023: driver, Docker, NVIDIA container
  toolkit) + the m5 bootstrap and compose agent + the KernelBench image pre-pulled. The image
  (`deploy/Dockerfile.kernelbench_ec2`, ECR `acs-tbhdk-kernelbench-gpu-client`) has CUDA torch **baked in**:
  no 2 GB cap here, so no ~90 s torch install per open.
- **Slots**: one episode per GPU (an L4 has no MIG), a fresh container per episode pinned to that GPU
  (compose device reservation) with a 1/GPUs share of vCPUs and memory. g6.xlarge = 1 slot (default);
  a g6.12xlarge would give 4 (but costs more per GPU: ~$4.60/h vs 4 x ~$0.80/h).
- **Reuse**: after an episode, `down -v` + a reset check scoped to its project + "no process left on that
  GPU"; clean -> the slot is reused (open ~1.3 s), dirty / dead agent -> the instance is terminated. An
  instance idle `TB_G6_IDLE_RETIRE_S` is terminated; the m5 dead-man switch, the exit hook and
  `5_run.sh` / `6_cleanup.sh` (`g6_host.py gc` / `down`) are the backstops.
- **Capacity**: each launch sweeps `instance_types` x AZ subnets (`g6.xlarge, g5.xlarge, g6e.xlarge, g6.2xlarge,
  g5.2xlarge, g6.4xlarge` x 4 AZs), skips a type/AZ that just said InsufficientInstanceCapacity for 60 s
  (process-wide), keeps at most `TB_G6_LAUNCH_CONC` RunInstances in flight, and re-sweeps for
  `TB_G6_LAUNCH_WAIT_S`. `TB_G6_CAPACITY_RESERVATION_ID` (+ `TB_G6_CAPACITY_BLOCK=1`) targets an ODCR / Capacity Block.
  The shortage itself is NOT fixed: on 2026-10-12 every g6/g6e size was starved in all 4 AZs and the pool ran on g5.
- **Isolation is weaker** than AgentCore's one-instance-per-session: episodes on a reused instance share its
  kernel and driver (own container, own GPU, no host or metadata access). Fine for KernelBench.
- The instances are in **your EC2 console** (tag `Name=acs-tbh-g6ec2`; `python3 deploy/g6_host.py status`).

    python3 deploy/g6_host.py image && python3 deploy/g6_host.py bake-ami   # once (~25 min)
    python3 deploy/g6_host.py probe                                          # optional: cold vs warm open
    bash 5_run.sh                       # split (default); TB_KB_SUBSTRATE=ec2 puts every level on the pool

**Measured (2026-10-12, `../demo_terminalbench_kernelbench/logs/kb-g6ec2-10step_20261012_091850`, on the first demo, KernelBench-only, qwen3-4B, RB=4 x NS=8,
`TB_G6_MAX_INSTANCES=16`):** 10/10 steps + eval in 26 min; 381 episodes, **0 infra errors**; open p50
**1.35 s** / p95 52 s (AgentCore g6, 9B epoch run 2026-10-11: p50 133 s, start wait p50 50 s); 95 % of opens
reused a warm instance (up to 28 episodes per instance); 18 launches, all g5.xlarge (g6 starved),
RunInstances incl. capacity sweeps p50 92 s; step time 34-72 s after the first (367 s, the fleet
launching); 9/10 steps with a gradient; eval/kernelbench 0.425. ~18 g5.xlarge x ~0.4 h ≈ $8.

## Warm m5 pool (how instances are reused and when they are terminated)

`WarmPool` (`agentcore_slime_demo/ec2_episode_sandbox.py`) serves one session per instance at a time:
- **Opening:** use an idle instance (prefer one already holding the task's images); else launch one if
  live + launching < `TB_M5_MAX_INSTANCES` (24); else wait. So the pool grows only to the peak concurrent
  demand.
- **After each episode:** `compose down -v`, then `reset_check`: no containers, volumes or non-default
  networks, an empty episode dir, and disk ≥ 8 GB (cached images are evicted otherwise).
  - Clean: the instance goes back to the pool.
  - Dirty, or a failed health check on reuse: it is terminated, never reused.
- **No periodic recycle:** the agent never touches the host, so a verified reset leaves nothing behind.
- **Termination** (AgentCore doesn't manage these instances; our code does):
  - the trainer's `atexit` hook;
  - `5_run.sh` → `deploy/m5_host.py gc-episodes` (every instance tagged `acs-tbh-m5-ep`);
  - the next run's pool start;
  - `6_cleanup.sh`;
  - a **dead-man switch**: an instance with no trainer call for `TB_M5_IDLE_SHUTDOWN_S` (1800) shuts
    itself down. Its `InstanceInitiatedShutdownBehavior=terminate` makes that a terminate.

## Isolation

| | microVM | m5 compose project |
|---|---|---|
| Per episode | its own microVM (own kernel) | per_episode: its own EC2 VM (own kernel), terminated after grading. shared: its own containers, networks, volumes and task-dir copy, removed with `down -v` |
| Shared | the read-only image | per_episode: nothing (images pulled from ECR). shared: the read-only images and the host kernel |
| Agent ↔ sidecars | n/a | container boundary; reachable only over the project network, as upstream |

The m5 is weaker than a microVM (shared kernel), and it is how Terminal-Bench itself runs. The host is
hardened:
- No IAM role.
- IMDSv2 with hop limit 1, and `DOCKER-USER` drops traffic to 169.254.169.254.
- Containers can't reach the host itself: an `INPUT` drop on `docker0`/`br-*` covers the agent, sshd
  and the daemon.
- The agent API needs a shared secret and is open only to the trainer's SG.
- Docker uses its own `10.200.0.0/16` address pool, so it never shadows the VPC's `172.31/16`.

## Rollout + sandbox knobs (env vars; defaults in `run.sh` / the deploy scripts)

| Knob | Default | Meaning |
|---|---|---|
| `NR` / `EVAL_INTERVAL` | `epoch` / 10 (5_run.sh) | rollouts (NR-1 train steps; `epoch` = train rows/RB + 1, one full pass) / eval cadence |
| `TB_PREOPEN_BATCHES` | 2 | batches of sandboxes booted ahead, both substrates (0 = open on demand) |
| `TB_M5_MODE` | warm | `warm` = one m5 per session, reused after a verified reset; `per_episode` = terminated after each episode; `shared` = one compose project each on the builder |
| `TB_M5_MAX_INSTANCES` | 24 | warm / per_episode: live m5 instances at once (~87 % of moments need ≤ 24) |
| `TB_PREOPEN_BATCHES_SLOW` | 5 | batches ahead for m5 VMs and g6 Instances (microVMs: `TB_PREOPEN_BATCHES`); a g6 needs ~2.5 min (boot + torch install) |
| `TB_DOCKER_MAX_PROJECTS` | 16 | shared mode: m5 episode projects at once (trainer-side gate; the host's `M5_MAX_PROJECTS` is the backstop) |
| `TB_KERNELBENCH` / `TB_KB_TRAIN` / `TB_KB_EVAL` | 1 / all (100) / all (20) | the KernelBench g6 Runtime (step 4) / KernelBench rows in train / eval (caps; 0 = leave out) |
| `TB_GPU_TASKS` / `TB_GPU_MAX_SESSIONS` / `TB_GPU_INSTANCE` | empty / 32 / g6.xlarge | Terminal-Bench tasks on the g6 tier / live g6 sessions at once, per pool (at 16 the gate was full 66 % of smoke 5 and 31/36 pre-opened sessions were still booting; ~4 KernelBench prompts × 8 are in the look-ahead window) / instance type |
| `TB_KB_SUBSTRATE` / `TB_KB_EC2_LEVELS` | split / 2,3,4 | KernelBench substrate per level (table in "KernelBench: all 270 problems"): `split` = L1 on the AgentCore g6 Runtime, these levels on the EC2 GPU pool (`deploy/g6_host.json`); `agentcore` = all on AgentCore (no L4); `ec2` = all on the pool |
| `TB_KB_LEVELS` | 1,2,3,4 | `kernelbench/build_kernelbench.py`: levels written to `kernelbench/*.jsonl` (then step 1) |
| `TB_G6_MAX_INSTANCES` / `TB_G6_IDLE_RETIRE_S` / `TB_G6_LAUNCH_WAIT_S` / `TB_G6_LAUNCH_CONC` | 16 / 900 / 900 / 2 | `ec2` substrate: live instances (x GPUs = episode slots) / terminate an instance idle this long / keep re-sweeping starved candidates this long / RunInstances in flight (more got the account throttled) |
| `TB_G6_CAPACITY_RESERVATION_ID` / `TB_G6_CAPACITY_BLOCK` | empty / 0 | `ec2` substrate: launch into this ODCR (or Capacity Block, =1) — pins type + AZ |
| `TB_KB_SPEEDUP_MIN` | 1.10 | KernelBench reward: 1 if correct and at least this much faster than the reference, 0.5 if only correct, 0 otherwise (0 = correctness only) |
| `TB_M5_IDLE_SHUTDOWN_S` | 1800 | m5 dead-man switch, armed from the instance's first boot (user-data systemd drop-in): an episode instance with no trainer call for this long terminates itself (0 = off). Must exceed the longest normal silence of an idle warm instance (final checkpoint save, eval: ~10 min); shorter = idle warm instances get replaced mid-run, longer = orphans bill longer after a crash |
| `RUNTIME_IDLE_TIMEOUT` / `TB_GPU_MAX_LIFETIME` | 900 / 14400 | step 4: AgentCore session idle timeout (every Runtime) / hard cap on one g6 session. A session whose stop was lost bills until the idle timeout; step 4 re-applies both to the g6 Runtime in place, microVM Runtimes pick the idle timeout up on their next create or digest update |
| `INVOKE_EXEC_MAX_RETRIES` | 1 | retries of a failed `exec` (not idempotent; 9 retries of a 4-min call stalled an eval) |
| `HEARTBEAT` | 1 | keep idle microVM sessions alive across trainer stalls |
| `EPISODE_MAX_TURNS` / `EPISODE_MAX_TOKENS` | 8 / 1024 | agent turns / tokens per turn |
| `HARNESS_SET` | `mini_claude` | per-episode harness routing (comma list of `kind[:fraction]`). Kinds: `mini_claude` (Python mimic, in-session — the cheap baseline), `mini_claude_rt` (same mimic, decoupled on its own harness Runtime), `claude_cli` (the **real Claude Code CLI**, decoupled). Recommended multi-harness demo: `mini_claude_rt:0.5,claude_cli:0.5` (both decoupled). Any decoupled kind needs the harness Runtime from step 4h (`HARNESS_SET=claude_cli bash 4_deploy_runtime.sh` → `deploy/harness_cli.json`) |
| `HARNESS_ASSIGN` | prompt | `prompt` = all `NS` samples of a prompt share a harness (clean GRPO within-group advantage); `episode` = per-episode random |
| `HARNESS_CLI_SESSIONS` | ≈ `HARNESS_WORKERS`×frac | warm reused sessions on the decoupled harness Runtime |
| `TB_EXEC_TIMEOUT` | 180 | per bash-tool command, seconds |
| `M5_INSTANCE_TYPE` / `M5_DISK_GB` | m5.8xlarge / 500 | the Docker host |
| `M5_CLIENT_MEM` / `M5_CLIENT_CPUS` | 8g / 4 | limits on each episode's client container |
| `TB_DOCKER_ORACLE_FALLBACK` | 1 | retry microVM oracle failures on the m5 (step 4) |
| `TB_RUNTIME_TAG` | tbh | microVM Runtime names `acs_<tag>_<task>` |
| `TB_VALIDATE` | 1 | step 4 validates every oracle (0 = skip; then step 1 trains on unvalidated tasks) |
| `SKILLS` / `SKILLS_SET` / `SKILLS_EVAL` | off / all / same | agent skills (see **Agent skills** below): `listed` installs the family's `skills/` in the sandbox and lists them in the system prompt / which skill names / `off` = evaluate without them |

## Agent skills (`SKILLS=listed`, default off)

Short how-to guides in `skills/<name>/SKILL.md` (+ scripts and templates), 5 for Terminal-Bench and 4 for
KernelBench. Design and evidence: [`docs/skills-in-agentic-rl.md`](../../../docs/skills-in-agentic-rl.md).
- **Use case:** give a weak policy family-level know-how (no nvcc → Triton; copy `Model.__init__`
  verbatim; start servers detached; verify each requirement before stopping) so more GRPO groups
  contain a success, i.e. a gradient.
- **How:** `agentcore_slime_demo/skills.py` writes the family's skills to `/opt/acs_skills` (outside the
  workdir) in one exec at episode start and appends a one-line-per-skill listing to that episode's
  system prompt. The agent reads a skill with its bash tool, so the text arrives as a loss-masked tool
  result. The folder is removed before grading. KernelBench's `kb-self-check` gets the prompt's
  reference `Model` as `reference.py`, so `python3 /opt/acs_skills/kb-self-check/check.py solution.py`
  runs the grader's checks and timing.
- **Metrics:** `[demo]` adds `skills_listed`, `skill_reads`, `skills_read=a+b` and `skills_version` (a content
  hash, also in each sample's metadata). Only the in-session `mini_claude` harness gets skills; the
  decoupled CLI kinds don't yet.

## Metrics

Every `[demo]` line carries `substrate=microvm|docker`. `metrics.py` prints a **by substrate** block
(solve rate, open/agent/grade/episode p50, infra errors), and m5 opens appear as `[sbx-op] op=open
arn=docker:<task>` (compose up + server ready), so `perf/sandbox_open_*` covers both substrates. Eval
reports `eval/terminalbench` (microVM held-out set, the one the microVM-only demo used) and
`eval/terminalbench_docker` (m5 held-out set).

