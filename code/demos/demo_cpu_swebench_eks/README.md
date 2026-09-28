# demo_cpu_swebench_eks — SWE-bench Verified RL on EKS: one fresh gVisor pod per episode

**Sandbox substrate: EKS + gVisor, not AgentCore.** This is the EKS twin of
[`../demo_cpu_swebench`](../demo_cpu_swebench/README.md). Dataset, images, harness, grader and RL
recipe are the same. Only the thing that hosts an episode changes. On AgentCore, an episode gets a
fresh `runtimeSessionId` on its instance's Runtime, which is its own m5 Instance. Here it gets a fresh
**Kubernetes pod** running under the **gVisor** runtime (`runtimeClassName: gvisor`, handler `runsc`)
on a shared m5 node. The two demos exist side by side so the substrates can be compared on the same
workload.

Created 2026-09-28 as a copy of `demo_cpu_swebench`. The vendored library is unchanged apart from:

- two new files: `eks_api.py` (a Kubernetes API client without kubectl) and `eks_pod_sandbox.py`
  (`EksPodSandbox`, a drop-in for `AgentCoreRuntimeSandbox`);
- `terminalbench_pool.py`, which now opens an `EksPodSandbox` per episode.

The in-image server (`deploy/acs_swe_http_server.py`, `POST /invocations` on :8080), the in-session
harness (`InSessionHarness`, loop dials the trainer's `AnthropicAdapter` back), the hidden-test grader
(`swebench.grade_swebench`) and the harness tier (`HarnessPool`, W = RB×NS workers,
least-loaded engine) are all identical. **REMOTE-ONLY** (`SANDBOX` must be `eks-gvisor`).

**Status (2026-09-28): code complete, not yet run on a cluster.**

- The EKS cluster does **not** exist yet. Only `render` (read-only) has run against the account.
- Unit-checked offline: `EksPodSandbox` was tested against a fake API plus the real server on :8080;
  the manifests, token format and pool routing were also checked.

---

## Use case covered

RL on real GitHub issues (django, sympy, sphinx, pytest, astropy, requests, xarray, pylint, flask).
The agent edits `/testbed` and is rewarded when the instance's hidden FAIL_TO_PASS + PASS_TO_PASS
tests pass. The episodes run on a **self-managed Kubernetes sandbox fleet** instead of AgentCore
Runtime. The subset is the same 20 instances by default (16 train + 4 eval).

## Why this demo

- **A like-for-like substrate comparison.** It answers "what would this look like on EKS instead of
  AgentCore?" with numbers, using the same images, tasks, policy and metrics:
  - open latency;
  - cost per episode;
  - isolation;
  - operational burden.
- **It removes AgentCore's limits for this dataset.** No 2 GB microVM image cap and no arm64
  requirement: the SWE-bench images are amd64 and run as-is. Also no capacity-provider quota and no
  one-machine-per-session rule. Several episodes share an m5 node, which is much denser than one
  m5 per episode. So all 500 Verified rows are possible, including the big matplotlib and sklearn
  images (see `SWE_QUOTA` / `SWE_MAX_IMAGE_MB`).
- **gVisor keeps it an honest sandbox.** Plain containers would share the node kernel with untrusted
  agent code. gVisor puts a user-space kernel between them.

## Design considerations

- **Isolation: gVisor per pod vs a machine per session.**
  - `runsc` intercepts every syscall in its own user-space kernel (platform `systrap`, so no KVM or
    nested virtualisation is needed on m5).
  - One pod per episode, never reused, so no state leaks between episodes.
  - It is weaker than AgentCore's one Instance per session: pods share a node's CPU, memory bandwidth,
    disk and page cache.
  - It is stronger than runc. `EKS_RUNTIME_CLASS=` (empty) runs plain runc, for an A/B only.
  - Pods get no Kubernetes credentials: `automountServiceAccountToken: false`, no service links.
- **Shared images.** `deploy/deploy_swebench.py` keeps the AgentCore demo's wrap recipe unchanged.
  It pushes to the same `acs-swe-*` ECR repos with the same recipe tag, so an image built by either
  demo serves both.
  - It registers **no** Runtimes. It writes `swebench/image_by_task.json` ({instance: uri@digest}),
    which the pool routes by. Pinning by digest means a re-push cannot change a running pool.
- **gVisor install.** EKS managed nodes do not ship gVisor. `deploy/eks/gvisor-installer.yaml` is a
  privileged DaemonSet that does the install on each node. It:
  1. downloads `runsc` + `containerd-shim-runsc-v1` and checks their sha512;
  2. adds the `runsc` runtime to containerd's config (containerd 2.x and 1.7 layouts);
  3. restarts containerd;
  4. labels the node `acs/gvisor=ready`.

  The `gvisor` RuntimeClass (`runtimeclass.yaml`) selects on that label, so a sandbox pod never lands
  on a node where gVisor is missing. Pod overhead is declared as 100m CPU / 200Mi.
- **Image cache per node + pre-pull.**
  - An image is pulled once per node, then cached (`IfNotPresent`). The first episode of an instance
    on a fresh node pays the pull (1-2 GB compressed).
  - `EKS_PREPULL=1` (the default) applies the DaemonSet `acs-swe-prepull`. Its init containers are
    the pool images, so every node pulls all of them once, at deploy time.
  - Disk budget: 3-6 GB per image per node. The default 300 GiB gp3 fits the 20-image subset.
    Kubelet evicts images above 85 % disk use, so size `EKS_NODE_DISK_GIB` to the pool.
- **No heartbeat.** Pods have no idle reclaim, so the pool does not register sessions with
  `SessionKeeper`.
  - A leaked pod (trainer crash) is capped by `activeDeadlineSeconds` (`EKS_POD_MAX_LIFETIME` 7200 s).
  - It can also be cleaned up by deleting the label `app=acs-swe-sandbox`.
- **Network.**
  - The pods sit in the `${NAME_PREFIX}-egress-*` NAT subnets of the trainer's VPC (VPC CNI: pods get
    VPC IPs). The trainer talks HTTP straight to `podIP:8080`, with no AWS data plane in the path.
  - The `acs-swe` namespace's NetworkPolicy (`enableNetworkPolicy` on the vpc-cni addon) sets:
    - **ingress**: :8080 only, from the trainer IP and the node subnets (for the kubelet's readiness
      probe);
    - **egress**: only the trainer's adapter ports (18100-18163, 18021). This is the AgentCore
      demo's "egress none": no PyPI and no internet from the agent.
  - Security groups: the cluster SG admits :8080 from the trainer SGs, and the trainer SGs admit the
    dial-back from the cluster SG.
- **No kubectl in the trainer.** The `agentcore_p4` container has boto3 but no aws CLI or kubectl.
  `eks_api.py` mints the EKS bearer token itself (a presigned STS GetCallerIdentity URL with
  `x-k8s-aws-id`) and calls the API server over stdlib HTTPS. The endpoint and CA come from
  `DescribeCluster`. Auth uses access entries: the creator role gets cluster admin, and the trainer
  runs as that same role.
- **Open timings.** The `open_sent_epoch` stamp is taken before the pod is created, so `open_ms` means
  the same thing as on AgentCore. The `[demo]` line adds:
  - `schedule_ms`: time until the pod is bound to a node;
  - `pod_ready_ms`: time until `/ping` is ready;
  - `node`.

  `warm_hit` is always 0: every episode is a brand-new pod.
- **Failure modes.** Fatal container states (`CreateContainerError`, `InvalidImageName`, …) raise
  `PodOpenError` at once. `ErrImagePull`/`ImagePullBackOff` fail after `EKS_PULL_FAIL_SEC` (180 s).
  A pod that is not ready within `EKS_OPEN_TIMEOUT` (1200 s) fails; that time includes waiting for
  the cluster autoscaler or for room on a node.
- **Sizing.**
  - Each pod requests 1.5 vCPU / 6 GiB (limit 8 GiB) / 10 GiB ephemeral storage.
  - An m5.2xlarge (8 vCPU / 32 GiB) holds 4 pods, so the default 4 nodes run about 16 concurrent
    episodes.
  - For `TB_PREOPEN_BATCHES=2 × RB × NS` = 64 pre-opened sessions, raise `EKS_NODES` (max 16), or
    lower the pre-open.
  - Pre-opened pods cost only their slice of a node, not a whole m5.

## How an episode flows

```
slime → terminalbench_pool_generate.generate → TerminalBenchPool.submit
  harness worker pulls episode → route by task.image (= instance id) → image_by_task.json → ECR uri@digest
  (at submit) EksPodSandbox: POST a BRAND-NEW pod (runtimeClassName gvisor) while the episode waits
    → scheduled → image (node cache / pre-pulled) → runsc starts the container → readiness /ping
      InSessionHarness writes session_runner.py into the pod (HTTP to podIP:8080) and polls it
        in-pod loop in /testbed: LLM via the least-loaded engine's adapter (dial-back, NetworkPolicy allows only this)
    grade: swebench.grade_swebench (reset test files → apply hidden test_patch → test_cmd → parse F2P/P2P)
  finish_session → ONE loss-masked Sample (the longest); adapter.drop_session; DELETE pod (grace 0)
```

## Step by step (numbered scripts; run from this dir on the HOST — container steps `docker exec` themselves)

```bash
bash 1_prepare_data.sh     # 1. container: fetch_swebench.py (select subset -> swebench/manifest.json) -> data/
bash 2_build_image.sh      # 2. deploy_swebench.py: wrap + push one amd64 image per instance (shared acs-swe-* repos)
                           #    -> swebench/image_by_task.json + images.csv
bash 3_push_ecr.sh         # 3. no-op: prints swebench/images.csv (step 2 already pushed)
bash 4_deploy_runtime.sh   # 4. eks_deploy.py up: RENDER ONLY unless EKS_CREATE=1 (see below);
                           #    then swebench_oracle_check.py (on gVisor pods) and fetch_swebench.py --data-only
bash 5_run.sh              # 5. container: 10 train steps + 1 eval through the pool; logs/<tag>.{log,csv,eval.csv,png}
EKS_DELETE=1 bash 6_cleanup.sh   # 6. eks_deploy.py down: revoke our SG rules, eksctl delete cluster
```

- **Creating the cluster is gated.** Step 4 without `EKS_CREATE=1` only renders
  `deploy/eks/cluster.yaml` + `deploy/eks/rendered/`, prints the plan, and stops. Nothing is created.
  Look at it first: `python3 deploy/eks_deploy.py render`.
- `EKS_CREATE=1 bash 4_deploy_runtime.sh` is billed: EKS control plane + nodes, about 15-20 minutes.
  It needs **`eksctl` on the host** and does the following:
  1. `eksctl create cluster`;
  2. opens the SG rules;
  3. applies the gVisor installer, RuntimeClass, namespace and NetworkPolicy;
  4. waits for the pre-pull;
  5. runs one smoke pod, which must show `gVisor` in `dmesg` and print the testbed Python version.

  The trainer container needs no kubectl.
- **Validate on 2 instances first**: `SWE_LIMIT=2 bash 2_build_image.sh && EKS_CREATE=1 bash 4_deploy_runtime.sh`.
  If the AgentCore demo already pushed the images, step 2 only resolves digests.
- `python3 deploy/eks_deploy.py status` lists nodes (with `acs/gvisor` label) and live sandbox pods.
  `python3 deploy/eks_deploy.py smoke` re-runs the smoke pod.
- Train to convergence (inside the container): `train.sh` (`STEPS`, `EVAL_EVERY`, `SANDBOX_CONC`).

## Knobs (env vars)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `eks-gvisor` | the only accepted value |
| `EKS_CLUSTER` / `EKS_NAMESPACE` | `acs-swe-eks` / `acs-swe` | cluster + sandbox namespace |
| `EKS_RUNTIME_CLASS` | `gvisor` | pod RuntimeClass; empty = runc (A/B only, no syscall sandbox) |
| `EKS_POD_CPU` / `EKS_POD_MEMORY` / `EKS_POD_MEMORY_LIMIT` | `1500m` / `6Gi` / `8Gi` | per-episode pod resources |
| `EKS_POD_EPHEMERAL` / `EKS_POD_MAX_LIFETIME` | `10Gi` / `7200` | scratch disk / hard cap on a leaked pod (s) |
| `EKS_OPEN_TIMEOUT` / `EKS_PULL_FAIL_SEC` / `EKS_POLL_SEC` | `1200` / `180` / `1.0` | open deadline / pull-error grace / status poll |
| `EKS_VERSION` / `EKS_NODE_TYPE` | `1.35` / `m5.2xlarge` | cluster (render time) |
| `EKS_NODES` / `EKS_NODES_MIN` / `EKS_NODES_MAX` / `EKS_NODE_DISK_GIB` | 4 / 1 / 16 / 300 | managed node group |
| `EKS_PUBLIC_CIDRS` | this box's public IP/32 | who may reach the public API endpoint |
| `EKS_PREPULL` / `GVISOR_RELEASE` | `1` / `latest` | pre-pull DaemonSet / gVisor release |
| `EKS_CREATE` / `EKS_DELETE` | unset | must be `1` to create / delete the cluster |
| `TB_PREOPEN_BATCHES` | `2` | upcoming batches kept open ahead (`× RB × NS` pods); size the node group to it |
| `HARNESS_WORKERS`, `CONC`, `EPISODE_MAX_TURNS`, `RB`/`NS`/`NR`, … | as in `demo_cpu_swebench` | RL + harness knobs are unchanged |

- Deploy / data: `SWE_TASKS`, `SWE_LIMIT`, `SWE_BUILD_CONC`, `SWE_REBUILD`, `SWE_NO_PUSH`, `SWE_QUOTA`,
  `SWE_INSTANCES`, `SWE_MAX_IMAGE_MB`, `SWE_TEST_TIMEOUT`, as in the AgentCore demo.
- The default subset matches the AgentCore demo on purpose, so results compare directly.

## What to watch

- `rollout/raw_reward` and `eval/swebench`: the learning signal and the held-out solve rate. Compare
  them against the AgentCore run; they should match within noise, since only the substrate differs.
- `[demo] fresh_session=1 … open_wait_ms=… schedule_ms=… pod_ready_ms=… node=…`:
  - large `schedule_ms` means the cluster is out of room (Pending pods, autoscaler);
  - large `pod_ready_ms − schedule_ms` means an image pull (not pre-pulled) or a slow gVisor start.
- `perf/sandbox_ms_*`: exec latency under gVisor. Syscall-heavy test suites (django's) run slower
  than on runc or AgentCore; `EKS_RUNTIME_CLASS=` gives the A/B.
- `swebench/oracle.csv`: the gold/empty check on gVisor pods. An instance that is usable on AgentCore
  but fails here points at a gVisor incompatibility in that test suite.

## Known risks (to check on the first real run)

- The containerd config path and version on the AL2023 EKS AMI. The installer handles containerd
  2.x (`version = 3`) and 1.7. Check its logs: `kube-system/acs-gvisor-installer`.
- Kubelet readiness probes under the NetworkPolicy. Ingress from the node subnets is allowed for them.
  If pods stay NotReady, check the policy first.
- SWE-bench test suites under gVisor. The oracle check in step 4 validates every instance.

## Folder layout (self-contained)

- `agentcore_slime_demo/`: vendored library (as in `demo_cpu_swebench`), plus:
  - `eks_api.py`: EKS token + Kubernetes API over stdlib HTTPS;
  - `eks_pod_sandbox.py`: `EksPodSandbox`, one gVisor pod per episode;
  - `terminalbench_pool.py`: patched to open `EksPodSandbox`, with no keeper.
- `deploy/eks_deploy.py`: render / up / smoke / status / down for the cluster.
- `deploy/eks/`:
  - `cluster.yaml.tmpl` (eksctl);
  - `gvisor-installer.yaml`;
  - `runtimeclass.yaml`;
  - `sandbox-namespace.yaml.tmpl` (namespace + NetworkPolicy);
  - `cluster.yaml` + `rendered/` (gitignored render output).
- `deploy/deploy_swebench.py`: wrap + push one image per instance, then write `image_by_task.json`.
- `deploy/acs_swe_http_server.py`: the in-image server (the same one AgentCore uses).
- `swebench_oracle_check.py`: gold/empty check per instance on its own pod → `swebench/oracle.csv`.
- `fetch_swebench.py` · `prepare_data.sh` (step 1) · `run.sh` (step 5) · `metrics.py` · `train.sh`.
- `1_…6_*.sh` + `deploy/demo.env`: the numbered steps (`SUBSTRATE=eks`, BUILD/PUSH/DEPLOY/CLEANUP_CMD).
- `swebench/` (gitignored): `manifest.json` (holds the hidden test patches), `image_by_task.json`,
  `images.csv`, `oracle.csv`, build logs.
- `data/`: `swebench_train.jsonl` / `swebench_eval.jsonl`.
