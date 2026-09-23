# Two-Runtime deployment (A3 / B3 mini-harness target)

Deploys the mini harness the **target** way: the harness gets its **own** AgentCore Runtime #1,
and the sandbox is a **separate** AgentCore Runtime #2 (CPU for A3, GPU for B3). One driver,
`deploy_two_runtime.py`, builds the three images, creates the roles/runtimes, opens the one SG
rule, and prints the export block for the demos.

## Hosting model (verified against the `bedrock-agentcore-control` API, not assumed)

| Component | Hosting | Network | Reached how | Why |
|---|---|---|---|---|
| **Trainer + inference + proxy** | this EC2 box (`PRIVATE-IP-f99864bc`, default VPC) | private | — | runs slime + SGLang + `AnthropicAdapter` on pinned `ADAPTER_PORT=18021` |
| **Harness #1** | managed **serverless** microVM | **`networkMode: VPC`** (default SG + subnets) | box → IAM `invoke` | it's the **only** thing that dials **back** to the private adapter → needs VPC egress + one SG rule, **no public port** |
| **CPU sandbox (A3)** | managed **serverless** microVM | `networkMode: PUBLIC` | IAM `invoke` | pulled via control plane; never dials back → **no VPC** |
| **GPU sandbox (B3)** | **capacity provider** `g6.xlarge` + EBS volume | CP's VPC (incidental) | IAM `invoke` | only way to get a GPU; torch (>2 GB) installed to the volume at `mode=prepare` |

The "no public port / SG-gated / VPC" requirement is driven by **one** thing: the harness dialing
back to the private adapter. Sandboxes are *pulled* over the IAM control plane, so they need no
inbound port and (for CPU) no VPC. The GPU sandbox is on an EC2 capacity-provider instance for the
**GPU**, independently of networking.

### Private VPC endpoints (why a VPC-mode harness needs them)

A `networkMode=VPC` runtime has **no internet route**, so — with nothing public — it can only reach
AWS through **private VPC endpoints**. `deploy_two_runtime.ensure_vpc_endpoints()` creates them
(all-private, mandate-aligned) and this is *required* for the harness to even start:

- **Interface endpoints** (`ecr.api`, `ecr.dkr` → image pull; `logs`; `sts`; `bedrock-agentcore` →
  the runtime's phone-home) in the default SG with `PrivateDnsEnabled`.
- **S3 gateway endpoint** on the VPC route tables (ECR image *layers* live in S3).
- One **`tcp/443` ingress from the VPC CIDR** on the endpoint SG — so both the harness ENI *and*
  this box (which sits in a different SG) can reach the endpoints. `PrivateDnsEnabled` overrides ECR
  DNS **VPC-wide**, so without this rule the box's own ECR access breaks.

Two gotchas the driver handles automatically: interface endpoints allow **one subnet per AZ** (a
default VPC can have two subnets in one AZ → `DuplicateSubnetsInSameZone`), and **`bedrock-agentcore`
is not offered in every AZ** (e.g. not `us-west-2d`). The harness is therefore pinned to the subnets
whose AZ is served by **every** endpoint — land it elsewhere and create fails with a **null reason
and an empty log group** (container never started).

### Data flow

```
 box `generate` driver ──IAM invoke──▶ Harness #1        (outbound; no inbound to the box)
 Harness #1 ──HTTP dial-back, private VPC, SG-gated :18021──▶ Adapter proxy on box   ◀── the ONLY inbound
 Harness #1 ──IAM invoke──▶ Sandbox #2 (CPU|GPU)          (tool exec)
 box `generate` driver ──IAM invoke──▶ Sandbox #2         (reward checker / scoring)
 Adapter ─▶ SGLang ─▶ policy under training (GRPO; TrajectoryManager keyed on Bearer session_id)
```

## Images (build context = `code/`)

- `Dockerfile.harness` — slim CPU Python + `bedrock-agentcore`/`boto3`/`aiohttp`; copies the package
  with a **blank** `__init__.py` (so `python -m agentcore_slime_demo.harness_runtime_agent` doesn't
  drag in `policy.py` → transformers). Well under the 2 GB image cap.
- `Dockerfile.sandbox.cpu` — slim + `bedrock-agentcore`; copies `sandbox_server_agent.py` flat.
  `SANDBOX_PYLIB` unset → the GPU/torch path stays dormant.
- `Dockerfile.sandbox.gpu` — slim + `bedrock-agentcore`; `ENV LD_LIBRARY_PATH=/usr/lib64` (set at
  process start so the loader finds the injected driver), `ENV SANDBOX_PYLIB=/mnt/models/pylib`.
  torch is **not** baked in — installed to the volume at `mode=prepare`.

## Deploy

```bash
cd code/deploy
export AWS_REGION=us-west-2
export ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc   # this box PRIVATE IP (harness dials it back)
python3 deploy_two_runtime.py all
```

Prints `AGENTCORE_HARNESS_RUNTIME_ARN`, `AGENTCORE_SANDBOX_CPU_RUNTIME_ARN`,
`AGENTCORE_SANDBOX_GPU_RUNTIME_ARN`, and a ready-to-paste export block. Idempotent: re-runs reuse
existing ECR repos / roles / runtimes / capacity provider (delete a runtime to pick up a new image).

Knobs: `SKIP_BUILD=1` (reuse pushed `:latest`), `SKIP_PREPARE=1` (skip the GPU cold-launch/torch
install — do it before the first B3 run), `GPU_INSTANCE` (`g6.xlarge`), `MODELS_GIB` (40),
`AGENTCORE_TRAINER_SG` (override the auto-detected trainer SG), `NAME_PREFIX` (`acs`).

**Cost:** the first B3 invoke (or `mode=prepare`) cold-launches a real `g6.xlarge` (~$0.80–1/hr) +
EBS. `python3 deploy_two_runtime.py cleanup` deletes the runtimes and the capacity provider.

## Then run the demos

This orchestrator provisions the **sandbox** Runtimes (the harness itself is in-process, not a
Runtime). The GPU sandbox ARN feeds the GPU demos — see `code/demos/demo_gpu_single/README.md`,
`code/demos/demo_gpu_shared/README.md`, and `code/demos/demo_gpu_snapshot/README.md`; the CPU
sandbox ARN feeds `code/demos/demo_cpu_datasets/README.md` (`SANDBOX=agentcore`).
