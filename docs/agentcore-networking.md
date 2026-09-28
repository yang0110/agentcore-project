# Handling AgentCore networking safely

How to connect a slime trainer to an AgentCore sandbox **without opening the box up**. The guiding
principle is *direction of traffic*: prefer the "pull" shape (you call AWS), and only accept inbound
("dial-back") traffic when a task genuinely needs it — then lock it down with a VPC, PrivateLink, and
security-group **references** (never CIDRs).

Companion tool: `code/demos/demo_cpu_tasks/netcheck.py` is an executable version of this checklist
(its large-image companion demo, `demo_cpu_instance_pool`, was removed 2026-09-28). Related deep dives: [fault-recovery](fault-recovery.md), [sandbox-comparison](sandbox-comparison.md),
and the VPC-mode war story lives in the `agentcore-vpc-mode-harness` project memory.

---

## The two connection shapes

| Shape | Who initiates | Ports **you** expose | Auth | When you need it |
|---|---|---|---|---|
| **Pull** (`invoke_agent_runtime`) | trainer → AWS | **none** | SigV4 (IAM), TLS | always — the default |
| **Dial-back** | sandbox → your private adapter | the adapter port (inbound to you) | SG→SG + private link | only when the in-sandbox agent must reach a model/service you host |

### 1. Pull — the safe default (no inbound at all)
The sandbox Runtime is reached with `invoke_agent_runtime`, a **SigV4-signed, TLS** AWS API call.
You open **zero inbound ports**; the trainer initiates every round-trip. This holds even for GPU /
capacity-provider Instances — they are reached by IAM invoke (pull), so they need **no VPC**.

- Keep the Runtime in the default `PUBLIC` network mode and gate it entirely with **IAM** — nothing
  is exposed but a signed API.
- Scope the trainer's IAM role to exactly `bedrock-agentcore:InvokeAgentRuntime` (+ the
  Create/Update actions only on the deploy role, not the trainer role).

### 2. Dial-back — the only shape that adds risk
The one time you accept inbound traffic is when the in-sandbox agent must reach a service **you**
host — in this project, the Qwen policy-model adapter on the trainer box. Traffic now flows *into*
your box, so treat it as hostile-by-default and constrain it hard (§ below).

---

## The dial-back checklist (do all of these)

1. **`networkMode=VPC` on the Runtime.** A VPC-mode Runtime has **no internet route**; it can reach
   AWS only through **private VPC (PrivateLink) endpoints**. This removes public exposure and gives
   you controllable egress.
2. **Security-group → security-group rules, never CIDR.** Allow the adapter port *from the Runtime's
   security group as the source*, not from an IP range and **never** `0.0.0.0/0`. Only that Runtime
   can then reach the adapter (least privilege).
3. **Private IP for the adapter.** Point the sandbox at the adapter's *private* address
   (`ADAPTER_PUBLIC_HOST` = private IP); trainer box and Runtime share the same VPC.
4. **PrivateLink endpoints for everything the Runtime touches.** Interface endpoints for
   `ecr.api`, `ecr.dkr`, `logs`, `sts`, **`bedrock-agentcore`** (phone-home) **plus an S3 *gateway***
   endpoint (ECR layers live in S3). Set `PrivateDnsEnabled=True`.
5. **`tcp/443` ingress from the VPC CIDR on the *endpoint* SG.** `PrivateDnsEnabled` overrides ECR
   DNS VPC-wide; without this rule even your own box's ECR calls silently hang.
6. **One subnet per AZ, and pin to AZs served by *every* endpoint.** Interface endpoints reject a
   second subnet in the same AZ (`DuplicateSubnetsInSameZone`), and `bedrock-agentcore` PrivateLink
   is **not offered in every AZ** — pick AZs covered by all of your endpoints.

### The failure signature to recognize
A VPC-mode Runtime that can't reach an endpoint fails with `status=CREATE_FAILED`,
`statusReason=null`, and a **0-byte log group** (only `otel-rt-logs`/`spans` streams) — the container
never started. Debug **in this order**: endpoints `available` → S3 gateway on the Runtime subnet's
route table → exec role has ENI perms + ECR pull → the Runtime's AZ is covered by all endpoints.

---

## Controlling what the sandbox can reach outbound

The code an RL policy executes inside the sandbox is arbitrary — treat its **egress** as a control
surface, not an afterthought:

- **Egress broker / allow-list** so executed code can't reach the open internet unless whitelisted.
- **Cedar policies + IAM** to gate what the sandbox identity is allowed to call.
- In VPC mode, outbound is already confined to your PrivateLink endpoints + whatever you route.

See [sandbox-comparison](sandbox-comparison.md) (`[A8]`) and the `agentcore-samples`
egress-broker / Cedar examples for concrete patterns.

---

## Self-hosting the Runtime container (your own EC2, e.g. `m5.xlarge`)

If you run the AgentCore Runtime image yourself (local-endpoint mode, `AGENTCORE_LOCAL_ENDPOINT`),
the `/invocations` server is plain HTTP — **do not expose it publicly**:

- Keep the instance in the **same VPC/subnet** as the trainer; SG→SG on the invocations port only.
  For dev, an **SSH tunnel / SSM port-forward** avoids any inbound rule at all.
- Add **TLS + a bearer token (or mTLS)** if the hop ever crosses a trust boundary — the local
  transport is plain `urllib` POST, acceptable only over a trusted private link.
- Restrict the instance's **egress** SG rules to just what the image needs.

---

## One-line summary

Default to **IAM-pull + PUBLIC mode + no inbound**. Introduce **VPC + SG→SG + PrivateLink** *only*
for the adapter dial-back, and layer **Cedar / egress-broker** to bound what executed code can reach.
Run `demo_cpu_tasks/netcheck.py` against your config to verify the checklist before you deploy.
