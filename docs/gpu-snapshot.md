# Can we snapshot a GPU Runtime instance? (No — and what to do instead)

> **Historical (2026-09-30).** This doc was written for demos and code dirs that were removed on 2026-09-30 (and the microVM-only `demo_cpu_terminalbench` on 2026-10-06); the only live demo is [`code/demos/demo_terminalbench_kernelbench`](../code/demos/demo_terminalbench_kernelbench/README.md).

**Question.** An AgentCore GPU Runtime session runs on an EC2 instance under the hood, and EC2
supports snapshots/AMIs — so can we snapshot the GPU Runtime instance to pause/resume state (e.g.
across a weight sync)?

**Short answer.** No usable path. The literal "snapshot the EC2" idea hits three walls; only an
**application-level** snapshot inside your own container is feasible, and even that loses to a plain
heartbeat for the one case we actually care about (weight-sync quiesce). Use the heartbeat +
re-seed model instead — see [`demos-program.md`](demos-program.md) and
the `HEARTBEAT` knob of every demo (on by default for remote own Runtimes since 2026-09-26) (formerly `demo_cpu_heartbeat`; offline probe
[`code/demos/demo_cpu_tasks/heartbeat_demo.py`](../code/demos/demo_cpu_tasks/heartbeat_demo.py)).

## Wall 1 — you don't own the instance
An AgentCore Runtime on a capacity provider runs your container image on **AWS-managed** compute.
AgentCore never exposes an `i-…` instance ID to you and owns session→instance placement (1:1). The
EC2 snapshot APIs (`ec2:CreateSnapshot`, `CreateImage`) need an instance/volume **you** can see and
control; you have neither. Closed at the ownership/IAM boundary.

*(Caveat to verify per setup: if a capacity provider were ever EC2 in your **own** account, this
softens — but you still could not bind a snapshot to a specific `runtimeSessionId`, because
AgentCore, not you, controls which session lands on which instance.)*

## Wall 2 — a disk snapshot is not a paused GPU
Even granted the snapshot, an EBS snapshot/AMI captures the **block device** only. A running GPU
job's live state is in **GPU VRAM + host RAM + the CUDA context**, none of which is on disk. Restore
gives you a filesystem, not a resumable process — you still cold-start CUDA and re-run. That
filesystem is exactly what **session storage / re-seed** already gives you, more cheaply. "EC2
snapshot" ≠ pause/resume.

## Wall 3 — the only real snapshot is application-level, in your own container (workable, rarely worth it)
You control the **Dockerfile and entrypoint** (`sandbox_server_agent.py`), so the honest version of
"a customized runtime with a snapshot function" is a snapshot **you implement above the OS**,
triggered by an op you define, persisted where you own it (S3 / session storage):

- **(a) State serialization — recommended.** On a `{"op":"quiesce"}` invoke, serialize the meaningful
  state (workspace files, partial results, seeds) to S3 / the session's persistent path; restore on
  the next invoke after a recycle. Cheap, robust, no exotic deps. This *is* the re-seed recovery
  model made explicit (`recovery.py`). **Built + demonstrated:**
  [`code/demos/demo_gpu_single/`](../code/demos/demo_gpu_single/) with `SNAPSHOT_EVERY=1` (formerly the
  separate `demo_gpu_snapshot`, merged into `demo_gpu_farm` 2026-09-28, back in `demo_gpu_single` since
  2026-09-30; snapshots in `demo_gpu_single/snapshots`; reusable mechanism: `agentcore_slime_demo/snapshot_store.py` +
  `app_snapshot.py`, offline animation `snapshot_demo.py`).
- **(b) Full process+GPU checkpoint — possible, heavy, fragile.** Bake NVIDIA `cuda-checkpoint`
  (evicts the CUDA context to host memory) + `CRIU` (dumps the process tree) into the image; on
  quiesce, checkpoint and push to S3; on restore, pull + `criu restore` + re-toggle CUDA. Real
  technique, but brittle across driver/instance mismatches, adds minutes, and a CRIU image only
  restores on a compatible instance — which AgentCore does not guarantee you'll get back.

## Where to put the app-level snapshot: S3 vs FSx (and EFS)
For tier (a), the state store is a real choice. The decider is a hard AgentCore constraint, not the
storage feature set: **S3 needs no mount; FSx/EFS need one AgentCore won't hand you.**

- **S3 (recommended).** `boto3.put_object`/`get_object` over HTTPS — **no mount**, works from any
  instance in any AZ, VPC or not. Fits the RL case exactly (small workspace: PROBLEM.md, solution
  files, partial results) and the from-scratch re-seed model. Dead simple.
- **FSx for Lustre.** POSIX + high throughput, so state is simply *present* on a fresh instance (no
  serialize/restore). But it's the wrong tool here for three reasons:
  1. **Mount.** AgentCore Runtime is a managed, container-image runtime with no declarative volume
     mount. Your only route is to mount at container startup from the entrypoint, and Lustre needs
     the **lustre client kernel module** — which you generally cannot load in a managed container.
  2. **AZ.** FSx for Lustre *scratch* is **single-AZ**; AgentCore picks your instance's AZ, so a
     recycle in another AZ may not reach the filesystem. (Persistent/multi-AZ deployments exist but
     cost more and still don't solve the mount.)
  3. **Fit.** FSx earns its keep only for **large, throughput-bound** state (multi-GB checkpoints /
     datasets). Our snapshot is tiny; Lustre throughput buys nothing, and FSx-linked-to-S3 or plain
     S3 is simpler.
- **EFS** (the NFS sibling, if you meant that): more plausible than FSx — regional/multi-AZ, no
  kernel module, just an NFS mount — and it removes the serialize step. But it **still requires
  `networkMode=VPC` + an NFS mount at startup + the SG/endpoint/AZ plumbing** (see the VPC-mode
  notes), which is exactly the operational surface S3 lets you skip.

Net: mounting FSx/EFS re-introduces VPC-mode complexity to save a serialize step on state that is
trivial to serialize. Use **S3** unless the snapshot is huge and throughput-bound *and* you're
already committed to VPC mode — then reconsider EFS (not FSx/Lustre).

## Why none of this matters for weight-sync quiesce
The pause we care about — trainer→inference weight sync — is **seconds**, never near max-lifetime
(8 h CPU / 24 h GPU). Snapshot-and-restore (even tier a) costs more wall-clock than simply keeping
the session warm and gating invokes. A snapshot would only win for a pause **longer than
max-lifetime**, which weight sync never is. So the architecture is:

- **Heartbeat** to survive short pauses (idle timer reset) — `SessionKeeper` / `KeptSandbox`.
- **Re-seed** (tier-a serialization) to recover from a genuine loss — `recovery.py`.
- **Snapshot never needed.**

Demonstrated end-to-end by the `HEARTBEAT` knob of every demo (on by default for remote own Runtimes since 2026-09-26) (formerly `demo_cpu_heartbeat`; offline probe
[`code/demos/demo_cpu_tasks/heartbeat_demo.py`](../code/demos/demo_cpu_tasks/heartbeat_demo.py)).
