# Can we snapshot a GPU Runtime instance? (No — and what to do instead)

**Question.** An AgentCore GPU Runtime session runs on an EC2 instance under the hood, and EC2
supports snapshots/AMIs — so can we snapshot the GPU Runtime instance to pause/resume state (e.g.
across a weight sync)?

**Short answer.** No usable path. The literal "snapshot the EC2" idea hits three walls; only an
**application-level** snapshot inside your own container is feasible, and even that loses to a plain
heartbeat for the one case we actually care about (weight-sync quiesce). Use the heartbeat +
re-seed model instead — see [`demos-program.md`](demos-program.md) and
[`code/demos/demo_cpu_quiesce/`](../code/demos/demo_cpu_quiesce/).

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
  model made explicit (`recovery.py`).
- **(b) Full process+GPU checkpoint — possible, heavy, fragile.** Bake NVIDIA `cuda-checkpoint`
  (evicts the CUDA context to host memory) + `CRIU` (dumps the process tree) into the image; on
  quiesce, checkpoint and push to S3; on restore, pull + `criu restore` + re-toggle CUDA. Real
  technique, but brittle across driver/instance mismatches, adds minutes, and a CRIU image only
  restores on a compatible instance — which AgentCore does not guarantee you'll get back.

## Why none of this matters for weight-sync quiesce
The pause we care about — trainer→inference weight sync — is **seconds**, never near max-lifetime
(8 h CPU / 24 h GPU). Snapshot-and-restore (even tier a) costs more wall-clock than simply keeping
the session warm and gating invokes. A snapshot would only win for a pause **longer than
max-lifetime**, which weight sync never is. So the architecture is:

- **Heartbeat** to survive short pauses (idle timer reset) — `SessionKeeper` / `KeptSandbox`.
- **Re-seed** (tier-a serialization) to recover from a genuine loss — `recovery.py`.
- **Snapshot never needed.**

Demonstrated end-to-end in [`code/demos/demo_cpu_quiesce/`](../code/demos/demo_cpu_quiesce/).
