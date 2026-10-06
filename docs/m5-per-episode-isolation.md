# m5 tier: one shared instance vs. one instance per episode

Context: `demo_terminalbench_kernelbench` runs the Terminal-Bench tasks AgentCore can't (multi-container
compose, x86-only, > 2 GB images, tasks needing root powers) on m5 EC2 instances. This note answers two
questions: do episodes share an m5, and which instances does AgentCore manage?

## Today's shared m5: one project per session, not one instance per session

When several episodes in a batch need the m5 (for example a group of 8 samples of one compose task),
up to 16 run there at once (`TB_DOCKER_MAX_PROJECTS`). Each gets its own compose project:
- fresh containers;
- its own networks and volumes;
- its own copy of the task folder.

All of it is deleted afterwards (`docker compose down -v`). But the episodes share one instance and
one kernel. That's "one project per session", not "one instance per session".

## g6 is managed by AgentCore; the m5 isn't

- **The g6 tier is an AgentCore capacity provider.** AgentCore launches the instance when a session
  opens and releases it afterwards: natively one session per instance. Our code only calls
  `InvokeAgentRuntime` / `StopRuntimeSession`.
- **The m5 is plain EC2** that our code launches and runs (`deploy/m5_host.py` + the compose agent).

**Why not use AgentCore-managed m5 Instances too?** They still don't allow Docker: every capability is
dropped, as on microVMs. That rules out:
- the 12 compose tasks;
- the ~55 tasks whose reference solution fails on a microVM for lack of root powers.

So for those tasks, one instance per session means our own EC2 per episode.

## The "EC2 per episode" mode (`TB_M5_MODE=per_episode`, the default)

- **Docker-ready AMI** (`deploy/m5_host.py bake-ami`). Baked once on a temporary `m5.large` running the
  same bootstrap: Docker, compose, the agent and the firewall rules, but no task images.
- **Per episode** (`agentcore_slime_demo/ec2_episode_sandbox.py`):
  1. Launch one `m5.xlarge` with 40 GB gp3 from that AMI, with a fresh agent secret in its user-data.
  2. Upload the task folder, without its tests or reference solution.
  3. Pull only that task's images from ECR (`acs-tbhdk-<task>-<service>`), then `compose up` as
     today.
  4. Run the episode, grade it, and terminate the instance.

  Nothing is ever reused.
- **Images come only from ECR.** The shared m5 is now just the builder: it pushes every image of a task
  (built ones and sidecars) to ECR. Pulling `postgres`, `redis`, `vulhub/*` and so on from Docker Hub
  per episode would hit its anonymous pull limit (100 per 6 h per IP) behind the one NAT IP.
- **Cold start:**
  - About 30–90 s launch plus 10 s–2 min pull (the mteb images are about 10 GB).
  - Hidden by the same look-ahead pre-open as the microVMs.
  - A gate (`TB_M5_MAX_INSTANCES`, default 16) caps the number of live instances.
- **Cost:** about $0.19/h per `m5.xlarge`, billed per second, so roughly $0.02–0.05 per episode.
- **Hardening:**
  - Instance metadata (IMDS) is blocked from containers.
  - Containers can't reach the host itself.
  - There's no privileged mode and no Docker socket.
  - New: a VM boundary per episode.

  One correction to "no IAM role": each episode instance has a pull-only ECR role, scoped to the
  `acs-tbhdk-*` repos. Containers can't reach the metadata endpoint, so agent code never sees those
  credentials. The builder has a push role on the same repos.
- **Oracle validation carries over.** Per-episode instances run the same images and the same compose
  files, so whether a task's oracle passes doesn't change. `validate_oracle.py` therefore validates on
  the shared builder (`TB_M5_MODE=shared` is its default).

| | shared (`TB_M5_MODE=shared`) | per episode (`TB_M5_MODE=per_episode`) |
|---|---|---|
| Unit per episode | a compose project | a whole EC2 instance |
| Shared between episodes | kernel, host, image layers | nothing (images pulled fresh from ECR) |
| Cold start | ~6–10 s (`compose up`) | ~1–3 min (launch + pull + `compose up`) |
| Capacity limit | one host (16 projects) | EC2 quota (`TB_M5_MAX_INSTANCES`) |
| Who manages it | our code | our code (AgentCore can't run Docker) |

## Who launches and terminates the per-episode m5 instances?

Our own code inside the trainer; AgentCore isn't involved. The m5 tier is driven by the same object that
drives the microVMs: `TerminalBenchPool`, inside slime's rollout process on the g6e. It calls the EC2 API
with boto3, using the g6e's IAM role. For one m5 episode in per-episode mode:

```
slime rollout (g6e)
 └─ look-ahead data source: "batch k+1 has 8 samples of postgres-csv-clean"
 └─ TerminalBenchPool pre-opens 8 sandboxes → Ec2EpisodeSandbox.__aenter__ for each:
      1. ec2.run_instances(baked AMI, m5.xlarge, 40 GB, fresh agent secret)     ~30-60 s
      2. wait for the compose agent on <new ip>:18300
      3. upload the task folder (no tests/solution) + its ECR image list
      4. agent: docker pull (from ECR) + docker compose up → client + sidecars   ~10 s-2 min
 └─ episode: the agent loop runs INSIDE the client container, calls back to the trainer's adapter
    (policy tokens recorded by slime), runs bash/write/read tools
 └─ grade: inject hidden tests, run run-tests.sh → reward 1/0 → slime Sample
 └─ Ec2EpisodeSandbox.__aexit__: ec2.terminate_instances(...)                     the VM is gone
```

Steps 1–4 run during the look-ahead window, so the episode normally finds its instance ready. If the
trainer crashes, leftover instances (tag `acs-tbh-m5-ep`) are terminated in three places:
- at the next pool start;
- by `6_cleanup.sh`;
- by `python3 deploy/m5_host.py gc-episodes`.

## One session = one machine, and the machine runs the whole compose project

The guarantee is **one episode (session) = one m5 instance**, the same as AgentCore gives the g6 tier.
It is NOT one instance per container. The episode's instance runs the task's entire compose project:
the `client` container where the agent works, plus every sidecar (database, API, redis, …). They are
connected by the project's own Docker network, exactly as upstream Terminal-Bench runs them. When the
episode ends, the whole instance is terminated, sidecars included.

| | one compose task, e.g. `postgres-csv-clean` (client + `db`) |
|---|---|
| per episode | 1 × m5.xlarge running both containers |
| a group of 8 samples | 8 × m5.xlarge, one per sample, each with its own client + `db` |

## Do we need a VM per episode? Compose already isolates

| | shared m5 (compose project per episode) | m5 per episode |
|---|---|---|
| Fresh environment per episode | ✅ new containers, networks, volumes, task copy; deleted afterwards | ✅ |
| One episode seeing/changing another's files or processes | not possible (verified) | not possible |
| Shared between concurrent episodes | the host kernel and the host's CPU/RAM/disk | nothing |
| Breakout risk | a kernel exploit from one container could reach the others running at that moment | contained to that VM |
| Cold start | ~6–10 s | ~1–3 min (hidden by pre-open) |
| Cost and moving parts | one instance, simple | N instances, AMI + ECR + launch/terminate code |

Shared compose is what Terminal-Bench itself uses, and it's enough for a correct reward. Per-episode VMs
add defence against the policy breaking out of its container. That matters if model-generated code is
treated as hostile, or if you want the same "one session = one machine" guarantee AgentCore gives the g6
tier. **Decision (2026-10-04): per-episode is the default (`TB_M5_MODE=per_episode`).**
