# Sandboxes for agent code-execution: AgentCore Runtime vs Docker vs raw microVM vs E2B

Scope: choosing an execution sandbox for **AI-agent / RL workloads** — running untrusted,
model-generated code and tool calls and getting back `stdout/stderr/exit_code`.

**Every claim in the tables below is cited.** Markers like `[A2]`, `[D1]`, `[F1]`, `[E3]` link to
the [References](#references) section, which lists the exact source URL (and, for AgentCore, the
`repo file:line`). AgentCore facts come from the cloned repos under `repos/` + AWS docs;
Docker/Firecracker/gVisor/E2B facts come from vendor primary sources.

---

## 1. What tasks AgentCore Runtime can do as a sandbox

All backed by real code in `repos/` (see [References](#references), `[A*]`).

| Task | How | Evidence |
|------|-----|----------|
| Run Python / JS / TS code | `CodeInterpreter.execute_code(code, language)` → `executeCode` | `code_interpreter_client.py:705-751` [A3] |
| Run shell commands | `invoke("executeCommand", {"command": ...})` | `run_commands.py` [A8] |
| Install packages (pip) | `install_packages(...)` → `executeCommand` | `code_interpreter_client.py` [A3] |
| Read / write files in sandbox | `writeFiles` / `readFiles` | `file_operations.py` [A8] |
| Data analysis out-of-the-box | numpy / pandas / matplotlib pre-installed | [A3] |
| Persist state across microVM restart | `/mnt` session storage, EFS, S3 | `commands.md:102` [A4]; EFS sample [A8] |
| Run a full agent harness (Claude Code / Codex) | custom Runtime container | coding-agents samples [A8] |
| Browser automation | Browser tool (`aws.browser.v1`, Playwright/CDP) | `browser_client.py:40-41` [A7] |
| Call AWS APIs from inside the sandbox | `aws s3 ...` via `executeCommand` | `run_commands.py` [A8] |
| Run containers *inside* the sandbox w/ egress control | containerd supervisor + broker | egress sample [A8] |
| Long-running jobs (8 h serverless / 14 days on instances) | serverless microVM / capacity-provider | `schema.py:193-232` [A6]; [A2] |
| GPU / memory / compute-optimized workloads | runtime instances via capacity provider | `commands.md:791-841` [A4]; [A2] |
| Deterministic reward for RL | parse `structuredContent.exitCode/stdout/stderr` | `run_commands.py` [A8] |
| Hard isolation between concurrent runs | one microVM session per `runtimeSessionId` | `models.py:17` [A1] |
| Policy-gated actions (allow/deny) | Cedar authorization in the entrypoint | durable sample [A8] |

**Not a fit for:** GUI desktop apps needing a display (beyond the headless Browser tool [A7]),
workloads needing host root, sub-10 ms inner loops, or fully offline / on-prem with no AWS [A1].

---

## 2. The four options in one line each

- **AgentCore Runtime** — a *managed* microVM sandbox **service** on AWS: call an API, AWS runs the
  isolated microVM (or your EC2 instance). Serverless microVMs on ARM64/Graviton; "runtime
  instances" run on your chosen EC2 incl. GPU. [A1][A2]
- **Docker** — OS-level **containers** sharing the host kernel: fast, local, ubiquitous, full
  control — but a **weaker security boundary** for hostile code. [D1]
- **Raw microVM (Firecracker / gVisor, self-run)** — the same VM tech AWS/E2B use underneath.
  Strongest DIY isolation; you own orchestration, networking, images, scaling. [F1][F3]
- **E2B** — an **open-source, purpose-built** code-interpreter sandbox for AI agents, on Firecracker
  microVMs. Managed cloud or self-hostable. slime's reference backend (`E2BSandbox`). [E1][E2][E6]

AgentCore serverless and E2B are *both* managed microVM sandboxes — the closest comparison. "Docker"
and "raw microVM" are the self-operated ends of the spectrum.

---

## 3. Comparison table (fully referenced)

| Dimension | **AgentCore Runtime** | **Docker** | **Raw microVM (Firecracker/gVisor)** | **E2B** |
|---|---|---|---|---|
| **Isolation boundary** | VM-level microVM, own kernel, managed [A1] | OS-level, **shared host kernel**; namespaces/cgroups [D1] | VM-level, own kernel (Firecracker/KVM) [F1][F2]; or user-space kernel (gVisor) [F3] | VM-level, Firecracker microVM [E2] |
| **Safe for hostile model code?** | Yes — per-session microVM [A1] | Risky by default (shared kernel); harden w/ rootless/gVisor [D1][D4][F3] | Yes — hardware/user-space isolation [F1][F3] | Yes — "made to run untrusted workflows" [E2] |
| **Startup latency** | Fast (managed microVM) [A1] | ~ms–seconds (image dependent) [D1] | **<125 ms** boot [F1] | **<200 ms**, 80 ms Pro [E2] |
| **Max duration** | 8 h serverless / **14 days** instances [A6][A2] | unlimited (you run it) [D1] | unlimited (you run it) [F1] | **24 h Pro / 1 h Base**; self-host unlimited [E3] |
| **Persistence** | `/mnt` session storage, EFS, S3 [A4][A8] | volumes / bind mounts [D5] | you build it [F1] | pause/resume, state kept **indefinitely** [E4] |
| **GPU** | Yes, via runtime instances (capacity provider) [A2][A4] | Yes, `--gpus` + NVIDIA toolkit [D2] | Hard — limited device passthrough [F1] | Deployment-dependent / limited [E1] |
| **Network control** | PUBLIC or VPC; egress broker pattern [A4][A8] | full (bridge/host/none) [D3] | full (you configure) [F1] | SDK-controlled; self-host full [E1] |
| **Ops burden** | Lowest — fully managed API [A1] | Low local; medium at scale [D1] | **Highest** — you run everything [F1] | Low cloud / medium self-host [E1] |
| **Scaling limit** | AWS service quotas (concurrent sessions) [A1] | your hardware / cluster [D1] | your hardware [F1] | provider quota / your hardware [E1] |
| **Languages** | Python/JS/TS (Code Interpreter); any (container) [A3] | any [D1] | any [F1] | Python/JS SDK; any in-sandbox [E5] |
| **Runs locally / offline** | No (cloud) [A1] | Yes [D1] | Yes [F1] | Self-host yes; cloud no [E1] |
| **Cost model** | vCPU-hr + GB-hr + mgmt fee; EC2+fee for instances [A5] | infra only (free engine) [D1] | infra only [F1] | usage-based cloud / infra self-host [E1] |
| **Auth / policy** | IAM + Cedar policies [A8] | none built-in [D1] | none built-in [F1] | API keys [E1] |
| **Built for AI agents?** | Yes (Code Interpreter, Browser, Memory) [A3][A7] | No — general purpose [D1] | No — primitive [F1] | **Yes** — its entire purpose [E1] |
| **Vendor lock-in** | AWS [A1] | none (open) [D1] | none (open) [F1] | Low — open-source, self-hostable [E1] |

---

## 4. Best-practice "slides" — when and for what to use each

> One card per option: **use it when… / great for… / avoid when… / rules of thumb.**

```
┌───────────────────────────────────────────────────────────────────────────┐
│  SLIDE 1 — AGENTCORE RUNTIME            managed microVM sandbox on AWS       │
├───────────────────────────────────────────────────────────────────────────┤
│  USE WHEN                                                                   │
│   • You run untrusted model code and want ZERO infra to manage        [A1]  │
│   • You're already on AWS and want IAM + Cedar policy gating + VPC    [A8]   │
│   • You need GPU or sessions > 8 h → runtime instances (up to 14 d)   [A2]   │
│   • Production agents / RL environments that must scale w/o a cluster [A1]   │
│  GREAT FOR                                                                  │
│   • Code Interpreter data analysis, tool execution, browser automation[A3][A7]│
│   • Deterministic RL reward from exitCode/stdout/stderr               [A8]   │
│  AVOID WHEN                                                                 │
│   • You need local/offline runs, or non-AWS portability               [A1]  │
│   • Ultra-high rollout fan-out before raising concurrency quotas       [A1]  │
│  RULES OF THUMB                                                             │
│   • Serverless microVM = default (CPU, cheap). GPU instance only if the     │
│     ENVIRONMENT itself needs GPU or >8 h.  • Request quota bumps early.[A2]  │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│  SLIDE 2 — DOCKER                       OS-level containers, shared kernel   │
├───────────────────────────────────────────────────────────────────────────┤
│  USE WHEN                                                                   │
│   • Code is TRUSTED (your own tools, CI, internal pipelines)          [D1]  │
│   • You want fast, free, local-first execution & full control         [D1]  │
│   • You already run Kubernetes and want cheap self-managed compute     [D1]  │
│   • You need GPU quickly on your own hardware (--gpus)                  [D2]  │
│  GREAT FOR                                                                  │
│   • Local dev, CI, reproducible builds, general-purpose workloads      [D1]  │
│  AVOID WHEN                                                                 │
│   • Running HOSTILE/unknown code without hardening — shared kernel =        │
│     escape risk. Add rootless + gVisor + seccomp + cap-drop.      [D1][D4]  │
│  RULES OF THUMB                                                             │
│   • `--network none --cap-drop=ALL --read-only --pids-limit` for untrusted. │
│   • Swap runtime to gVisor (`--runtime=runsc`) to get VM-like safety. [F3]  │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│  SLIDE 3 — RAW microVM (Firecracker / gVisor)   the tech under the others   │
├───────────────────────────────────────────────────────────────────────────┤
│  USE WHEN                                                                   │
│   • You're BUILDING a sandbox platform at scale and own the eng team  [F1]  │
│   • You need custom isolation economics (density, boot <125 ms)        [F1]  │
│   • You want VM safety with container speed, self-operated            [F1]  │
│  GREAT FOR                                                                  │
│   • Multi-tenant execution backends; the layer E2B/AgentCore sit on   [F1]  │
│   • gVisor when you want strong isolation without full HW virt        [F3]  │
│  AVOID WHEN                                                                 │
│   • You just need to run code today — highest ops burden (rootfs,           │
│     networking, snapshotting, orchestration, autoscaling all on you)  [F1]  │
│   • You need easy GPU passthrough (Firecracker support is limited)     [F1]  │
│  RULES OF THUMB                                                             │
│   • Don't build this unless a sandbox IS your product. Otherwise use a      │
│     managed layer (AgentCore/E2B) that already wraps Firecracker.     [E2]  │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│  SLIDE 4 — E2B                          open-source agent code sandbox      │
├───────────────────────────────────────────────────────────────────────────┤
│  USE WHEN                                                                   │
│   • You want a batteries-included agent sandbox with a clean SDK      [E5]  │
│   • You want cloud simplicity BUT the option to self-host (no lock-in)[E1]  │
│   • You need pause/resume — snapshot memory+fs, keep state forever    [E4]  │
│   • You're not committed to AWS / want multi-cloud                     [E1]  │
│  GREAT FOR                                                                  │
│   • AI code interpreters, RL rollouts, long agent sessions (24 h Pro) [E3]  │
│   • Fast iteration: sandbox starts in <200 ms (80 ms Pro)             [E2]  │
│  AVOID WHEN                                                                 │
│   • You need deep AWS-native IAM/VPC/Cedar integration → AgentCore     [A8]  │
│   • Heavy GPU workloads in-sandbox (support is limited)                [E1]  │
│  RULES OF THUMB                                                             │
│   • It's slime's reference backend (`E2BSandbox`) — fastest way to a        │
│     working RL sandbox; swap to AgentCore later if you go AWS-native. [E6]  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Examples (same task, four ways)

**Task:** run model-generated Python, capture stdout + exit code (the RL reward primitive).

**AgentCore Runtime** — managed microVM; parsing from `file_operations.py` [A8]:
```python
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
c = CodeInterpreter("us-west-2"); c.start()
r = next(iter(c.invoke("executeCode", {"code": code, "language": "python",
                                        "clearContext": True})["stream"]))["result"]
stdout, exit_code = r["structuredContent"]["stdout"], r["structuredContent"]["exitCode"]
c.stop()
```

**Docker** — shared kernel; harden for untrusted code [D1][D2][D4]:
```bash
docker run --rm --network none --cap-drop=ALL --pids-limit=128 \
  python:3.12-slim python -c "$CODE"      # $? = exit code
```

**Raw Firecracker** — you own everything [F1]:
```text
build rootfs w/ python → boot microVM via API socket → push code over vsock →
run, read stdout/exit → tear VM down   (an orchestration + rootfs pipeline you maintain)
```

**E2B** — purpose-built SDK [E5]:
```python
from e2b_code_interpreter import Sandbox
with Sandbox() as sbx:
    exec = sbx.run_code(code)              # exec.logs.stdout / exec.error
```

**In slime's RL loop:** all four sit behind slime's `Sandbox` Protocol
(`slime/agent/sandbox.py:27-59`); swapping backends = swapping one class — `E2BSandbox` (shipped)
[E6], `AgentCoreSandbox` (`agentcore-slime-integration.md` §4), or a Docker/Firecracker impl.

---

## 6. Quick decision guide

- **Untrusted model code + zero infra + AWS-native + policy control** → **AgentCore Runtime** [A1][A8].
- **Batteries-included agent sandbox, cloud or self-host, not AWS-locked** → **E2B** [E1].
- **Trusted code, local dev/CI, you already run k8s** → **Docker** (harden w/ gVisor/rootless if
  untrusted) [D1][D4].
- **Building a sandbox *platform* at scale, can run VM infra** → **raw Firecracker/gVisor** [F1][F3].
- **RL specifically:** policy on your slime GPUs; use any of these as the CPU environment. Start with
  AgentCore serverless or E2B; move to a GPU AgentCore instance only if the *environment* needs GPU
  or >8 h sessions [A2].

---

## References

### AgentCore (repos under `repos/` + AWS docs)
- **[A1]** Runtime overview & session isolation — AWS docs: <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime.html> · repo `bedrock-agentcore-sdk-python/src/bedrock_agentcore/runtime/models.py:17` (session-id header), `runtime/app.py:195-199` (routes)
- **[A2]** "AgentCore runtime instances are now generally available" (GPU-accelerated families, sessions up to 14 days), 2026-08-06 — <https://aws.amazon.com/about-aws/whats-new/2026/08/aws-bedrock-agentcore-runtime-instances-generally-available/>
- **[A3]** Code Interpreter tool (languages python/js/ts, pre-installed libs, 8 h max) — AWS docs: <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html> · repo `.../tools/code_interpreter_client.py:705-751`; `bedrock-agentcore-starter-toolkit/documentation/docs/user-guide/builtin-tools/quickstart-code-interpreter.md:328-354`
- **[A4]** Capacity providers / instance types / session storage — repo `agentcore-cli/docs/commands.md:791-841, :102`
- **[A5]** AgentCore pricing (vCPU-hr, GB-hr, mgmt fee, G-series) — <https://aws.amazon.com/bedrock/agentcore/pricing/>
- **[A6]** Lifecycle bounds (60–28 800 s = 8 h) — repo `bedrock-agentcore-starter-toolkit/.../utils/runtime/schema.py:193-232`
- **[A7]** Browser tool (`aws.browser.v1`, Playwright/CDP) — AWS docs: <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/browser-tool.html> · repo `.../tools/browser_client.py:40-41,312`
- **[A8]** Sample code (executeCommand, writeFiles/readFiles, EFS, egress broker, Cedar policy, durable sandbox) — repo `agentcore-samples/01-features/...` (`03-connect-your-agent-to-anything/01-code-interpreter/{01-file-operations,04-run-commands}`, `02-host-your-agent/01-runtime/{03-advanced/05-execute-commands,03-advanced/12-egress-coding-execution,04-coding-agents/05-autonomous-coding-agent-durable}`) · upstream: <https://github.com/awslabs/agentcore-samples>

### Docker
- **[D1]** Docker security (shared kernel, namespaces/cgroups, root daemon, capabilities) — <https://docs.docker.com/engine/security/>
- **[D2]** GPU access (`--gpus`, NVIDIA Container Toolkit) — <https://docs.docker.com/engine/containers/resource_constraints/gpu/>
- **[D3]** Network drivers (bridge/host/none) — <https://docs.docker.com/engine/network/>
- **[D4]** Rootless mode (hardening) — <https://docs.docker.com/engine/security/rootless/>
- **[D5]** Volumes / persistence — <https://docs.docker.com/engine/storage/volumes/>

### Firecracker / gVisor
- **[F1]** Firecracker — boot **<125 ms**, up to **150 microVMs/s/host**, **<5 MiB** overhead per VM — <https://firecracker-microvm.github.io/>
- **[F2]** Firecracker README (microVMs = HW-virtualization security + container speed; minimal device model) — <https://github.com/firecracker-microvm/firecracker>
- **[F3]** gVisor (user-space application kernel intercepting syscalls; "strong layer of isolation") — <https://gvisor.dev/docs/>

### E2B
- **[E1]** E2B — open-source secure isolated sandboxes; self-host via Terraform (AWS/GCP) — <https://github.com/e2b-dev/E2B> · self-hosting: <https://docs.e2b.dev/self-hosting>
- **[E2]** Firecracker microVM ("made to run untrusted workflows"); start **<200 ms** (80 ms Pro) — <https://e2b.dev/>
- **[E3]** Session duration — **24 h (Pro) / 1 h (Base)** — <https://docs.e2b.dev/sandbox>
- **[E4]** Persistence — pause/resume snapshots memory+filesystem, kept **indefinitely** — <https://docs.e2b.dev/sandbox/persistence>
- **[E5]** SDKs — Python <https://pypi.org/project/e2b> · JS <https://www.npmjs.com/package/e2b>
- **[E6]** slime reference backend `E2BSandbox` — repo `slime/slime/agent/sandbox.py:160`
