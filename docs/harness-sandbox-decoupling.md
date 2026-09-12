# Decoupling & disaggregating harness and sandbox (A2 / A3 / B2 / B3)

**Status: target architecture, not yet wired.** In (harnessed) agentic RL the **harness** and the
**sandbox** are two distinct component types (Agent Lightning v1.0, §1). The **target for all four
harness rows — A2, A3, B2, B3 — is fully decoupled and disaggregated**: the harness runs on **its
own AgentCore Runtime** and the sandbox is a **separate AgentCore Runtime**, with the slime trainer
(policy) and the proxy (`AnthropicAdapter`) as the other two components. See **§7 for the canonical
target**; §1–§3 walk through the intermediate topologies and the mechanism (the MCP bridge for the
CLI rows).

The current implementations are all bring-up shortcuts, not the target:
- **A2 / B2** co-locate the Claude Code CLI *inside* the sandbox (§1A).
- **A3 / B3** run the mini harness *in the trainer process* (§7).

See `demos/demo_a2/README.md` / `demos/demo_b2/README.md` for the four-component wiring and the
harness-vs-sandbox definitions this builds on.

---

## 0. Alignment with Agent Lightning (Figure 1)

This design *is* the Agent Lightning v1.0 architecture. AL calls for **three** decouplings; we have
all three:

| Agent Lightning component (Fig. 1) | Our component | Decoupled from |
|---|---|---|
| **Training Engine** | slime Megatron actor | the Inference Engine |
| **Inference Engine** | SGLang | the Training Engine |
| **LLM API Proxy** | slime `AnthropicAdapter` | *sits between* harness ↔ inference |
| **Agents with Harness** | Claude Code (A2/B2) / mini harness (A3/B3) | the Inference Engine (via the proxy) *and* the sandbox |
| *(execution environment)* | AgentCore sandbox (Code Interpreter / GPU Runtime) | the harness |
| **Sample Adapter** | slime `TrajectoryManager` | — |

1. **Train ⟂ inference.** slime's async **decoupled placement** (disjoint GPUs, `not
   args.colocate`) = AL Training Engine ⟂ Inference Engine.
2. **Harness ⟂ inference engine — *via the proxy*.** The core AL idea: *"connect arbitrary agents
   to RL training through an LLM endpoint proxy."* The harness calls the **proxy**
   (`ANTHROPIC_BASE_URL` → `AnthropicAdapter`), never SGLang directly; the proxy routes to the
   inference engine **and** records every request/response pair. That indirection is what lets the
   **training engine observe only LLM request/response pairs** — AL's defining property of
   *harnessed* agentic RL.
3. **Harness ⟂ sandbox.** AL: *"the harness owns tools, execution environments, and the interaction
   loop."* Harness = the loop; sandbox = the environment → the two-Runtime target of §7.

Net: five independently-scalable services — trainer · inference · proxy · harness · sandbox. The
only gap vs. AL is **implementation maturity** (the current rows are bring-up shortcuts, §1A/§7),
not architecture.

---

## 1. Topologies (bring-up → target)

All keep the same four components (training engine · proxy · harness · sandbox); they differ only
in **where the harness runs** and therefore **how tool actions reach the sandbox**. §1A/§1B below
are stepping stones; **§7 is the target for every row.**

### A. Co-located (what the code does today)

```
  host / rollout worker            AgentCore sandbox (custom multi-user container)
  ┌───────────────────┐            ┌──────────────────────────────────────────┐
  │ slime generate()  │──install──▶│  claude -p   (HARNESS)                     │
  │ AnthropicAdapter  │◀─Anthropic─┤   └─ built-in Bash/Edit exec LOCALLY here  │
  └───────────────────┘   (proxy)  │  run_checks.py  (reward)                   │
                                    └──────────────────────────────────────────┘
```

- slime's `ClaudeCodeHarness.launch_and_wait` → `run_agent(sb, start_cmd="claude -p …")` runs the
  **CLI inside `sb`**; its built-in Bash/Edit tools execute in that same container.
- **Pro:** minimal — literally slime's stock harness with `E2BSandbox → AgentCoreSandbox`.
- **Con:** the sandbox must host a full multi-user CLI (Node + `claude` + `useradd`/`setsid`), so
  it **cannot** be the managed serverless Code Interpreter — it needs a **custom Runtime
  container**. The harness and the execution environment are conceptually two components but
  physically one container.

### B. Decoupled (this sketch — the "pure" split)

```
  host / rollout worker                              AgentCore sandbox (any type)
  ┌────────────────────────────────────┐            ┌───────────────────────────┐
  │ claude -p   (HARNESS, on host)      │            │  bash / write / read      │
  │   built-in Bash/Edit DISABLED       │            │  run in the sandbox       │
  │   tools = mcp__sandbox__*           │──MCP call──▶│  (tool execution only)    │
  │ in-proc MCP server ── wraps ─ sb ───┼────────────┤  run_checks.py (reward)   │
  │ AnthropicAdapter (proxy)            │◀─Anthropic─ │                           │
  └────────────────────────────────────┘            └───────────────────────────┘
```

- The **CLI runs on the host** (the `agentcore_p4` rollout container, which already has Node +
  `claude` staged). Its built-in `Bash`/`Edit`/`Write`/`Read` tools are **disabled**; the only
  tools it has are an **MCP "sandbox" server** whose `bash`/`write_file`/`read_file` execute in the
  AgentCore sandbox.
- **Pro:** the sandbox only ever *executes tool calls*, so it can be the **vanilla managed Code
  Interpreter** again (A2) — no custom multi-user container needed just to host the CLI. Harness
  and sandbox are now genuinely separate components. Matches Agent Lightning's disaggregation
  exactly.
- **Con:** deviates from "only swap the sandbox class" — we no longer use slime's
  `ClaudeCodeHarness` (which is hard-wired to run in-sandbox); we write a small host-side launcher.
  Every tool call is now a network round-trip (host → MCP → AgentCore SDK → sandbox).

> **B2 caveat:** the KernelBench *checker* still needs CUDA, so B2's **sandbox** is still a GPU
> Runtime. Decoupling moves the *harness* off the GPU box, but the GPU sandbox stays (that's the
> task's requirement, not the harness's). See §6 — the target GPU sandbox is a **real AgentCore
> GPU Runtime instance**, not the local box.

---

## 2. Why MCP (and not hooks / SDK)

`claude -p` executes its built-in tools on whatever host it runs on; there is no built-in "run my
Bash tool in a remote sandbox" flag. Three ways to redirect execution, and why MCP wins here:

| Approach | Verdict |
|---|---|
| **PreToolUse hooks** | Can *block/observe* a tool, not cleanly *substitute* its result with remote output. Wrong tool for redirection. |
| **Claude Agent SDK custom tools** | Clean, but replaces the *real CLI* with an SDK harness — loses the "real Claude Code, zero agent changes" demo point. |
| **MCP sandbox server + disable built-ins** ✅ | Keeps the **real CLI**; every tool call is an MCP tool backed by the AgentCore sandbox. The harness is unmodified Claude Code; only its *tool wiring* changes. |

---

## 3. Implementation sketch

Two new pieces in `agentcore_slime_demo/`, plus a new generate path. Nothing here is tested — the
A2/B2 real-run prerequisites (adapter host, tarballs, AWS) still apply.

### 3.1 In-process MCP server over a live sandbox — `sandbox_mcp.py`

One MCP server **per episode**, running **in the rollout process** (like the adapter's aiohttp app
via `run_app_in_thread`), backed by the *already-open* `sb`. This is what keeps a **single sandbox
session** shared between the agent's tool calls and the reward checker.

```python
# agentcore_slime_demo/sandbox_mcp.py  (SKETCH)
"""Expose a live Sandbox as MCP tools so a host-side Claude Code can execute in it."""
from mcp.server.fastmcp import FastMCP


def build_sandbox_mcp(sb, workdir: str) -> FastMCP:
    mcp = FastMCP("agentcore-sandbox")

    def _abs(p: str) -> str:
        return p if p.startswith("/") else f"{workdir}/{p}"

    @mcp.tool()
    async def bash(command: str, timeout: int = 120) -> str:
        """Run a shell command in the AgentCore sandbox; returns exit code + stdout + stderr."""
        code, out, err = await sb.exec(command, timeout=timeout)
        return f"exit={code}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"

    @mcp.tool()
    async def write_file(path: str, content: str) -> str:
        """Write a file in the sandbox workspace (path is relative to the workdir)."""
        await sb.write_file(_abs(path), content)
        return f"wrote {path}"

    @mcp.tool()
    async def read_file(path: str) -> str:
        """Read a file from the sandbox workspace."""
        return await sb.read_file(_abs(path))

    return mcp
```

> The tool set intentionally mirrors the `Sandbox` protocol (`exec`/`write_file`/`read_file`,
> `sandbox.py:49-54`). For B2, the `bash` tool's CUDA work runs on the remote GPU sandbox
> Runtime (`AgentCoreRuntimeSandbox`), so it needs no local GPU handling here.

### 3.2 Host-side launcher — replaces `harness.run(sb, …)`

```python
# in a new agentcore_slime_demo/claude_code_generate_decoupled.py  (SKETCH)
import json, asyncio

_MCP_TOOLS = "mcp__sandbox__bash,mcp__sandbox__write_file,mcp__sandbox__read_file"

_AGENT_PROMPT = (  # the workspace lives in the SANDBOX, not on the host FS — say so
    "You have an `sandbox` MCP server with bash/write_file/read_file tools that run in a remote "
    "sandbox. Use ONLY those tools. Read PROBLEM.md, write solution.py, then run "
    "`python3 run_checks.py` via the bash tool and iterate until it exits 0."
)

async def _run_claude_on_host(*, prompt, adapter_url, session_id, model, mcp_url) -> int:
    mcp_config = {"mcpServers": {"sandbox": {"type": "http", "url": mcp_url}}}
    cmd = [
        "claude", "-p", prompt,
        "--permission-mode", "bypassPermissions",
        "--output-format", "stream-json", "--verbose",
        "--mcp-config", json.dumps(mcp_config),
        "--allowedTools", _MCP_TOOLS,           # only the sandbox tools
        "--disallowedTools", "Bash,Edit,Write,Read,WebFetch",  # kill built-ins → all exec is remote
    ]
    env = {
        "ANTHROPIC_BASE_URL": adapter_url,      # → slime AnthropicAdapter → SGLang policy
        "ANTHROPIC_AUTH_TOKEN": session_id,
        "ANTHROPIC_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    proc = await asyncio.create_subprocess_exec(*cmd, env={**os.environ, **env})
    return await proc.wait()
```

### 3.3 The generate — sandbox owned by generate(), harness on host

```python
async def generate(args, base_sample, sampling_params, evaluation=False):
    task = _task_from_sample(base_sample)
    # ... build AnthropicAdapter + run_app_in_thread, open_session (identical to today) ...

    async with AgentCoreSandbox(region=..., workdir=_WORKDIR) as sb:   # NO install_cli
        await _prepare_workspace(sb, task)             # PROBLEM.md, run_checks.py IN the sandbox

        mcp = build_sandbox_mcp(sb, _WORKDIR)
        mcp_handle = run_app_in_thread(mcp.streamable_http_app(), host="127.0.0.1", port=0)
        mcp_url = f"http://127.0.0.1:{mcp_handle.port}/mcp"

        exit_code = await _run_claude_on_host(          # HARNESS runs on the host now
            prompt=_AGENT_PROMPT, adapter_url=adapter_url,
            session_id=session_id, model=os.environ["ANTHROPIC_MODEL"], mcp_url=mcp_url,
        )
        reward = task.reward((await sb.exec("python3 run_checks.py", timeout=120))[0])

    return await adapter.finish_session(session_id, base_sample=base_sample, reward=float(reward), ...)
```

Wire it in `run.sh` with
`--custom-generate-function-path agentcore_slime_demo.claude_code_generate_decoupled.generate`.

---

## 4. What changes vs. co-located — checklist

- **Sandbox container:** A2 can drop the custom multi-user image and use the **managed Code
  Interpreter** (the CLI no longer lives inside it). B2 still needs CUDA+torch in the sandbox for
  the *checker*, but no longer Node/`claude`.
- **Host requirement (new):** Node 22 + `claude` must be installed on the **rollout container**
  (`agentcore_p4`), not the sandbox. `SLIME_AGENT_*_TARBALL` semantics move host-side.
- **Prompt:** must tell the agent the workspace is remote (MCP tools only) — it has no local FS.
- **Concurrency:** N concurrent episodes = N host `claude` processes + N sandbox sessions + N
  in-proc MCP servers. Cap with `--sglang-server-concurrency` / `CONC` as today; watch host
  CPU/mem for the CLI fan-out.
- **`ADAPTER_PUBLIC_HOST`:** if the CLI is on the same host as the adapter, this can be
  `127.0.0.1` — the sandbox no longer needs to dial *back* to the adapter (only the host-side CLI
  does), which removes the public-reachability requirement that A2/B2 have today.
- **Not using `ClaudeCodeHarness`:** we hand-roll the launcher, so the "only-swap-the-sandbox"
  property is lost. That's the deliberate trade for a real component split.

---

## 5. Open questions / risks

- **MCP transport:** `streamable-http` on localhost is assumed; stdio would avoid a port but
  complicates the in-process-server + live-`sb` sharing. Verify the installed `mcp` SDK's
  `FastMCP.streamable_http_app()` surface.
- **Tool-name allowlisting:** confirm `--allowedTools mcp__<server>__<tool>` naming and that
  `--disallowedTools` fully suppresses built-ins under `bypassPermissions`.
- **Latency:** every tool call is host→MCP→AgentCore SDK→sandbox. For chatty agents this adds
  round-trips vs. in-sandbox local exec. Measure before committing.
- **Token capture unaffected:** the *proxy* path (adapter → SGLang) is identical in both
  topologies, so GRPO sample capture is unchanged — only tool execution moved.

---

## 6. B / B2 / B3 sandbox = a real AgentCore GPU Runtime instance (not the local box)

**Current reality:** Demo B / B2 / B3 run the GPU sandbox as its **own remote AgentCore GPU
Runtime instance** (`AgentCoreRuntimeSandbox`, `SANDBOX=agentcore-runtime`) — a capacity-provider
EC2 GPU (a `g6`) launched by AgentCore. The checker's CUDA work (`python3 run_checks.py`) runs on
that **remote managed instance**, reached over `invoke_agent_runtime`, so **no** local GPUs are
held back for env checkers. slime runs **4 train + 4 infer** on all 8 local GPUs (like A/A2).

### What running the sandbox off-box gives us

- **GPU accounting is 4+4+0.** Because the sandbox lives on a *separate* AgentCore GPU instance,
  the trainer box reserves nothing for envs → slime uses all 8 local GPUs.
- **Fully physical split.** With the sandbox on a remote AgentCore GPU instance (and, for B2, the
  harness decoupled per §1B / §7), the split is fully physical: trainer box = policy (8 GPU) +
  proxy + harness; AgentCore GPU instance = sandbox only. This is the cleanest form of the
  four-component architecture.
- **Sharing lives on the remote Runtime.** If many episodes must share one g6, that is a concern
  of the remote Runtime image (Demo B4 runs a Ray GPU-sharing farm *inside* the sandbox Runtime),
  not of the trainer box.
- **Cost/lifecycle:** a GPU Runtime instance bills while up (~$0.80–1/hr + EBS on `g6.xlarge`) and
  must be torn down — see the 08/09 `OBSERVATIONS.md` and the
  capacity-provider/`securityGroups`/`LD_LIBRARY_PATH` gotchas found there. A rollout that
  launches one instance per episode would be far too slow/expensive; the realistic design is a
  **pool of long-lived GPU Runtime sessions** the rollouts share.

> **Honest status:** this is item (b) of demos-guide.md §8 "Not yet" — a full logged *Demo B on a
> real AgentCore GPU Runtime instance* training run. The `AgentCoreRuntimeSandbox` backend and the
> GPU sandbox Runtime image are built and deploy; the end-to-end logged run is the pending piece.

---

## 7. The real target: TWO AgentCore Runtimes — one harness, one sandbox

**Correction to §1B.** Putting the harness "on the host / rollout worker" (§1B), and running
A3/B3's mini harness **in-process**, are both the *wrong* way. AgentCore is not only a sandbox —
it can host the **harness** too (its own Runtime running the agent loop, or the managed harness
product, use case 10). So the target topology puts **each component type on its own AgentCore
Runtime**:

```
  trainer box (slime)                AgentCore Runtime #1              AgentCore Runtime #2
  ┌────────────────────┐             ┌──────────────────────┐         ┌──────────────────────┐
  │ Megatron + SGLang  │             │  HARNESS             │         │  SANDBOX             │
  │ AnthropicAdapter   │◀─Anthropic──┤  agent loop:         │──tool──▶│  bash/write/read     │
  │ (proxy endpoint)   │   (LLM)     │  mini harness (A3/B3)│  calls  │  run_checks.py       │
  │                    │             │  or Claude Code(A2/B2)│         │  (Code Interp / GPU) │
  └────────────────────┘             └──────────────────────┘         └──────────────────────┘
       policy GPUs                    AgentCore-hosted harness          AgentCore-hosted sandbox
                                      (NOT in the trainer process)      (GPU Runtime for B)
```

- **Harness = AgentCore Runtime #1.** The agent loop is deployed *as an AgentCore Runtime* (or the
  managed harness), **not** run inside the slime rollout process. It reaches the policy through the
  proxy and issues tool calls to Runtime #2.
- **Sandbox = AgentCore Runtime #2.** A *separate* AgentCore Runtime executes the tool calls —
  Code Interpreter for CPU rows, a **GPU Runtime instance** for B rows (§6).
- **Two Runtimes, two roles.** This is the clean four-component split with AgentCore owning both
  the harness and the sandbox as distinct managed services.

### What's wrong with the current implementations (all four rows)

| Row | Current (non-target) | Why it's not the target |
|---|---|---|
| **A2 / B2** | Claude Code CLI runs **inside** the sandbox (§1A co-location) | harness and sandbox share one container — not two Runtimes |
| **A3 / B3** | mini harness runs **in the trainer/rollout process** | harness isn't on AgentCore at all — it's in slime's process |

So **every row needs work** to reach the target; none is "already there." The mini harness being
in-process is convenient for bring-up but is explicitly *not* the intended deployment.

### How each row gets to the two-Runtime target

- **A3 / B3 (mini harness):** deploy `MiniClaudeHarness` as a `BedrockAgentCoreApp` on its own
  AgentCore Runtime. Its LLM calls go to the proxy; its tool calls go to the **sandbox Runtime**
  (Code Interpreter for A3, GPU Runtime for B3) instead of a local `sb` object. This is the
  cleanest port since we own the harness code — the tool-dispatch just changes from an in-process
  `sb.exec` to an invoke of the sandbox Runtime.
- **A2 / B2 (Claude Code):** run Claude Code on **harness Runtime #1**, with its built-in tools
  disabled and pointed (via the §3 MCP bridge) at **sandbox Runtime #2**. The MCP bridge is still
  the mechanism for the black-box CLI; the difference from §3 is that the harness itself now lives
  on AgentCore, not on the trainer host.

> **Takeaway:** the goal for **all four rows** is *harness on one AgentCore Runtime, sandbox on
> another*. In-process (A3/B3) and in-sandbox (A2/B2) are both bring-up shortcuts to be replaced.
