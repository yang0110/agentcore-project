# Using AWS Bedrock AgentCore Runtime as a Sandbox for Agentic RL with slime

**Question this answers:** How do you use AgentCore Runtime as the *execution sandbox* for
agent reinforcement learning, and how do you connect it to the **slime** RL framework
(THUDM/slime)?

Everything below is backed by the actual source of the cloned repos under `repos/`. Every
non-obvious claim has a `file:line` citation you can open and verify. Where a capability does
**not** exist in the code, that is stated explicitly rather than invented.

Repos referenced (all cloned locally):

| Repo | Local path | Role |
|------|-----------|------|
| slime | `repos/slime` | RL trainer (SGLang rollout + Megatron train) |
| AgentCore Python SDK | `repos/bedrock-agentcore-sdk-python` | Runtime server + sandbox tool clients |
| AgentCore starter toolkit | `repos/bedrock-agentcore-starter-toolkit` | deploy/invoke helpers, docs |
| AgentCore CLI | `repos/agentcore-cli` | current CLI (`@aws/agentcore`), capacity providers |
| AgentCore samples | `repos/agentcore-samples` | sandbox / code-interpreter examples |

---

## 1. TL;DR and mental model

- **slime owns the GPUs and the learning.** slime runs the policy model on SGLang for
  generation and trains with Megatron. The GPUs live on *your* slime cluster, not on AgentCore.
- **AgentCore Runtime is the *environment*, not the trainer.** It gives you an isolated,
  per-session microVM sandbox where the policy's proposed actions (shell commands, code, tool
  calls) are actually executed, returning deterministic `stdout / stderr / exit_code`.
- **The seam between them is a single function.** slime already defines a `Sandbox` Protocol
  and a per-sample `generate` hook. You implement one of those against AgentCore's sandbox API.
  No fork of slime is required.

```
            slime cluster (GPUs)                         AWS (no GPU needed)
 ┌────────────────────────────────────────┐   ┌──────────────────────────────────┐
 │  Megatron trainer  ◄── train data ──┐   │   │  AgentCore sandbox (microVM)     │
 │                                     │   │   │  per rollout = 1 session         │
 │  SGLang engine (policy)  ──gen──►  generate()│  executeCode / executeCommand    │
 │        ▲                            │   │   │  returns stdout/stderr/exitCode  │
 │        └──── action tokens ─────────┘   │   │            ▲                     │
 │             observation tokens ◄────────┼───┼────────────┘                     │
 └────────────────────────────────────────┘   └──────────────────────────────────┘
```

The rollout loop is: **SGLang generates an action → AgentCore executes it → result is fed back
as observation tokens (masked out of the loss) → reward is computed from the sandbox result.**

---

## 2. What AgentCore Runtime actually gives you as a sandbox

AgentCore exposes **two** sandbox primitives. For RL you almost always want the first.

### 2.1 Code Interpreter — the managed sandbox (recommended)

A fully managed, session-isolated sandbox. No container to build, no runtime to deploy.

Client: `repos/bedrock-agentcore-sdk-python/src/bedrock_agentcore/tools/code_interpreter_client.py`

- Default identifier `aws.codeinterpreter.v1`, default session timeout `900`s (line 22, 30).
- `start(identifier, name, session_timeout_seconds=900)` → `start_code_interpreter_session(...)`,
  returns a `sessionId` (lines 308-348).
- `invoke(method, params)` → `invoke_code_interpreter(codeInterpreterIdentifier, sessionId,
  name=method, arguments=params)` (lines 442-470).
- Convenience wrappers: `execute_code(code, language="python", clear_context=False)` with
  `valid_languages = ["python","javascript","typescript"]` (lines 705-751); plus
  `execute_command`, `install_packages` (pip), `upload_file(s)` (→ `writeFiles`),
  `download_file(s)` (→ `readFiles`), and a `code_session(...)` context manager (lines 806-839).
- Limits (docs `.../builtin-tools/quickstart-code-interpreter.md:328-354`): default 15 min,
  **max 8 h (28,800 s)**; sessions auto-terminate at timeout; numpy/pandas/matplotlib etc.
  pre-installed.

The response is a stream; the sample repos parse it exactly like this
(`repos/agentcore-samples/.../01-code-interpreter/01-file-operations/file_operations.py`):

```python
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

def call_tool(client, tool_name, arguments):
    response = client.invoke(tool_name, arguments)
    for event in response["stream"]:
        return json.dumps(event["result"])          # single result event

code_client = CodeInterpreter(REGION)
code_client.start()
exec_json = call_tool(code_client, "executeCode",
                      {"code": src, "language": "python", "clearContext": True})
stdout = json.loads(exec_json)["structuredContent"]["stdout"]   # .stderr / .exitCode too
code_client.stop()
```

Raw boto3 equivalent (from `.../01-code-interpreter/04-run-commands/run_commands.py`):
`start_code_interpreter_session` → `invoke_code_interpreter(name="executeCommand",
arguments={"command": ...})` → `stop_code_interpreter_session`. IAM actions needed:
`bedrock-agentcore:CreateCodeInterpreter / StartCodeInterpreterSession / InvokeCodeInterpreter /
StopCodeInterpreterSession / DeleteCodeInterpreter / List* / Get*`.

**Why this is the right primitive for RL:** each `start()` is an isolated microVM, so N concurrent
rollouts = N sessions that cannot see each other. `executeCode`/`executeCommand` are deterministic
and return a structured `exitCode` you can turn into a reward.

### 2.2 Custom Runtime — a container/agent you deploy (heavier)

If your environment needs a custom image (specific tools, your own harness), you host it as an
AgentCore Runtime. Server: `repos/bedrock-agentcore-sdk-python/src/bedrock_agentcore/runtime/app.py`.

- Starlette + uvicorn app. Routes: `POST /invocations`, `GET /ping`, `WS /ws` (app.py:195-199).
- Define the agent with `@app.entrypoint` (app.py:216-233); handler is `(payload)` or
  `(payload, context)` — `context.session_id` exposes the session (app.py:534-539).
- Serves on **port 8080** (`run(port=8080,...)`, app.py:649-676).
- Invoke from a client with boto3 (`quickstart.md:180-206`):

```python
client = boto3.client("bedrock-agentcore")
resp = client.invoke_agent_runtime(
    agentRuntimeArn=arn,
    runtimeSessionId=str(uuid.uuid4()),   # session affinity → same microVM
    payload=json.dumps({"prompt": ...}).encode(),
    qualifier="DEFAULT",
)
body = resp["response"].read()
```

- Session affinity: pass the *same* `runtimeSessionId` to reuse the same microVM. AgentCore
  requires session ids **≥ 33 chars** (validated in the durable sample's
  `.../05-autonomous-coding-agent-durable/coding-agent/sandbox_client.py`).
- Live shell exec inside a session: `invoke_agent_runtime_command(agentRuntimeArn,
  runtimeSessionId, body={"command":..., "timeout":...})` streams `stdout/stderr` and an
  `exitCode`; release with `stop_runtime_session(...)`
  (`.../03-advanced/05-execute-commands/invoke.py`).

Deploy with the current CLI (`repos/agentcore-cli/README.md:36-61`):
`npm i -g @aws/agentcore` → `agentcore create` → `agentcore deploy` → `agentcore invoke`.
Containers build remotely via CodeBuild → ECR (ARM64/Graviton).

### 2.3 Session isolation, statefulness

- Session id header: `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` (SDK `runtime/models.py:17`).
- Per-request/session state is held in `contextvars` (`runtime/context.py`), so concurrent
  sessions are isolated in-process too.
- Documented as "complete session isolation … long-running workloads up to 8 hours"
  (AWS launch announcement) and "true session isolation" (SDK README). Idle session timeout is
  15 min (`runtime/overview.md`); lifecycle max is 8 h (`schema.py:193-232`, bounds 60–28,800 s).

### 2.4 The GPU / compute reality (answering the original question)

**AgentCore Runtime *does* support GPU** — via **"AgentCore Runtime instances," which went GA on
2026-08-06** ("AgentCore runtime instances are now generally available", AWS What's New,
`/about-aws/whats-new/2026/08/aws-bedrock-agentcore-runtime-instances-generally-available/`).
Two compute models exist:

- **Default serverless microVMs (managed fleet):** ARM64/Graviton, fast startup, sessions up to
  **8 h**; consumption-billed (`$0.0895/vCPU-hr`, `$0.00945/GB-hr` per the pricing page). No GPU.
- **Runtime instances (customer-managed EC2 via a capacity provider):** you "run agents on your
  own Amazon EC2 instances without managing infrastructure," specifying the "EC2 instance types
  your agents need, **including GPU-accelerated, memory-optimized, and compute-optimized
  families**," with long-running sessions up to **14 days**. GA regions: N. Virginia, Ohio,
  Oregon, Mumbai, Singapore, Sydney, Tokyo, Frankfurt, Ireland.

This is exactly the capacity-provider primitive in the AgentCore CLI
(`repos/agentcore-cli/docs/commands.md:791-841`), now confirmed by the announcement as the
official GPU path — the CLI docs just never use the word "GPU":

```bash
agentcore add capacity-provider \
  --instance-types g6.xlarge,g6.2xlarge \   # GPU-accelerated family (1–30 types)
  --os LINUX_X86_64 \
  --subnets <...> --security-groups <...>
agentcore add agent --capacity-provider <name-or-arn>   # attach the agent to it
```

`--max-lifetime` / `--idle-instance-timeout` on a capacity provider go up to **1,209,600 s
(= 14 days)**, matching the announced session limit (vs. 8 h / 28,800 s for the serverless
lifecycle). The pricing page also lists a reduced 7.8% management fee for **G-series (incl.
Graviton gr6)** under the "Instances" model. Note: the announcement names *categories*
("GPU-accelerated") rather than an exact allow-list; the console/CLI presents the concrete
instance types available in your region.

- **For RL, you still usually don't need GPU in the sandbox.** The policy runs on your slime
  GPUs; the sandbox executes code and returns text, so CPU serverless microVMs are the cheap
  default. Reach for a **GPU runtime instance** only when the *environment itself* runs a model
  or CUDA workload (e.g. a GPU-backed verifier/simulator, or an agent that does local inference),
  or when you need sessions longer than 8 h.

---

## 3. slime's extension points (where AgentCore plugs in)

All paths under `repos/slime`.

### 3.1 The `Sandbox` Protocol — the cleanest hook

`slime/agent/sandbox.py:27-59` defines a `@runtime_checkable Protocol`:

```python
class Sandbox(Protocol):
    sandbox_id: str
    async def __aenter__(self) -> "Sandbox": ...
    async def __aexit__(self, *exc) -> None: ...
    async def exec(self, cmd: str, *, user="root", env=None, timeout=120,
                   check=False, idempotent=True) -> ExecResult: ...      # (exit_code, stdout, stderr)
    async def write_file(self, sandbox_path: str, content, *, user="root") -> None: ...
    async def read_file(self, sandbox_path: str, *, user="root") -> str: ...
```

`ExecResult = tuple[int, str, str]`. The reference implementation is `E2BSandbox`
(`sandbox.py:160`). **To use AgentCore, implement this Protocol against the Code Interpreter
client** (Section 4). The multi-turn machinery that consumes a `Sandbox` — `TrajectoryManager`
(`slime/agent/trajectory.py:269`) and the OpenAI/Anthropic adapters (`slime/agent/adapters/`) —
is sandbox-agnostic, so swapping the backend is all that's needed.

### 3.2 The per-sample `generate` hook — the simplest hook

`slime/utils/arguments.py:486-496` defines `--custom-generate-function-path`; it is loaded per
sample and dispatched at `slime/rollout/sglang_rollout.py:283-294`. Signature you implement:

```python
async def generate(args, sample: Sample, sampling_params: dict,
                   evaluation: bool = False) -> Sample | list[Sample]: ...
```

Return `list[Sample]` for fan-out/multi-turn. Set `generate.abort_mode = "request"` if you
support request-level abort. Wire it with
`--custom-generate-function-path your.module.generate`.

Inside `generate` you call SGLang directly for the policy tokens:

```python
from slime.utils.http_utils import post
url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
out = await post(url, {"input_ids": sample.tokens,
                       "sampling_params": sampling_params,
                       "return_logprob": True})
# token ids + logprobs in out["meta_info"]["output_token_logprobs"]
# finish reason in out["meta_info"]["finish_reason"]["type"]  ("length"|"stop"|"abort")
```

(Verbatim pattern from `slime/rollout/sglang_rollout.py:154-223` and the search example
`examples/search-r1/generate_with_search.py:150,179-188`.)

### 3.3 The `Sample` contract (what you must fill)

`slime/utils/types.py:93-151`. Minimal fields the trainer requires: `tokens`,
`response_length`, `reward`, `status`, and `loss_mask` (masks tool/observation tokens out of the
loss). Enforced in `slime/ray/rollout.py:_convert_samples_to_train_data:306-354`: if `loss_mask`
is `None` it defaults to all-ones, and `len(loss_mask) == response_length` is asserted.

**Use the sanctioned helper** `Sample.append_response_tokens(...)` (`types.py:253-314`) to keep
`tokens / response_length / loss_mask / rollout_log_probs` in lockstep. For sandbox output
(observation) tokens pass `trainable=False` so they get `loss_mask=0` — exactly what search-r1
does (`generate_with_search.py:244`).

### 3.4 Reward wiring

`slime/rollout/rm_hub/__init__.py` (`async_rm`): a custom reward is
`--custom-rm-path your.module.custom_rm` with `async def custom_rm(args, sample) -> float`, OR
you fill `sample.reward` inside `generate` and slime skips the RM
(`sglang_rollout.py:317`). For a sandbox, computing reward from the `exit_code` /
test result *inside* `generate` is usually simplest.

### 3.5 Reference examples to copy

- `examples/coding_agent_rl/generate.py` — full SWE sandbox agent, wired via
  `--custom-generate-function-path examples.coding_agent_rl.generate.generate`. Boots an **E2B**
  sandbox, runs an agent harness, computes reward from an evaluator, returns `list[Sample]`.
  **This is the closest template — swap E2B for AgentCore.**
- `examples/search-r1/generate_with_search.py` — minimal multi-turn tool loop; best for
  understanding token/loss_mask bookkeeping.

---

## 4. The integration: an `AgentCoreSandbox` for slime

This is the concrete adapter. It implements slime's `Sandbox` Protocol
(`slime/agent/sandbox.py:27-59`) on top of the AgentCore **Code Interpreter** client
(§2.1). The boto3 client is synchronous, so calls are offloaded with `asyncio.to_thread`.

```python
# slime_plugins/agentcore_sandbox.py
import asyncio, json, uuid
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

class AgentCoreSandbox:
    """AgentCore Code Interpreter as a slime Sandbox (slime/agent/sandbox.py:27-59).

    One instance == one isolated microVM session == one rollout.
    """
    def __init__(self, region: str = "us-west-2", session_timeout_seconds: int = 900):
        self._region = region
        self._timeout = session_timeout_seconds
        self._client: CodeInterpreter | None = None
        self.sandbox_id: str = f"ac-{uuid.uuid4().hex}"

    # --- lifecycle -------------------------------------------------------
    async def __aenter__(self) -> "AgentCoreSandbox":
        self._client = CodeInterpreter(self._region)
        await asyncio.to_thread(self._client.start,
                                name=self.sandbox_id,
                                session_timeout_seconds=self._timeout)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.stop)
            self._client = None

    # --- helper: invoke a tool and return its single result dict ---------
    def _call(self, tool: str, args: dict) -> dict:
        resp = self._client.invoke(tool, args)          # code_interpreter_client.py:442-470
        for event in resp["stream"]:
            return event["result"]
        return {"isError": True}

    # --- Sandbox Protocol ------------------------------------------------
    async def exec(self, cmd: str, *, user="root", env=None, timeout=120,
                   check=False, idempotent=True):
        # wrap with coreutils `timeout` since executeCommand has no per-call timeout
        wrapped = f"timeout {int(timeout)} bash -lc {json.dumps(cmd)}"
        result = await asyncio.to_thread(self._call, "executeCommand", {"command": wrapped})
        sc = result.get("structuredContent", {})        # file_operations.py parsing shape
        exit_code = int(sc.get("exitCode", 1))
        stdout, stderr = sc.get("stdout", ""), sc.get("stderr", "")
        if check and exit_code != 0:
            raise RuntimeError(f"exec failed ({exit_code}): {stderr}")
        return (exit_code, stdout, stderr)               # ExecResult

    async def run_python(self, code: str, clear_context: bool = False):
        """Extra convenience: executeCode, not part of the Protocol."""
        result = await asyncio.to_thread(
            self._call, "executeCode",
            {"code": code, "language": "python", "clearContext": clear_context})
        sc = result.get("structuredContent", {})
        return (int(sc.get("exitCode", 1)), sc.get("stdout", ""), sc.get("stderr", ""))

    async def write_file(self, sandbox_path: str, content, *, user="root") -> None:
        text = content.decode() if isinstance(content, (bytes, bytearray)) else content
        await asyncio.to_thread(self._call, "writeFiles",
                                {"content": [{"path": sandbox_path, "text": text}]})

    async def read_file(self, sandbox_path: str, *, user="root") -> str:
        result = await asyncio.to_thread(self._call, "readFiles", {"paths": [sandbox_path]})
        # readFiles returns file contents in result["content"]; shape per samples repo
        return json.dumps(result.get("content", ""))
```

> Field shapes (`structuredContent.stdout/.stderr/.exitCode`, `writeFiles` with a `content`
> list, `readFiles` with `paths`) mirror the sample code in
> `repos/agentcore-samples/.../01-code-interpreter/`. Verify them against your account's SDK
> version before training at scale; they are the documented shapes but the API surface evolves.

### 4.1 Two ways to consume it

**Option A — reuse slime's agent scaffolding.** Point the coding-agent example at
`AgentCoreSandbox` instead of `E2BSandbox` and let `TrajectoryManager` +
adapters build the loss-masked `list[Sample]`. Best when you want a real multi-turn agent
harness. (Mirror `examples/coding_agent_rl/generate.py`.)

**Option B — a direct custom `generate`** (simplest; see the demo doc). You drive the loop
yourself: SGLang generates code → `AgentCoreSandbox` runs it → reward from `exit_code` →
one `Sample`. This is what `docs/minimal-demo.md` implements.

### 4.2 Concurrency & session mapping

- slime runs many rollouts concurrently via asyncio with a semaphore sized
  `args.sglang_server_concurrency * num_engines` (`sglang_rollout.py` `GenerateState`). Each
  concurrent `generate` should own its **own** `AgentCoreSandbox` (= its own microVM session),
  so environments never bleed into each other.
- Starting a session has latency; for high rollout throughput, pool a fixed set of sessions and
  check them out per rollout, or raise the session timeout and reuse across a group
  (`n_samples_per_prompt`) — but reset state with `clearContext=True` / a fresh working dir
  between samples to preserve on-policy independence.
- Watch AgentCore service quotas (concurrent Code Interpreter sessions) — that, not GPUs, is the
  scaling limit on the environment side.

---

## 5. End-to-end wiring (launch)

1. Install the SDK in the slime rollout image: `pip install bedrock-agentcore boto3`.
2. Provide AWS creds/region to the rollout workers (env or instance role) with the Code
   Interpreter IAM actions from §2.1.
3. Put `AgentCoreSandbox` on the Python path (e.g. `slime_plugins/agentcore_sandbox.py`) and your
   `generate` in a module (see demo).
4. Add to your slime launch script:

```bash
python train.py \
  ... your model / data / megatron args ... \
  --custom-generate-function-path slime_plugins.agentcore_generate.generate \
  --rm-type none        # reward is filled inside generate; or use --custom-rm-path
```

(`--custom-generate-function-path` is consumed at `sglang_rollout.py:283-294`; the default
full-rollout function stays in place.)

---

## 6. Honest caveats / gaps (no hand-waving)

- **No RL code in AgentCore samples.** A full grep of `repos/agentcore-samples` for `gym`,
  `PPO`, `GRPO`, `rollout buffer`, `reward model` found only false positives. AgentCore provides
  the *sandbox*; the RL loop is entirely slime's. Do not expect a ready-made RL example on the
  AWS side.
- **GPU IS supported** via AgentCore Runtime *instances* (capacity providers), GA 2026-08-06 —
  see §2.4. GPU-accelerated EC2 families, sessions up to 14 days. Configurable RAM/CPU sizing for
  the *serverless* fleet is still not exposed; there you get consumption-billed vCPU/GB.
- **API field shapes** for `writeFiles`/`readFiles`/`executeCommand` results are taken from the
  samples repo, not a formal schema — pin your SDK version and smoke-test §4's adapter first.
- **Session-start latency and service quotas**, not GPUs, are the environment-side scaling limit.
- **Per-command timeout** isn't native to `executeCode`; §4 wraps commands with coreutils
  `timeout`. The *session* timeout maxes at 8 h.
- slime has **no generic "sandbox registry"** — backend selection in the coding example is a
  hard-coded dict + env var (`examples/coding_agent_rl/generate.py:45-52`). You wire
  `AgentCoreSandbox` in explicitly.

---

## 7. File-reference index

**AgentCore**
- Code Interpreter client: `repos/bedrock-agentcore-sdk-python/src/bedrock_agentcore/tools/code_interpreter_client.py`
- Runtime server: `repos/bedrock-agentcore-sdk-python/src/bedrock_agentcore/runtime/app.py:195-199,216-233,649-676`
- Session header/context: `.../runtime/models.py:17`, `.../runtime/context.py`
- Code Interpreter samples: `repos/agentcore-samples/01-features/03-connect-your-agent-to-anything/01-code-interpreter/`
- Runtime shell exec: `repos/agentcore-samples/01-features/02-host-your-agent/01-runtime/03-advanced/05-execute-commands/`
- Durable sandbox pattern: `repos/agentcore-samples/.../04-coding-agents/05-autonomous-coding-agent-durable/`
- Capacity providers: `repos/agentcore-cli/docs/commands.md:791-841`

**slime**
- Sandbox Protocol: `repos/slime/slime/agent/sandbox.py:27-59` (E2B ref at :160)
- Custom generate arg + dispatch: `repos/slime/slime/utils/arguments.py:486-496`, `repos/slime/slime/rollout/sglang_rollout.py:283-294`
- Sample type: `repos/slime/slime/utils/types.py:93-151,253-314`
- Train-data contract: `repos/slime/slime/ray/rollout.py:306-354`
- SGLang call: `repos/slime/slime/rollout/sglang_rollout.py:154-223`
- Reward hub: `repos/slime/slime/rollout/rm_hub/__init__.py`
- Templates: `repos/slime/examples/coding_agent_rl/generate.py`, `repos/slime/examples/search-r1/generate_with_search.py`
