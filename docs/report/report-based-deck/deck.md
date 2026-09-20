
**Kaige Yang** — Senior Applied Scientist, AWS

AgentCore Runtime as the execution sandbox for agentic RL, integrated with slime.

---

## Outline

```{=latex}
\tableofcontents
```

# 1. Why sandboxes — what frontier labs run in them

## Purpose & roadmap — AgentCore as the sandbox for agentic RL

**One idea:** in agentic RL the policy acts over many turns against an executable environment, and a **programmatic checker** — not an LLM judge — turns the outcome into a scalar reward (RLVR).

**Roadmap (6 sections):**

1. **Why sandboxes** — what frontier labs run in sandboxes, and how they score it.
2. **Architecture & pipeline** — reference architecture, the five components, slime + its async rollout.
3. **Foundations of AgentCore** — Runtime / session / microVM, CPU vs GPU compute, session lifecycle & launch/terminate.
4. **Integrating into slime** — the call path, one hook, the episode loop, and the key code.
5. **The demos** — all 21 at a glance, then **one slide per demo** (config, checker, measured eval/latency, honest status, real use case).
6. **Cross-cutting mechanisms** — episode shape, TITO, the scaling `min()` law, one-episode→many-samples, and the resilience levers.

*Scope: AgentCore = the execution **sandbox**; **slime** = the trainer; the **harness stays in-process**.*

## Executive summary — the one slide

- **Problem.** Agentic RL runs **untrusted model output** in an isolated environment thousands of times per run and scores it with a **programmatic checker** — the *sandbox* layer, not the GPU, is often the bottleneck (Kimi K2 [5]: **>10,000 concurrent sandboxes**).
- **Solution.** **AgentCore Runtime is that sandbox** — one isolated microVM session per episode, **in your own AWS account** (IAM/SigV4, VPC), bring-your-own container, CPU by default and **GPU on Instances** — wired into the **slime** trainer through **one hook**, harness in-process.
- **Proven.** The full loop runs **end-to-end on real hardware across 21 workloads** (code, math, SQL, RAG, tool-use, CUDA kernels), each a standard **10-step run** with real rewards from an in-sandbox checker.
- **How you adopt it.** Build a sandbox image → deploy a Runtime → seed a task → train via one `generate` hook. **CPU↔GPU is one env var.**

## What tasks do frontier labs run in sandboxes

Untrusted output runs in an isolated sandbox; a **programmatic verifier** produces the reward. This is the dominant recipe in recent tech reports — Kimi K2 [5] reports **>10,000 concurrent sandboxes** for agentic tool-use / code RL.

**Two verifier styles:** *verifiable / rule-based* (binary, ungameable) vs *reference-based judge* (open-ended). This project uses **RLVR only**.

| Model | Verifier | Tasks | Reward |
|---|---|---|---|
| DeepSeek-R1 [2] | rule verifier + compiler | math, competitive code | tests pass / match |
| Qwen3 [3] | RLVR + live tool env | math, code, tool-use, RAG | rule + judge |
| Llama-Nemotron [4] | LLM-judge | science, math, instruction | match + format |
| Kimi K2 [5] | simulator + real exec sandboxes | agentic tool use, code, SWE | tests + checks |

*Same recipe throughout this deck: **one isolated sandbox per episode**, a **programmatic checker** for the reward.*

# 2. Agentic RL architecture & the training pipeline

## The reference architecture — one diagram

```{=latex}
\begin{center}
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
  font=\small,
  box/.style={draw, rounded corners, minimum height=1.15cm, minimum width=2.4cm, align=center, thick, fill=white},
  gpu/.style={box, fill=blue!8},
  cpu/.style={box, fill=black!4},
  focus/.style={box, fill=orange!18},
  lbl/.style={font=\scriptsize, align=center, fill=white, inner sep=1.5pt},
  ptitle/.style={font=\scriptsize\bfseries, text=black!55},
  panel/.style={draw, rounded corners, dashed, thick, inner sep=0.34cm},
  >={Stealth[]},
]
\node[gpu] (trainer) at (0,0)    {\textbf{Training Engine}\\[1pt]\scriptsize Megatron};
\node[gpu] (infer)   at (5.0,0)  {\textbf{Inference Engine}\\[1pt]\scriptsize SGLang};
\node[cpu] (buffer)  at (0,-3.4) {\textbf{Buffer}\\[1pt]\scriptsize training samples};
\node[cpu] (adapter) at (5.0,-3.4){\textbf{Sample Adapter}\\[1pt]\scriptsize retokenize + reward};
\node[cpu] (proxy)   at (10.0,0)   {\textbf{LLM API Proxy}\\[1pt]\scriptsize policy under training};
\node[cpu] (rollout) at (10.0,-3.4){\textbf{Rollout API}\\[1pt]\scriptsize collect trajectory};
\node[cpu]   (harness) at (15.0,0) {\textbf{Harness}\\[1pt]\scriptsize agent loop};
\node[focus] (sandbox) at (20.0,0){\textbf{Sandbox}\\[1pt]\scriptsize AgentCore:\\tools + checker};
\draw[<->, thick] (trainer) -- node[lbl,above]{weight sync} (infer);
\draw[->, thick]  (infer)   -- node[lbl,above]{serve} (proxy);
\draw[->, thick]  (proxy)   -- node[lbl,above]{turn} (harness);
\draw[<->, thick] (harness) -- node[lbl,above]{tool call / reward} (sandbox);
\draw[->, thick]  (harness) -- node[lbl,pos=0.32]{rollout} (rollout);
\draw[->, thick]  (rollout) -- node[lbl,above]{trajectory} (adapter);
\draw[->, thick]  (adapter) -- node[lbl,above]{samples} (buffer);
\draw[->, thick]  (buffer)  -- node[lbl,left]{sample batch} (trainer);
\begin{scope}[on background layer]
  \node[panel, fill=blue!3,  fit=(trainer)(infer)(buffer)(adapter)] (ptrain) {};
  \node[panel, fill=black!3, fit=(proxy)(rollout)] (pgw) {};
  \node[panel, fill=orange!4, fit=(harness)(sandbox)] (pagent) {};
\end{scope}
\node[ptitle, anchor=south west] at ([yshift=2pt]ptrain.north west) {Training server (GPU) --- slime};
\node[ptitle, anchor=south] at ([yshift=2pt]pgw.north) {API gateway};
\node[ptitle, anchor=south west] at ([yshift=2pt]pagent.north west) {Agent runtime (CPU + sandbox)};
\end{tikzpicture}%
}
\end{center}
```

*Generation lane (top, →); training-data lane (bottom, ←); disaggregated design after Agent Lightning [1], slime [9] as trainer. **AgentCore fills the sandbox seam.***

## The pipeline — five components, one job each

- **Trainer / Inference** (GPU) — learn, and generate turns at the *current* policy (kept fresh by **weight sync**).
- **Proxy** (CPU) — makes every model call hit the *policy under training*, not an external API.
- **Harness** (CPU) — the agent loop that owns one episode end-to-end.
- **Sandbox** (CPU/GPU) — the isolated box untrusted output runs in; the **checker scores the reward**.

**Two lanes, one cycle:**

- **Generation lane (→):** trainer weights → SGLang → proxy → harness turn → sandbox tool call → observation → next turn.
- **Training-data lane (←):** trajectory → reward attached → loss-masked sample → buffer → trainer step → weight sync.

**AgentCore fills the sandbox seam** — ad-hoc + isolated per episode. The harness needs *warmth*, not isolation, so it stays **in-process**.

## slime — the trainer, and its asynchronous rollout

```{=latex}
\begin{center}
\resizebox{0.5\textwidth}{!}{%
\begin{tikzpicture}[
  font=\small,
  box/.style={draw, rounded corners, minimum height=1.2cm, minimum width=3.0cm, align=center, thick, fill=white},
  gpu/.style={box, fill=blue!8},
  cpu/.style={box, fill=black!4},
  focus/.style={box, fill=orange!18},
  lbl/.style={font=\scriptsize, align=center, fill=white, inner sep=1.5pt},
  ptitle/.style={font=\scriptsize\bfseries, text=black!55},
  panel/.style={draw, rounded corners, dashed, thick, inner sep=0.34cm},
  >={Stealth[]},
]
\node[cpu]   (buffer)  at (0,0)      {\textbf{Data Buffer}\\[1pt]\scriptsize prompts + rollout data};
\node[focus] (rollout) at (6.8,0)    {\textbf{Custom rollout gen.}\\[1pt]\scriptsize our harness + AgentCore};
\node[cpu]   (router)  at (6.8,-2.6) {\textbf{SGLang router}};
\node[gpu]   (sg1)     at (5.0,-5.0) {\textbf{SGLang server}\\[1pt]\scriptsize GPU};
\node[gpu]   (sg2)     at (8.6,-5.0) {\textbf{SGLang server}\\[1pt]\scriptsize GPU};
\node[gpu]   (mega)    at (0,-5.0)   {\textbf{Megatron}\\[1pt]\scriptsize training group (GPU)};
\draw[->, thick] ([yshift=4pt]buffer.east) -- node[lbl,above]{send init prompt} ([yshift=4pt]rollout.west);
\draw[->, thick] ([yshift=-4pt]rollout.west) -- node[lbl,below]{recv rollout data} ([yshift=-4pt]buffer.east);
\draw[->, thick] (rollout) -- node[lbl,right]{LLM calls} (router);
\draw[->, thick] (router)  -- (sg1);
\draw[->, thick] (router)  -- (sg2);
\draw[->, thick] (buffer)  -- node[lbl,left]{send train data} (mega);
\draw[->, thick] (mega)    -- node[lbl,above]{update weights} (sg1);
\begin{scope}[on background layer]
  \node[panel, fill=blue!3,   fit=(mega)] (ptrain) {};
  \node[panel, fill=orange!4, fit=(router)(sg1)(sg2)] (proll) {};
\end{scope}
\node[ptitle, anchor=south west] at ([yshift=2pt]ptrain.north west) {Training (Megatron)};
\node[ptitle, anchor=south west] at ([yshift=2pt]proll.north west) {Rollout (SGLang)};
\end{tikzpicture}%
}
\end{center}
```

**slime** [9] = Megatron (training) + SGLang [27] (rollout), bridged by a **Data Buffer**; the agentic loop plugs into its rollout **generation hook** — no fork of the training kernel.

- **Sync mode** alternates train / rollout — each side's GPUs idle while the other runs.
- **Async driver** (`train_async.py`) overlaps them: Megatron trains batch *k* while the rollout side already generates batch *k+1*.
- **Fully-async rollout** (`fully_async_rollout`) keeps a *fixed pool of in-flight episodes across rollout boundaries*, so a step never waits on the slowest trajectory — the **long-tail** problem of agentic RL.
- **Why it matters here:** AgentCore sandboxes have **variable, sometimes long** per-episode latency; async rollout keeps the trainer fed while they run — at the cost of **staleness** (rollouts generated on an older policy version). This project runs the fully-async path (`MODE=B`).

# 3. Foundations of AgentCore Runtime

## AgentCore basics — three concepts, precisely

- **Runtime** — a versioned, containerized host behind an ARN (`CreateAgentRuntime` [D9]). The deployable unit; holds no per-user state.
- **Session** — one episode, addressed by a `runtimeSessionId`. First invoke cold-starts compute; the same id re-routes to that warm compute [D3].
- **microVM** — the unit of isolation: own kernel, CPU, memory, filesystem; sanitized on termination [D2].

**Cardinality is 1 : N : 1** — one Runtime serves many sessions; each session = exactly one microVM. **The number of Runtimes tracks distinct sandbox *images*, not tasks.**

## Concurrency — episodes, sessions & the Runtime together

**`N` concurrent episodes run as `N` session ids on ONE Runtime — not `N` Runtimes.**

- The `generate` hook opens **one session per episode** (`CONC` in flight at once); each session id pins that episode to **its own microVM** (or its own GPU instance).
- Every tool op of an episode carries the *same* `runtimeSessionId`, so it returns to the **same warm microVM** — state persists within the episode, isolation holds across episodes.
- The capacity provider places those `N` sessions for you; **one image (one ARN)** backs them all. Add a **second ARN** only for a different image/substrate (e.g. GPU), then round-robin the pool.

*So concurrency = more sessions, not more Runtimes.*

## Why a microVM (not just a container)

Running **untrusted, model-generated code** — isolation is the whole point: one episode's code can't see another's files, the trainer's weights, or the host, and the box is wiped at session end.

| | Docker container | **AgentCore microVM** | E2B |
|---|---|---|---|
| Boundary | shared host kernel | **own kernel** (Firecracker-class) | own kernel |
| Runs your image? | yes | **your OCI image in a microVM** | Dockerfile templates |
| Managed by | you | **AWS, in your account** (IAM/VPC) | E2B service |
| GPU | host-dependent | **via Instances** | not core |

## Two compute types — CPU microVM vs GPU Instances

**CPU work belongs on the microVM; Instances exist to add a GPU** [D5] (EC2 in your account via a capacity provider). **No cross-session GPU time-slicing** — scale by adding sessions/instances, not by fractioning one (no MIG on L4).

| Property | Serverless microVM | Instances (GPU) |
|---|---|---|
| GPU | none | g4dn/g5/g6/g6e… + inf2 |
| Max session | 8 h | 14 d |
| Isolation | 1 microVM / session | 1 instance / session (**1 GPU**) |
| Filesystem | ephemeral | persistent EBS |
| Agents / session | 1 | many |

**One session = one whole instance** (session→instance is 1:1 — you get the *instance*, not a GPU). Two ids acquire two *separate* instances, so you can't split one multi-GPU box across sessions. To use every GPU of one instance: **one shared session + a self-managed in-instance Ray farm** (`demo_gpu_shared` / `demo_gpu_local`).

## The microVM's resource envelope — three limits that bite

Serverless isolation is cheap, but the microVM is **small and ephemeral**. Three limits shape every image:

| Limit | Value | Why it bites | Work-around |
|---|---|---|---|
| **Image size** | **≤ ~2 GB** (arm64, project-measured) | a CUDA/torch wheel alone is ~2.7 GB → build fails | install heavy wheels **at `CMD`**, not baked (GPU image: ~90 s cold) |
| **Disk** | **ephemeral only** — no persistent volume [D3] | a `git clone`d repo / installed pkg is **wiped at session end** | durable state → S3 / session Memory, or move to **Instances** (EBS) [D5] |
| **vCPU / RAM** | **metered, not sized** (vCPU-hr + GB-hr) [D13] — no fixed knob | a data-heavy checker can exceed the account quota | check **Service Quotas**; big fixed footprint → **Instances** (EC2 spec) |

- **Example:** the g6 sandbox image can't bake torch (>2 GB) → `pip install torch` at `CMD` so the wheel lands in the ephemeral workspace at boot; a big SWE repo + persistent workspace → move to **Instances** (EBS, 14 d).
- **Rule of thumb:** *files-in-a-session* for light deps → *`CMD`-time install* for medium → *Instances* when the footprint is large and **fixed**.

## Building & deploying a Runtime — three ways, one contract

**Contract [D17]:** ARM64 image, serve `0.0.0.0:8080` with `/invocations` + `/ping`. Three ways to put your code on a Runtime (increasing control): the **`@aws/agentcore` CLI** [D16] (zero-Docker zip) → the **`bedrock-agentcore` SDK** [D15] (`@app.entrypoint` implements the contract) → **bring-your-own container** (max control — this project).

```dockerfile
FROM public.ecr.aws/docker/library/python:3.12-slim   # CPU sandbox
RUN pip install --no-cache-dir bedrock-agentcore
COPY agentcore_slime_demo/sandbox_server_agent.py /app/
CMD ["python","-m","agentcore_slime_demo.sandbox_server_agent"]  # serves :8080
```

- **GPU image:** point `LD_LIBRARY_PATH=/usr/lib64` at the injected `libcuda.so.1`, and `pip install torch` **at `CMD`** (not baked) to stay under the **2048 MB** cap.
- **Deploy:** ECR create → `buildx --platform` (arm64 microVM / amd64 GPU) → push → `CreateAgentRuntime` → (GPU) `CreateCapacityProvider` (`g6.xlarge`, VPC, `idleInstanceTimeout=600`) + attach.

## Session lifecycle — launch, idle timer, terminate

```{=latex}
\begin{center}
\resizebox{0.80\textwidth}{!}{%
\begin{tikzpicture}[
  font=\small,
  box/.style={draw, rounded corners, minimum height=1.15cm, minimum width=2.7cm, align=center, thick, fill=white},
  rt/.style={box, fill=blue!8},
  act/.style={box, fill=orange!18},
  idle/.style={box, fill=black!5},
  term/.style={box, fill=red!8},
  lbl/.style={font=\scriptsize, align=center, fill=white, inner sep=1.5pt},
  >={Stealth[]},
]
\node[rt]   (rt)   at (0,0)    {\textbf{Runtime (ARN)}\\[1pt]\scriptsize deployed once\\\scriptsize no per-user state};
\node[act]  (act)  at (5.6,0)  {\textbf{Active session}\\[1pt]\scriptsize microVM = 1 episode};
\node[idle] (idle) at (11.2,0) {\textbf{Idle}\\[1pt]\scriptsize warm, no work};
\node[term] (term) at (11.2,-3.4){\textbf{Terminated}\\[1pt]\scriptsize microVM wiped};
\draw[->, thick] (rt)  -- node[lbl,above]{first invoke:\\cold start} (act);
\draw[->, thick] (act) -- node[lbl,above]{$\sim$15 min\\no invoke} (idle);
\draw[->, thick] (idle) -- node[lbl,right,align=left]{idle reclaim /\\max lifetime /\\failed health} (term);
\draw[->, thick] (act.north east) .. controls +(1.1,1.6) and +(-1.1,1.6) ..
      node[lbl,above]{each invoke resets the idle timer} (act.north west);
\end{tikzpicture}%
}
\end{center}
```

*Launch is **implicit** (first invoke); terminate is **implicit** (idle / max-lifetime / health). A heartbeat ping resets the timer (`demo_cpu_heartbeat`).*

## How to launch & terminate a session — the mechanics

- **Launch = the first `invoke_agent_runtime` on a fresh `runtimeSessionId`.** There is no explicit "start" call; the id (≥33 chars) *is* the launch. The first op cold-starts the microVM (or GPU instance — minutes on a cold GPU); every later op with that id re-routes to the same warm box.
- **Terminate = passive.** No explicit stop on the happy path — the session ends on **idle reclaim (~15 min)**, **max lifetime** (8 h microVM / 14 d instance), or **failed health**. Each invoke resets the idle timer.
- **Consequence in the RL loop:** a **100-row dataset does not launch 100 instances** — only `CONC` episodes are in flight, recycling through rows over steps. **Concurrency multiplies instances, not data.**
- The **weight-sync pause can trip the idle clock** and silently lose a warm session → the resilience levers in §5.
- **Latency: measure, don't assume** — every op is instrumented (`sandbox_ms`, `sglang_gen_ms`).

## Why AgentCore vs. rolling your own sandbox

Every agentic-RL team must solve "run untrusted code, isolated, at scale." The options and the trade:

```{=latex}
\scriptsize
```

| | Self-managed K8s / Firecracker | E2B / Modal (SaaS) | **AgentCore Runtime** |
|---|---|---|---|
| Isolation | microVM (you operate) | microVM (their cloud) | **microVM, per session** |
| In your account / VPC | yes — you build it | **no** — their cloud | **yes** — IAM/SigV4, VPC |
| Bring-your-own container | yes | limited | **yes** (OCI image) |
| GPU sandbox | you wire it | limited / none | **yes, on Instances** |
| Ops burden | **high** (run the fleet) | low | **low** (managed) |
| Session length | you choose | provider cap | 8 h µVM / **14 d Instance** |

```{=latex}
\normalsize
```

*The client point: AgentCore gives SaaS-level "no fleet to run" **without** your code or data leaving your account.*

## Security & isolation — safe to run untrusted output

Every episode executes model-generated code you did not write. AgentCore makes that safe **by construction**:

- **Per-session microVM isolation** — a hardware-virtualized boundary (not a shared container); one episode's code cannot see another's.
- **In your account, in your VPC** — the sandbox runs under **your IAM**, authenticated with **SigV4**; VPC-mode Runtimes keep traffic on **private endpoints** — **nothing leaves your account**.
- **Bring-your-own image** — you own every dependency, user, and filesystem, and control what the sandbox can reach on the network.
- **Ephemeral by default** — a fresh session per episode, reclaimed on idle; no state bleeds across rollouts unless you deliberately snapshot it (`gpu_snapshot`).

*This is the enterprise reason to run agentic RL on AWS rather than a third-party sandbox cloud.*

## Cost & economics — what you pay, and the packing win

- **Billing model:** you pay for sandbox **vCPU-seconds + GB-hours** while a session is live [D13], plus **Instance** hours for GPU sandboxes — not a flat per-sandbox fee. Idle sessions are reclaimed (~15 min), so **warm-but-unused time is bounded**.
- **CPU sandboxes are cheap; the bottleneck is elsewhere:** for cheap-checker families (math, code, SQL, tool-use) the measured sandbox op is **~0.1 s** (p50 123 ms / p95 900 ms) — generation dominates, so you scale inference, not sandboxes.
- **GPU is where cost concentrates — and where packing pays:** KernelBench is sandbox-bound (`exec` p50 **8.5 s** ≫ generation 2.5 s). Isolated needs **N whole g6 at once**; the `gpu_shared` Ray farm packs **N episodes onto one GPU** (`num_gpus=0.5` → 2×), so you provision **one** instance, not N.
- **Primary lever is capacity, not just cost:** on a capacity-starved GPU fleet you often *cannot* launch N instances at all — the farm makes the run *possible*, and cheaper.

# 4. Integrating AgentCore into slime as the sandbox

## How AgentCore sessions integrate into slime — the call path

```{=latex}
\begin{center}
\resizebox{0.80\textwidth}{!}{%
\begin{tikzpicture}[
  font=\small,
  box/.style={draw, rounded corners, minimum height=1.2cm, minimum width=3.0cm, align=center, thick, fill=white},
  gpu/.style={box, fill=blue!8},
  cpu/.style={box, fill=black!4},
  focus/.style={box, fill=orange!18},
  lbl/.style={font=\scriptsize, align=center, fill=white, inner sep=1.5pt},
  ptitle/.style={font=\scriptsize\bfseries, text=black!55},
  panel/.style={draw, rounded corners, dashed, thick, inner sep=0.34cm},
  >={Stealth[]},
]
\node[cpu]   (slime)  at (0,0)     {\textbf{slime}\\[1pt]\scriptsize Data Buffer $\leftrightarrow$ Megatron\\\scriptsize + SGLang router};
\node[focus] (hook)   at (5.4,0)   {\textbf{generate() hook}\\[1pt]\scriptsize run\_episode\\\scriptsize (in-process harness)};
\node[cpu]   (proxy)  at (5.4,3.0) {\textbf{AnthropicAdapter}\\[1pt]\scriptsize proxy $\rightarrow$ SGLang};
\node[focus] (sbx)    at (12.6,0)  {\textbf{AgentCore Runtime}\\[1pt]\scriptsize sandbox microVM\\\scriptsize \texttt{@app.entrypoint}:\\\scriptsize write / exec / read + checker};
\draw[->, thick] ([yshift=5pt]slime.east) -- node[lbl,above]{prompt} ([yshift=5pt]hook.west);
\draw[->, thick] ([yshift=-5pt]hook.west) -- node[lbl,below]{Sample + reward} ([yshift=-5pt]slime.east);
\draw[->, thick] ([xshift=-6pt]hook.north) -- node[lbl,left,pos=0.55]{model turn} ([xshift=-6pt]proxy.south);
\draw[->, thick] ([xshift=6pt]proxy.south) -- node[lbl,right,pos=0.45]{tokens + logprobs} ([xshift=6pt]hook.north);
\draw[->, thick] ([yshift=5pt]hook.east) -- node[lbl,above,pos=0.6]{invoke\_agent\_runtime (op)} ([yshift=5pt]sbx.west);
\draw[->, thick] ([yshift=-5pt]sbx.west) -- node[lbl,below,pos=0.6]{exit\_code / stdout} ([yshift=-5pt]hook.east);
\begin{scope}[on background layer]
  \node[panel, fill=blue!3,   fit=(slime)(proxy)(hook)] (phost) {};
  \node[panel, fill=orange!4, fit=(sbx)]                (psbx)  {};
\end{scope}
\node[ptitle, anchor=south west] at ([yshift=2pt]phost.north west) {Trainer host --- in-process (warm)};
\node[ptitle, anchor=north] at ([yshift=-2pt]psbx.south) {AgentCore session = 1 episode (isolated)};
\end{tikzpicture}%
}
\end{center}
```

*One `generate()` hook, one **`runtimeSessionId` per episode**; each tool call = one SigV4 `invoke_agent_runtime`.*

## Plugging into slime — one hook, no fork

slime exposes the rollout as a **custom generation function**, so the agentic workload plugs in without touching the training kernel:

- `--custom-generate-function-path …generate.generate` — drives the in-process harness; each tool call → one `invoke_agent_runtime` keyed by that episode's **`runtimeSessionId`**.
- **Design choice:** AgentCore is the **sandbox, not the harness**. The harness stays in-process (warm); the sandbox is the isolated per-session microVM. (It *can* host the harness too — `demo_cpu_branching` runs the real Claude Code CLI inside a Runtime.)
- The returned **loss-masked `Sample` + reward** flows straight back into slime's buffer — no fork of Megatron/SGLang.

## Who owns the lifecycle

slime's training kernel controls **neither** the Runtime nor the session.

- **Runtime** (create/delete the ARN) — the **deploy driver**, once before training. slime just gets the ARN.
- **Session** (per episode) — the `generate` hook's sandbox client: **launch implicit** (first invoke cold-starts the microVM); **terminate implicit** (idle reclaim, no explicit stop on the happy path).

Because termination is *passive*, the weight-sync pause can trip the idle clock and lose a warm session — the failure mode the resilience demo (`cpu_heartbeat`) addresses.

## The episode, end to end — the loop

```{=latex}
\begin{center}
\resizebox{0.78\textwidth}{!}{%
\begin{tikzpicture}[
  font=\small,
  box/.style={draw, rounded corners, minimum height=1.15cm, minimum width=2.7cm, align=center, thick, fill=white},
  svc/.style={box, fill=blue!8},
  focus/.style={box, fill=orange!18},
  stage/.style={box, fill=black!4},
  lbl/.style={font=\scriptsize, align=center, fill=white, inner sep=1.5pt},
  ptitle/.style={font=\scriptsize\bfseries, text=black!55},
  panel/.style={draw, rounded corners, dashed, thick, inner sep=0.34cm},
  >={Stealth[]},
]
\node[stage]  (seed)   at (0,0)     {\textbf{Seed}\\[1pt]\scriptsize PROBLEM.md +\\run\_checks.py};
\node[stage]  (ask)    at (4.0,0)   {\textbf{Ask}\\[1pt]\scriptsize model turn\\(parse action)};
\node[stage]  (run)    at (8.6,0)   {\textbf{Run}\\[1pt]\scriptsize exec action +\\checker};
\node[stage]  (reward) at (13.2,0)  {\textbf{Reward}\\[1pt]\scriptsize 1.0 / 0.0 $\rightarrow$\\Sample};
\node[svc]   (proxy)  at (4.0,3.5) {\textbf{Proxy} $\rightarrow$ policy\\[1pt]\scriptsize SGLang (in training)};
\node[focus] (sbx)    at (8.6,-2.8){\textbf{AgentCore sandbox}\\[1pt]\scriptsize tools + checker};
\draw[->, thick] (seed) -- (ask);
\draw[->, thick] (ask)  -- node[lbl,above]{action} (run);
\draw[->, thick] (run)  -- node[lbl,above]{solved} (reward);
\draw[<->, thick] (ask) -- node[lbl,left]{turn} (proxy);
\draw[<->, thick] (run) -- node[lbl,right]{write+exec / stdout} (sbx);
\draw[->, thick] (run.north) .. controls +(0,2.0) and +(0,2.0) ..
      node[lbl,above,pos=0.5]{observation $\rightarrow$ next turn (if unsolved)} (ask.north east);
\begin{scope}[on background layer]
  \node[panel, fill=black!3, fit=(ask)(run)] (ploop) {};
\end{scope}
\node[ptitle, anchor=north west] at ([yshift=-2pt]ploop.south west) {turn loop ($\leq$ max\_turns)};
\end{tikzpicture}%
}
\end{center}
```

*Seed → Ask (proxy) → Run (sandbox + checker) → Reward; unsolved loops back.*

## Step-by-step — one MBPP episode

1. **Deploy (once):** build the arm64 sandbox image → `CreateAgentRuntime` → get the ARN (the deploy driver, not slime).
2. **Launch run:** `run.sh` sets `SANDBOX=agentcore`, `FAMILY`, `CONC`, `EPISODE_MAX_TURNS`, then `ray job submit … train_async.py --custom-generate-function-path …generate.generate`.
3. **Open session:** slime hands a prompt to `generate()`, which rebuilds the `Task` and calls `make_sandbox("agentcore")` → **cold-starts** that episode's microVM (`runtimeSessionId`).
4. **Seed:** `write_file` `PROBLEM.md` + hidden `run_checks.py` into the session.
5. **Turn loop (≤ `EPISODE_MAX_TURNS`):** `policy.complete` via the **proxy** (TITO tokens) → extract `solution.py` → `exec("python3 run_checks.py")` in the sandbox → on failure, feed stdout back.
6. **Reward & learn:** `exit_code == 0` → **reward 1.0/0.0** from the in-sandbox checker → `build_token_stream` → loss-masked `Sample` → buffer → Megatron step → **weight sync** to SGLang.

## Key code (1/2) — the in-process generate hook

slime calls this once per prompt: build the policy at the SGLang router, open **one sandbox per episode**, run the loop, return one loss-masked `Sample`.

```python
# agentcore_slime_demo/generate.py   (--custom-generate-function-path)
async def generate(args, sample, sampling_params, evaluation=False):
    policy  = SGLangPolicy(router_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
                           model_path=args.hf_checkpoint)
    task    = _task_from_sample(sample)            # rebuild Task from row metadata
    sandbox = make_sandbox(os.environ.get("SANDBOX", "agentcore"))
    async with sandbox as sb:                      # one session == one episode
        traj = await run_episode(policy, sb, task, sampling_params)
    tokens, resp_len, loss_mask, logprobs = build_token_stream(traj, tokenizer)
    return Sample(tokens=tokens, response_length=resp_len, reward=traj.reward,
                  loss_mask=loss_mask, rollout_log_probs=logprobs)
```

- Loss mask = **1 on the policy's own tokens, 0 on prompt / observation / tool tokens**.
- `build_token_stream` reuses the **recorded `token_ids` per turn** — no re-tokenization (TITO); returns a *bare* `Sample` (slime's async path chokes on nested lists).

## Key code (2/2) — the sandbox client & the Runtime server

Each `write_file` / `exec` / `read_file` = **one SigV4 `invoke_agent_runtime`** [D8] keyed by the episode's session id; boto3 offloaded with `asyncio.to_thread` so `N` episodes run concurrently (with retries for transient blips + the GPU cold launch).

```python
# client — agentcore_runtime_sandbox.py
def _invoke_sync(self, payload):
    resp = self._client.invoke_agent_runtime(agentRuntimeArn=self._arn,
        runtimeSessionId=self._session_id,        # >=33 chars; one per episode
        payload=json.dumps(payload).encode())
    return json.loads(resp["response"].read().decode())
```

```python
# server (deployed image) — the checker runs HERE, so the microVM is the reward source
app = BedrockAgentCoreApp()
@app.entrypoint
def entrypoint(request):        # dispatch on op: ping / write_file / read_file / exec
    return handle(request)
app.run()                       # serves 0.0.0.0:8080  (/invocations + /ping)
```

**CPU → GPU is one env var** (`SANDBOX=`) — zero change to the loop.

## One image, many tasks & dependencies

**One microVM = one image per session.** Task diversity is absorbed *inside* one generic image, in increasing order of how structural the difference is:

1. **Files, not images** — most CPU families (math `sympy`, code `unittest`, SQL `sqlite3`, tool-use, light CTF) share **one generic Python image**; prompt, seed files, and hidden checker are written into *that episode's session*.
2. **Per-session install** — heavier deps go into the ephemeral workspace: **`CMD`-time** (the GPU image installs the ~2.7 GB torch wheel at boot to stay under the ~2 GB arm64 cap) or **seed-time** `git clone`/`pip`.
3. **Per-capability image** — only a genuine substrate difference (chiefly the **GPU** toolchain: `nvcc` + a real GPU, or a prebuilt dependency set) needs a **second Runtime ARN** (`demo_cpu_imagepool`, the GPU demos).

# 5. The demos

## Find your workload — the demos are client use cases

The 21 demos aren't toys; each stands in for a **real agent a client would train**. Find yours:

- **Coding agents** → `cpu_mbpp` (MBPP repair), `cpu_humaneval` (function synthesis), `cpu_branching` (a real Claude Code CLI agent) — test-driven assistants that iterate against your test runner.
- **Reasoning / math** → `cpu_math`, `cpu_gsm8k`, `cpu_math500`, `cpu_deepscaler` — reasoning-model post-training with a one-line numeric checker.
- **Data & analytics** → `cpu_gretelsql` (text-to-SQL), `cpu_imagepool` (pandas / scipy / sklearn) — agents whose reward is *running the real data stack*.
- **RAG / search** → `cpu_hotpotqa` (multi-hop) — retrieval agents over your corpus.
- **Tool / API use** → `cpu_xlam`, `cpu_mcp` — function-calling and MCP-schema agents for your APIs.
- **GPU kernels / systems** → `gpu_single`, `gpu_shared`, `gpu_local` — agents whose checker compiles and runs CUDA on a real GPU.

*The rest (`harness_scale`, `mixtask`, `customruntime`, `heartbeat`, `snapshot`) show **how to scale, mix, and operate** these in production.*

## The demo set — 21 demos, one design

Policy always **Qwen3-4B**, harness **in-process** (except `cpu_branching`), AgentCore the **sandbox**; only the *substrate* and *task* vary — in **four layers**:

- **6 canonical** — `cpu_mbpp` (MBPP), `cpu_math`, `gpu_single` (KernelBench), `gpu_shared` (Ray farm), `cpu_heartbeat` + `gpu_snapshot` (resilience).
- **8 per-task CPU fan-out** — one real HF dataset each: gsm8k, math500, deepscaler, humaneval, gretelsql, hotpotqa, xlam, mcp.
- **2 scale a new dimension** — `harness_scale` (the driver), `mixtask` (the batch).
- **5 substrate/design variants** — `cpu_branching`, `cpu_imagepool`, `cpu_customruntime`, `gpu_customruntime`, `gpu_local`.

**Honest status — uniform.** Every training demo is **one 10-step run**: 10 train steps · 1 held-out eval · 1 checkpoint · a weight-sync each step (10) — **loop-complete, not convergence**. The four non-training demos are the exceptions: `cpu_heartbeat`/`gpu_snapshot` run offline (no policy), `gpu_customruntime` is a capability probe, and `gpu_local` attempted the run but hit farm instability before a clean finish. *One slide per demo follows the summary.*

## All 21 demos at a glance

```{=latex}
\scriptsize
```

| Demo | Substrate · task | Checker | Status / measured |
|---|---|---|---|
| *canonical (6)* | | | |
| `cpu_mbpp` | CPU µVM · MBPP repair | hidden asserts | 10-step; `eval` 0.56 |
| `cpu_math` | CPU µVM · math words | numeric assert | 10-step; `eval/gsm8k` 0.94 |
| `gpu_single` | GPU g6 · KernelBench | CUDA `allclose` | 10-step; `eval` 0.65 |
| `gpu_shared` | GPU g6 farm · KernelBench | CUDA `allclose` | 10-step (local GPU) |
| `cpu_heartbeat` | CPU µVM · heartbeat | — (no policy) | offline sim |
| `gpu_snapshot` | GPU inst. · snapshot | — (no policy) | offline sim; real recycle unverified |
| *per-task CPU fan-out (8)* | | | |
| `cpu_gsm8k` | CPU µVM · GSM8K | exact int match | 10-step; `eval` 0.91 |
| `cpu_math500` | CPU µVM · MATH-500 | numeric 1e-4 | 10-step; `eval` 0.83 |
| `cpu_deepscaler` | CPU µVM · DeepScaleR | numeric 1e-4 | 10-step; `eval` 0.70 |
| `cpu_humaneval` | CPU µVM · HumanEval | row `check()` test | 10-step; `eval` 0.81 |
| `cpu_gretelsql` | CPU µVM · Gretel SQL | result-set eq (sqlite3) | 10-step; `eval` 0.65 |
| `cpu_hotpotqa` | CPU µVM · HotpotQA | SQuAD-norm EM | 10-step; `eval` 0.21 |
| `cpu_xlam` | CPU µVM · xLAM-60k | arg-norm call match | 10-step; `eval` 1.00 |
| `cpu_mcp` | CPU µVM · ToolACE (MCP) | arg-norm call match | 10-step; `eval` 1.00 |
| *scale a dimension (2)* | | | |
| `cpu_harness_scale` | CPU µVM · DeepScaleR (fixed) | numeric assert | offline sweep + 10-step; `eval` 0.33 |
| `cpu_mixtask` | CPU µVM · math+SQL+CTF+tool | per-row checker | 10-step; `eval/mixed` 0.67 |
| *substrate / design variants (5)* | | | |
| `cpu_branching` | CPU CLI Runtime · SWE + Claude Code CLI | SWE checker | 10-step; fan-out ≥ 1, reward 0 |
| `cpu_imagepool` | CPU µVM ×3 · datasci/scipy/sklearn | per-family checker | 10-step; `eval/mixed` 1.00 |
| `cpu_customruntime` | CPU µVM · DeepScaleR, both backends | numeric assert | 10-step; `eval` 0.69 |
| `gpu_customruntime` | GPU inst. · torch-on-CUDA probe | `cuda.is_available()` | capability probe; no run |
| `gpu_local` | local whole-GPU farm · KernelBench | CUDA `allclose` | run attempted; real rewards, no clean success |

*Every training demo = one **10-step run** (10 train · 1 eval · 1 ckpt · 10 weight syncs) — loop-complete, not convergence.*

```{=latex}
\normalsize
```

## What the 21 demos demonstrate — in four layers

- **6 canonical — the core claim.** `cpu_mbpp` (MBPP) and `gpu_single` (KernelBench) are the two reference training loops, on a CPU and a GPU sandbox; `cpu_math` adds reasoning; `gpu_shared` packs many episodes onto one GPU; `cpu_heartbeat` + `gpu_snapshot` are the resilience levers. *Shows the loop works on both substrates and survives real faults.*
- **8 per-task CPU fan-out — breadth across real workloads.** One real Hugging Face dataset each — GSM8K, MATH-500, DeepScaleR, HumanEval, Gretel SQL, HotpotQA, xLAM, ToolACE — spanning math, code, SQL, RAG, and tool-use. *Shows the same task-agnostic sandbox + checker covers every frontier family, with no image change.*
- **2 scale a new dimension.** `harness_scale` makes the `min()` throughput law concrete (scale the rollout driver); `mixtask` trains one generalist across families in a single stream. *Shows how you scale and how you mix — not just what you run.*
- **5 substrate / design variants — the AgentCore mechanics.** `cpu_branching` (real Claude Code CLI as harness), `cpu_customruntime` / `gpu_customruntime` (managed vs. your own Runtime), `cpu_imagepool` (capability-routed image pool), `gpu_local` (whole-GPU farm). *Shows the substrate choices — BYO container, GPU, image routing — a production deployment actually makes.*

## `cpu_mbpp` — MBPP code repair · CPU microVM · 10-step run

**Substrate** CPU µVM · **Data** MBPP 500/8 · **Checker** hidden asserts, 1.0 iff exit 0 · **Use case** unit-test-driven coding assistant.

- **Measured (10-step run):** `raw_reward` 0.5–0.94, `eval/mbpp` 0.56 @ step 9; `step_time` 23–78 s, `wait_time_ratio` 0.39–0.84, MFU 0.018–0.043 (2–4 % of L40S peak).
- **Latency:** sandbox p50 123 ms / p95 900 ms; SGLang decode p50 1.76 s / p95 4.1 s.
- **Bottleneck:** generation (1.76 s) ≫ sandbox (0.12 s) → **generation-bound**, scale inference. The reference CPU demo the eight fan-outs clone.

## `cpu_math` — math word problems · CPU microVM · 10-step run

**Substrate** CPU µVM · **Data** GSM8K 200/40, MATH-500 160/40, DeepScaleR 400/40 (real HF) · **Checker** numeric assert, 1.0/0.0 · **Use case** reasoning-model math post-training.

- **Measured (10-step run):** `raw_reward` 0.94→1.0; `eval/gsm8k` 0.94 — a *sanity signal* the loop + numeric checker work, not a curve.
- **Bottleneck:** long reasoning traces dominate, checker is milliseconds → **inference-bound**; `[sglang-gen]`, not `[sbx-op]`, sets step time.
- **To convergence:** `NR≈150–200`/split, size `--rollout-max-response-len` to the trace p95, scale SGLang.

## `gpu_single` — KernelBench CUDA · GPU g6 Runtime · 10-step run

**Substrate** remote g6 (L4) custom Runtime; trainer keeps all 8 L40S · **Data** KernelBench 100 L1 / 20 L2 · **Checker** rebuild ref on CUDA, matched seeds, `allclose` + speedup · **Use case** CUDA kernel-optimization agent.

- **Measured (10-step run):** `raw_reward` 0.28–0.94, `eval/kernelbench` 0.65 (csv)/0.72 (log); `step_time` 110–804 s, `wait_time_ratio` 0.79–0.96 (mostly waiting on the remote checker).
- **Latency (remote g6):** `exec` p50 8.5 s / p95 17.3 s; first `ping` ~123 s (one-time g6 cold launch); SGLang decode p50 2.5 s / p95 6.4 s.
- **Bottleneck:** `exec` (8.5 s) ≫ generation (2.5 s) → **sandbox-bound**, scale the sandbox (more g6, or the farm). Re-verified on hardware via `LOCAL_GPU` (`Job succeeded`, ckpt 2).

## `gpu_shared` — pack many episodes onto one GPU · 10-step run (local GPU)

**Diff from `gpu_single`:** one line — `SANDBOX=agentcore-ray-farm`. All episodes invoke **one** GPU Runtime under a shared `runtimeSessionId`; inside, `_handle_farm` leases one **`num_gpus=0.5`** Ray actor per episode (per-episode `sub_session`). **Use case** GPU packing on a capacity-starved fleet.

- **Measured (10-step run, local-GPU fallback):** `Job succeeded` + ckpt 2 (fraction 0.5 → 2 warm-CUDA actors); the client-side open-concurrency gate capped opens at the farm's 2 slots — **10 opens succeeded, 0 refused**.
- **Why:** isolated needs *N whole g6 at once* (stalls on a starved fleet); the farm needs **one** instance. No MIG on L4 → software-only isolation (`nvcc` `flock`). No convergence curve yet.

## `cpu_heartbeat` — heartbeat keeps the sandbox warm · offline

**Mechanism, not a training run.** During slime's weight-sync pause generation stalls and an idle session is reclaimed (~15 min); a `SessionKeeper` heartbeat resets the idle clock so the *same* warm microVM survives. **Use case** any long RL run whose warm, stateful session must outlive the sync pause.

- **Three acts:** (1) naive "stop invoking" → reclaimed; (2) heartbeat holds it warm across an arbitrary pause; (3) recover-by-reseed if it dies anyway.
- Lives **outside** the harness by design; runs offline in seconds (no policy). Paired with `gpu_snapshot`: *heartbeat prevents loss; snapshot recovers from it.*

## `gpu_snapshot` — application-level snapshot / resume · offline

**Mechanism, not a training run.** A GPU instance can be recycled and come back blank (AgentCore has no VM/EBS snapshot), so workspace state is serialized to a store you own (S3 in prod) and restored onto a fresh session. **Use case** surviving an instance recycle mid-episode in a long GPU run.

- **A/B arc:** Run A (no snapshot) redoes from step 0; Run B restores and prints the rework saved.
- **Status — honest:** the **offline sim** is committed (~1 s, `LocalSnapshotStore`); the `REAL_SNAPSHOT=1` on-hardware recycle is a **documented recipe, not yet verified** (`snapshots/` is empty).

## `cpu_gsm8k` — grade-school math · 10-step run

**Family** math · **Data** GSM8K (`openai/gsm8k`) 200/40 · **Checker** exact integer match, 1.0/0.0 · **Use case** grade-school word-problem solver.

- **Measured (10-step run):** `eval/gsm8k` **0.91** — the base policy already solves most of this easy split, so it is the fan-out's *sanity anchor*; its curve should flatten fast.
- **Bottleneck:** generation-bound (cheap checker) → scale inference, never sandboxes.

## `cpu_math500` — competition-style math · 10-step run

**Family** math · **Data** MATH-500 (`HuggingFaceH4/MATH-500`) 160/40, `EPISODE_MAX_TURNS=4` · **Checker** numeric match (tol 1e-4) · **Use case** competition-math solver.

- **Measured (10-step run):** `eval/math500` **0.83** — harder than GSM8K, so more headroom to climb.
- **Note:** its longer reasoning traces make `--rollout-max-response-len` the first knob to size up. Generation-bound.

## `cpu_deepscaler` — RL-scale math (largest split) · 10-step run

**Family** math · **Data** DeepScaleR (`agentica-org/DeepScaleR-Preview-Dataset`) 400/40 — the largest fan-out split · **Checker** numeric match (tol 1e-4), `EPISODE_MAX_TURNS=4` · **Use case** RL-scale math post-training.

- **Measured (10-step run):** `eval/deepscaler` **0.70** — lowest of the math three (hardest cold), the most room to show learning.
- Also the fixed workload `cpu_harness_scale` reuses. Generation-bound.

## `cpu_humaneval` — function-synthesis coding · 10-step run

**Family** coding · **Data** HumanEval (`openai/openai_humaneval`) 140/24 · **Checker** the row's own `check()` unit test, run in the sandbox · **Use case** function-synthesis coding assistant.

- **Measured (10-step run):** `eval/humaneval` **0.81** — same shape as `cpu_mbpp` (MBPP) but with the canonical HumanEval tests (real code execution, not a numeric assert).
- Generation-bound.

## `cpu_gretelsql` — text-to-SQL · 10-step run

**Family** sql · **Data** Gretel synthetic text-to-SQL (`gretelai/synthetic_text_to_sql`) 300/40 · **Checker** result-set equality — query executed against **`sqlite3`** (stdlib), rows vs gold · **Use case** text-to-SQL analytics agent.

- **Measured (10-step run):** `eval/gretelsql` **0.65**.
- **Cheapest checker of all** (in-process SQLite query) → textbook generation-bound.

## `cpu_hotpotqa` — multi-hop RAG · 10-step run

**Family** rag · **Data** HotpotQA (`hotpotqa/hotpot_qa`, distractor; 10 passages/question) 300/40 · **Checker** SQuAD-normalized exact match · **Use case** multi-hop retrieval / search agent.

- **Measured (10-step run):** `eval/hotpotqa` **0.21** — the **hardest cold** of the eight, so the most interesting convergence target.
- **Bottleneck:** the mild exception — 10-passage observations grow the *context* each turn, but the checker is trivial → generation/**context**-bound, still not sandbox-bound.

## `cpu_xlam` — function-calling / tool use · 10-step run

**Family** tool use · **Data** xLAM-60k (`minpeter/xlam-function-calling-60k-parsed`) 300/40 · **Checker** arg-normalized gold-call match (order-insensitive), RLVR · **Use case** function-calling tool agent.

- **Measured (10-step run):** `eval/xlam` **1.00** — the base policy nails this slice, so the value is verifying the **tool-call parsing/checker path** end-to-end, not showing a climb.
- Generation-bound.

## `cpu_mcp` — MCP-style tool schemas · 10-step run

**Family** tool use · **Data** ToolACE (`Team-ACE/ToolACE`, MCP-shaped schemas) 300/40 · **Checker** arg-normalized gold-call match, RLVR · **Use case** MCP tool-schema agent.

- **Measured (10-step run, `eval-interval 2`):** `eval/mcp` **1.00** on the two-rollout held-out set — a *baseline solve rate*: read it as "the MCP-style tool loop + checker run end-to-end," not a learned gain.
- Same family as `xlam`, against MCP-shaped tool definitions. Generation-bound.

## `cpu_harness_scale` — scale the rollout driver · offline sweep + 10-step run

**Fixes task + sandbox, scales the *harness*.** A pool of `W` warm in-process workers (`HARNESS_WORKERS`, default 4) over one `asyncio.Queue`, on DeepScaleR (400/40), 2 CPU sandbox ARNs round-robined. **Use case** lifting the easily-forgotten third dial once inference + sandbox are fast.

- **Offline sweep (illustrates the `min()` law):** throughput 5.3 / 10.5 / 20.9 / 29.6 / 36.5 ep/s at `W`=1/2/4/6/8 while **harness-bound**, then **plateaus ~37 ep/s** once **inference-bound** (`W`=12 → 36.6). 48/48 solved. Unseeded sim — shape, not absolute rate.
- **Loop-verified on hardware:** the real slime integration reaches `Job succeeded` at ckpt 10 (the standard 10-step run), `eval/deepscaler` 0.33.

## `cpu_mixtask` — one interleaved generalist stream · 10-step run

**Substrate** CPU µVM, MODE B fully-async · **Data** `build_mixed_datasets.py` interleaves **math + SQL + CTF + tool-use** into one 174-row `mixed_train.jsonl` · **Checker** each row is self-describing (its own family + hidden checker) · **Use case** post-training **one generalist policy** across skills every batch (the frontier norm).

- **Why nothing changes:** the sandbox is task-agnostic and each row rebuilds its own `EpisodicTask`; GRPO normalizes advantage **within a same-prompt group**, so families' reward scales never clash. Still **one Runtime, many sessions**.
- **Measured (10-step run):** `Job succeeded` at ckpt 10, **`eval/mixed` 0.67** (single aggregate metric; per-family breakdown is the open work). Generation-bound.

## `cpu_branching` — real Claude Code CLI as harness · 10-step run

**The only demo where the harness is the real Claude Code CLI**, co-located with slime; its **tools run in a remote CPU Runtime** (image bakes node22 + `claude`). One episode fans out (subagents + context summarization) into a `list[Sample]`. **Use case** RL on a real production coding agent whose episodes spawn subagents and compact context.

- **Reconciliation:** all leaves tagged with the shared `rollout_id ρ`; advantage baseline + token-mean loss normalized **per rollout, not per sample**, so an incidental fan-out can't reweight the update. `MODE=A`, `dp=1` (TP2×PP2), sync save.
- **Measured (10-step run):** `Job succeeded` at ckpt 10; **`fan_out=0` blocker resolved** (`=1` ×315, `=2` ×8, `=3` ×1) — but `raw_reward` and `eval/swe` are still **0.0** (the SWE checker earns no reward yet: the residual open item).

## `cpu_customruntime` — managed CI vs. your own Runtime · 10-step run

**Both ways to get a CPU sandbox** expose the *identical* `write_file`/`exec`/`read_file` contract: the **managed Code Interpreter** (`SANDBOX=agentcore`, zero-ops) vs **your own custom Runtime** (`SANDBOX=agentcore-runtime`, your OCI image, VPC/IAM posture). **Use case** deciding whether a rollout needs a custom image at all.

- `compare_backends.py` still runs the same episode on both — **reward identical, only operational trade-offs differ** — but `run.sh` now drives a real DeepScaleR training run on the custom Runtime.
- **Measured (10-step run):** `Job succeeded` at ckpt 10, **`eval/deepscaler` 0.69** (per-rollout `raw_reward` 0.72–0.91).

## `cpu_imagepool` — capability-routed image pool · 10-step run

**One prebuilt image per capability**, each its own Runtime ARN — `datasci`→pandas, `scipy`→scipy, `sklearn`→sklearn — and the harness **routes each episode to the image its family needs**. **Use case** a portfolio whose families pull heavy, incompatible toolkits that can't co-install under the ~2 GB cap.

- Routing is not cosmetic: send a `scipy` episode to the pandas image and `import scipy` fails → reward 0. The third rung of the dependency ladder (files → CMD-install → **per-capability ARN**).
- **Measured (10-step run):** `Job succeeded` at ckpt 10 driving the **real 3-image ARN pool** (per-family routing confirmed in the runtime-env), **`eval/mixed` 1.00**.

## `gpu_customruntime` — a GPU under the sandbox · capability probe

**The managed Code Interpreter has no GPU.** If a checker needs CUDA, the sandbox must be **your own custom Runtime on an AgentCore capacity-provider Instance** (a g6 box). The demo `write_file`s a torch-on-CUDA probe, `exec`s it, `read_file`s the result. **Use case** confirming a CUDA-dependent reward can only be scored on the Instances substrate.

- **Measured (probe):** `torch.cuda.is_available()` **False on managed CI** vs **True on the custom Runtime** (`torch 2.11+cu129`, `device: NVIDIA L40S`).
- **Status:** capability probe only — the `LOCAL_GPU` training attempt was `Killed`; no committed training run. States the rule: one session = one whole Instance, no cross-session GPU time-slicing.

## `gpu_local` — KernelBench on a local whole-GPU farm · run attempted

**The whole-GPU counterpart to `gpu_shared`:** the same Ray-farm Runtime served **locally** on the box's reserved GPUs 6,7 with **fraction 1.0** — a whole dedicated GPU per episode (`floor(2/1.0)=2` actors); slime keeps the other six. **Use case** GPU packing when a whole GPU per episode *is* available locally and the fractional compile-wedge risk isn't worth it.

- Runtime config confirmed: `SANDBOX=agentcore-ray-farm`, fraction 1.0, GPUs 6,7, open-concurrency capped at 2, shared `runtimeSessionId`.
- **Status — honest:** produced **real KernelBench rewards** (`raw_reward` 0.59–0.81) but **neither attempt reached a clean success** — both hit repeated farm-endpoint `Connection refused` (the local farm container dies under load) before a final checkpoint.

# 6. Cross-cutting mechanisms

## Episode shape decides what you scale

The task's shape — not the domain — sets the config. **One microVM serves many families**: the sandbox is task-agnostic (`write_file` / `exec` / `read_file`); task specificity is *files seeded per episode*, not a new image.

| Regime | Example | Bottleneck | Scale |
|---|---|---|---|
| single-turn, long trace | math, reasoning | **generation** | inference |
| few-turn, heavy checker | KernelBench (CUDA) | **sandbox** | sandbox / GPU |
| multi-turn, big observations | SWE, RAG, web | context growth | both + warm session |

- **`--rollout-max-response-len`** — per-turn generation budget; set from the family's p95 per-turn response (math 1k–32k; code 100–800; CUDA kernel 200–1k; tool call 50–300).
- **`EPISODE_MAX_TURNS`** (default 3) — the single biggest lever on episode cost: single-turn → 1; iterative code/SQL → 3–5; long-horizon SWE/web → 15–50+.

## Mechanism spotlight — TITO & why the proxy exists

A harness works in **text**; RL trains on **token IDs**. Re-tokenizing decoded text can drift from what the model actually sampled — corrupting the action, the importance ratio, and the loss mask.

```{=latex}
\begin{center}
\resizebox{0.82\textwidth}{!}{%
\begin{tikzpicture}[
  font=\small,
  box/.style={draw, rounded corners, minimum height=1.15cm, minimum width=2.8cm, align=center, thick, fill=white},
  svc/.style={box, fill=blue!8},
  focus/.style={box, fill=orange!18},
  cpu/.style={box, fill=black!4},
  lbl/.style={font=\scriptsize, align=center, fill=white, inner sep=1.5pt},
  >={Stealth[]},
]
\node[cpu]   (harness) at (0,0)     {\textbf{Harness}\\[1pt]\scriptsize agent loop (text)};
\node[focus] (proxy)   at (5.5,0)   {\textbf{Proxy}\\[1pt]\scriptsize LLM API $\rightarrow$ policy};
\node[svc]   (infer)   at (11.0,0)  {\textbf{Inference engine}\\[1pt]\scriptsize SGLang (GPU)};
\node[cpu]   (store)   at (5.5,-2.7) {\textbf{Rollout store}\\[1pt]\scriptsize per \texttt{session\_id}};
\node[cpu]   (adapter) at (11.0,-2.7){\textbf{Sample Adapter}\\[1pt]\scriptsize + reward $\rightarrow$ trainer};
\draw[->, thick] ([yshift=4pt]harness.east) -- node[lbl,above]{POST /v1/messages (text)} ([yshift=4pt]proxy.west);
\draw[->, thick] ([yshift=-4pt]proxy.west)  -- node[lbl,below]{Anthropic-style reply} ([yshift=-4pt]harness.east);
\draw[->, thick] ([yshift=4pt]proxy.east) -- node[lbl,above]{prompt\_ids} ([yshift=4pt]infer.west);
\draw[->, thick] ([yshift=-4pt]infer.west) -- node[lbl,below]{response\_ids, logprobs} ([yshift=-4pt]proxy.east);
\draw[->, thick] (proxy) -- node[lbl,right]{record exact tokens} (store);
\draw[->, thick] (store) -- node[lbl,above]{on episode end} (adapter);
\end{tikzpicture}%
}
\end{center}
```

**Token-in-token-out (TITO):** the proxy records the **exact sampled `token_ids` + logprobs** at generation time and forwards them intact, so the trainer learns from the tokens that were really sampled.

- The proxy is the only component straddling both sides (text ↔ tokens), so it is the only correct place to capture them.
- For binary RLVR rewards this token-level exactness is not optional — slop silently corrupts the signal.

## The scaling demo (1/2) — the concrete setup & the three tiers

On the 8×L40S box the demos **fix trainer, inference, and harness** and change only the sandbox ARN + task — the reference-architecture disaggregation made concrete:

| Component | Where | Config (shared) |
|---|---|---|
| **Trainer** | 4 GPUs | Megatron, GRPO, `global_batch_size=32`, `ACTOR_GPUS=4` |
| **Inference** | 4 GPUs | SGLang, 2 engines, one **cache-aware router**, `ROLLOUT_GPUS=4` |
| **Harness** | in-process (CPU) | `run_episode`, fully-async `MODE=B`, `CONC=4`, `EPISODE_MAX_TURNS=3` |
| **Sandbox** | **remote AgentCore** | CPU microVM (arm64) **or** GPU `g6` (amd64); one session/episode |

```{=latex}
\begin{center}
\resizebox{0.6\textwidth}{!}{%
\begin{tikzpicture}[
  font=\small,
  box/.style={draw, rounded corners, minimum height=1.0cm, minimum width=2.5cm, align=center, thick, fill=white},
  gpu/.style={box, fill=blue!8},
  cpu/.style={box, fill=black!4},
  focus/.style={box, fill=orange!18},
  wrk/.style={draw, rounded corners, thick, fill=green!10, minimum height=0.72cm, minimum width=2.3cm, align=center, font=\scriptsize},
  lbl/.style={font=\scriptsize, align=center, fill=white, inner sep=1.5pt},
  ptitle/.style={font=\scriptsize\bfseries, text=black!55, align=center},
  panel/.style={draw, rounded corners, dashed, thick, inner sep=0.34cm},
  >={Stealth[]},
]
\node[wrk] (w1) at (0,0.95) {worker 1 $\times$ CONC};
\node[wrk] (w2) at (0,0.05) {worker 2 $\times$ CONC};
\node[font=\scriptsize] (wd) at (0,-0.7) {$\vdots$};
\node[wrk] (wN) at (0,-1.4) {worker $N$ $\times$ CONC};
\begin{scope}[on background layer]
  \node[panel, fill=green!4, fit=(w1)(w2)(wN)] (harness) {};
\end{scope}
\node[ptitle, anchor=south] at ([yshift=2pt]harness.north) {Harness --- replicate\\warm workers (pull / shard, no LB)};
\node[cpu] (router) at (5.4,2.7) {\textbf{SGLang router}\\[1pt]\scriptsize cache-aware};
\node[gpu, minimum width=2.1cm] (sgA) at (10.0,3.8) {\scriptsize replica A};
\node[gpu, minimum width=2.1cm] (sgB) at (10.0,2.7) {\scriptsize replica B};
\node[gpu, minimum width=2.1cm] (sgC) at (10.0,1.6) {\scriptsize replica C};
\node[cpu] (disp) at (5.4,-2.7) {\textbf{Sandbox dispatcher}\\[1pt]\scriptsize round-robin, sticky};
\node[focus, minimum width=2.1cm] (sbA) at (10.0,-1.6) {\scriptsize sandbox A};
\node[focus, minimum width=2.1cm] (sbB) at (10.0,-2.7) {\scriptsize sandbox B};
\node[focus, minimum width=2.1cm] (sbC) at (10.0,-3.8) {\scriptsize sandbox C};
\draw[->, thick] (harness.east) -- node[lbl,above,pos=0.6]{model turns} (router.west);
\draw[->, thick] (harness.east) -- node[lbl,below,pos=0.6]{tool ops} (disp.west);
\draw[->, thick] (router) -- (sgA);
\draw[->, thick] (router) -- node[lbl,pos=0.45]{by prompt prefix} (sgB);
\draw[->, thick] (router) -- (sgC);
\draw[->, thick] (disp) -- (sbA);
\draw[->, thick] (disp) -- node[lbl,pos=0.45]{by session id} (sbB);
\draw[->, thick] (disp) -- (sbC);
\begin{scope}[on background layer]
  \node[panel, fill=blue!3,   fit=(router)(sgA)(sgC)] (pinf) {};
  \node[panel, fill=orange!4, fit=(disp)(sbA)(sbC)]   (psbx) {};
\end{scope}
\node[ptitle, anchor=south west] at ([yshift=2pt]pinf.north west) {Inference --- cache-aware router (prefix affinity)};
\node[ptitle, anchor=north west] at ([yshift=-2pt]psbx.south west) {Sandbox --- round-robin by session id, sticky};
\end{tikzpicture}%
}
\end{center}
```

*Each tier balances load by a **different rule** — and none needs a load balancer you build.*

## The scaling demo (2/2) — the min() law & `cpu_harness_scale`

**The min() law — `throughput = min(inference, sandbox, harness)`.** Scale two and the bottleneck moves to the third.

- **Diagnose first:** compare `perf/sglang_gen_ms_*` vs `perf/sandbox_ms_*`. Generation dominates → add SGLang replicas; sandbox dominates → add sessions / ARNs / the GPU farm.
- **`cpu_harness_scale`** targets the **easily-forgotten third dial**: the warm in-process harness (GIL, `to_thread` pool, FDs). Scale it by **replicating warm workers** (`N × S`), not by making it ad-hoc. **Problem targeted:** rollout driver becomes the bottleneck once inference and sandbox are both fast.
- **Load-balancing differs by side:** inference → target slime's cache-aware router (prefix affinity); sandbox → the **session id *is* the balancer** (sticky round-robin across a pool of ARNs). You never stand up your own load balancer.

## Mechanism spotlight — one episode → many samples

The reconciliation `cpu_branching` implements, and the reason a **real** agent needs it. A production agent (Claude Code CLI) **fans one episode out into many samples** — it spawns subagents (the `Task` tool) and summarizes context (resetting the token prefix). Scored naively per-sample, that fan-out silently reweights the GRPO baseline and the loss [1].

```{=latex}
\begin{center}
\resizebox{0.66\textwidth}{!}{%
\begin{tikzpicture}[
  font=\small,
  box/.style={draw, rounded corners, minimum height=1.05cm, minimum width=2.2cm, align=center, thick, fill=white},
  cpu/.style={box, fill=black!4},
  focus/.style={box, fill=orange!18},
  call/.style={draw, rounded corners, thick, fill=blue!8, minimum height=0.72cm, minimum width=1.15cm, align=center, font=\scriptsize},
  smp/.style={draw, rounded corners, thick, fill=green!10, minimum height=0.9cm, minimum width=2.1cm, align=center, font=\scriptsize},
  lbl/.style={font=\scriptsize, align=center, fill=white, inner sep=1.5pt},
  op/.style={font=\scriptsize\itshape, text=black!60, align=center},
  panel/.style={draw, rounded corners, dashed, thick, inner sep=0.32cm},
  >={Stealth[]},
]
\node[cpu]   (proxy)   at (-3.6,0.1) {\textbf{LLM Proxy}\\[1pt]\scriptsize records\\per-call tokens};
\node[focus] (sandbox) at (15.6,0.1) {\textbf{CLI Runtime}\\[1pt]\scriptsize tools + checker\\$\Rightarrow$ reward $R$};
\node[call] (c1) at (2.2,1.8) {call 1};
\node[call] (c2) at (4.5,1.8) {call 2};
\node[call] (c3) at (8.2,1.8) {call 4};
\node[call] (s1) at (5.6,-1.5) {sub 1};
\node[call] (s2) at (7.8,-1.5) {sub 2};
\draw[->,thick] (c1)--(c2);
\draw[->,thick,dashed] (c2) -- node[op,above]{summarize:\\prefix reset} (c3);
\draw[->,thick,dashed] (c2.south) -- node[op,fill=white,inner sep=2pt,pos=0.5]{spawn\\subagent} (s1.north);
\draw[->,thick] (s1)--(s2);
\begin{scope}[on background layer]
  \node[panel, fill=orange!4, fit=(c1)(c2)(c3)(s1)(s2)] (harness) {};
\end{scope}
\node[op, anchor=south west, font=\scriptsize\bfseries] at ([yshift=2pt]harness.north west) {Harness --- ONE episode (rollout $\rho$): the only whole-episode view};
\draw[<->,thick] (proxy) -- node[lbl,pos=0.6]{turn + tokens} (harness.west);
\draw[<->,thick] (harness.east) -- node[lbl]{tool / reward} (sandbox);
\node[smp] (S1) at (2.6,-4.2) {$S_1$: calls 1--2\\reward $R$,\ id $\rho$};
\node[smp] (S2) at (6.7,-4.2) {$S_2$: sub 1--2\\reward $R$,\ id $\rho$};
\node[smp] (S3) at (10.6,-4.2) {$S_3$: call 4\\reward $R$,\ id $\rho$};
\node[cpu] (adv) at (15.6,-4.2) {\textbf{Advantage + loss}\\[1pt]\scriptsize grouped over\\\emph{rollouts}, not samples};
\draw[->,thick] (c2) -- (S1);
\draw[->,thick] (s2) -- (S2);
\draw[->,thick] (c3) -- (S3);
\draw[->,thick] (S3) -- node[lbl,above]{group by id $\rho$} (adv);
\end{tikzpicture}%
}
\end{center}
```

**Solution:** the CLI runs **inside a CLI-capable CPU Runtime** (image bakes node22 + `claude`); one episode returns a `list[Sample]`, all tagged with the shared `rollout_id ρ` and the episode's terminal reward $R$. **Advantage baseline + loss are normalized per rollout, not per sample** (token-mean loss), so an incidental fan-out can't reweight the update. `MODE=A`, `dp=1` (TP2×PP2), sync save.

*The `cpu_branching` demo (§5) now trains to ckpt 10 with `fan_out ≥ 1` — the mechanism runs end-to-end; its SWE reward is still 0.*

## Mechanism spotlight — resilience: heartbeat & snapshot

Two levers a long RL run composes, demonstrated by `cpu_heartbeat` and `gpu_snapshot` (§5). **Use case:** any multi-hour run whose warm, stateful sessions (installed deps, seeded repo) must survive both the periodic pause and a genuine instance recycle.

- **Heartbeat (`cpu_heartbeat`)** — during weight sync generation stalls; an idle session is **reclaimed (~15 min)** [D3] and the next round pays a cold start. A `SessionKeeper` pings on a timer, **resetting the idle clock**, so the *same* warm microVM survives. Lives **outside the harness** (`KeptSandbox`), because the stall is a training-loop property, not an episode one.
- **Snapshot (`gpu_snapshot`)** — a recycled GPU instance comes back blank (no VM/EBS snapshot), so the workspace is serialized to a store you own (S3) and restored onto a fresh session.
- **The pairing:** *heartbeat prevents loss; snapshot recovers from it.*

## Adoption path & limits — from pilot to production

**Day-1 pilot (a few hours):**

1. Build a sandbox image (`Dockerfile.sandbox.cpu`, arm64) and push it; deploy a Runtime (`CreateAgentRuntime` [D9]).
2. Point slime at it with **one env var** (`SANDBOX=agentcore-runtime`) + the `--custom-generate-function-path` hook — **no fork of slime**.
3. Seed a task family, run the standard 10-step loop, watch `eval/<task>` and the `[sbx-op]` / `[sglang-gen]` latencies.
4. Flip **CPU↔GPU** with one env var when your checker needs CUDA.

**Limits & quotas to plan around:**

- **Session length:** 8 h on a CPU microVM, 14 d on a GPU Instance; **idle ~15 min → reclaimed** (use the `cpu_heartbeat` heartbeat to survive the weight-sync pause).
- **Image size:** ~2 GB arm64 cap — install heavy deps at `CMD`-time, or route a per-capability image (`cpu_imagepool`).
- **GPU sharing:** **no MIG** on L4 / L40S — cross-session GPU sharing is software-only (the Ray farm), not hardware-partitioned.
- **Next step after training:** serve the resulting policy on Bedrock / SageMaker — the trained checkpoint is a standard model artifact.

## Summary

- **Can:** isolated per-session sandbox for untrusted output (CPU default), fully in-account (IAM/SigV4, VPC), long sessions, bring-your-own container, **GPU on Instances**; a one-hook slime integration with the harness in-process.
- **Cannot:** a microVM has no GPU; no cross-session GPU fractioning (no MIG on L4); a session can't exceed 8 h without moving to Instances (14 d).
- **How to run:** build a sandbox image → deploy a Runtime (+ GPU capacity provider) → seed a task → `invoke_agent_runtime` per tool call → in-sandbox checker gives the reward → slime trains via one `generate` hook. **CPU↔GPU is one env var.**

*One image, `N` sessions, one hook: AgentCore is the isolated per-episode sandbox; slime does the learning.*

# References

## References — papers, models & tech reports

- **[1] Agent Lightning** — harnessed agentic RL · <https://arxiv.org/abs/2508.03680>
- **[2] DeepSeek-R1** · <https://arxiv.org/abs/2501.12948>
- **[3] Qwen3 Technical Report** · <https://arxiv.org/abs/2505.09388>
- **[4] Llama-Nemotron** · <https://arxiv.org/abs/2505.00949>
- **[5] Kimi K2** · <https://arxiv.org/abs/2507.20534>
- **[7] KernelBench** · <https://huggingface.co/datasets/ScalingIntelligence/KernelBench>
- **[8] MBPP** · <https://arxiv.org/abs/2108.07732>
- **[9] slime** — the trainer · <https://github.com/THUDM/slime> · docs <https://thudm.github.io/slime/>
- **[27] SGLang / RadixAttention** · <https://arxiv.org/abs/2312.07104> · cache-aware router <https://lmsys.org/blog/2024-12-04-sglang-v0-4/>

## References — RL algorithms & systems (further reading)

- **[13] GRPO (DeepSeekMath)** · <https://arxiv.org/abs/2402.03300>
- **[14] DAPO** · <https://arxiv.org/abs/2503.14476>
- **[15] GSPO** · <https://arxiv.org/abs/2507.18071>
- **[18] AReaL** — asynchronous RL system · <https://arxiv.org/abs/2505.24298>
- **[19] HybridFlow / veRL** · <https://arxiv.org/abs/2409.19256>
- **[22] Truncated Importance Sampling** — train/inference mismatch · <https://fengyao.notion.site/off-policy-rl>
- **[28] APRIL** — active partial rollouts · <https://arxiv.org/abs/2509.18521>

## References — AgentCore docs, SDK & CLI

- **[D2] Host with Runtime** — isolation, compute types · <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html>
- **[D3] Runtime — how it works** — idle 15 m / max 8 h · <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-how-it-works.html>
- **[D5] Instances (GPU)** · <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-instances-how-it-works.html>
- **[D8] InvokeAgentRuntime** · <https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_InvokeAgentRuntime.html>
- **[D9] CreateAgentRuntime** · <https://docs.aws.amazon.com/bedrock-agentcore-control/latest/APIReference/API_CreateAgentRuntime.html>
- **[D13] Observability** — vCPU / GB-hour metrics · <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html>
- **[D15] Python SDK** — `@app.entrypoint` · <https://github.com/aws/bedrock-agentcore-sdk-python>
- **[D16] AgentCore CLI** — `@aws/agentcore` · <https://www.npmjs.com/package/@aws/agentcore>
- **[D17] Runtime service contract** — :8080, /invocations, /ping · <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-service-contract.html>
