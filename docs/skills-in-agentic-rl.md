# Skills in agentic RL on AgentCore — research note

> Status (2026-10-12): Part 2's design is **implemented** for the in-session mini_claude harness
> (`SKILLS=listed`, default `off`): `agentcore_slime_demo/skills.py` + 9 skills in the demo's `skills/`.
> The decoupled Claude Code CLI path (2.3) is not wired yet; the A/B/C runs (2.4) are not done yet.
> Part 1 is general (any AgentCore-sandboxed RL setup). Part 2 maps it onto the live demo,
> [`code/demos/demo_terminalbench_kernelbench`](../code/demos/demo_terminalbench_kernelbench/README.md).

## TL;DR

- A **skill** is a small folder (`SKILL.md` with a `name` + `description` header, plus optional
  scripts or templates) that the agent can **discover cheaply and load on demand**. Only the one-line
  descriptions are in the prompt. The body is read only when the agent decides it needs it
  ("progressive disclosure", the Anthropic Agent Skills format).
- In RL this changes three things: **what is in the context** (more tokens, but they are tool
  results, so loss-masked), **what the policy can learn** (when to load a skill and how to follow
  it), and **what the eval measures** (policy + skill library, not policy alone).
- For our demo the strongest motivation is the **no-gradient problem**: most GRPO groups have all-0
  rewards (39/48 steps in train48 run 2; train reward is flat for the 9B too). Skills that raise
  the pass rate on hard tasks from 0 to "sometimes" create within-group variance, and that is
  where the gradient comes from.
- Recommended first step: **static, hand-written, family-level skills, read with the existing
  `bash` tool** (`cat /opt/acs_skills/<name>/SKILL.md`). No new tool schema, no image rebuild. Run
  an A/B against `SKILLS=off`.

---

## Part 1 — Skills in agentic RL, in general

### 1.1 What a skill is (and isn't)

| | System prompt | Tool | Skill |
|---|---|---|---|
| What it is | always-on instructions | a callable action with a schema | a document (+ files) the agent pulls in when relevant |
| Context cost | full, every turn | schema, every turn | ~1 line until loaded; then the body, once |
| Who executes code | — | the harness | the agent, via its normal tools (e.g. runs `scripts/check.py` with bash) |
| Scales to | a few rules | tens of tools | hundreds of skills (only descriptions are listed) |

The standard format (Anthropic Agent Skills, also what Claude Code loads from `.claude/skills/`):

```
my-skill/
  SKILL.md          # YAML front matter: name, description; then markdown instructions
  scripts/foo.py    # optional, run by the agent with its own shell tool
  templates/...     # optional
```

The harness lists `name: description` for every skill. When the agent decides a skill applies, it
reads `SKILL.md`, and from there any referenced file. Nothing about this needs a special model
feature: a model that can call `bash` can read a skill.

### 1.2 Where skills sit in the RL loop

```
 trainer (slime) ── weights ──► SGLang ◄── /v1/messages ── AnthropicAdapter (records tokens)
                                                                 ▲
                                                                 │ dial-back
 sandbox session (AgentCore microVM / m5 / g6) ──────────────────┘
   ├── harness loop (session_runner.py)
   ├── task environment
   └── skills dir   ◄── written at episode start, removed before grading
```

- **Delivery**: skills are files in the sandbox. You can bake them into the image, or write them at
  episode open. Writing at open lets you change the library without rebuilding images, which
  matters with one AgentCore Runtime per task.
- **Discovery**: the listing goes in the system prompt (or the first user turn).
- **Loading**: the model reads the body with a tool call. In the token stream this is a
  **tool_result**, so it is **loss-masked**: the policy is never trained to reproduce skill text.
  It is only trained on its own actions (deciding to read the skill, then following it).
- **Execution**: skill scripts run **in the sandbox**, the same place as the task. That matters for
  isolation and for grading (see 1.5).

### 1.3 Four ways to use skills in RL (increasing ambition)

1. **Skills as a fixed scaffold.** A hand-written library, frozen for the whole run. RL learns *when*
   to load a skill and *how* to follow it. This is the simplest and most stable option, and the
   right first experiment.
2. **Skill selection as a learned action.** Add a `load_skill(name)` tool, or detect `cat …/SKILL.md`
   in bash. Load decisions become explicit, countable actions in the trajectory, so you can measure
   and reward-shape them (carefully, see 1.5).
3. **Skills as training wheels (internalization).** Train with skills, then evaluate **without**
   them, or anneal how many are listed over training. The goal is a policy that has absorbed the
   procedure. The comparison to report: eval(no skills) after training with skills vs after
   training without.
4. **Evolving skill library.** The agent (or an offline step) writes new skills from successful
   trajectories (Voyager-style skill libraries; "agent workflow memory"), and the library grows
   between steps. This is the most powerful option but also non-stationary: the environment
   changes under the policy. It also interacts badly with fully-async rollout (1.4). Keep it for later.

### 1.4 RL-specific design considerations

- **Context budget.** The listing costs tokens on every turn of every episode, so keep each
  description to about one line and the whole listing short. Bodies should be short too, with
  details pushed into referenced files the agent reads only if needed.
- **Loss masking.** Confirm that skill text enters only as tool_result or prompt, never as assistant
  tokens. Otherwise you are doing SFT on the skill text by accident.
- **Off-policy / staleness.** With fully-async rollout, samples in one train step can come from
  different library versions if the library changes mid-run. **Version the library** (a content
  hash), stamp the version on every sample's metadata, and change it only at epoch boundaries.
  Mode 4 needs this.
- **GRPO groups.** All `n_samples_per_prompt` rollouts of a prompt must see the **same** skill
  set. Otherwise the advantage mixes two conditions. Assign skill conditions **per prompt**, not per
  episode (like `HARNESS_ASSIGN=prompt`).
- **Reward.** Keep the task oracle as the only reward to begin with. A bonus for "loaded a skill"
  invites load-everything spam. If anything, use a tiny token-cost penalty instead.
- **Model priors.** Open models were not trained on a `Skill` tool. Exposing skills through a tool
  the model already uses well (`bash cat`) avoids a cold start on tool-calling.

### 1.5 Pitfalls

- **Answer leakage / reward hacking.** A skill written from a task's reference solution or tests
  turns RL into lookup. Rules: skills are **family-level** (how to debug pytest, how to write a
  `load_inline` kernel), never task-level. Never author them from eval tasks. Never let them
  mention `/tests`, `run-tests.sh` or solution files.
- **Grader contamination.** Some Terminal-Bench oracles inspect the working directory. Keep skills
  **outside the workdir** and delete them before grading, the same way `.acs_h/` is removed.
- **Eval semantics.** "Eval with skills" measures policy + library. Always report both with-skills
  and without-skills eval numbers.
- **Scripts run where the agent's tools run.** If the harness runs in a different machine from
  the sandbox (our decoupled Claude CLI path), a skill script the harness executes locally runs
  in the **wrong machine**. Skills must either be instruction-only or have their scripts placed in the
  sandbox.

---

## Part 2 — How to use skills in `demo_terminalbench_kernelbench`

### 2.1 What the demo has today (the hook points)

| Piece | File | Relevant fact |
|---|---|---|
| System prompt (TB persona) | `agentcore_slime_demo/task_families.py` `TerminalBenchTask.SYSTEM` | set once on the shared harness in `terminalbench_pool.py` (`self._harness.system_prompt = …`) |
| Tools | `agentcore_slime_demo/mini_agent.py` `TOOLS` | `write_file` / `bash` / `read_file` only |
| Loop config | `agentcore_slime_demo/in_session_harness.py` `_run_in_session` | builds `cfg.json` (`system`, `tools`, budgets) per episode, writes `.acs_h/runner.py` |
| Loop | `agentcore_slime_demo/session_runner.py` | stdlib, runs in the sandbox, dials the adapter |
| Episode | `agentcore_slime_demo/terminalbench_pool.py` `_run_episode` | open → `harness.run` → `_grade` (hidden oracle injected after the agent) |
| Cleanup before grade | `in_session_harness.py` | `kill … ; rm -rf .acs_h` |
| Real Claude Code path | `harness_runtime_agent.py` `_run_claude` (`HARNESS_SET=…claude_cli`) | `claude -p` in a separate harness microVM, built-ins disabled, sandbox reached over MCP (`mcp__sandbox__*`) |
| Budgets | `run.sh` | `EPISODE_MAX_TURNS=8`, `EPISODE_MAX_TOKENS=1024`; the 9B packs at 4096 tokens (8192 OOMed) |

Three substrates (AgentCore microVM, m5 compose client, g6 GPU Instance) all speak the same
`write_file` / `exec` protocol, so **one delivery path covers all of them**.

### 2.2 Proposed design (mode 1 + the countable-load part of mode 2)

**Library**: `code/demos/demo_terminalbench_kernelbench/skills/<name>/SKILL.md` (+ scripts/templates),
front matter `name` / `description` / `families`. All are family-level, and each SKILL.md is
< 2,000 chars (the `MINI_AGENT_MAX_TOOL_CHARS` clip in `run.sh`; the loader warns above it).

What the epoch-9b run (2026-10-11, 1,357 KernelBench episodes) says KernelBench needs. The checker
runs its 6 checks independently, so the pass count tells you which failed:

| Outcome | Episodes | Meaning |
|---|---|---|
| reward 1.0 | 30 | correct and >= 1.10x faster |
| reward 0.5 | 685 | correct, not fast enough (median speedup of correct solutions: 1.00x; only 7% reach 1.10x) |
| 0, 1/6 checks | 239 | `ModelNew` fails to build or run |
| 0, 5/6 checks | 179 | values differ (typically different weights: `ModelNew` doesn't re-create `Model`'s parameters in order under the grader's seed) |
| 0, 4/6 checks | 170 | wrong shape or CPU output (each fails two checks) |

Also: the g6 image is `python:3.12-slim` + pip torch 2.5.1/cu124. It has **no nvcc**, so
`load_inline(cuda_sources=…)` cannot build, but Triton (a torch dependency) can. And ~90% of
episodes end on the turn budget.

| Skill | Family | Targets |
|---|---|---|
| `kb-modelnew-contract` | kernelbench | the grader's checks in order + the 3 failure classes above + "a correct copy already earns 0.5" |
| `kb-self-check` | kernelbench | `check.py`: the grader's seeds, checks (device, shape, allclose) and timing on the agent's `solution.py`. The installer saves the prompt's reference `Model` as `reference.py` next to it, so the agent never re-types it (it is in the prompt already, so not a leak) |
| `kb-triton-kernels` | kernelbench | "no nvcc, use Triton" + a fused-elementwise template (`cp` it to `solution.py`, edit one line) |
| `kb-speedup-tactics` | kernelbench | which rewrites beat 1.10x (fuse elementwise chains, epilogue fusion, `addmm`, SDPA, row kernels) and which waste turns |
| `tb-explore-first` | terminalbench | a 1-call survey of the container (files, tools, services) |
| `tb-long-running` | terminalbench | start servers with `nohup … &` / `setsid`, poll readiness, never block a bash call (the play-lord hang, 2026-10-04) |
| `tb-python-env` | terminalbench | `python3 -m pip`, `--break-system-packages`, offline wheels, checking the interpreter |
| `tb-build-from-source` | terminalbench | make/cmake triage: read the first error; an error → fix table |
| `tb-verify-before-stop` | terminalbench | list each concrete requirement, check each with one command, before stopping |

Verified locally (L40S, torch in `slime_trainer`): the Triton template and a reference copy pass
`check.py` (1.00x on ReLU, as expected for one memory-bound op), and a CPU output and wrong weights
are reported as `FAIL device` / `FAIL allclose`.

**Delivery** (no image rebuild):
1. `agentcore_slime_demo/skills.py`: load the library, filter by family (KernelBench rows carry the
   generic `coding` family, so they are recognized by their `kernelbench*` image), render the listing
   (`- <name>: <description> (read /opt/acs_skills/<name>/SKILL.md)`), compute a content hash
   (`skills_version`).
2. In `terminalbench_pool._run_episode` (TB) and `mini_claude_episode.run` (KB), after open and before
   the agent: write the selected skills in ONE exec (a base64 tar that `python3` unpacks) to
   `/opt/acs_skills` (or `/tmp/acs_skills` if `/opt` is not writable), **outside** the task workdir.
   Every file's literal `SKILL_DIR` is replaced by the installed path.
3. Pass the listing **per episode**: add a `system_prompt=` kwarg to `InSessionHarness.run` (and
   `DecoupledCliHarness.run`). **Do not** mutate `self._harness.system_prompt`: the harness object
   is shared by all concurrent episodes and would race.
4. Before `_grade` / `_hidden_grade`: `rm -rf` the skills root.
5. `skills_version` (a content hash) goes into each sample's metadata.
6. `run.sh` forwards the knobs into the Ray job, and `deploy/steps_common.sh` `PASS_RE` forwards
   `SKILLS*` from `5_run.sh` into the container.

**Loading**: through `bash` (`cat /opt/acs_skills/<name>/SKILL.md`). This adds no tool schema,
works with the `qwen3_coder` parser we already use for the 9B, and needs no change to
`session_runner.py`. Loads are counted afterwards from the transcript (commands matching
`/opt/acs_skills/`).

**Knobs** (`run.sh`, all defaults off, so current runs are unchanged):

| Knob | Values | Meaning |
|---|---|---|
| `SKILLS` | `off` (default) \| `listed` | `listed` = write the files + add the listing |
| `SKILLS_SET` | comma list \| `all` | which skills to offer |
| `SKILLS_DIR` | path | library location (default: the demo's `skills/`) |
| `SKILLS_EVAL` | `same` \| `off` | eval with the same library, or without it (internalization check) |

**Metrics** (`[demo]` channel, so they reach `.demo.csv`): `skills_listed=N`, `skill_reads=N`,
`skills_read=a,b`, `skills_version=<hash>`. Analysis: reward conditional on reading each skill,
and the change in the zero-std group count.

**GRPO**: the skill condition is a run-level knob, so every rollout in a group sees the same library
automatically. If a per-prompt A/B within one run is ever wanted, key it on the prompt (like
`HARNESS_ASSIGN=prompt`), never per episode.

### 2.3 Real Claude Code path (`claude_cli`)

Claude Code loads skills natively from `<cwd>/.claude/skills/`, and `_run_claude` already uses a
per-episode `ep_cwd`, so writing the library there is all it takes. It is not quite that simple:
- The CLI runs in the **harness microVM**, not the sandbox. Skills there must be **instruction-only**,
  or their scripts must also be written into the sandbox (`/opt/acs_skills/...`) and the
  `SKILL.md` must say "run it with `mcp__sandbox__bash`".
- The `Skill` tool is not in `_CLAUDE_DISALLOWED`, but `--allowedTools` lists only the three MCP
  tools. Check on hardware whether the CLI exposes skills under this flag set, and update the
  `_CLAUDE_STEER` prompt, which currently says "EXACTLY three tools".
- Do this after the mini_claude A/B shows the skills help. The CLI path is the secondary harness.

### 2.4 Experiment plan

Same seed, same task set (full set, no history filter), Qwen3.5-9B, 20 steps (the length of the
`lr3e-6-9b` run), eval every 10 steps:

| Run | `SKILLS` | `SKILLS_EVAL` | Question |
|---|---|---|---|
| A (baseline) | `off` | — | existing `lr3e-6-9b` / `epoch-9b` numbers can serve if the config matches |
| B | `listed` | `same` | do skills raise reward and cut zero-std groups? |
| C | `listed` | `off` | does any gain survive without the skills (internalization)? |

Primary signals: `eval/kernelbench`, `eval/terminalbench`, the per-step zero-std group count,
`skill_reads` per episode. Secondary: turns per episode and response length, since skills cost
context inside the 4096 packing budget.

Expectations to test: KernelBench should benefit most (`kb-load-inline` + `kb-self-check` attack
the dominant failure mode directly). Terminal-Bench gains will be smaller and task-dependent.

### 2.5 Risks specific to this demo

- **Packing budget.** The 9B is capped at 4096 tokens per packed sequence, and a 1–2k-token skill
  body in a tool result eats a big share of it. Keep bodies short and check the truncation ratio.
- **Oracle interference.** Some TB tests list or inspect directories. `/opt/acs_skills` is outside
  the workdir and removed before grading, but grep the oracles for `/opt` to be sure.
- **m5 compose tasks.** Skills are written into the `client` container only. A skill that assumes
  the sidecars are on localhost would be wrong there.
- **g6 KB sessions.** `kb-self-check` uses the GPU and adds wall time to each episode. Bound its
  runtime inside the script.

### 2.6 Implementation checklist

1. [x] `skills/` library (9 skills) + `skills.py` (loader, family filter, listing, hash, install/uninstall, read counting).
2. [x] `system_prompt=` kwarg on `InSessionHarness.run` (per episode; the harness object is shared).
3. [x] Install before the agent and remove before grading, on both the TB and the KB episode paths.
4. [x] `[demo]` fields `skills_listed` / `skill_reads` / `skills_read` / `skills_version`, plus `skills_version` in sample metadata.
5. [x] `run.sh` knobs (default `SKILLS=off`) + the `PASS_RE` forward + a README section.
6. [ ] 2-step smoke (`data/smoke_skills_train.jsonl`: 2 microVM TB tasks + 2 KB problems), then the A/B/C runs.
7. [ ] Claude Code CLI path (2.3).
