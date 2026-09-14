# Figures — agentcore_slime_deck

34 figures/diagrams extracted from `../agentcore_slime_deck.pdf`, cropped to the
diagram (or table) region and rendered at ~300 DPI PNG for pasting into other decks.

| File | Fig | Slide | Description |
|---|---|---|---|
| `fig01_project-theme-two-roles.png` | 1 | 3 | Project theme — AgentCore fills the two RL roles around a slime trainer |
| `fig02_act-observe-revise-loop.png` | 2 | 5 | Multi-turn act→observe→revise loop (a POMDP) |
| `fig03_harness-control-loop.png` | 3 | 7 | Harness control loop: model ↔ tool |
| `fig04_harness-as-hub-dataflow.png` | 4 | 8 | The harness as hub — generation, action, learning |
| `fig05_sandbox-isolated-exec.png` | 5 | 9 | Agent actions run in the isolated sandbox |
| `fig06_sandbox-isolation-options.png` | 6 | 10 | Sandbox isolation options — container vs. microVM (table) |
| `fig07_agent-lightning-architecture.png` | 7 | 12 | Agent Lightning reference architecture — components & data flow |
| `fig08_scaled-rollout-C-E-C.png` | 8 | 16 | Scaled rollout — C harnesses & sandboxes, shared pool of E engines |
| `fig09_usecase-catalog.png` | 9 | 19 | Use-case catalog (table) |
| `fig10_usecase-concrete-examples.png` | 10 | 20 | Concrete example tasks — prompt → sandbox → reward (table) |
| `fig11_usecase-arch-1-2.png` | 11 | 21 | Use-case architectures 1–2 |
| `fig12_usecase-arch-3-4.png` | 12 | 22 | Use-case architectures 3–4 |
| `fig13_usecase-arch-5-6.png` | 13 | 23 | Use-case architectures 5–6 |
| `fig14_usecase-arch-7-8.png` | 14 | 24 | Use-case architectures 7–8 |
| `fig15_slime-architecture.png` | 15 | 28 | slime architecture — rollout → data buffer → training |
| `fig16_agentcore-two-roles.png` | 16 | 31 | AgentCore's two RL roles — harness (Runtime) & sandbox |
| `fig17_runtime-hosting-harness.png` | 17 | 32 | AgentCore Runtime hosting the RL harness |
| `fig18_disjoint-train-infer-pools.png` | 18 | 35 | Disjoint train/infer GPU pools, decoupled by a buffer |
| `fig19_three-decoupling-seams.png` | 19 | 37 | AL mapped onto slime + AgentCore — the three decoupling seams |
| `fig20_five-scalable-services.png` | 20 | 38 | Five independently-scalable services (table) |
| `fig21_generate-hook.png` | 21 | 39 | The generate hook: harness → adapter → policy + trajectory |
| `fig22_demo-matrix.png` | 22 | 41 | Demo matrix — two harnesses × two sandboxes (table) |
| `fig23_all-rows-have-harness.png` | 23 | 42 | All rows have a harness — they differ in which & where (table) |
| `fig24_demo-a-inprocess-cpu.png` | 24 | 44 | Demo a — in-process harness, CPU sandbox data flow |
| `fig25_demo-b-inprocess-gpu.png` | 25 | 47 | Demo b — in-process harness, remote GPU sandbox data flow |
| `fig26_demo-a2-colocated.png` | — | — | **REMOVED (obsolete)** — demo_a2 (real Claude Code CLI) deleted; slide no longer in the deck. Image file kept but unused. |
| `fig27_demo-b2-two-runtimes.png` | — | — | **REMOVED (obsolete)** — demo_b2 (real CLI, two-Runtime) deleted; slide no longer in the deck. Image file kept but unused. |
| `fig28_demo-a3-two-runtimes-cpu.png` | 28 | 56 | Demo a3 — mini harness (Runtime #1) ⊥ CPU sandbox (Runtime #2) |
| `fig29_demo-b3-two-runtimes-gpu.png` | 29 | 59 | Demo b3 — mini harness (Runtime #1) ⊥ GPU sandbox (Runtime #2) |
| `fig30_colocated-vs-two-runtime.png` | 30 | 62 | In-process harness vs. two-Runtime topology & trade-offs |
| `fig31_per-tier-replication.png` | 31 | 63 | Per-tier replication — C : E : C concurrency & GPU split (table) |
| `fig32_network-posture-per-component.png` | 32 | 67 | Hosting & network posture per component (table) |
| `fig33_vpc-endpoint-checklist.png` | 33 | 68 | VPC-endpoint checklist for a networkMode=VPC Runtime (table) |
| `fig34_session-id-routing.png` | 34 | 70 | Everything keyed by per-episode session_id — routing & isolation |

## Regenerating

Figures were extracted with PyMuPDF (`fitz`): each figure page is cropped to the
union bbox of its vector drawings (diagrams) or its body text (tables), excluding
the slide header, footer, and "Figure N:" caption, then rasterized at a 4× matrix
(~300 DPI).
