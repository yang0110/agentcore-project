#!/usr/bin/env python3
"""Render the 4 demo-catalogue tables as slide-ready PNGs."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

# palette (brand-neutral, light theme, readable on white slides)
HEADER_BG = "#1f3a5f"
HEADER_FG = "#ffffff"
ROW_A = "#ffffff"
ROW_B = "#eef2f7"
GRID = "#c9d3df"
TEXT = "#1a1a1a"


def render(filename, title, columns, rows, col_widths):
    n_rows = len(rows) + 1  # + header
    row_h = 0.52
    title_h = 0.75
    fig_h = row_h * n_rows + title_h
    fig_w = 13
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off")

    # title sits in the top band; table fills the rest
    title_frac = title_h / fig_h
    ax.text(0.0, 1.0, title, transform=ax.transAxes, fontsize=19,
            fontweight="bold", color=TEXT, ha="left", va="top")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colWidths=col_widths,
        cellLoc="left",
        bbox=[0, 0, 1, 1 - title_frac],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.5)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color=HEADER_FG, fontweight="bold", fontsize=13)
            cell.set_height(cell.get_height() * 1.05)
        else:
            cell.set_facecolor(ROW_A if r % 2 else ROW_B)
            cell.set_text_props(color=TEXT)
            if c == 0:
                cell.set_text_props(color=TEXT, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT, filename)
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


# ---- Table 1: Canonical ----
render(
    "1_canonical.png",
    "Canonical — the core story",
    ["#", "Demo", "Use case it demonstrates"],
    [
        ["1", "demo_cpu_datasets", "Reference — RL on verifiable tasks in a CPU microVM; DATASET= picks the data (default MBPP)"],
        ["2", "demo_gpu_single", "GPU workload in the sandbox (KernelBench — writing / optimizing CUDA kernels)"],
        ["3", "demo_gpu_farm", "GPU packing on a local Ray farm — FARM=fractional (2 episodes, 1 GPU) | whole (1 GPU each)"],
        ["4", "demo_cpu_heartbeat", "Long-running / offline resilience — keeping a session alive via heartbeat outside the harness"],
        ["5", "demo_gpu_snapshot", "Offline resilience for GPU — snapshot / restore of sandbox state"],
    ],
    [0.04, 0.20, 0.76],
)

# ---- Table 2: Real datasets (one demo, DATASET= selector — not separate demos) ----
render(
    "2_datasets.png",
    "demo_cpu_datasets — real agentic-RL datasets via DATASET=",
    ["DATASET", "Use case"],
    [
        ["mbpp (default)", "Python code repair against hidden asserts"],
        ["math", "Math word problems (synthetic family)"],
        ["gsm8k", "Grade-school math reasoning"],
        ["math500", "Competition math"],
        ["deepscaler", "Scaled math reasoning dataset"],
        ["humaneval", "Python code generation"],
        ["gretelsql", "Text-to-SQL generation"],
        ["hotpotqa", "Multi-hop retrieval / RAG QA"],
        ["xlam", "Function / tool calling"],
        ["mcp", "MCP tool-use (agent calling sandbox tools)"],
    ],
    [0.22, 0.78],
)

# ---- Table 3: Scaling ----
render(
    "3_scaling.png",
    "Scaling a new dimension",
    ["#", "Demo", "Use case"],
    [
        ["6", "demo_cpu_scale_lanes", "Scaling as 4 × 4 × 4 lanes — harness worker : SGLang engine : microVM (was harness_scale)"],
        ["7", "demo_cpu_mixtask", "Mixed-family batch — one training batch spans multiple task types"],
    ],
    [0.05, 0.26, 0.69],
)

# ---- Table 4: Substrate & design variants ----
render(
    "4_variants.png",
    "Substrate & design variants",
    ["#", "Demo", "Use case"],
    [
        ["8", "demo_cpu_branching", "RL on the real Claude Code CLI; one episode fans out into subagents (rollout-level GRPO)"],
        ["9", "demo_cpu_customruntime", "Managed Code Interpreter vs. a custom Runtime — capability comparison"],
        ["10", "demo_gpu_customruntime", "Same comparison for GPU — GPU only reachable via a custom Runtime on a g6 Instance"],
        ["11", "demo_cpu_imagepool", "Capability-routed image pool — one prebuilt image per family (pandas / scipy / sklearn)"],
        ["12", "demo_cpu_terminalbench", "Terminal-Bench as a large image pool — one Runtime per task (~229), fresh microVM per episode"],
        ["13", "demo_cpu_bigimage", "Large (~4 GB) image exceeding the microVM cap → deploys on a CPU capacity-provider Instance"],
    ],
    [0.04, 0.22, 0.74],
)
