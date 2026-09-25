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
        ["1", "demo_cpu_tasks", "Reference — RL on verifiable tasks in a CPU microVM; DATASET= picks or mixes the data"],
        ["2", "demo_gpu_single", "KernelBench CUDA on one warm g6; + SNAPSHOT_EVERY=1 snapshot/resume, custom-Runtime angle"],
        ["3", "demo_gpu_farm", "GPU packing on a local Ray farm — FARM=fractional (2 episodes, 1 GPU) | whole (1 GPU each)"],
    ],
    [0.04, 0.20, 0.76],
)

# ---- Table 2: Real datasets (one demo, DATASET= selector — not separate demos) ----
render(
    "2_datasets.png",
    "demo_cpu_tasks — DATASET= (one, a,b,c, mixed) + variants",
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
        ["a,b,c", "Round-robin mix of any datasets above (eval/mix)"],
        ["mixed", "Math + SQL + CTF + datasci, one batch (was demo_cpu_mixtask)"],
        ["VARIANT=bigimage", "~4 GB image on a warm CPU Instance (was demo_cpu_bigimage)"],
        ["VARIANT=imagepool", "3 images (pandas/scipy/sklearn), per family (was demo_cpu_imagepool)"],
        ["compare_backends.py", "Managed CI vs. your own Runtime (was demo_cpu_customruntime)"],
        ["HEARTBEAT=1", "Any demo: warm across weight sync (was demo_cpu_heartbeat)"],
    ],
    [0.27, 0.73],
)

# ---- Table 3: Scaling ----
render(
    "3_scaling.png",
    "Scaling a new dimension",
    ["#", "Demo", "Use case"],
    [
        ["4", "demo_cpu_scale_lanes", "Scaling as 4 × 4 × 4 lanes — harness worker : SGLang engine : microVM (was harness_scale)"],
    ],
    [0.05, 0.26, 0.69],
)

# ---- Table 4: Substrate & design variants ----
render(
    "4_variants.png",
    "Substrate & design variant (+ experimental)",
    ["#", "Demo", "Use case"],
    [
        ["5", "demo_cpu_terminalbench", "Terminal-Bench as a large image pool — one Runtime per task (~229), fresh microVM per episode"],
        ["exp", "demo_cpu_branching", "EXPERIMENTAL / blocked — real Claude Code CLI fan-out; CLI exits 1 before any model call"],
    ],
    [0.04, 0.22, 0.74],
)
