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


# ---- Table 1: Canonical storyline (CPU) ----
render(
    "1_canonical.png",
    "Storyline — CPU demos (default runtime → custom images → Instances)",
    ["#", "Demo", "Use case it demonstrates"],
    [
        ["1", "demo_cpu_tasks", "Literature RLVR tasks on the DEFAULT managed Code Interpreter — no image, no Runtime"],
        ["2", "demo_cpu_terminalbench", "Custom images at scale — one image + Runtime per task (~229), fresh microVM per episode"],
        ["", "  + warmstart/ probe", "Pre-open (boot next batch ahead) + heartbeat (survive weight-sync idle), cold vs warm"],
        ["note", "demo_cpu_instance_pool", "SIDE NOTE: > 2 GB image on a warm m5 pool, Landlock-jailed sub_sessions (shared kernel: not full isolation)"],
    ],
    [0.04, 0.24, 0.72],
)

# ---- Table 2: Real datasets (one demo, DATASET= selector — not separate demos) ----
render(
    "2_datasets.png",
    "demo_cpu_tasks — DATASET= (one, a,b,c, mixed, libs) on the Code Interpreter",
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
        ["libs", "pandas / scipy / sklearn tasks — preinstalled in the Code Interpreter, no image"],
        ["a,b,c", "Round-robin mix of any datasets above (eval/mix)"],
        ["mixed", "Math + SQL + CTF + datasci, one batch"],
        ["SANDBOX=agentcore-runtime", "Opt-in: your own Runtime (acs_tasks_cpu, stdlib-only image)"],
        ["compare_backends.py", "Same episode, same reward: Code Interpreter vs. your own Runtime"],
        ["HEARTBEAT=1", "Own Runtime only: warm across weight sync (no-op on the Code Interpreter)"],
    ],
    [0.33, 0.67],
)

# ---- Table 3: Scaling ----
render(
    "3_scaling.png",
    "Scaling the tiers together",
    ["#", "Demo", "Use case"],
    [
        ["3", "demo_cpu_scale_lanes", "Scaling as 4 × 4 × 4 lanes — harness worker : SGLang engine : microVM, strictly 1:1:1"],
    ],
    [0.05, 0.26, 0.69],
)

# ---- Table 4: GPU demos (+ experimental) ----
render(
    "4_variants.png",
    "GPU demos (+ experimental)",
    ["#", "Demo", "Use case"],
    [
        ["4", "demo_gpu_single", "KernelBench CUDA on one warm g6; + SNAPSHOT_EVERY=1, LOCAL_GPU=1"],
        ["5", "demo_gpu_farm", "ONE GPU instance, many sessions — FARM=fractional (0.5 + 0.5) | whole (1 GPU each); trusted-only"],
        ["exp", "demo_cpu_branching", "EXPERIMENTAL / blocked — real Claude Code CLI fan-out; CLI exits 1 before any model call"],
    ],
    [0.05, 0.25, 0.70],
)
