#!/usr/bin/env python3
"""Plain diagrams for the Medium blog (light theme), redrawn from the deck's TikZ slides.

    python3 make_blog_diagrams.py    # -> figures/diag_{loop,routing,architecture}.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
INK, MUTED, LINE = "#16191F", "#5F6B7A", "#9AA4B1"
C = {"microvm": "#2F7FDE", "docker": "#E2602C", "gpu": "#12936A", "gpu_side": "#7B4FD6", "neutral": "#5F6B7A"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 10); ax.set_ylim(0, h * 10); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body="", color=C["neutral"], fill=None, fs=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                lw=1.8, ec=color, fc=fill or color + "14"))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=fs, weight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 2.6 - 3.6 * (title.count("\n") + 1), body, ha="center", va="top",
                fontsize=10, color=INK, linespacing=1.35)


def note(ax, x, y, text, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=9.5, color=MUTED, linespacing=1.3)


def arrow(ax, a, b, color=LINE, label="", dx=0, dy=1.2, style="-|>", ls="-", rad=0.0, fs=9.5):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=16, lw=1.6, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, label, ha="center", va="bottom",
                fontsize=fs, color=MUTED)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ 1. the RL loop
fig, ax = canvas(12, 6.0)
box(ax, 2, 38, 22, 16, "Task data", "Terminal-Bench +\nKernelBench\n251 tasks per epoch")
box(ax, 2, 13, 22, 16, "Rollout manager", "4 tasks × 8 tries\n= 32 episodes\nper training step")
arrow(ax, (13, 38), (13, 29.6))
for k in (2, 1):
    ax.add_patch(FancyBboxPatch((38 + k * 1.2, 14 + k * 1.2), 34, 36, boxstyle="round,pad=0.4,rounding_size=1.2",
                                lw=1.4, ec=C["microvm"], fc="white"))
box(ax, 38, 14, 34, 36, "Sandbox", "one fresh machine per episode\n\nThe agent loop runs inside:\nask the model\n"
    "→ run a shell command\n→ read the output\n→ repeat (up to 8 turns)\n\nThen hidden tests score it: 0 to 1",
    color=C["microvm"], fill="white")
ax.text(75, 53, "× 32 at once", fontsize=13, weight="bold", color=C["microvm"], ha="right")
box(ax, 95, 38, 23, 16, "Policy model", "Qwen3-4B on SGLang\n2 engines, 4 GPUs", color=C["gpu_side"])
box(ax, 95, 13, 23, 16, "Trainer", "Megatron, GRPO\n4 GPUs", color=C["gpu_side"])
arrow(ax, (24.6, 21), (37.5, 21)); note(ax, 31, 17, "open\nsandbox")
arrow(ax, (76, 45), (94.5, 47), color=C["microvm"]); note(ax, 85, 50.5, "conversation\nso far")
arrow(ax, (94.5, 42), (76, 40), color=C["microvm"]); note(ax, 85, 36.5, "next\ncommand")
arrow(ax, (76, 21), (94.5, 21)); note(ax, 85, 25, "transcript\n+ reward")
arrow(ax, (106.5, 29.6), (106.5, 37.6), color=C["gpu_side"]); note(ax, 113.5, 33.6, "new\nweights")
arrow(ax, (106.5, 12.4), (13, 12.4), color=LINE, ls="--", rad=-0.12); note(ax, 60, 2, "next training step: repeat")
save(fig, "diag_loop.png")

# ------------------------------------------------------------------ 2. routing
fig, ax = canvas(13, 5.2)
qs = ["Needs an\nNVIDIA GPU?", "One container\nonly?", "Builds for arm64\nand ≤ 2 GB?", "Reference solution\npasses on\na microVM?"]
for i, q in enumerate(qs):
    x = 2 + i * 27
    box(ax, x, 34, 21, 13, q, color=C["neutral"], fill="white", fs=10.5)
    if i:
        arrow(ax, (x - 5.6, 40.5), (x - 0.4, 40.5)); note(ax, x - 3, 43.5, "no" if i == 1 else "yes")
box(ax, 110, 34, 18, 13, "AgentCore\nmicroVM", "162 tasks", color=C["microvm"], fs=11)
arrow(ax, (104.4, 40.5), (109.6, 40.5)); note(ax, 107, 43.5, "yes")
outs = [("AgentCore g6\nGPU Instance", "KernelBench\n(120 tasks)", C["gpu"], "yes"),
        ("m5 EC2", "multi-container\n(12 tasks)", C["docker"], "no"),
        ("m5 EC2", "x86-only or big\n(11 tasks)", C["docker"], "no"),
        ("m5 EC2", "retry there\n(56 tasks)", C["docker"], "no")]
for i, (t, b, col, lab) in enumerate(outs):
    x = 2 + i * 27
    box(ax, x, 12, 21, 15, t, b, color=col, fs=11)
    arrow(ax, (x + 10.5, 33.6), (x + 10.5, 27.4)); note(ax, x + 13, 30.5, lab)
ax.add_patch(FancyBboxPatch((27, 0.5), 79, 7, boxstyle="round,pad=0.4,rounding_size=1.2", lw=1.4,
                            ec=C["docker"], fc=C["docker"] + "14"))
ax.text(66.5, 4, "each m5 candidate is built, and its reference solution re-checked, on an m5  →  24 usable",
        ha="center", va="center", fontsize=10, color=INK)
for i in (1, 2, 3):
    arrow(ax, (2 + i * 27 + 10.5, 11.6), (2 + i * 27 + 10.5, 8.2), color=C["docker"])
save(fig, "diag_routing.png")

# ------------------------------------------------------------------ 3. architecture
fig, ax = canvas(13, 6.2)
ax.add_patch(FancyBboxPatch((1, 4), 38, 54, boxstyle="round,pad=0.4,rounding_size=1.5", lw=1.2, ls="--",
                            ec=LINE, fc="none"))
ax.text(20, 59.5, "Trainer host (one machine, 8 GPUs)", ha="center", fontsize=11, weight="bold", color=MUTED)
box(ax, 4, 41, 32, 13, "slime", "trainer (4 GPUs) +\npolicy model (4 GPUs)", color=C["gpu_side"])
box(ax, 4, 24, 32, 13, "Model endpoint", "one per engine; records\nevery token the agent sees", color=C["gpu_side"])
box(ax, 4, 7, 32, 13, "Sandbox pool (our code)", "route, pre-open, heartbeat,\ncaps, grade", color=C["neutral"])
ax.add_patch(FancyBboxPatch((60, 25), 48, 33, boxstyle="round,pad=0.4,rounding_size=1.5", lw=1.2, ls="--",
                            ec=C["microvm"], fc="none"))
ax.text(84, 59.5, "AgentCore (AWS manages these)", ha="center", fontsize=11, weight="bold", color=C["microvm"])
box(ax, 63, 42, 42, 12, "microVM Runtimes (one per task)", "fresh microVM per episode, < 1 s to open", color=C["microvm"], fs=11)
box(ax, 63, 27, 42, 12, "GPU capacity provider (g6)", "fresh GPU Instance per episode, ~2 min", color=C["gpu"], fs=11)
ax.add_patch(FancyBboxPatch((60, 4), 48, 16, boxstyle="round,pad=0.4,rounding_size=1.5", lw=1.2, ls="--",
                            ec=C["docker"], fc="none"))
ax.text(84, 21.5, "Plain EC2 (our code manages these)", ha="center", fontsize=11, weight="bold", color=C["docker"])
box(ax, 63, 6, 42, 12, "m5 warm pool (up to 24 VMs)", "Docker Compose; reused only after\na verified reset", color=C["docker"], fs=11)
arrow(ax, (36.5, 17), (62.5, 31), color=C["gpu"])
arrow(ax, (36.5, 15), (62.5, 13), color=C["docker"])
arrow(ax, (36.5, 19), (62.5, 46), color=C["microvm"])
note(ax, 50, 7, "open, run commands,\ngrade, close")
arrow(ax, (62.5, 50), (36.5, 33), color=INK, ls="--")
note(ax, 50, 55, "the agent asks for\nits next step")
ax.text(120, 31, "Everything is private:\nsandboxes live in\nour VPC, no public\nports", ha="center",
        va="center", fontsize=10, color=MUTED, linespacing=1.4)
save(fig, "diag_architecture.png")
