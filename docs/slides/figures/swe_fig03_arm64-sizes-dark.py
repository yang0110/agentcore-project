#!/usr/bin/env python3
# Figure for aws/swebench.tex (dark theme): per SWE-bench Verified row, the smallest prebuilt arm64 image
# (what a microVM can run) vs the 2 GB AgentCore cap; rows with no arm64 image show their amd64 size, hollow.
# Data: code/demos/demo_cpu_swebench/swebench/verified_image_sizes.csv (survey_image_sizes.py).
# Build: python3 swe_fig03_arm64-sizes-dark.py  (-> swe_fig03_arm64-sizes-dark.pdf)
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "..", "code", "demos", "demo_cpu_swebench", "swebench", "verified_image_sizes.csv")
CAP_GB = 2048 / 1024
ARM, NONE = "#007DFF", "#D9591A"            # awsblue / orange; validated on the dark surface (CVD floor band -> labels)
INK, INK2, GRID, SURFACE = "#E6E9EE", "#9AA4B2", "#2E3A4A", "#161D26"

d = pd.read_csv(CSV)
for c in ("hub_x86_mb", "hub_arm64_mb", "ghcr_arm64_mb"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["arm_gb"] = d[["hub_arm64_mb", "ghcr_arm64_mb"]].min(axis=1) / 1024
d["x86_gb"] = d.hub_x86_mb / 1024
d["repo"] = d.repo.str.split("/").str[1]
order = d.groupby("repo").x86_gb.median().sort_values().index.tolist()

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(8.6, 3.9), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
rng = np.random.default_rng(0)
for y, repo in enumerate(order):
    g = d[d.repo == repo]
    j = rng.uniform(-0.22, 0.22, len(g))
    has = g.arm_gb.notna()
    ax.scatter(g.arm_gb[has], y + j[has], s=14, color=ARM, alpha=0.8, linewidths=0.4, edgecolors=SURFACE)
    ax.scatter(g.x86_gb[~has], y + j[~has], s=16, facecolors="none", edgecolors=NONE, linewidths=1.0)
    ax.text(3.28, y, f"{int(has.sum())}/{len(g)}", va="center", ha="right", color=INK2, fontsize=8)

ax.axvline(CAP_GB, color=INK, lw=1.2, ls=(0, (4, 3)))
ax.text(CAP_GB + 0.04, 5, "AgentCore image cap 2 GB\n(quota L-0A9E32B3, not adjustable)",
        va="center", ha="left", color=INK, fontsize=8)
ax.text(3.28, len(order) - 0.35, "arm64", va="bottom", ha="right", color=INK2, fontsize=8, fontweight="bold")

ax.set_yticks(range(len(order)), [f"{r}  ({(d.repo == r).sum()})" for r in order])
ax.set_ylim(-0.6, len(order) - 0.3)
ax.set_xlim(0.6, 3.3)
ax.set_xlabel("compressed image size, GB", color=INK2)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(INK2)
ax.tick_params(colors=INK2, length=0)
ax.scatter([], [], s=14, color=ARM, label="arm64 image (420 rows)")
ax.scatter([], [], s=16, facecolors="none", edgecolors=NONE, linewidths=1.0,
           label="no arm64: amd64 size (80)")
ax.legend(loc="lower left", bbox_to_anchor=(0.56, 0.0), frameon=False, fontsize=8, labelcolor=INK)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "swe_fig03_arm64-sizes-dark.pdf"), facecolor=SURFACE)
fig.savefig(os.path.join(HERE, "swe_fig03_arm64-sizes-dark.png"), dpi=160, facecolor=SURFACE)
print("wrote swe_fig03_arm64-sizes-dark.{pdf,png}")
