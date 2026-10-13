#!/usr/bin/env python3
"""fig_concurrency.png: live sandbox sessions per substrate over the full-epoch run (13 Oct).

    python3 fig_concurrency.py

A session is live from its start to its end in <tag>.sessions.csv; its substrate comes from the
Runtime/route of its ops in <tag>.sandbox_ops.csv. Colours: the deck's substrate slots plus yellow for
the EC2 GPU pool, validated with dataviz validate_palette.js (dark, #161D26): all checks pass.
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "..", "..", "..", "code", "demos", "demo_terminalbench_kernelbench", "logs")
RUN = os.environ.get("RUN", "fullepoch-allkb-hybrid_20261013_020139")
BG, INK, MUTED, GRID = "#161D26", "#F2F4F7", "#A3ACB9", "#333E4C"
SUB = [("microvm", "AgentCore microVM", "#3B8BEB"), ("m5", "m5 pool (EC2)", "#E2602C"),
       ("gpu", "AgentCore GPU Instance", "#1BAF7A"), ("gpu_ec2", "GPU pool (EC2)", "#c98500")]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "font.family": "DejaVu Sans",
                     "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

kind = {}
for x in csv.DictReader(open(os.path.join(LOGS, f"{RUN}.sandbox_ops.csv"))):
    a = x["arn"]
    kind.setdefault(x["session"], "gpu_ec2" if "g6ec2" in a else "m5" if "ec2warm" in a else "gpu" if "tbhgpu" in a else "microvm")
ses = [(float(r["start_ts"]), float(r["end_ts"]), kind.get(r["session"])) for r in
       csv.DictReader(open(os.path.join(LOGS, f"{RUN}.sessions.csv"))) if r["session"] in kind]
t0, t1 = min(s for s, _, _ in ses), max(e for _, e, _ in ses)
grid = [t0 + i * 30 for i in range(int((t1 - t0) / 30) + 1)]
fig, axs = plt.subplots(2, 2, figsize=(11, 4.6), sharex=True)
peaks = {}
for ax, (key, name, col) in zip(axs.flat, SUB):
    iv = [(s, e) for s, e, k in ses if k == key]
    y = [sum(1 for s, e in iv if s <= t < e) for t in grid]
    peaks[key] = max(y) if y else 0
    x = [(t - t0) / 60 for t in grid]
    ax.fill_between(x, y, color=col, alpha=0.25, lw=0)
    ax.plot(x, y, color=col, lw=1.6)
    ax.set_title(f"{name}: peak {peaks[key]}", loc="left", fontsize=12, color=INK)
    ax.set_ylim(0, None)
for ax in axs[1]:
    ax.set_xlabel("minutes since the first sandbox opened")
for ax in axs[:, 0]:
    ax.set_ylabel("live sessions")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_concurrency.png"), dpi=200)
print("peaks", peaks, "minutes", round((t1 - t0) / 60))
