#!/usr/bin/env python3
"""fig_ev_ops.png: machine lifetimes and sessions served per machine, full-epoch run (13 Oct).

    python3 fig_ops.py

Machines come from <tag>.instances.csv (the demo's instance_timeline.py, from CloudTrail): AgentCore GPU
Instances are the ones launched by acs-operator-role; the other g5/g6/g6e are our EC2 GPU pool, and m5.*
are the m5 hosts. Only machines launched while the run had sessions count. Sessions and their substrate
come from <tag>.sessions.csv + <tag>.sandbox_ops.csv, as in fig_concurrency.py.
"""
from __future__ import annotations

import csv
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "..", "..", "..", "code", "demos", "demo_terminalbench_kernelbench", "logs")
RUN = os.environ.get("RUN", "fullepoch-allkb-hybrid_20261013_020139")
BG, INK, MUTED, GRID = "#161D26", "#F2F4F7", "#A3ACB9", "#333E4C"
SUB = [("gpu", "AgentCore GPU Instance", "#1BAF7A"), ("gpu_ec2", "GPU pool (EC2)", "#c98500"),
       ("m5", "m5 pool (EC2)", "#E2602C")]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "font.family": "DejaVu Sans",
                     "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

kind = {}
for x in csv.DictReader(open(os.path.join(LOGS, f"{RUN}.sandbox_ops.csv"))):
    a = x["arn"]
    kind.setdefault(x["session"], "gpu_ec2" if "g6ec2" in a else "m5" if "ec2warm" in a else "gpu" if "tbhgpu" in a else "microvm")
ses = [(float(r["start_ts"]), float(r["end_ts"]), kind[r["session"]]) for r in
       csv.DictReader(open(os.path.join(LOGS, f"{RUN}.sessions.csv"))) if r["session"] in kind]
t0, t1 = min(s for s, _, _ in ses), max(e for _, e, _ in ses)
n_ses = {k: sum(1 for *_, kk in ses if kk == k) for k, _, _ in SUB}
gpu_ses_min = [(e - s) / 60 for s, e, k in ses if k == "gpu"]

life = {k: [] for k, _, _ in SUB}
n_mach = {k: 0 for k, _, _ in SUB}
for r in csv.DictReader(open(os.path.join(LOGS, f"{RUN}.instances.csv"))):
    if not t0 - 600 <= float(r["launch_ts"]) <= t1:
        continue
    k = "m5" if r["type"].startswith("m5") else "gpu" if r["launched_by"] == "acs-operator-role" else "gpu_ec2"
    n_mach[k] += 1
    if r["terminate_ts"]:
        life[k].append((float(r["terminate_ts"]) - float(r["launch_ts"])) / 60)

fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.3), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axs[0]
bins = [i * 2.5 for i in range(33)]
ax.hist([min(x, 80) for x in gpu_ses_min], bins=bins, color=MUTED, alpha=0.45, edgecolor=BG, linewidth=0.4,
        label=f"AgentCore GPU session (n={len(gpu_ses_min)})")
for k, lab, col in SUB:
    ax.hist([min(x, 80) for x in life[k]], bins=bins, color=col, alpha=0.75, edgecolor=BG, linewidth=0.4,
            label=f"{lab} machine (n={len(life[k])}, p50 {statistics.median(life[k]):.0f} min)")
ax.set_yscale("log"); ax.set_xlabel("lifetime (min, capped at 80)"); ax.set_ylabel("count")
ax.set_title("An AgentCore GPU machine serves one session; a pool machine serves many", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7.5)
ax = axs[1]
per = [n_ses[k] / max(n_mach[k], 1) for k, _, _ in SUB]
ax.barh(range(len(SUB)), per, color=[c for *_, c in SUB])
ax.set_yticks(range(len(SUB))); ax.set_yticklabels([lab for _, lab, _ in SUB], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("sessions per machine"); ax.set_title("Reuse: sessions served per machine", fontsize=10, color=INK, loc="left")
for i, (k, _, _) in enumerate(SUB):
    ax.text(per[i] + 0.15, i, f"{per[i]:.1f}  ({n_ses[k]} / {n_mach[k]})", va="center", fontsize=8, color=INK)
ax.set_xlim(0, max(per) * 1.45)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ev_ops.png"), dpi=200)
print({k: (n_ses[k], n_mach[k], round(statistics.median(life[k]), 1) if life[k] else None) for k, _, _ in SUB},
      "gpu session p50 min", round(statistics.median(gpu_ses_min), 1))
