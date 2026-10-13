#!/usr/bin/env python3
"""fig_time.png: where the time goes, per training step and per episode (full-epoch run, 13 Oct).

    python3 fig_time_breakdown.py

Left: median step split (trainer waiting for rollouts / training / weight sync), from the trainer's
`perf N: {...}` lines. Right: median per-episode sandbox time by substrate (open, busy = every
exec/write/read/upload incl. the agent loop and grading, close), from <tag>.sandbox_ops.csv.
Phase colours: busy = yellow, overhead = magenta, waiting = grey + hatch (off the critical path;
direct-labelled, so identity is never colour alone). Validated with dataviz validate_palette.js
(dark, #161D26): band/CVD/contrast pass; the grey is deliberately low-chroma.
"""
from __future__ import annotations

import ast
import csv
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "..", "..", "..", "code", "demos", "demo_terminalbench_kernelbench", "logs")
RUN = os.environ.get("RUN", "fullepoch-allkb-hybrid_20261013_020139")
BG, INK, MUTED, GRID = "#161D26", "#F2F4F7", "#A3ACB9", "#333E4C"
BUSY, OVER, WAIT = "#c98500", "#d55181", "#7B8594"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "text.color": INK, "font.family": "DejaVu Sans",
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})

# ---- per step (trainer perf lines; step 1 = cold start, dropped)
train = {}
for line in open(os.path.join(LOGS, f"{RUN}.log"), errors="replace"):
    m = re.search(r"- perf (\d+): (\{.*\})", line)
    if m and "perf/train_time" in m.group(2):
        train[int(m.group(1))] = ast.literal_eval(m.group(2))
S = [d for k, d in sorted(train.items()) if k > 1]
med = lambda key: st.median(d[key] for d in S)  # noqa: E731
step = {"waits for rollouts": med("perf/train_wait_time"), "train": med("perf/train_time"),
        "weight sync": med("perf/update_weights_time")}

# ---- per episode (sandbox ops per session)
sub = lambda a: ("GPU pool (EC2)" if "g6ec2" in a else "m5 pool (EC2)" if "ec2warm" in a  # noqa: E731
                 else "GPU Instance (AgentCore)" if "tbhgpu" in a else "microVM (AgentCore)")
ses, kind = defaultdict(lambda: defaultdict(float)), {}
first_ping = {}
for x in csv.DictReader(open(os.path.join(LOGS, f"{RUN}.sandbox_ops.csv"))):
    if x["status"] != "ok":
        continue
    s, ms = x["session"], float(x["ms"])
    kind[s] = sub(x["arn"])
    if x["op"] == "open" or (x["op"] == "ping" and s not in first_ping and "ec2" not in x["arn"] and "g6ec2" not in x["arn"]):
        first_ping.setdefault(s, ms)
    ses[s][x["op"]] += ms
rows = defaultdict(lambda: defaultdict(list))
for s, d in ses.items():
    busy = d["exec"] + d["write_file"] + d["read_file"] + d["put_tar"]
    if busy == 0:
        continue
    r = rows[kind[s]]
    r["open"].append(first_ping.get(s, 0.0) / 1000); r["busy"].append(busy / 1000); r["close"].append(d["stop_session"] / 1000)
ORDER = ["microVM (AgentCore)", "m5 pool (EC2)", "GPU Instance (AgentCore)", "GPU pool (EC2)"]
ep = {k: {p: st.median(rows[k][p]) for p in ("open", "busy", "close")} for k in ORDER}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 1.5]})
GAP = dict(edgecolor=BG, linewidth=2)
# left: one stacked bar
left = 0
for (lab, v), col, hatch in zip(step.items(), (WAIT, BUSY, OVER), ("//", None, None)):
    a1.barh([0], [v], left=left, color=col, hatch=hatch, height=0.5, **GAP)
    if v > 6:
        a1.text(left + v / 2, 0, f"{lab}\n{v:.0f} s", ha="center", va="center", fontsize=10,
                color=BG if col != WAIT else INK)
    left += v
a1.annotate(f"weight sync {step['weight sync']:.1f} s", xy=(left, 0.25), xytext=(left - 20, 0.55),
            fontsize=9, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a1.set_yticks([0]); a1.set_yticklabels([f"one step\n({sum(step.values()):.0f} s)"])
a1.set_ylim(-0.6, 0.9); a1.set_xlabel("seconds (median per step)")
a1.set_title("Training step: the trainer waits for rollouts", loc="left", fontsize=12, color=INK)
# right: per substrate
for i, k in enumerate(ORDER):
    left = 0
    for p, col, hatch in (("open", WAIT, "//"), ("busy", BUSY, None), ("close", OVER, None)):
        v = ep[k][p]
        a2.barh([i], [v], left=left, color=col, hatch=hatch, height=0.6, **GAP)
        left += v
    a2.text(left + 2, i, f"open {ep[k]['open']:.1f} s · busy {ep[k]['busy']:.0f} s · close {ep[k]['close']:.1f} s",
            va="center", fontsize=9.5, color=MUTED)
a2.set_yticks(range(len(ORDER))); a2.set_yticklabels(ORDER); a2.invert_yaxis()
a2.set_xlim(0, 260); a2.set_xlabel("seconds (median per episode)")
a2.set_title("Episode: busy time is similar everywhere; opening is not", loc="left", fontsize=12, color=INK)
from matplotlib.patches import Patch  # noqa: E402
fig.legend(handles=[Patch(facecolor=WAIT, hatch="//", label="waiting: trainer idle, or sandbox opening (hidden by pre-open)"),
                    Patch(facecolor=BUSY, label="busy: train, or agent loop + grading"),
                    Patch(facecolor=OVER, label="overhead: weight sync, close")],
           loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(os.path.join(HERE, "fig_time.png"), dpi=200)
print("step", {k: round(v, 1) for k, v in step.items()})
print("episode", {k: {p: round(v, 1) for p, v in d.items()} for k, d in ep.items()})
