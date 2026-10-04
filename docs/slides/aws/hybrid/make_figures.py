#!/usr/bin/env python3
"""Figures + numbers for the hybrid Terminal-Bench / KernelBench deck (reads the demo's logs and CSVs).

    python3 make_figures.py        # -> fig_*.png + data.json next to this file

Colours are the deck's categorical slots, validated for the dark surface #161D26 (dataviz
validate_palette.js, all checks pass): microVM = blue, m5 = orange, g6 = aqua — fixed per substrate.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "..", "..", "..", "code", "demos", "demo_hybrid_terminalbench")
OLD = os.path.join(HERE, "..", "..", "..", "..", "code", "demos", "demo_cpu_terminalbench")
TASKS = os.path.join(DEMO, "terminalbench", "tasks")
LOGS = os.path.join(DEMO, "logs")

C = {"microvm": "#3B8BEB", "docker": "#E2602C", "gpu": "#1BAF7A"}
NAME = {"microvm": "AgentCore microVM", "docker": "m5 EC2 (warm)", "gpu": "AgentCore g6 Instance"}
BG, PANEL, INK, MUTED, GRID = "#161D26", "#232F3E", "#F2F4F7", "#A3ACB9", "#333E4C"

plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "font.family": "DejaVu Sans",
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})


def rd(p):
    return open(p, errors="replace").read()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


data: dict = {}

# ------------------------------------------------------------------ corpus + routing
manifest = json.load(open(os.path.join(TASKS, "manifest.json")))
routes = json.load(open(os.path.join(TASKS, "route_by_image.json")))
img = list(csv.DictReader(open(os.path.join(TASKS, "images.csv"))))
dock = list(csv.DictReader(open(os.path.join(TASKS, "images_docker.csv"))))
orc = list(csv.DictReader(open(os.path.join(TASKS, "oracle.csv"))))
sub = lambda r: "docker" if str(r).startswith("docker://") else "gpu" if str(r).startswith("gpu:") else "microvm"  # noqa: E731
data["corpus"] = {
    "total": manifest["counts"]["total"], "single": manifest["counts"]["single_container"],
    "multi": manifest["counts"]["multi_container"],
    "microvm_status": dict(Counter(r["status"] for r in img)),
    "m5_status": dict(Counter(r["status"] for r in dock)),
    "m5_reason_built": dict(Counter(r["reason"] for r in dock if r["status"] == "built")),
    "m5_failed": sorted(r["task"] for r in dock if r["status"] != "built"),
    "routes": dict(Counter(sub(v) for k, v in routes.items() if k != "kernelbench-gpu")),
    "usable": dict(Counter(sub(routes.get(r["task"], "")) for r in orc if r.get("usable") == "1")),
    "usable_total": sum(r.get("usable") == "1" for r in orc),
}
mfail = [r["task"] for r in orc if r.get("substrate") == "docker" and r.get("usable") == "1"
         and dict((x["task"], x["reason"]) for x in dock).get(r["task"]) == "microvm_oracle_fail"]
data["corpus"]["m5_rescued_oracle"] = len(mfail)

# ------------------------------------------------------------------ image sizes
arm = [(r["task"], float(r["size_mb"]), r["status"]) for r in img if r.get("size_mb")]
m5 = [(r["task"], float(r["size_mb"]), r["reason"]) for r in dock if r["status"] == "built" and r.get("size_mb")]
fits = sorted(s for _, s, st in arm if st == "registered")
big = sorted(s for _, s, st in arm if st == "too_large")
data["sizes"] = {"arm_n": len(arm), "arm_p50": pct(fits, .5), "arm_p95": pct(fits, .95), "arm_max_fit": max(fits),
                 "too_large": sorted((t, s) for t, s, st in arm if st == "too_large"),
                 "m5_n": len(m5), "m5_p50": pct([s for _, s, _ in m5], .5), "m5_max": max(s for _, s, _ in m5),
                 "kb_image_mb": 145, "kb_torch_mb": 2700}

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axs[0]
bins = [2 ** (i / 2) * 32 for i in range(0, 17)]
ax.hist(fits, bins=bins, color=C["microvm"], edgecolor=BG, linewidth=2, label=f"fits a microVM ({len(fits)})")
ax.hist(big, bins=bins, color=C["docker"], edgecolor=BG, linewidth=2, label=f"> 2 GB → m5 ({len(big)})")
ax.axvline(2048, color=INK, linewidth=1.2, linestyle="--")
ax.text(2048 * 1.07, ax.get_ylim()[1] * 0.9, "microVM cap\n2048 MB", color=INK, fontsize=9, va="top")
ax.set_xscale("log")
ax.set_xticks([64, 128, 256, 512, 1024, 2048, 4096, 8192])
ax.set_xticklabels(["64", "128", "256", "512", "1 G", "2 G", "4 G", "8 G"])
ax.set_xlabel("compressed arm64 image size (MB, log)")
ax.set_ylabel("Terminal-Bench tasks")
ax.set_title("Terminal-Bench single-container images (ECR, arm64)", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.38, 0.80))
ax = axs[1]
order = sorted(m5, key=lambda x: x[1])
ys = range(len(order))
colors = [C["docker"]] * len(order)
ax.scatter([s for _, s, _ in order], list(ys), s=14, color=colors, edgecolors=BG, linewidths=0.6, zorder=3,
           label=f"m5 task images ({len(order)}, unpacked amd64)")
ax.scatter([145], [-6], s=60, marker="D", color=C["gpu"], edgecolors=BG, zorder=3, label="KernelBench image 145 MB")
ax.scatter([145 + 2700], [-6], s=60, marker="D", facecolors="none", edgecolors=C["gpu"], linewidths=1.6, zorder=3,
           label="+ CUDA torch at start (~2.7 GB)")
ax.plot([145, 2845], [-6, -6], color=C["gpu"], linewidth=1.2, zorder=2)
ax.set_xscale("log")
ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
ax.set_xticklabels(["128", "256", "512", "1 G", "2 G", "4 G", "8 G"])
ax.set_yticks([])
ax.set_xlabel("image size on the instance (MB, log)")
ax.set_title("m5 tasks and the KernelBench (g6) image", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 0.10))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_image_sizes.png"), dpi=200)
plt.close(fig)


# ------------------------------------------------------------------ runs
def episodes(t):
    out = []
    for l in t.splitlines():
        if "[demo]" not in l or "solved=" not in l or ("episode_end" not in l and "family=coding" not in l):
            continue
        s = re.search(r"substrate=(\w+)", l)
        if not s:
            continue
        inf = re.search(r"infra_error=(\w+)", l)
        rw = re.search(r"reward=([\d.]+)", l)
        out.append({"sub": s.group(1), "solved": int(re.search(r"solved=(\d)", l).group(1)),
                    "infra": int(bool(inf) and inf.group(1) != "none"), "reward": float(rw.group(1)) if rw else None})
    return out


def opens(t):
    """(substrate, preopened, open_wait_ms, open_ms) per episode, from the [demo] worker line."""
    return [(m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))) for m in re.finditer(
        r"substrate=(\w+) preopened=(\d) open_wait_ms=([\d.]+) preopen_idle_ms=[\d.]+ open_attempts=\d+ open_ms=([\d.]+)", t)]


def evals(log):
    p = log[:-4] + ".eval.csv"
    if not os.path.exists(p):
        return {}
    r = list(csv.DictReader(open(p)))
    return {k: round(float(v), 4) for k, v in r[-1].items()
            if re.fullmatch(r"eval/(terminalbench(_docker)?|kernelbench)", k)} if r else {}


runs = {}
for tag in ("smoke2", "smoke3", "smoke4mix", "smoke5", "smoke6"):
    fs = sorted(glob.glob(os.path.join(LOGS, f"{tag}-hybrid_*.log")))
    if not fs:
        continue
    t = rd(fs[-1])
    eps = episodes(t)
    per = {}
    for s in ("microvm", "docker", "gpu"):
        e = [x for x in eps if x["sub"] == s]
        if e:
            per[s] = {"n": len(e), "solved": sum(x["solved"] for x in e), "infra": sum(x["infra"] for x in e)}
    rw = Counter(x["reward"] for x in eps if x["sub"] == "gpu" and x["reward"] is not None)
    runs[tag] = {"per": per, "eval": evals(fs[-1]), "kb_rewards": {str(k): v for k, v in sorted(rw.items())},
                 "succeeded": bool(re.search(r"Job '[^']+' succeeded", t)),
                 "speedup_max": max([float(x) for x in re.findall(r"speedup=([\d.]+)", t)] or [0])}
data["runs"] = runs

# ------------------------------------------------------------------ pre-open: with vs without
t6 = rd(sorted(glob.glob(os.path.join(LOGS, "smoke6-hybrid_*.log")))[-1])
o6 = opens(t6)
pre = {}
for s in ("microvm", "docker", "gpu"):
    xs = [(w, o) for ss, p, w, o in o6 if ss == s]
    if not xs:
        continue
    waits, cold = [w / 1000 for w, _ in xs], [o / 1000 for _, o in xs]
    pre[s] = {"n": len(xs), "with_p50": pct(waits, .5), "with_p90": pct(waits, .9),
              "without_p50": pct(cold, .5), "without_p90": pct(cold, .9),
              "zero": sum(w == 0 for w in waits), "saved_s": sum(max(0.0, c - w) for w, c in zip(waits, cold))}
data["preopen_smoke6"] = pre
# empirical no-pre-open microVM run (demo_cpu_terminalbench, TB_PREOPEN_BATCHES=0, 2026-10-01)
np_logs = sorted(glob.glob(os.path.join(OLD, "logs", "smoke_nopreopen-*.log")))
if np_logs:
    o = re.findall(r"preopened=\d open_wait_ms=([\d.]+) preopen_idle_ms=[\d.]+ open_attempts=\d+ open_ms=([\d.]+)",
                   rd(np_logs[-1]))       # that demo's lines carry no substrate= (microVM only)
    w = [float(x) / 1000 for x, _ in o]
    data["nopreopen_microvm"] = {"n": len(w), "wait_p50": pct(w, .5), "wait_p90": pct(w, .9)}
# g6 cap history (smoke4: cap 8, smoke5: cap 16, smoke6: cap 32 + 5 batches ahead)
gcap = {}
for tag, cap in (("smoke4mix", 8), ("smoke5", 16), ("smoke6", 32)):
    t = rd(sorted(glob.glob(os.path.join(LOGS, f"{tag}-hybrid_*.log")))[-1])
    w = [x[2] / 1000 for x in opens(t) if x[0] == "gpu"]
    gcap[tag] = {"cap": cap, "n": len(w), "p50": pct(w, .5), "zero": sum(v == 0 for v in w)}
data["g6_cap"] = gcap
# warm m5 pool (smoke6): reuse and pool size
data["warm_smoke6"] = {"reused": len(re.findall(r"warm_reused=1", t6)), "launched": len(re.findall(r"warm_reused=0", t6)),
                       "pool_max": max([int(x) for x in re.findall(r"warm_instances=(\d+)", t6)] or [0]),
                       "dirty": t6.count("warm_reset_dirty"), "retired": t6.count("warm_retire")}

fig, ax = plt.subplots(figsize=(10.5, 3.3))
labels, yy = [], []
for i, s in enumerate(("microvm", "docker", "gpu")):
    if s not in pre:
        continue
    p = pre[s]
    y = 2 - i
    ax.barh(y + 0.18, p["without_p50"], height=0.32, color=MUTED, edgecolor=BG, linewidth=2)
    ax.barh(y - 0.18, p["with_p50"], height=0.32, color=C[s], edgecolor=BG, linewidth=2)
    ax.text(p["without_p50"] + 1.5, y + 0.18, f"without pre-open  {p['without_p50']:.0f} s", va="center", fontsize=9, color=MUTED)
    ax.text(p["with_p50"] + 1.5, y - 0.18, f"with pre-open  {p['with_p50']:.0f} s", va="center", fontsize=9, color=INK)
    labels.append(NAME[s]); yy.append(y)
ax.set_yticks(yy)
ax.set_yticklabels(labels)
ax.set_xlabel("median time an episode waits for its sandbox (s)")
ax.set_xlim(0, max(p["without_p50"] for p in pre.values()) * 1.45)
ax.grid(axis="y", visible=False)
ax.set_title("Pre-open removes the cold start from the episode's critical path (smoke 6)", fontsize=10, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_preopen.png"), dpi=200)
plt.close(fig)

json.dump(data, open(os.path.join(HERE, "data.json"), "w"), indent=1, default=str)
print(json.dumps(data, indent=1, default=str)[:6000])
