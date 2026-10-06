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
DEMO = os.path.join(HERE, "..", "..", "..", "..", "code", "demos", "demo_terminalbench_kernelbench")
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
SHUTDOWN_MARK = "cannot schedule new futures after shutdown"


def episodes(t):
    """One record per finished episode. Episodes killed with no sandbox fault are dropped: tagged
    cancelled=abort|shutdown (terminalbench_pool._cancel_reason), or — logs from before that tag —
    any non-grade error logged after the first executor-shutdown traceback (process teardown)."""
    out = []
    cut = t.find(SHUTDOWN_MARK)
    pos = 0
    for l in t.splitlines(keepends=True):
        at, pos = pos, pos + len(l)
        if "[demo]" not in l or "solved=" not in l or ("episode_end" not in l and "family=coding" not in l):
            continue
        if "cancelled=" in l or (cut >= 0 and at > cut and "infra_error=none" not in l
                                 and "infra_error=grade" not in l):
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
RUN_LOGS = [(tag, sorted(glob.glob(os.path.join(LOGS, f"{tag}-hybrid_*.log")))[-1:])
            for tag in ("smoke2", "smoke3", "smoke4mix", "smoke5", "smoke6")]
# both 48-step training runs: run 1 (eval deadlock at step 20), run 2 (with the fix, completed)
RUN_LOGS += [(f"train48_run{i}", [f]) for i, f in
             enumerate(sorted(glob.glob(os.path.join(LOGS, "train48-hybrid_*.log"))), 1)]
for tag, fs in RUN_LOGS:
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
                 "cancelled": len(re.findall(r"\[demo\] (?:episode_end|task=).*cancelled=", t))
                              + (sum(1 for x in re.finditer(r"\[demo\] (?:episode_end|task=)[^\n]*infra_error=(?!none|grade)", t)
                                     if x.start() > t.find(SHUTDOWN_MARK)) if SHUTDOWN_MARK in t else 0),
                 "grade_errors": Counter(re.findall(r"episode_end task=(\S+)[^\n]*infra_error=grade", t)),
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

# ------------------------------------------------------------------ instances alive during training (run 2)
# microVM = AgentCore sessions (one microVM each, pre-opened ones included); m5 and g6 = real EC2
# lifetimes from CloudTrail (demo's instance_timeline.py -> <log>.instances.csv; g6 launched by AgentCore).
tl = sorted(glob.glob(os.path.join(LOGS, "train48-hybrid_*.instances.csv")))
if tl:
    base = tl[-1][:-len(".instances.csv")]
    kind = {}
    for r in csv.DictReader(open(base + ".sandbox_ops.csv")):
        if r["session"] not in kind and r["arn"].startswith("acs_tbh_"):
            kind[r["session"]] = "microvm"
    life = {"microvm": [(float(r["start_ts"]), float(r["end_ts"])) for r in csv.DictReader(open(base + ".sessions.csv"))
                        if kind.get(r["session"]) == "microvm"]}
    inst = list(csv.DictReader(open(tl[-1])))
    for s, ty in (("docker", "m5.xlarge"), ("gpu", "g6.xlarge")):
        life[s] = [(float(r["launch_ts"]), float(r["terminate_ts"])) for r in inst if r["type"] == ty and r["terminate_ts"]]
    step0 = {}
    for r in csv.DictReader(open(base + ".sandbox_ops.csv")):
        if not r["step"]:
            continue
        st = int(r["step"]); step0[st] = min(float(r["ts"]), step0.get(st, 9e18))
    t0 = min(a for v in life.values() for a, _ in v)
    t_end = max(step0.values())
    t1 = max(b for v in life.values() for _, b in v)
    grid = [t0 + 10 * i for i in range(int((t1 - t0) / 10) + 2)]
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    inst_tl = {}
    for s in ("microvm", "docker", "gpu"):
        ys = [sum(a <= g < b for a, b in life[s]) for g in grid]
        ax.plot([(g - t0) / 60 for g in grid], ys, color=C[s], lw=1.6,
                label=f"{NAME[s].replace(' (warm)', '')}  (peak {max(ys)}, {len(life[s]):,} total)")
        inst_tl[s] = {"total": len(life[s]), "peak": max(ys),
                      "mean_training": sum(y for g, y in zip(grid, ys) if g <= t_end) / max(1, sum(g <= t_end for g in grid))}
    inst_tl["gpu"]["after_run_min"] = (t1 - t_end) / 60
    gs = {r["session"] for r in csv.DictReader(open(base + ".sandbox_ops.csv")) if "tbhgpu" in r["arn"]}
    inst_tl["gpu"]["session_p50_min"] = pct([float(r["lifetime_s"]) / 60 for r in csv.DictReader(open(base + ".sessions.csv"))
                                             if r["session"] in gs], .5)
    inst_tl["gpu"]["instance_p50_min"] = pct([(b - a) / 60 for a, b in life["gpu"]], .5)
    ax.set_xlabel("minutes since the first sandbox opened")
    ax.set_ylabel("alive at once")
    ax.set_xlim(0, (t1 - t0) / 60)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=3, bbox_to_anchor=(0, 1.0))
    ax.axvline((t_end - t0) / 60, color=MUTED, lw=0.8, ls="--")
    ax.text((t_end - t0) / 60 + 1, ax.get_ylim()[1] * 0.70, "training ends", fontsize=8.5, color=MUTED)
    ax.set_title("Sandboxes alive during the 48-step run (train48 run 2)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_instances.png"), dpi=200)
    plt.close(fig)
    data["instances_train48"] = inst_tl

# ------------------------------------------------------------------ learning curves (train48 run 2)
base = os.path.join(LOGS, "train48-hybrid_20261005_005629")
if os.path.exists(base + ".csv"):
    steps = list(csv.DictReader(open(base + ".csv")))
    ev = list(csv.DictReader(open(base + ".eval.csv")))
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else 0.0  # noqa: E731
    x = list(range(1, len(steps) + 1))
    rew = [f(r, "rollout/raw_reward") for r in steps]
    flat = [f(r, "rollout/zero_std/count_0.0") + f(r, "rollout/zero_std/count_0.5") + f(r, "rollout/zero_std/count_1.0")
            for r in steps]                                     # groups whose 8 rewards are all equal (RB = 4)
    gn = [f(r, "train/grad_norm") for r in steps]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
    ax = axs[0]
    ax.plot(x, rew, color=INK, lw=1.0, alpha=0.45, label="per step")
    k = 5
    ax.plot(x[k - 1:], [sum(rew[i - k + 1:i + 1]) / k for i in range(k - 1, len(rew))], color=C["microvm"], lw=2,
            label=f"{k}-step mean")
    ax.set_title("Training reward (mean of 32 episodes)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("training step"); ax.set_ylim(0, max(rew) * 1.25)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    ax.bar(x, [4] * len(x), color=GRID, width=0.85, label="groups with a gradient")
    ax.bar(x, flat, color=C["docker"], width=0.85, label="groups with no gradient (all 8 equal)")
    ax.scatter([s for s, g in zip(x, gn) if g == 0], [4.25] * sum(g == 0 for g in gn), marker="v", s=12, color=INK,
               label="grad norm = 0 (step learns nothing)")
    ax.set_title("GRPO signal: 4 groups per step", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("training step"); ax.set_ylim(0, 7.2); ax.set_yticks([0, 1, 2, 3, 4])
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    ax = axs[2]
    es = [int(float(r["eval/step"])) for r in ev]
    for key, lab, col in (("eval/kernelbench", "KernelBench (mean reward, 0 / 0.5 / 1)", C["gpu"]),
                          ("eval/terminalbench", "Terminal-Bench microVM (solve rate)", C["microvm"]),
                          ("eval/terminalbench_docker", "Terminal-Bench m5 (solve rate)", C["docker"])):
        ax.plot(es, [f(r, key) for r in ev], marker="o", ms=4, lw=1.8, color=col, label=lab)
    ax.set_title("Held-out eval", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("training step"); ax.set_ylim(0, 0.65)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_learning.png"), dpi=200)
    plt.close(fig)
    data["learning_train48"] = {"steps": len(x), "no_grad_steps": sum(g == 0 for g in gn),
                                "flat_groups": sum(flat), "groups": 4 * len(x),
                                "reward_first10": sum(rew[:10]) / 10, "reward_last10": sum(rew[-10:]) / 10,
                                "eval": {r["eval/step"]: {k: r.get(k) for k in ("eval/kernelbench", "eval/terminalbench",
                                                                               "eval/terminalbench_docker")} for r in ev}}

# ------------------------------------------------------------------ evidence figures (one per challenge group)
import statistics as stx  # noqa: E402

RUNS = sorted(p[:-len(".demo.csv")] for p in glob.glob(os.path.join(LOGS, "*.demo.csv")))
fv = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else None  # noqa: E731
ev: dict = {}


def _hist(ax, xs, bins, color, label, alpha=0.85):
    ax.hist(xs, bins=bins, color=color, alpha=alpha, label=label, edgecolor=BG, linewidth=0.4)


# Build: how many images, how big, how long ------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
ax = axs[0]
mv_st = Counter(r["status"] for r in img)
dk_st = Counter(r["status"] for r in dock)
bars = [("microVM\n(arm64)", mv_st["registered"], len(img) - mv_st["registered"], C["microvm"]),
        ("m5 tasks\n(amd64)", dk_st["built"], len(dock) - dk_st["built"], C["docker"]),
        ("g6\n(KernelBench)", 1, 0, C["gpu"])]
for i, (lab, ok, bad, col) in enumerate(bars):
    ax.bar(i, ok, color=col, width=0.6)
    ax.bar(i, bad, bottom=ok, color=GRID, width=0.6)
    ax.text(i, ok + bad + 4, f"{ok} built" + (f"\n+{bad} failed" if bad else ""), ha="center", fontsize=8, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([b[0] for b in bars], fontsize=8)
ax.set_ylim(0, 290); ax.set_title("Images built, per substrate", fontsize=10, color=INK, loc="left")
ax = axs[1]
mv_sz = [float(r["size_mb"]) for r in img if r["size_mb"]]
dk_sz = [float(r["size_mb"]) for r in dock if r["size_mb"]]
lb = [10 ** (1.6 + i * 0.1) for i in range(26)]
_hist(ax, mv_sz, lb, C["microvm"], f"microVM, compressed (n={len(mv_sz)})")
_hist(ax, dk_sz, lb, C["docker"], f"m5, unpacked (n={len(dk_sz)})", alpha=0.7)
ax.axvline(2048, color=INK, lw=1, ls="--"); ax.text(2150, ax.get_ylim()[1] * 0.85, "2 GB\nmicroVM cap", fontsize=7.5, color=INK)
ax.set_xscale("log"); ax.set_xlabel("image size (MB)"); ax.set_ylabel("images")
ax.set_title("Image size distribution", fontsize=10, color=INK, loc="left"); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax = axs[2]
dk_bt = [float(r["build_s"]) / 60 for r in dock if r["build_s"]]
_hist(ax, dk_bt, [i * 0.5 for i in range(16)], C["docker"], f"m5 task builds (n={len(dk_bt)})")
ax.set_xlabel("build time (min)"); ax.set_ylabel("tasks")
ax.set_title("Build time per task (m5, amd64 native)", fontsize=10, color=INK, loc="left")
ax.text(0.98, 0.97, "microVM arm64 builds (QEMU, 23 Sep):\nper-task times not kept;\np50 6.8 min, max 2.0 h,\n39.6 h total, 16 in parallel",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
ax.legend(frameon=False, fontsize=7.5, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_build.png"), dpi=200); plt.close(fig)
ev["build"] = {"microvm": dict(mv_st), "docker": dict(dk_st), "mv_size_p50": pct(mv_sz, .5), "mv_size_max": max(mv_sz),
               "dk_size_p50": pct(dk_sz, .5), "dk_size_max": max(dk_sz), "dk_build_p50_min": pct(dk_bt, .5),
               "dk_build_max_min": max(dk_bt), "dk_build_total_h": sum(dk_bt) / 60, "ecr_repos_m5": 89}

# Speed: open time, tool-call latency, pre-open idle ---------------------------------
T48 = os.path.join(LOGS, "train48-hybrid_20261005_005629")
ops = list(csv.DictReader(open(T48 + ".sandbox_ops.csv")))
dem = list(csv.DictReader(open(T48 + ".demo.csv")))
subst = lambda a: "gpu" if a.startswith("acs_tbhgpu") else "docker" if a.startswith("ec2") else "microvm"  # noqa: E731
first_ping: dict = {}
for r in ops:
    if r["op"] == "ping" and r["session"] not in first_ping:
        first_ping[r["session"]] = (subst(r["arn"]), float(r["ms"]))
open_s = {"microvm": [ms / 1000 for s, ms in first_ping.values() if s == "microvm"],
          "gpu": [ms / 1000 for s, ms in first_ping.values() if s == "gpu"],
          "docker": [float(r["ms"]) / 1000 for r in ops if r["op"] == "open" and r["arn"].startswith("ec2")]}
idle_m = {s: [float(r["preopen_idle_ms"]) / 60000 for r in dem if r["preopen_idle_ms"] and r["substrate"] == s] for s in C}
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
ax = axs[0]
order = ["microvm", "docker", "gpu"]
bp = ax.boxplot([open_s[s] for s in order], vert=False, widths=0.55, patch_artist=True, showfliers=True,
                flierprops={"marker": ".", "markersize": 2, "markerfacecolor": MUTED, "markeredgecolor": MUTED},
                medianprops={"color": INK})
for patch, s in zip(bp["boxes"], order):
    patch.set_facecolor(C[s]); patch.set_edgecolor(C[s])
ax.set_yticks([1, 2, 3]); ax.set_yticklabels([f"{NAME[s].split(' (')[0]}\nn={len(open_s[s])}" for s in order], fontsize=7.5)
ax.set_xscale("log"); ax.set_xlabel("time to open a sandbox (s)")
ax.set_title("Cold start (env.reset)", fontsize=10, color=INK, loc="left")
ax = axs[1]
# The agent's own tool calls run INSIDE the sandbox (in-session harness), so they are not trainer-side ops;
# per episode we have the agent-loop wall time and its turns (model generation + tool execution).
agent_s = {s: [float(r["agent_ms"]) / 1000 for r in dem if r["agent_ms"] and r["substrate"] == s] for s in C}
ab = [10 ** (0 + i * 0.1) for i in range(29)]
for s in ("microvm", "docker"):
    _hist(ax, agent_s[s], ab, C[s], f"{NAME[s].split(' (')[0]} (n={len(agent_s[s]):,})", alpha=0.65)
ax.set_xscale("log"); ax.set_xlabel("agent loop per episode (s): all turns, model + tools"); ax.set_ylabel("episodes")
ax.set_title("Episode time in the sandbox (env.step $\\times$ turns)", fontsize=10, color=INK, loc="left"); ax.legend(frameon=False, fontsize=7)
ax.text(0.98, 0.62, "g6: not logged per episode", transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
ax = axs[2]
ib = [i * 0.5 for i in range(41)]
for s in order:
    if idle_m[s]:
        _hist(ax, [min(x, 20) for x in idle_m[s]], ib, C[s], f"{NAME[s].split(' (')[0]} (n={len(idle_m[s]):,})", alpha=0.65)
ax.axvline(7.5, color=MUTED, lw=1, ls=":"); ax.axvline(15, color=INK, lw=1, ls="--")
ax.text(7.7, 3, "heartbeat\nping", fontsize=7.5, color=MUTED); ax.text(15.2, 3, "15-min\nidle reclaim", fontsize=7.5, color=INK)
ax.set_yscale("log"); ax.set_xlabel("pre-opened sandbox idle before use (min)"); ax.set_ylabel("episodes")
ax.set_title("Idle time of pre-opened sandboxes", fontsize=10, color=INK, loc="left"); ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.8))
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ev_speed.png"), dpi=200); plt.close(fig)
ev["speed"] = {s: {"open_p50_s": pct(open_s[s], .5), "open_p95_s": pct(open_s[s], .95), "n_open": len(open_s[s]),
                   "agent_p50_s": pct(agent_s[s], .5), "agent_p95_s": pct(agent_s[s], .95), "n_agent": len(agent_s[s]),
                   "idle_p50_min": pct(idle_m[s], .5), "idle_max_min": max(idle_m[s]) if idle_m[s] else None,
                   "idle_over_7_5": sum(x > 7.5 for x in idle_m[s]), "idle_over_15": sum(x > 15 for x in idle_m[s])}
               for s in order}

# Rewards: per-task solve rate, partial progress, KernelBench speedup ----------------
eps = []
for b in RUNS:
    for r in csv.DictReader(open(b + ".demo.csv")):
        if r.get("infra_error") == "none" and (r.get("tests_total") or r.get("checker")):
            eps.append(r)
per_task: dict = {}
for r in eps:
    if r["substrate"] != "gpu" and r.get("tests_total"):
        per_task.setdefault((r["substrate"], r["task"]), []).append(int(r["solved"] or 0))
kb_sp = [float(r["speedup"]) for r in eps if r["substrate"] == "gpu" and r.get("speedup")]
uns = [int(r["tests_passed"] or 0) / int(r["tests_total"]) for r in eps
       if r["substrate"] != "gpu" and r.get("tests_total") and int(r["tests_total"]) > 0 and r["solved"] == "0"]
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
ax = axs[0]
rb = [i / 20 for i in range(21)]
for s in ("microvm", "docker"):
    xs = [sum(v) / len(v) for (ss, t), v in per_task.items() if ss == s and len(v) >= 4]
    _hist(ax, xs, rb, C[s], f"{NAME[s].split(' (')[0]} tasks (n={len(xs)})", alpha=0.75)
    ev.setdefault("reward", {})[f"{s}_tasks"] = len(xs)
    ev["reward"][f"{s}_never_solved"] = sum(x == 0 for x in xs)
    ev["reward"][f"{s}_always_solved"] = sum(x == 1 for x in xs)
ax.set_yscale("log"); ax.set_xlabel("solve rate per task (all runs, tasks with $\\geq$ 4 episodes)"); ax.set_ylabel("tasks")
ax.set_title("Most tasks are never solved", fontsize=10, color=INK, loc="left"); ax.legend(frameon=False, fontsize=7.5)
ax = axs[1]
_hist(ax, uns, rb, C["microvm"], f"unsolved episodes (n={len(uns):,})")
ax.set_yscale("log"); ax.set_xlabel("fraction of hidden tests passed"); ax.set_ylabel("episodes")
ax.set_title("Unsolved is often partly right", fontsize=10, color=INK, loc="left"); ax.legend(frameon=False, fontsize=7.5)
ax = axs[2]
sb = [0.8 + i * 0.025 for i in range(25)]
_hist(ax, [min(max(x, 0.8), 1.39) for x in kb_sp], sb, C["gpu"], f"correct KernelBench kernels (n={len(kb_sp)})")
ax.axvline(1.10, color=INK, lw=1, ls="--"); ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
ax.text(0.62, 0.45, "1.10$\\times$ gate:\nreward 1 above,\n0.5 below", transform=ax.transAxes, fontsize=7.5, color=INK)
ax.set_yscale("log"); ax.set_xlabel("speedup vs the PyTorch reference"); ax.set_ylabel("episodes")
ax.set_title("KernelBench: correct is easy, faster is rare", fontsize=10, color=INK, loc="left"); ax.legend(frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.9))
ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]); ax.set_xticklabels(["$\\leq$0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4"])
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ev_reward.png"), dpi=200); plt.close(fig)
ev["reward"].update({"unsolved_eps": len(uns), "unsolved_partial": sum(x > 0 for x in uns),
                     "kb_correct": len(kb_sp), "kb_over_gate": sum(x >= 1.10 for x in kb_sp),
                     "kb_speedup_p50": pct(kb_sp, .5), "runs": [os.path.basename(b) for b in RUNS]})

# Operations: instance lifetimes vs session lifetimes, infra errors ------------------
inst = list(csv.DictReader(open(T48 + ".instances.csv")))
ses = {r["session"]: float(r["lifetime_s"]) / 60 for r in csv.DictReader(open(T48 + ".sessions.csv"))}
gses = {r["session"] for r in ops if r["arn"].startswith("acs_tbhgpu")}
life = {"g6": [(float(r["terminate_ts"]) - float(r["launch_ts"])) / 60 for r in inst if r["type"].startswith("g6") and r["terminate_ts"]],
        "m5": [(float(r["terminate_ts"]) - float(r["launch_ts"])) / 60 for r in inst if r["type"].startswith("m5") and r["terminate_ts"]]}
g_sess = [v for k, v in ses.items() if k in gses]
infra = Counter()
for b in RUNS:
    for r in csv.DictReader(open(b + ".demo.csv")):
        if r.get("infra_error") not in (None, "", "none"):
            infra[r["infra_error"]] += 1
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.3), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axs[0]
lbins = [i * 2.5 for i in range(41)]
_hist(ax, [min(x, 100) for x in g_sess], lbins, INK, f"g6 session (n={len(g_sess)})", alpha=0.45)
_hist(ax, [min(x, 100) for x in life["g6"]], lbins, C["gpu"], f"g6 instance (n={len(life['g6'])})", alpha=0.7)
_hist(ax, [min(x, 100) for x in life["m5"]], lbins, C["docker"], f"m5 instance, warm pool (n={len(life['m5'])})", alpha=0.6)
ax.set_xlabel("lifetime (min, capped at 100)"); ax.set_ylabel("count")
ax.set_title("A g6 instance outlives its session; an m5 serves many", fontsize=10, color=INK, loc="left"); ax.legend(frameon=False, fontsize=7.5)
ax = axs[1]
ILAB = {"grade": "grading time-out", "URLError": "network (URLError)", "ValueError": "bad reply (ValueError)",
        "RuntimeError": "killed at exit (not counted)"}
ks = [k for k, _ in infra.most_common()]
ax.barh(range(len(ks)), [infra[k] for k in ks], color=[MUTED if k == "RuntimeError" else C["docker"] for k in ks])
ax.set_yticks(range(len(ks))); ax.set_yticklabels([ILAB.get(k, k) for k in ks], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("episodes"); ax.set_title(f"Infra errors by cause, all {len(RUNS)} runs", fontsize=10, color=INK, loc="left")
for i, k in enumerate(ks):
    ax.text(infra[k] + 0.3, i, str(infra[k]), va="center", fontsize=8, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ev_ops.png"), dpi=200); plt.close(fig)
ev["ops"] = {"g6_session_p50_min": pct(g_sess, .5), "g6_instance_p50_min": pct(life["g6"], .5),
             "g6_instance_p90_min": pct(life["g6"], .9), "m5_instance_p50_min": pct(life["m5"], .5),
             "m5_instances": len(life["m5"]), "g6_instances": len(life["g6"]), "infra": dict(infra),
             "episodes_all_runs": sum(1 for b in RUNS for r in csv.DictReader(open(b + ".demo.csv")) if r.get("infra_error"))}
data["evidence"] = ev

# Why episodes end: the turn / token budget (all runs) -------------------------------
stop = {s: Counter() for s in C}
solv = {}
for b in RUNS:
    for r in csv.DictReader(open(b + ".demo.csv")):
        if r.get("infra_error") == "none" and r.get("stop") in ("end_turn", "budget", "max_tokens"):
            stop[r["substrate"]][r["stop"]] += 1
            if r["substrate"] != "gpu":
                a = solv.setdefault(r["stop"], [0, 0]); a[0] += int(r["solved"] or 0); a[1] += 1
SLAB = {"end_turn": "agent finished", "budget": "hit the 8-turn limit", "max_tokens": "hit 1,024 tokens in a turn"}
SCOL = {"end_turn": C["gpu"], "budget": C["docker"], "max_tokens": MUTED}
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
for i, s in enumerate(order):
    n = sum(stop[s].values()); left = 0
    for k in ("end_turn", "budget", "max_tokens"):
        v = 100 * stop[s][k] / n
        ax.barh(i, v, left=left, color=SCOL[k], height=0.55, label=SLAB[k] if i == 0 else None)
        if v > 6:
            ax.text(left + v / 2, i, f"{v:.0f} %", ha="center", va="center", fontsize=8.5, color=BG)
        left += v
ax.set_yticks(range(3)); ax.set_yticklabels([f"{NAME[s].split(' (')[0]}\nn={sum(stop[s].values()):,}" for s in order], fontsize=8)
ax.set_xlim(0, 100); ax.set_xlabel("% of episodes"); ax.invert_yaxis()
ax.set_title("How episodes end (all 9 runs)", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
ax = axs[1]
ks = ["end_turn", "budget", "max_tokens"]
rates = [100 * solv[k][0] / solv[k][1] for k in ks]
ax.bar(range(3), rates, color=[SCOL[k] for k in ks], width=0.55)
for i, k in enumerate(ks):
    ax.text(i, rates[i] + 0.15, f"{solv[k][0]}/{solv[k][1]:,}", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([SLAB[k].replace(" in a turn", "") for k in ks], fontsize=7.5)
ax.set_ylabel("% solved"); ax.set_ylim(0, max(rates) * 1.35)
ax.set_title("Terminal-Bench solve rate by how it ended", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ev_stop.png"), dpi=200); plt.close(fig)
data["stop_reasons"] = {"by_substrate": {s: dict(v) for s, v in stop.items()}, "tb_solved_by_stop": solv}

# Cost of train48 run 2 in dollars ---------------------------------------------------
# us-west-2 On-Demand (= us-east-1 for these types), checked 6 Oct 2026; AgentCore [P1]: microVM $0.1276/vCPU-h +
# $0.0169/GB-h, active CPU only (so the microVM figure is an UPPER bound at 2 vCPU / 8 GB busy all session);
# Instances = EC2 On-Demand + 7.8 % management fee for G-series.
PRICE = {"g6e.48xlarge": 30.131, "g6.xlarge": 0.805, "m5.xlarge": 0.192}
sess_h = Counter()
sub_of = {}
for r in ops:
    sub_of.setdefault(r["session"], subst(r["arn"]))
for r in csv.DictReader(open(T48 + ".sessions.csv")):
    if r["session"] in sub_of:
        sess_h[sub_of[r["session"]]] += float(r["lifetime_s"]) / 3600
inst_h = Counter()
for r in inst:
    if r["terminate_ts"]:
        inst_h[r["type"]] += (float(r["terminate_ts"]) - float(r["launch_ts"])) / 3600
wall_h = 2.97
eps_n = Counter(r["substrate"] for r in dem if r.get("infra_error"))
cost = {"trainer (g6e.48xlarge)": wall_h * PRICE["g6e.48xlarge"],
        "g6 Instances": inst_h["g6.xlarge"] * PRICE["g6.xlarge"] * 1.078,
        "microVM sessions (upper bound)": sess_h["microvm"] * (2 * 0.1276 + 8 * 0.0169),
        "m5 warm pool": inst_h["m5.xlarge"] * PRICE["m5.xlarge"]}
fig, axs = plt.subplots(1, 2, figsize=(11.5, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
cols = [MUTED, C["gpu"], C["microvm"], C["docker"]]
left = 0
for (k, v), col in zip(cost.items(), cols):
    ax.barh(0, v, left=left, color=col, height=0.5, label=f"{k}: \\${v:.0f}")
    if v > 20:
        ax.text(left + v / 2, 0, f"\\${v:.0f}", ha="center", va="center", fontsize=9, color=BG)
    left += v
ax.set_xlim(0, left * 1.02); ax.set_ylim(-0.5, 1.3); ax.set_yticks([])
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
ax.set_xlabel("US dollars"); ax.set_title(f"One 48-step run (train48 run 2): about \\${left:.0f} in {wall_h:.1f} h", fontsize=10, color=INK, loc="left")
ax = axs[1]
per = {"microVM": cost["microVM sessions (upper bound)"] / eps_n["microvm"] * 1000,
       "m5": cost["m5 warm pool"] / eps_n["docker"] * 1000, "g6": cost["g6 Instances"] / eps_n["gpu"] * 1000}
ax.bar(list(per), list(per.values()), color=[C["microvm"], C["docker"], C["gpu"]], width=0.55)
for i, v in enumerate(per.values()):
    ax.text(i, v + 6, f"\\${v:.0f}", ha="center", fontsize=9, color=INK)
ax.set_ylabel("\\$ per 1,000 episodes"); ax.set_ylim(0, max(per.values()) * 1.25)
ax.set_title("Sandbox cost per 1,000 episodes", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cost.png"), dpi=200); plt.close(fig)
data["cost_train48"] = {"usd": cost, "total": left, "per_1000_eps": per, "session_h": dict(sess_h),
                        "instance_h": dict(inst_h), "episodes": dict(eps_n)}

json.dump(data, open(os.path.join(HERE, "data.json"), "w"), indent=1, default=str)
print(json.dumps(data, indent=1, default=str)[:6000])
