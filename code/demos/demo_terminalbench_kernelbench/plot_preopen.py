#!/usr/bin/env python3
"""Bar plot: what pre-open saves per substrate (microVM / m5 / GPU Instance).

Reads the per-episode `[demo] worker=… substrate=… preopened=… open_ms=… open_wait_ms=…` records and
plots, per substrate, the mean time an episode WAITED for its sandbox with pre-open off vs on; a tick
marks the mean boot (open) cost, which pre-open tries to hide.

    # the A/B: the TB_PREOPEN_BATCHES=0 run vs the first STEPS steps of a pre-open-on run
    python3 plot_preopen.py --on logs/fullepoch-hybrid_….log --off logs/preopen-off-hybrid_….log --steps 5 \
        --out logs/preopen_ab.png
    # one log only: "off" = the episodes that could not be pre-opened (the first batch)
    python3 plot_preopen.py --on logs/train48-hybrid_20261005_005629.log --out logs/preopen_inrun.png

Each panel has its own y scale (a microVM boots in ~1 s, a GPU Instance in minutes); --dark renders
for the deck's dark slides.
"""
from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SUBSTRATES = [("microvm", "microVM"), ("docker", "m5 EC2 (Docker)"), ("gpu", "GPU Instance")]
REC = re.compile(r"\[demo\] worker=\S+ (.*)")
STEP = re.compile(r"rollout (\d+): \{'rollout/")
# Validated categorical slots 2 (orange) and 1 (blue), light / dark (dataviz validate_palette.js: all PASS).
THEME = {
    False: dict(off="#eb6834", on="#2a78d6", surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                muted="#898781", grid="#e1e0d9", axis="#c3c2b7"),
    True: dict(off="#d95926", on="#3987e5", surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7",
               muted="#898781", grid="#2c2c2a", axis="#383835"),
}


def records(path: str, steps: int | None):
    """Yield {field: value} per episode, only while fewer than `steps` rollouts have completed."""
    done = 0
    with open(path, errors="replace") as f:
        for line in f:
            if (m := STEP.search(line)):
                done = max(done, int(m.group(1)) + 1)
            if steps is not None and done >= steps:
                break
            if (m := REC.search(line)):
                yield dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)


def collect(on_log, off_log, steps):
    """{substrate: {"off"|"on": {"wait": [s], "open": [s]}}}"""
    out = defaultdict(lambda: {"off": {"wait": [], "open": []}, "on": {"wait": [], "open": []}})
    sources = [(on_log, None)] + ([(off_log, "off")] if off_log else [])
    for path, force in sources:
        for r in records(path, steps):
            try:
                wait, opn = float(r["open_wait_ms"]) / 1000, float(r["open_ms"]) / 1000
            except (KeyError, ValueError):
                continue
            arm = force or ("on" if r.get("preopened") == "1" else "off")
            out[r.get("substrate", "?")][arm]["wait"].append(wait)
            out[r.get("substrate", "?")][arm]["open"].append(opn)
    return out


def fmt(s: float) -> str:
    return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True, help="log of a pre-open-on run")
    ap.add_argument("--off", help="log of a TB_PREOPEN_BATCHES=0 run (else: the on log's preopened=0 episodes)")
    ap.add_argument("--steps", type=int, help="only the first N steps of each log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--title", default="Pre-open hides the sandbox boot")
    a = ap.parse_args()
    t = THEME[a.dark]
    data = collect(a.on, a.off, a.steps)
    subs = [(k, lbl) for k, lbl in SUBSTRATES if any(data[k][arm]["wait"] for arm in ("off", "on"))]

    plt.rcParams.update({"font.size": 11, "text.color": t["ink"], "axes.labelcolor": t["ink2"],
                         "xtick.color": t["ink2"], "ytick.color": t["muted"]})
    fig, axes = plt.subplots(1, len(subs), figsize=(3.6 * len(subs), 4.2), facecolor=t["surface"], squeeze=False)
    for ax, (k, lbl) in zip(axes[0], subs):
        ax.set_facecolor(t["surface"])
        arms = [arm for arm in ("off", "on") if data[k][arm]["wait"]]
        xs = [0 if arm == "off" else 1 for arm in arms]   # fixed slots: a missing arm leaves a gap
        waits = [statistics.mean(data[k][arm]["wait"]) for arm in arms]
        opens = [statistics.mean(data[k][arm]["open"]) for arm in arms]
        ax.bar(xs, waits, width=0.62, color=[t[arm] for arm in arms], edgecolor=t["surface"], linewidth=2)
        ax.scatter(xs, opens, marker="_", s=900, linewidths=2, color=t["ink2"], zorder=3)
        top = max(waits + opens)
        for x, w, o, arm in zip(xs, waits, opens, arms):
            n = len(data[k][arm]["wait"])
            y = w + top * 0.03                      # on the bar; above the boot tick only if they'd collide
            if -top * 0.02 < o - y < top * 0.09:
                y = o + top * 0.03
            ax.text(x, y, fmt(w), ha="center", va="bottom", color=t["ink"],
                    fontsize=11, weight="bold")
            ax.text(x, -top * 0.13, f"n={n}", ha="center", va="top", color=t["muted"], fontsize=9)
        for x, arm in ((0, "off"), (1, "on")):
            if arm not in arms:
                ax.text(x, top * 0.05, "no data", ha="center", va="bottom", color=t["muted"], fontsize=10)
        if len(arms) == 2 and waits[0] > 0:
            ax.set_title(f"{lbl}\nwait {100 * (1 - waits[1] / waits[0]):.0f}% lower with pre-open",
                         color=t["ink"], fontsize=11)
        else:
            ax.set_title(lbl, color=t["ink"], fontsize=11)
        ax.set_xticks([0, 1], ["pre-open off", "pre-open on"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, top * 1.18)
        ax.set_ylabel("episode wait for sandbox (s)" if ax is axes[0][0] else "")
        ax.grid(axis="y", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=t["off"]), plt.Rectangle((0, 0), 1, 1, color=t["on"]),
               plt.Line2D([], [], color=t["ink2"], linewidth=2)]
    fig.legend(handles, ["pre-open off", "pre-open on", "mean boot (open) time"], loc="lower center",
               ncol=3, frameon=False, labelcolor=t["ink2"], bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(a.title, color=t["ink"], fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(a.out, dpi=180, facecolor=t["surface"])
    for k, lbl in subs:
        print(lbl, {arm: (len(v["wait"]), round(statistics.mean(v["wait"]), 2) if v["wait"] else None,
                          round(statistics.mean(v["open"]), 2) if v["open"] else None)
                    for arm, v in data[k].items()})
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
