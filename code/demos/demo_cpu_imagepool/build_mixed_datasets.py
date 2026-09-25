#!/usr/bin/env python3
"""Build a MIXED-task slime dataset for demo_cpu_imagepool.

The other CPU demos pin one task family per run; demo_cpu_tasks:mixed mixes families but on ONE image
(plus a single pandas add-on). This demo goes further: it interleaves families each of which needs
its OWN prebuilt sandbox image, into ONE `mixed_train.jsonl` / `mixed_eval.jsonl`, so every GRPO
rollout step draws a batch of prompts that spans multiple tasks AND multiple *container images*
(datasci/pandas + scipy + sklearn by default). This is a genuine POOL of images — one image per
family — routed per episode by task.image (capability-routing by family; see
agentcore_slime_demo.sandbox._runtime_arn_for_image + AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE).

Nothing in the harness or the sandbox has to change for this — the reason it works:

  * Each row is SELF-DESCRIBING. It carries `metadata.task = {family, task_id, prompt, spec, ...}`,
    and `agentcore_slime_demo.generate._task_from_sample` reads `metadata.task.family` and rebuilds
    the right `EpisodicTask` subclass per prompt. So `run_episode` already dispatches prompt +
    seed files + hidden checker + reward PER ROW; a step being multi-family is transparent to it.
  * The sandbox is TASK-AGNOSTIC (write_file / exec / read_file). All task specificity lives in the
    per-episode files written into that episode's isolated remote AgentCore runtime SESSION. A math
    episode, a sql episode and a ctf episode are just different files in three identical sessions.

GRPO note: advantage is normalized WITHIN a group of `n_samples_per_prompt` repeats of the SAME
prompt (one family). Mixing happens ACROSS groups in a step, so the different families' reward
SCALES never need reconciling — within-group normalization handles it. Keep n-samples intact; only
the prompt SAMPLER mixes.

Offline + stdlib-only (the seed families are hand-authored/templated, no network/GPU). Row schema is
identical to the other demos' builders:

    {"prompt": <task text>, "label": <task_id>, "metadata": {"task": {<family task kwargs>}}}

Run inside the container:
    python3 /work/code/demos/demo_cpu_imagepool/build_mixed_datasets.py \
        --families datasci,scipy,sklearn --out-dir /work/code/demos/demo_cpu_imagepool/data
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Import ONLY the vendored copy in this demo dir (self-contained; no /work/code on the path).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentcore_slime_demo.task_families import FAMILY_CLASSES

# family label -> EpisodicTask subclass (the same registry the single-family builder uses)
_BY_FAMILY = {cls.family: cls for cls in FAMILY_CLASSES}


def _row(task, max_turns: int | None) -> dict:
    prompt = getattr(task, "initial_user_prompt", lambda: task.prompt)()
    return {"prompt": prompt, "label": task.task_id,
            "metadata": {"task": task.to_row_kwargs(max_turns=max_turns)}}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _interleave(per_family: dict[str, list[dict]]) -> list[dict]:
    """Round-robin across families so any window of the stream spans them evenly (balanced)."""
    out: list[dict] = []
    pools = {fam: list(rows) for fam, rows in per_family.items()}
    while any(pools.values()):
        for fam in list(pools):
            if pools[fam]:
                out.append(pools[fam].pop(0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the mixed-task dataset for demo_cpu_imagepool.")
    ap.add_argument("--families", default="datasci,scipy,sklearn",
                    help="comma-separated family labels to mix (default: datasci,scipy,sklearn — "
                         "the three custom-image families, one prebuilt sandbox image each: "
                         "pandas / scipy / scikit-learn). "
                         f"available: {','.join(sorted(_BY_FAMILY))}")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--max-turns", type=int, default=None,
                    help="override each task's max_turns (fewer => more reward variance in GRPO)")
    ap.add_argument("--eval-frac", type=float, default=0.25,
                    help="fraction of each family's seeds held out for eval (min 2 per family)")
    ap.add_argument("--balanced", action="store_true",
                    help="round-robin interleave families (every step spans them evenly); "
                         "default is a plain shuffle (proportional to each family's seed count)")
    ap.add_argument("--repeat", type=int, default=6,
                    help="replicate the mixed TRAIN pool N times so dataset_size >= NR*RB and one "
                         "epoch spans the whole run => the only checkpoint written is the final one")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in _BY_FAMILY]
    if unknown:
        print(f"unknown families {unknown}; available: {sorted(_BY_FAMILY)}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    train_by_fam: dict[str, list[dict]] = {}
    eval_by_fam: dict[str, list[dict]] = {}

    for fam in families:
        cls = _BY_FAMILY[fam]
        seeds = cls.seeds()
        n_eval = max(2, int(len(seeds) * args.eval_frac))
        eval_seeds, train_seeds = seeds[:n_eval], seeds[n_eval:]
        train_by_fam[fam] = [_row(t, args.max_turns) for t in train_seeds]
        eval_by_fam[fam] = [_row(t, args.max_turns) for t in eval_seeds]
        # per-family eval file too, so a client can score each task in isolation if they want
        _write(out / f"{fam}_eval.jsonl", eval_by_fam[fam])

    # ---- mixed TRAIN: combine, (optionally) balance, replicate, shuffle order -------------------
    if args.balanced:
        base_train = _interleave(train_by_fam)
    else:
        base_train = [r for rows in train_by_fam.values() for r in rows]
        rng.shuffle(base_train)
    mixed_train = list(base_train) * max(1, args.repeat)
    rng.shuffle(mixed_train)  # replicas land at different step positions

    # ---- mixed EVAL: one interleaved held-out set spanning every family -------------------------
    mixed_eval = _interleave(eval_by_fam)

    _write(out / "mixed_train.jsonl", mixed_train)
    _write(out / "mixed_eval.jsonl", mixed_eval)

    print(f"families mixed: {families}  (balanced={args.balanced}, repeat={args.repeat})")
    for fam in families:
        print(f"  {fam:9s} -> {len(train_by_fam[fam]):2d} train seeds + {len(eval_by_fam[fam])} eval")
    print(f"mixed_train.jsonl -> {len(mixed_train)} prompts  "
          f"({len(base_train)} unique x {max(1, args.repeat)})")
    print(f"mixed_eval.jsonl  -> {len(mixed_eval)} prompts")
    print(f"out -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
