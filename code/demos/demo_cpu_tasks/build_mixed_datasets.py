#!/usr/bin/env python3
"""Build the prebuilt MIXED-task slime dataset for demo_cpu_tasks (`DATASET=mixed`).

A single-`DATASET` run pins one task family. `DATASET=mixed` does the opposite: it interleaves
SEVERAL families into ONE `mixed_train.jsonl` / `mixed_eval.jsonl`, so that every GRPO rollout
step draws a batch of prompts that spans multiple tasks (math + sql + ctf + datasci by default).
The default mix includes `datasci` (pandas), which the default managed Code Interpreter already
ships (pandas/scipy/sklearn preinstalled), so no custom image is needed. Rows still carry `task.image`;
the vendored lib's image routing (agentcore_slime_demo.sandbox._runtime_arn_for_image) only matters
when you run on your own Runtimes with a per-image ARN map.

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
    python3 /work/code/demos/demo_cpu_tasks/build_mixed_datasets.py \
        --families math,sql,ctf,datasci --out-dir /work/code/demos/demo_cpu_tasks/data
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
    ap = argparse.ArgumentParser(description="Build the mixed-task dataset (DATASET=mixed) for demo_cpu_tasks.")
    ap.add_argument("--name", default="mixed",
                    help="output name: <name>_train.jsonl / <name>_eval.jsonl / <name>_<fam>_eval.jsonl "
                         "(default mixed; DATASET=libs uses --name libs --families datasci,scipy,sklearn)")
    ap.add_argument("--families", default="math,sql,ctf,datasci",
                    help="comma-separated family labels to mix (default: math,sql,ctf,datasci; "
                         "datasci needs pandas — preinstalled in the default Code Interpreter). "
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
        # per-family eval file too, so a client can score each task in isolation if they want.
        # Prefixed mixed_ so it never overwrites a same-named single-DATASET eval file (e.g. the
        # synthetic-zoo data/ctf_eval.jsonl) that lives in the same data/ dir.
        _write(out / f"{args.name}_{fam}_eval.jsonl", eval_by_fam[fam])

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

    _write(out / f"{args.name}_train.jsonl", mixed_train)
    _write(out / f"{args.name}_eval.jsonl", mixed_eval)

    print(f"families mixed: {families}  (balanced={args.balanced}, repeat={args.repeat})")
    for fam in families:
        print(f"  {fam:9s} -> {len(train_by_fam[fam]):2d} train seeds + {len(eval_by_fam[fam])} eval")
    print(f"{args.name}_train.jsonl -> {len(mixed_train)} prompts  "
          f"({len(base_train)} unique x {max(1, args.repeat)})")
    print(f"{args.name}_eval.jsonl  -> {len(mixed_eval)} prompts")
    print(f"out -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
