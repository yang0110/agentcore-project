---
name: kb-self-check
description: One command that runs the grader's checks (build, shape, allclose, cuda device, speedup) on your solution.py before you stop.
families: [kernelbench]
---
# Self-check

The reference `Model` from the task is already saved next to this file as `reference.py`.

```bash
python3 SKILL_DIR/check.py solution.py
```

It prints one line per check and ends with `ALL OK  speedup=1.23x` or `FAIL <check>: <reason>`.
It uses the same seeding and timing as the hidden grader. Treat the speedup as an estimate (±5%).

- `FAIL build`: read the traceback. Usually a signature mismatch or a Triton compile error.
- `FAIL allclose`: almost always different weights. Copy `Model.__init__` verbatim (see `kb-modelnew-contract`).
- `FAIL device`: you returned a CPU tensor.
- `ALL OK` but speedup < 1.10x: correct = 0.5 reward. Try one idea from `kb-speedup-tactics`, re-check,
  and keep the faster correct version.
