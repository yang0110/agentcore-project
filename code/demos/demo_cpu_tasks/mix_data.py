"""Build a TASK-INTERLEAVED train file so every rollout batch mixes tasks (called by run.sh; stdlib only).

    python3 mix_data.py OUT SRC [SRC ...]          # MIX_MIN_ROWS=N: repeat passes until >= N rows

Rows are grouped into strata by (source file, task family) — e.g. DATASET=mbpp,gsm8k,gretelsql,hotpotqa
gives coding / math / sql / rag, and DATASET=mixed gives its sql / math / ctf / datasci rows. Each stratum
is shuffled with a fixed seed, truncated to the smallest one (balanced), and the strata are dealt
round-robin: row i belongs to stratum i mod K. run.sh then launches slime WITHOUT --rollout-shuffle, so a
batch of RB consecutive prompts spans min(RB, K) tasks (RB=4, K=4: one prompt of each task per step).
Since slime no longer reshuffles per epoch, the file holds as many passes as the run needs (MIX_MIN_ROWS =
NR x RB, set by run.sh), each pass reshuffled within its strata.

Prints the stratum sizes to stderr and the number of strata K to stdout (1 = nothing to mix).
"""

from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def family_of(row: dict) -> str:
    from agentcore_slime_demo.generate import _task_from_sample

    class _Row:   # the slice of a slime Sample that _task_from_sample reads
        def __init__(self, r: dict) -> None:
            self.metadata, self.prompt = r.get("metadata") or {}, r.get("prompt")

    return getattr(_task_from_sample(_Row(row)), "family", "coding") or "coding"


def main(out: str, srcs: list[str], seed: int = 1234) -> int:
    strata: dict[str, list[str]] = {}
    for p in srcs:
        name = os.path.basename(p).rsplit("_", 1)[0]
        for line in open(p):
            if line.strip():
                key = f"{name}:{family_of(json.loads(line))}" if len(srcs) > 1 else family_of(json.loads(line))
                strata.setdefault(key, []).append(line if line.endswith("\n") else line + "\n")
    keys = sorted(strata)
    n = min(len(strata[k]) for k in keys)
    passes = max(1, -(-int(os.environ.get("MIX_MIN_ROWS", "0") or 0) // (n * len(keys))))
    rng = random.Random(seed)
    with open(out, "w") as f:
        for _ in range(passes):
            for k in keys:
                rng.shuffle(strata[k])
            for i in range(n):
                for k in keys:
                    f.write(strata[k][i])
    sizes = ", ".join(f"{k}={len(strata[k])}" for k in keys)
    print(f"[mix] {out}: {len(keys)} task strata x {n} rows x {passes} pass(es) (balanced; from {sizes})",
          file=sys.stderr)
    print(len(keys))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
