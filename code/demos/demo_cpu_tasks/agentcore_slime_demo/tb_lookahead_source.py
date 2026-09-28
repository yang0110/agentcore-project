"""Look-ahead data source for demo_cpu_tasks' Terminal-Bench rows — lets the pool pre-open the NEXT batches' sessions.

Wire with:
    --data-source-path agentcore_slime_demo.tb_lookahead_source.LookaheadDataSource

The pool alone can only pre-open episodes slime has already handed it, and slime hands over at most
`CONC x num_engines` of the current batch (the rest wait on its semaphore) and does not draw batch N+1
until batch N's rollout is done. So the pool can never see far enough ahead to hide a cold start of
minutes. This source can: slime's prompt order is deterministic (sequential offset through a dataset
shuffled with `random.seed(seed + epoch)` each epoch), so on every draw it also PEEKS the prompts of
the next TB_PREOPEN_BATCHES rollouts — without advancing its own state — and tells the pool the
ordered list of upcoming episodes (each prompt x n_samples_per_prompt). The pool then keeps sessions
booted for the first TB_PREOPEN_BATCHES x RB x NS of them (TerminalBenchPool.reserve / _refill).
In a mixed batch only the Terminal-Bench rows are announced; the other rows open their own sessions.

Assumes one draw per rollout (over_sampling_batch_size == rollout_batch_size, no dynamic filter),
which is how run.sh launches it: each draw REPLACES the pool's upcoming-episode list, so episodes left
over from an earlier draw are never held open. If that assumption breaks, the only cost is that some
episodes open their session on demand instead of pre-opened.
"""

from __future__ import annotations

import logging
import os
import random

from slime.rollout.data_source import RolloutDataSourceWithBuffer

logger = logging.getLogger(__name__)


class LookaheadDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        super().__init__(args)
        self.lookahead_batches = int(os.environ.get("TB_PREOPEN_BATCHES", "2"))
        self._draws = int(getattr(args, "start_rollout_id", 0) or 0)

    def _epoch_samples(self, epoch_id: int) -> list:
        """The dataset's prompt order in `epoch_id`, reproducing Dataset.shuffle without mutating it."""
        ds = self.dataset
        if not self.args.rollout_shuffle:
            return ds.origin_samples
        if epoch_id == ds.epoch_id:
            return ds.samples
        perm = list(range(len(ds.origin_samples)))
        random.Random(ds.seed + epoch_id).shuffle(perm)   # == random.seed(seed+epoch); random.shuffle
        return [ds.origin_samples[i] for i in perm]

    def _peek(self, num_prompts: int) -> list:
        """The next `num_prompts` prompts get_samples WOULD return, without advancing any state."""
        out, off, epoch = [], self.sample_offset, self.epoch_id
        samples = self._epoch_samples(epoch)
        while len(out) < num_prompts and samples:
            if off >= len(samples):
                epoch, off = epoch + 1, 0
                samples = self._epoch_samples(epoch)
            take = min(num_prompts - len(out), len(samples) - off)
            out += samples[off:off + take]
            off += take
        return out

    def get_samples(self, num_samples: int):
        groups = super().get_samples(num_samples)
        self._draws += 1
        if self.dataset is None or self.lookahead_batches <= 0:
            return groups
        try:
            from .terminalbench_pool import get_tb_pool

            pool = get_tb_pool(self.args)
            rb, ns = self.args.rollout_batch_size, self.args.n_samples_per_prompt
            remaining = max(0, self.args.num_rollout - self._draws) * rb   # don't boot past the last step
            future = self._peek(min(self.lookahead_batches * rb, remaining))
            drawn = [a for g in groups for a in [pool.arn_for_sample(g[0])] * len(g) if a]
            upcoming = [a for s in future for a in [pool.arn_for_sample(s)] * ns if a]
            for g in groups:
                if pool.arn_for_sample(g[0]):
                    for s in g:
                        s.metadata = dict(s.metadata or {}, tb_reserved=1)
            pool.reserve(drawn, upcoming)
            logger.info("[tb_lookahead] draw %d: %d episodes drawn + %d upcoming (%d batches ahead)",
                        self._draws, len(drawn), len(upcoming), self.lookahead_batches)
        except Exception as e:  # noqa: BLE001 — look-ahead is an optimisation; never break the draw
            logger.warning("[tb_lookahead] look-ahead failed (episodes will open on demand): %s", e)
        return groups
