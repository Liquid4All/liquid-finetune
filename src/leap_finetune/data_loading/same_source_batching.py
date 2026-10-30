"""Global batches built from same-source sub-batches across Ray-sharded ranks.

Every rank holds its own shard of the rows, but the loss runs over the
gathered global batch, so the source of each sub-batch must be agreed on by
all ranks. Ranks exchange their per-source row counts once; every epoch they
derive the same seeded slot schedule from the global counts, and each rank
fills its share of a slot from its local rows of that source.

A slot holds ``slot_rows`` global rows (``slot_rows / world_size`` per rank)
of one source. Each source gets ``global_count // slot_rows`` slots; the rows
left over after the full slots of every source fill mixed slots, so an epoch
keeps the overall source mixture. A rank whose local share of a source runs
short draws again from a reshuffled pass over that source's rows.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Sequence

import torch.distributed as dist

MIXED = "\x00mixed"


def gather_source_counts(local: Counter[str]) -> Counter[str]:
    """Sum per-source row counts over all ranks (identity without torch.distributed)."""

    if not (dist.is_available() and dist.is_initialized()):
        return Counter(local)
    gathered: list[dict[str, int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(local))
    total: Counter[str] = Counter()
    for counts in gathered:
        total.update(counts or {})
    return total


def slot_schedule(
    counts: Counter[str], slot_rows: int, slots_per_step: int, seed: int
) -> list[str]:
    """Seeded order of slot sources for one epoch, trimmed to whole steps."""

    slots: list[str] = []
    leftover = 0
    for source in sorted(counts):
        full, rest = divmod(counts[source], slot_rows)
        slots.extend([source] * full)
        leftover += rest
    slots.extend([MIXED] * (leftover // slot_rows))
    random.Random(seed).shuffle(slots)
    return slots[: len(slots) - len(slots) % slots_per_step]


class _Cycle:
    """Endless reshuffled passes over a list of row indices."""

    def __init__(self, indices: Sequence[int], rng: random.Random) -> None:
        self._indices = list(indices)
        self._rng = rng
        self._order: list[int] = []

    def take(self, count: int) -> list[int]:
        taken: list[int] = []
        while len(taken) < count:
            if not self._order:
                self._order = self._indices[:]
                self._rng.shuffle(self._order)
            taken.append(self._order.pop())
        return taken


class SameSourceBatchSampler:
    """Per-rank batch sampler whose global batches consist of same-source slots."""

    def __init__(
        self,
        sources: Sequence[str],
        *,
        per_device_batch_size: int,
        slot_rows: int,
        world_size: int,
        rank: int,
        seed: int,
    ) -> None:
        if slot_rows <= 0 or slot_rows % world_size:
            raise ValueError("slot_rows must be a positive multiple of the world size")
        global_batch = per_device_batch_size * world_size
        if global_batch % slot_rows:
            raise ValueError("the global batch size must be a multiple of slot_rows")
        self.rows_per_slot = slot_rows // world_size
        self.slots_per_step = global_batch // slot_rows
        self.slot_rows = slot_rows
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self._local: dict[str, list[int]] = {}
        for index, source in enumerate(sources):
            self._local.setdefault(source, []).append(index)
        self.counts = gather_source_counts(
            Counter({source: len(rows) for source, rows in self._local.items()})
        )
        self._steps = (
            len(slot_schedule(self.counts, slot_rows, self.slots_per_step, seed))
            // self.slots_per_step
        )

    def __len__(self) -> int:
        return self._steps

    def _pools(self, rng: random.Random) -> dict[str, _Cycle]:
        """Split local rows into per-source pools and the mixed leftover pool."""

        pools: dict[str, _Cycle] = {}
        leftover: list[int] = []
        for source in sorted(self._local):
            rows = self._local[source][:]
            rng.shuffle(rows)
            # The local share of the source's full slots; the remainder joins the mixed pool.
            full = self.counts[source] // self.slot_rows * self.slot_rows
            keep = round(len(rows) * full / self.counts[source]) if full else 0
            leftover.extend(rows[keep:])
            if full:
                pools[source] = _Cycle(rows[:keep] or rows, rng)
        pools[MIXED] = _Cycle(
            leftover or [i for r in self._local.values() for i in r], rng
        )
        return pools

    def __iter__(self) -> Iterator[list[int]]:
        # Each pass over the loader is one epoch; all ranks advance together.
        epoch_seed = self.seed + self.epoch
        self.epoch += 1
        schedule = slot_schedule(
            self.counts, self.slot_rows, self.slots_per_step, epoch_seed
        )
        pools = self._pools(random.Random(epoch_seed * 1_000_003 + self.rank))
        for start in range(0, len(schedule), self.slots_per_step):
            batch: list[int] = []
            for source in schedule[start : start + self.slots_per_step]:
                batch.extend(pools[source].take(self.rows_per_slot))
            yield batch
