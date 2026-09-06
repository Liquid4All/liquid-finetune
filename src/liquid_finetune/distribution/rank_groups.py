from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class WorkerTopology:
    rank: int
    node_id: str
    local_order: int


def build_node_local_blocks(
    workers: Sequence[WorkerTopology],
    group_size: int,
) -> list[tuple[str, list[int], int]]:
    """Build node-local rank groups of a fixed size.

    Returns tuples of `(node_id, ranks, local_group_index)`, where `local_group_index`
    is the ordinal of the group within that node.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")

    by_node: dict[str, list[WorkerTopology]] = defaultdict(list)
    for worker in workers:
        by_node[worker.node_id].append(worker)

    blocks: list[tuple[str, list[int], int]] = []
    for node_id in sorted(by_node):
        ordered_workers = sorted(
            by_node[node_id],
            key=lambda worker: (worker.local_order, worker.rank),
        )
        ordered_ranks = [worker.rank for worker in ordered_workers]
        if len(ordered_ranks) % group_size != 0:
            raise ValueError(
                f"node {node_id} has {len(ordered_ranks)} ranks, which is not divisible "
                f"by group_size ({group_size})"
            )

        for local_group_index, start in enumerate(
            range(0, len(ordered_ranks), group_size)
        ):
            blocks.append(
                (
                    node_id,
                    ordered_ranks[start : start + group_size],
                    local_group_index,
                )
            )

    return blocks


def build_replica_group_blocks(
    world_size: int,
    replica_group_size: int,
    worker_node_ids: Sequence[str] | None,
    *,
    require_rank_contiguous: bool = False,
) -> list[tuple[str, list[int], int]]:
    """Build node-local replica groups for rank-based worker layouts."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if replica_group_size < 1:
        raise ValueError(f"replica_group_size must be >= 1, got {replica_group_size}")
    if world_size % replica_group_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by replica_group_size "
            f"({replica_group_size})"
        )

    if worker_node_ids is not None and len(worker_node_ids) != world_size:
        raise ValueError(
            f"worker_node_ids length ({len(worker_node_ids)}) must equal world_size "
            f"({world_size})"
        )

    node_ids = (
        list(worker_node_ids)
        if worker_node_ids is not None
        else ["default"] * world_size
    )
    workers = [
        WorkerTopology(rank=rank, node_id=node_id, local_order=rank)
        for rank, node_id in enumerate(node_ids)
    ]
    blocks = build_node_local_blocks(workers, replica_group_size)
    if require_rank_contiguous:
        expected = [
            list(range(start, start + replica_group_size))
            for start in range(0, world_size, replica_group_size)
        ]
        actual = [ranks for _, ranks, _ in sorted(blocks, key=lambda item: item[1][0])]
        if actual != expected:
            raise ValueError(
                "EP Ray data replica groups must match rank-contiguous EP process "
                f"groups. Expected {expected}, got {actual}. This protects against "
                "replicating different dataset shards than the ranks that exchange "
                "expert activations."
            )

    return blocks
