import pytest

from liquid_finetune.distribution.data_sharding import ExpertParallelDataConfig
from liquid_finetune.distribution.rank_groups import build_replica_group_blocks


def test_build_replica_group_blocks_chunks_workers_per_node():
    blocks = build_replica_group_blocks(
        world_size=8,
        replica_group_size=2,
        worker_node_ids=["node-a"] * 4 + ["node-b"] * 4,
    )

    assert blocks == [
        ("node-a", [0, 1], 0),
        ("node-a", [2, 3], 1),
        ("node-b", [4, 5], 0),
        ("node-b", [6, 7], 1),
    ]


def test_build_replica_group_blocks_handles_interleaved_nodes():
    blocks = build_replica_group_blocks(
        world_size=4,
        replica_group_size=2,
        worker_node_ids=["node-a", "node-b", "node-a", "node-b"],
    )

    assert blocks == [
        ("node-a", [0, 2], 0),
        ("node-b", [1, 3], 0),
    ]


def test_rank_contiguous_replica_groups_reject_interleaved_nodes():
    with pytest.raises(ValueError, match="rank-contiguous EP process groups"):
        build_replica_group_blocks(
            world_size=4,
            replica_group_size=2,
            worker_node_ids=["node-a", "node-b", "node-a", "node-b"],
            require_rank_contiguous=True,
        )


def test_expert_parallel_data_config_requires_ep_group_alignment():
    config = ExpertParallelDataConfig(expert_parallel_size=2)

    with pytest.raises(ValueError, match="rank-contiguous EP process groups"):
        config.configure(
            datasets={},
            world_size=4,
            worker_handles=None,
            worker_node_ids=["node-a", "node-b", "node-a", "node-b"],
        )


def test_replica_groups_share_the_same_dp_shard_assignment():
    blocks = build_replica_group_blocks(
        world_size=8,
        replica_group_size=2,
        worker_node_ids=["node-a"] * 4 + ["node-b"] * 4,
    )
    rank_to_shard_index = {
        rank: shard_index
        for shard_index, (_, ranks, _) in enumerate(blocks)
        for rank in ranks
    }

    assert rank_to_shard_index[0] == rank_to_shard_index[1]
    assert rank_to_shard_index[2] == rank_to_shard_index[3]
    assert rank_to_shard_index[4] == rank_to_shard_index[5]
    assert rank_to_shard_index[6] == rank_to_shard_index[7]

    assert rank_to_shard_index[0] != rank_to_shard_index[2]
    assert rank_to_shard_index[2] != rank_to_shard_index[4]
    assert rank_to_shard_index[4] != rank_to_shard_index[6]


def test_build_replica_group_blocks_requires_full_groups_per_node():
    with pytest.raises(
        ValueError,
        match="node node-a has 3 ranks, which is not divisible by group_size \\(2\\)",
    ):
        build_replica_group_blocks(
            world_size=4,
            replica_group_size=2,
            worker_node_ids=["node-a", "node-a", "node-a", "node-b"],
        )
