import copy
from typing import Dict, List, Optional

from leap_finetune.distribution.ray_runtime import normalize_visible_devices  # noqa: F401

from ray.actor import ActorHandle
from ray.data import DataIterator, Dataset
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.train import DataConfig

from leap_finetune.distribution.rank_groups import build_replica_group_blocks


class ReplicaGroupDataConfig(DataConfig):
    """Shard datasets by DP group and replicate each shard across a parallel group."""

    def __init__(
        self,
        replica_group_size: int = 1,
        datasets_to_split="all",
        execution_options=None,
        enable_shard_locality: bool = True,
    ):
        super().__init__(
            datasets_to_split=datasets_to_split,
            execution_options=execution_options,
            enable_shard_locality=enable_shard_locality,
        )
        self._replica_group_size = replica_group_size

    def configure(
        self,
        datasets: Dict[str, Dataset],
        world_size: int,
        worker_handles: Optional[List[ActorHandle]],
        worker_node_ids: Optional[List[str]],
        **kwargs,
    ) -> List[Dict[str, DataIterator]]:
        if self._replica_group_size <= 1:
            return super().configure(
                datasets,
                world_size,
                worker_handles,
                worker_node_ids,
                **kwargs,
            )

        replica_blocks = build_replica_group_blocks(
            world_size=world_size,
            replica_group_size=self._replica_group_size,
            worker_node_ids=worker_node_ids,
            require_rank_contiguous=False,
        )
        output = [{} for _ in range(world_size)]

        for dataset_name, dataset in datasets.items():
            if dataset.name is None:
                dataset.set_name(dataset_name)

        if self._datasets_to_split == "all":
            datasets_to_split = set(datasets.keys())
        else:
            datasets_to_split = set(self._datasets_to_split)

        locality_hints = None
        if self._enable_shard_locality and worker_handles is not None:
            locality_hints = [
                worker_handles[group_ranks[0]] for _, group_ranks, _ in replica_blocks
            ]

        for name, ds in datasets.items():
            execution_options = copy.deepcopy(self._execution_options)

            if execution_options.is_resource_limits_default():
                execution_options.exclude_resources = (
                    execution_options.exclude_resources.add(
                        ExecutionResources(
                            cpu=self._num_train_cpus, gpu=self._num_train_gpus
                        )
                    )
                )

            ds = ds.copy(ds)
            ds.context.execution_options = execution_options

            if name not in datasets_to_split:
                for rank in range(world_size):
                    output[rank][name] = ds.iterator()
                continue

            dp_shards = ds.split(
                len(replica_blocks),
                equal=True,
                locality_hints=locality_hints,
            )
            for (_, group_ranks, _), dp_shard in zip(replica_blocks, dp_shards):
                for worker_rank in group_ranks:
                    output[worker_rank][name] = dp_shard.iterator()

        return output


class ExpertParallelDataConfig(ReplicaGroupDataConfig):
    """Shard datasets by DP group and replicate each shard across EP peers."""

    def __init__(self, expert_parallel_size: int = 1, **kwargs):
        super().__init__(replica_group_size=expert_parallel_size, **kwargs)

    def configure(
        self,
        datasets: Dict[str, Dataset],
        world_size: int,
        worker_handles: Optional[List[ActorHandle]],
        worker_node_ids: Optional[List[str]],
        **kwargs,
    ) -> List[Dict[str, DataIterator]]:
        if self._replica_group_size <= 1:
            return super().configure(
                datasets,
                world_size,
                worker_handles,
                worker_node_ids,
                **kwargs,
            )

        # === EP data safety ===
        # Runtime EP uses rank-contiguous DeviceMesh groups. The Ray data replica
        # groups must match those groups exactly, otherwise EP peers exchange expert
        # activations for different logical batches.
        build_replica_group_blocks(
            world_size=world_size,
            replica_group_size=self._replica_group_size,
            worker_node_ids=worker_node_ids,
            require_rank_contiguous=True,
        )
        return super().configure(
            datasets,
            world_size,
            worker_handles,
            worker_node_ids,
            **kwargs,
        )
