import os
from pathlib import Path

from leap_finetune.distribution.ray_runtime import (
    build_scaling_config,
    get_ray_env_vars,
    get_requested_ray_address,
    normalize_uv_project_path,
    normalize_visible_devices,
    resolve_local_ray_num_cpus,
    resolve_local_object_store_memory,
    resolve_num_workers,
    select_object_spilling_dir,
    select_ray_temp_dir,
    worker_process_setup_hook,
)

import ray
import ray.data
from accelerate.utils import set_seed
from ray.runtime_env import RuntimeEnv
from ray.train import CheckpointConfig, DataConfig, RunConfig
from ray.train.torch import TorchConfig, TorchTrainer
from torch import cuda

from leap_finetune import RUNTIME_DIR
from leap_finetune.checkpointing.model_info import is_moe_model_from_name
from leap_finetune.checkpointing.model_loading import _resolve_model_id, load_tokenizer
from leap_finetune.checkpointing.callback import hydrate_missing_ray_metrics
from leap_finetune.data_loading.dataset_loader import DatasetLoader
from leap_finetune.data_loading.ray_data_utils import create_ray_datasets
from leap_finetune.distribution.data_sharding import ExpertParallelDataConfig
from leap_finetune.distribution.distributed_configs import (
    strip_distributed_training_config,
)
from leap_finetune.distribution.vllm_server import (
    launch_vllm_server,
    resolve_server_host,
    resolve_vllm_rollout_plan,
)
from leap_finetune.rl.judge import (
    build_judge_runtime_config,
    export_judge_runtime_config,
    get_judge_config,
    judge_needs_local_server,
)
from leap_finetune.training import (
    PRETOKENIZED_TRAINING_TYPES,
    TRAINING_DATASETS_TO_SPLIT,
    TRAINING_LOOPS,
)
from leap_finetune.training.utils.logging import print_next_steps_panel


def ray_trainer(job_config: dict) -> None:
    """Driver entrypoint: prepare Ray/Data, then launch the per-worker train loop."""
    # ==== 1. Resolve runtime ====
    # Pick the train loop, connect to local/external Ray, and decide worker count.
    training_type = job_config["training_type"]

    if training_type in ("sft", "dpo") and is_moe_model_from_name(
        job_config["model_name"]
    ):
        training_type = f"moe_{training_type}"

    output_dir = job_config["training_config"]["output_dir"]

    set_seed(42)
    normalize_uv_project_path()

    ray_config = job_config.get("ray_config")
    ray_address = get_requested_ray_address(ray_config)
    connect_existing_cluster = ray_address is not None
    normalize_visible_devices()
    local_num_gpus = cuda.device_count()
    training_config = job_config["training_config"]
    is_grpo = training_type in ("grpo", "vlm_grpo")
    grpo_rollout_cfg = job_config.get("grpo_rollout") or {}
    rewards_cfg = job_config.get("rewards")
    judge_cfg = get_judge_config(rewards_cfg)
    rollout_plan = None
    server_handles = []

    if is_grpo:
        rollout_plan = resolve_vllm_rollout_plan(
            local_num_gpus,
            grpo_rollout_cfg,
            vllm_mode=training_config.get("vllm_mode", "colocate"),
            is_multi_node=connect_existing_cluster,
            reserve_judge=judge_needs_local_server(rewards_cfg),
        )

        if rollout_plan.uses_custom_training_visibility:
            if ray.is_initialized():
                raise RuntimeError(
                    "GRPO vLLM GPU splitting must be resolved before Ray starts. "
                    "Run this job in a fresh process or use an externally managed "
                    "vLLM server."
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                rollout_plan.training_cuda_visible_devices
            )

        if rollout_plan.launches_local_server:
            host = resolve_server_host(training_config.get("vllm_server_host"))
            port = int(training_config.get("vllm_server_port", 8000))
            server_handle = launch_vllm_server(
                model_id=_resolve_model_id(job_config["model_name"]),
                vllm_gpu_ids=rollout_plan.server_gpu_ids,
                grpo_rollout_cfg=grpo_rollout_cfg,
                host="0.0.0.0",
                port=port,
                vllm_cuda_visible_devices=rollout_plan.server_cuda_visible_devices,
                log_path=Path(output_dir) / "vllm_server.log",
            )
            server_handles.append(server_handle)
            training_config["vllm_server_base_url"] = f"http://{host}:{port}"
            training_config["vllm_server_host"] = host
            training_config["vllm_server_port"] = port

        if judge_cfg is not None:
            judge_base_url = judge_cfg.get("base_url")
            model_id = _resolve_model_id(
                judge_cfg.get("model") or job_config["model_name"]
            )

            if rollout_plan.launches_local_judge:
                host = resolve_server_host(judge_cfg.get("host"))
                port = int(judge_cfg.get("port", 8001))
                judge_handle = launch_vllm_server(
                    model_id=model_id,
                    vllm_gpu_ids=rollout_plan.judge_gpu_ids,
                    grpo_rollout_cfg=_judge_server_config(
                        grpo_rollout_cfg,
                        judge_cfg,
                        len(rollout_plan.judge_gpu_ids),
                    ),
                    host="0.0.0.0",
                    port=port,
                    vllm_cuda_visible_devices=rollout_plan.judge_cuda_visible_devices,
                    log_path=Path(output_dir) / "judge_vllm_server.log",
                )
                server_handles.append(judge_handle)
                judge_base_url = f"http://{host}:{port}"

            export_judge_runtime_config(
                build_judge_runtime_config(
                    rewards_cfg,
                    default_model=model_id,
                    base_url=judge_base_url,
                )
            )
        else:
            export_judge_runtime_config(None)
    else:
        export_judge_runtime_config(None)

    if not ray.is_initialized():
        normalize_visible_devices()
        ray_temp_dir = select_ray_temp_dir(os.path.expanduser("~/tmp-ray"))
        runtime_env = RuntimeEnv(
            working_dir=str(RUNTIME_DIR),
            env_vars=get_ray_env_vars(ray_temp_dir),
            worker_process_setup_hook=(
                "leap_finetune.distribution.ray_runtime.worker_process_setup_hook"
            ),
        )

        if connect_existing_cluster:
            ray.init(address=ray_address or "auto", runtime_env=runtime_env)
        else:
            if not cuda.is_available():
                raise ValueError(
                    "No local GPU available for training. "
                    "Either run on a GPU node or connect to an existing Ray cluster via RAY_ADDRESS."
                )

            spill_dir = select_object_spilling_dir(ray_temp_dir)

            object_store_mem = resolve_local_object_store_memory()
            ray_num_cpus = resolve_local_ray_num_cpus()

            ray_init_kwargs = {
                "address": "local",
                "runtime_env": runtime_env,
                "_temp_dir": ray_temp_dir,
                "object_spilling_directory": spill_dir,
                "object_store_memory": object_store_mem,
            }
            if ray_num_cpus is not None:
                ray_init_kwargs["num_cpus"] = ray_num_cpus

            ray.init(**ray_init_kwargs)

        print("[ray-trainer] Ray initialized", flush=True)

        # Also suppress on driver (must be after ray.init)
        worker_process_setup_hook()

        # Disable progress bar name truncation warning
        ray.data.DataContext.get_current().enable_progress_bar_name_truncation = False

    num_workers = resolve_num_workers(
        ray_config,
        local_num_gpus=local_num_gpus,
        connected_to_existing_cluster=connect_existing_cluster,
    )
    if num_workers < 1:
        raise ValueError("No GPU workers available for Ray training")

    training_config = strip_distributed_training_config(
        training_config,
        num_workers=num_workers,
    )

    train_loop = TRAINING_LOOPS.get(training_type)
    if train_loop is None:
        raise ValueError(
            f"Invalid training type: {training_type}. "
            f"Available: {list(TRAINING_LOOPS.keys())}"
        )

    print("[ray-trainer] Building Ray datasets", flush=True)

    # ==== 2. Build Ray datasets ====
    # The driver validates/tokenizes/packs text data once. Ray shards those
    # materialized datasets across workers when TorchTrainer starts.
    dataset_config = job_config["dataset"]
    use_pretokenize = training_type in PRETOKENIZED_TRAINING_TYPES
    tokenizer = None
    if use_pretokenize:
        tokenizer = load_tokenizer(
            job_config["model_name"],
            chat_template=training_config.get("chat_template"),
            chat_template_path=training_config.get("chat_template_path"),
        )

    if isinstance(dataset_config, DatasetLoader):
        # Pre-tokenize SFT/DPO; retrieval, VLM, and GRPO pass raw rows through.
        train_ds, eval_ds = create_ray_datasets(
            dataset_config,
            tokenizer=tokenizer if use_pretokenize else None,
            training_config=training_config if use_pretokenize else None,
        )
        datasets = {"train": train_ds}
        if eval_ds is not None:
            datasets["eval"] = eval_ds
    elif isinstance(dataset_config, tuple):
        # Legacy path: pre-loaded (Dataset, Dataset) tuple (deprecate eventually)
        train_hf, eval_hf = dataset_config
        train_ds = ray.data.from_arrow(train_hf.data.table)
        datasets = {"train": train_ds}
        if eval_hf is not None:
            datasets["eval"] = ray.data.from_arrow(eval_hf.data.table)
    else:
        raise ValueError(f"Invalid dataset type: {type(dataset_config)}")

    print("[ray-trainer] Ray datasets ready", flush=True)

    # ==== 3. Configure distributed training ====
    # EP uses a custom Ray DataConfig so ranks in each EP group receive the same
    # logical batch. Non-EP lets Ray split one shard per worker normally.
    train_loop_config = {
        "model_name": job_config["model_name"],
        "job_name": job_config.get("job_name", "leap-ft-run"),
        "train_config": training_config,
        "peft_config": job_config["peft_config"],
        "model_config": job_config.get("model_config"),
        "benchmark_configs": job_config.get("benchmark_configs"),
        "rewards": rewards_cfg,
        "async_eval": job_config.get("async_eval"),
        "rl_env": job_config.get("rl_env"),
        "grpo_rollout": job_config.get("grpo_rollout"),
        "config_dir": job_config.get("config_dir"),
    }

    moe_training = training_config.get("moe_training", {})
    ep_size = moe_training.get("expert_parallel_size", 1) or 1
    ray_dataset_config = None
    if training_type in ("moe_sft", "moe_dpo") and ep_size > 1:
        ray_dataset_config = ExpertParallelDataConfig(expert_parallel_size=ep_size)
    elif training_type in TRAINING_DATASETS_TO_SPLIT:
        ray_dataset_config = DataConfig(
            datasets_to_split=list(TRAINING_DATASETS_TO_SPLIT[training_type])
        )

    resources_per_worker = None
    if is_grpo and rollout_plan is not None and not connect_existing_cluster:
        num_workers = rollout_plan.num_training_workers
        resources_per_worker = rollout_plan.resources_per_worker

    async_eval_cfg = job_config.get("async_eval") or {}
    if async_eval_cfg.get("mode") == "reserved":
        if connect_existing_cluster:
            raise NotImplementedError(
                "async_eval mode=reserved is single-node only. "
                "Use mode=sidecar for multi-node training."
            )

        eval_gpu_count = int(async_eval_cfg.get("vllm_gpus", 1))
        current_train_gpus = list(range(num_workers))
        if eval_gpu_count >= len(current_train_gpus):
            raise ValueError(
                f"async_eval.vllm_gpus={eval_gpu_count} leaves no GPUs for training "
                f"(remaining={len(current_train_gpus)})."
            )

        eval_gpus_local = current_train_gpus[:eval_gpu_count]
        train_gpus_local = current_train_gpus[eval_gpu_count:]
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            visible_tokens = [token for token in visible_devices.split(",") if token]
            eval_gpu_ids = [visible_tokens[i] for i in eval_gpus_local]
            train_gpu_ids = [visible_tokens[i] for i in train_gpus_local]
        else:
            eval_gpu_ids = [str(gpu_id) for gpu_id in eval_gpus_local]
            train_gpu_ids = [str(gpu_id) for gpu_id in train_gpus_local]

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(train_gpu_ids)
        num_workers = len(train_gpu_ids)

        eval_port = int((async_eval_cfg.get("reserved") or {}).get("server_port", 8100))
        eval_host = resolve_server_host(None)
        train_loop_config["async_eval_server_url"] = f"http://{eval_host}:{eval_port}"
        train_loop_config["async_eval_gpu_ids"] = ",".join(eval_gpu_ids)
        print(
            f"[async_eval/reserved] reserved GPU(s) {eval_gpu_ids} for vLLM at "
            f"{train_loop_config['async_eval_server_url']}; training on {train_gpu_ids}"
        )

    scale_config = build_scaling_config(
        ray_config,
        num_workers=num_workers,
        resources_per_worker=resources_per_worker,
    )

    run_config = RunConfig(
        storage_path=output_dir,
        name="ray_logs",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
        ),
    )

    if connect_existing_cluster:
        print(
            f"\nTraining on {num_workers} Ray workers "
            f"(RAY_ADDRESS={ray_address or 'auto'})"
        )
    else:
        print(f"\nTraining on {num_workers} GPUs")

    # ==== 4. Launch workers ====
    # Each Ray worker runs TRAINING_LOOPS[training_type] with NCCL initialized by
    # Ray Train. Model loading, FSDP/EP wrapping, training, and saves happen there.
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=train_loop_config,
        scaling_config=scale_config,
        run_config=run_config,
        torch_config=TorchConfig(backend="nccl", timeout_s=7200),
        datasets=datasets,
        dataset_config=ray_dataset_config,
    )

    print("[ray-trainer] TorchTrainer constructed; starting fit", flush=True)

    result = None
    try:
        result = trainer.fit()
    finally:
        for handle in reversed(server_handles):
            handle.stop()

        # Ensure failed pytest cases and repeated in-process launches do not
        # inherit a stale local Ray runtime from the previous training attempt.
        try:
            ray.shutdown()
        except Exception:
            pass

    result = hydrate_missing_ray_metrics(result, output_dir)

    print_next_steps_panel(output_dir)

    return result


def _judge_server_config(
    grpo_rollout_cfg: dict,
    judge_cfg: dict,
    default_tensor_parallel_size: int,
) -> dict:
    return {
        "tensor_parallel_size": int(
            judge_cfg.get(
                "tensor_parallel_size",
                grpo_rollout_cfg.get(
                    "judge_tensor_parallel_size",
                    default_tensor_parallel_size,
                ),
            )
        ),
        "dtype": judge_cfg.get(
            "dtype", grpo_rollout_cfg.get("judge_dtype", "bfloat16")
        ),
        "gpu_memory_utilization": float(
            judge_cfg.get(
                "gpu_memory_utilization",
                grpo_rollout_cfg.get("judge_gpu_memory_utilization", 0.9),
            )
        ),
        "max_model_len": judge_cfg.get(
            "max_model_len",
            grpo_rollout_cfg.get("judge_max_model_len"),
        ),
    }
