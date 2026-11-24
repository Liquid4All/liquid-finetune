from __future__ import annotations

import argparse
import json
import os
import socket

import torch
import torch.distributed as dist
from transformers import TrainingArguments

from liquid_finetune.checkpointing.manual_sharded import (
    export_manual_sharded_checkpoint_as_hf,
    load_manual_sharded_checkpoint_metadata,
)
from liquid_finetune.checkpointing.model_loading import load_model
from liquid_finetune.training.sft import build_sft_data_collator
from liquid_finetune.training.moe_utils.ep_runtime import apply_fsdp2, create_dp_mesh


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_distributed_for_export() -> None:
    if dist.is_available() and dist.is_initialized():
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Manual-sharded export requires CUDA. Run this command on a GPU host "
            "or inside an allocated GPU shell."
        )

    if os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
        dist.init_process_group(backend="nccl")
    else:
        init_method = os.environ.get(
            "LIQUID_EXPORT_INIT_METHOD",
            f"tcp://127.0.0.1:{_free_local_port()}",
        )
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=0,
            world_size=1,
        )

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _load_model_config(args: argparse.Namespace, metadata: dict) -> dict | None:
    if args.model_config_json and args.model_config_file:
        raise ValueError("Use only one of --model-config-json or --model-config-file")
    if args.model_config_file:
        with open(args.model_config_file) as handle:
            return json.load(handle)
    if args.model_config_json:
        return json.loads(args.model_config_json)
    return metadata.get("model_config")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a manual-sharded MoE checkpoint into a standalone HF model "
            "directory. This uses a local single-process CUDA group; no Slurm is "
            "required if you are already on a GPU host."
        )
    )
    parser.add_argument("--model-name", "--base-model", dest="model_name", default=None)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chat-template-path", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--model-config-json", default=None)
    parser.add_argument("--model-config-file", default=None)
    parser.add_argument("--checkpoint-staging-dir", default=None)
    parser.add_argument("--reshard-after-forward", action="store_true", default=True)
    parser.add_argument(
        "--no-reshard-after-forward",
        dest="reshard_after_forward",
        action="store_false",
    )
    parser.add_argument("--cpu-offload", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _init_distributed_for_export()
    metadata = load_manual_sharded_checkpoint_metadata(args.checkpoint_dir)
    model_config = _load_model_config(args, metadata)
    model_name = args.model_name or metadata.get("base_model_name")
    if not model_name:
        raise ValueError(
            "--model-name is required for checkpoints that do not record base_model_name"
        )
    max_length = args.max_length or metadata.get("max_length") or 120000
    chat_template = None if args.chat_template_path else metadata.get("chat_template")

    model, tokenizer = load_model(
        model_name,
        model_config=model_config,
        chat_template=chat_template,
        chat_template_path=args.chat_template_path,
    )

    dp_mesh = create_dp_mesh()
    model = apply_fsdp2(
        model,
        dp_mesh,
        reshard_after_forward=args.reshard_after_forward,
        cpu_offload=args.cpu_offload,
    )

    data_collator = build_sft_data_collator(
        tokenizer,
        {
            "assistant_only_loss": True,
            "max_length": max_length,
            "packing": False,
            "padding_free": False,
        },
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        bf16=torch.cuda.is_available(),
        per_device_train_batch_size=1,
        learning_rate=args.learning_rate,
        save_strategy="no",
        report_to=[],
    )

    export_manual_sharded_checkpoint_as_hf(
        model=model,
        accelerator=None,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        processing_class=tokenizer,
        data_collator=data_collator,
        training_args=training_args,
        checkpoint_staging_dir=args.checkpoint_staging_dir,
        export_metadata={
            **metadata,
            "base_model_name": model_name,
            "model_config": model_config,
            "max_length": max_length,
            "chat_template": tokenizer.chat_template,
        },
    )

    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Exported HF checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
