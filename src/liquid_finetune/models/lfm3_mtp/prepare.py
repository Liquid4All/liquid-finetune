"""Turn an LFM3 checkpoint into a starting point for MTP training (``liquid-prepare-mtp``).

The output is a regular checkpoint directory that trains like any other model:
``config.json`` maps ``AutoModelForCausalLM`` to
``modeling_lfm3_mtp.Lfm3ForCausalLMWithMTP`` and carries the ``mtp_*`` settings, and
the weights gain ``mtp.*`` (and ``mtp2.*`` / ``mtp3.*``) tensors using Qwen3.5-style
component names. An inference engine still needs an LFM3-specific MTP implementation;
the prepared checkpoint is not by itself sufficient for vLLM speculative decoding.

Modes:
  heads       frozen backbone, MTP loss only (the served model's outputs do not change)
  joint       main loss + ``loss_weight`` * MTP loss, backbone trains too
  add-module  append the next sequential module (2 or 3), initialised from the last one,
              and train only that module (backbone and earlier modules stay frozen)
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shutil

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

REMOTE_CODE = pathlib.Path(__file__).parent / "remote_code" / "modeling_lfm3_mtp.py"
MTP_CLASS = "modeling_lfm3_mtp.Lfm3ForCausalLMWithMTP"
MODES = ("heads", "joint", "add-module")
_MODULE_PREFIX = {1: "mtp.", 2: "mtp2.", 3: "mtp3."}


def _resolve_source(source: str, revision: str | None) -> pathlib.Path:
    path = pathlib.Path(source).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(source, revision=revision))


def _load_weights(model_dir: pathlib.Path) -> dict[str, torch.Tensor]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(str(model_dir / shard)))
    return state


def _save_weights(model_dir: pathlib.Path, state: dict[str, torch.Tensor]) -> None:
    for stale in list(model_dir.glob("model*.safetensors")) + list(
        model_dir.glob("model.safetensors.index.json")
    ):
        stale.unlink()
    save_file(
        {k: v.contiguous() for k, v in state.items()},
        str(model_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )


def _present_modules(state: dict[str, torch.Tensor]) -> list[int]:
    return [
        index
        for index, prefix in _MODULE_PREFIX.items()
        if any(key.startswith(prefix) for key in state)
    ]


def _default_init_layer(config: dict) -> int:
    return max(
        index
        for index, layer_type in enumerate(config["layer_types"])
        if layer_type == "full_attention"
    )


def _init_first_module(
    state: dict[str, torch.Tensor], config: dict, source_layer: int
) -> int:
    """``mtp.*`` from decoder layer ``source_layer``; fc averages the two inputs."""
    hidden = config["hidden_size"]
    prefix = f"model.layers.{source_layer}."
    layer = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not layer:
        raise ValueError(f"No tensors under {prefix}")
    dtype = next(iter(layer.values())).dtype
    new = {f"mtp.layers.0.{k}": v.clone() for k, v in layer.items()}
    eye = torch.eye(hidden, dtype=torch.float32)
    new["mtp.fc.weight"] = (0.5 * torch.cat([eye, eye], dim=1)).to(dtype)
    new["mtp.pre_fc_norm_embedding.weight"] = torch.ones(hidden, dtype=dtype)
    new["mtp.pre_fc_norm_hidden.weight"] = torch.ones(hidden, dtype=dtype)
    new["mtp.norm.weight"] = state["model.embedding_norm.weight"].clone()
    state.update(new)
    return len(new)


def prepare_lfm3_mtp_checkpoint(
    source: str,
    output_dir: str,
    *,
    mode: str = "heads",
    loss_weight: float = 0.3,
    init_from_layer: int | None = None,
    objective: str = "hard",
    revision: str | None = None,
) -> dict:
    """Write an MTP training start point for ``source`` to ``output_dir``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if objective not in ("hard", "soft"):
        raise ValueError(f"objective must be 'hard' or 'soft', got {objective!r}")
    if loss_weight < 0:
        raise ValueError(f"loss_weight must be non-negative, got {loss_weight}")

    source_dir = _resolve_source(source, revision)
    out = pathlib.Path(output_dir).expanduser()
    if out.resolve().is_relative_to(source_dir.resolve()):
        raise ValueError("output_dir must not be inside the source checkpoint")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    shutil.copytree(source_dir, out, dirs_exist_ok=True)
    shutil.copy2(REMOTE_CODE, out / REMOTE_CODE.name)

    config_path = out / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "lfm3":
        raise ValueError(
            f"Expected an LFM3 checkpoint, got {config.get('model_type')!r}"
        )
    layer = _default_init_layer(config) if init_from_layer is None else init_from_layer
    if not 0 <= layer < len(config["layer_types"]):
        raise ValueError(
            f"init_from_layer must be between 0 and {len(config['layer_types']) - 1}, got {layer}"
        )
    if config["layer_types"][layer] not in ("full_attention", "sliding_attention"):
        raise ValueError(
            f"init_from_layer {layer} is a {config['layer_types'][layer]} layer"
        )

    state = _load_weights(out)
    present = _present_modules(state)
    if mode == "joint" and len(present) > 1:
        raise ValueError("joint mode supports a single MTP module")
    added = 0
    if mode == "add-module":
        if not present:
            raise ValueError(
                "add-module needs a checkpoint with a trained mtp.* module"
            )
        new_index = max(present) + 1
        if new_index > 3:
            raise ValueError("At most 3 MTP modules are supported")
        last = _MODULE_PREFIX[max(present)]
        copied = {
            _MODULE_PREFIX[new_index] + k[len(last) :]: v.clone()
            for k, v in state.items()
            if k.startswith(last)
        }
        state.update(copied)
        added = len(copied)
        num_modules, train_modules = new_index, [new_index]
    else:
        if not present:
            added = _init_first_module(state, config, layer)
        num_modules, train_modules = max(present or [1]), None

    config.setdefault("auto_map", {})["AutoModelForCausalLM"] = MTP_CLASS
    config["architectures"] = [MTP_CLASS.split(".")[1]]
    config.update(
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        mtp_init_from_layer=config.get("mtp_init_from_layer", layer),
        mtp_loss_weight=loss_weight,
        mtp_train_only=mode != "joint",
        mtp_soft=objective == "soft",
        mtp_num_modules=num_modules,
    )
    if train_modules is not None:
        config["mtp_train_modules"] = train_modules
    else:
        config.pop("mtp_train_modules", None)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if added:
        _save_weights(out, state)

    summary = {
        "output_dir": str(out),
        "mode": mode,
        "objective": objective,
        "mtp_modules": num_modules,
        "added_tensors": added,
        "init_from_layer": config["mtp_init_from_layer"],
    }
    logger.info("Prepared LFM3 MTP checkpoint: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("source", help="LFM3 checkpoint directory or Hub model id")
    parser.add_argument("output_dir", help="New directory for the MTP start point")
    parser.add_argument("--mode", choices=MODES, default="heads")
    parser.add_argument("--loss-weight", type=float, default=0.3)
    parser.add_argument("--init-from-layer", type=int, default=None)
    parser.add_argument("--objective", choices=("hard", "soft"), default="hard")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(
        json.dumps(
            prepare_lfm3_mtp_checkpoint(
                args.source,
                args.output_dir,
                mode=args.mode,
                loss_weight=args.loss_weight,
                init_from_layer=args.init_from_layer,
                objective=args.objective,
                revision=args.revision,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
