"""Turn an LFM3 or LFM3-VL checkpoint into a starting point for MTP training (``liquid-prepare-mtp``).

The output is a regular checkpoint directory that trains like any other model:
``config.json`` maps ``AutoModelForCausalLM`` to ``modeling_lfm3_mtp.Lfm3ForCausalLMWithMTP``
(text) or ``AutoModelForImageTextToText`` to
``modeling_lfm3_vl_mtp.Lfm3VlForConditionalGenerationWithMTP`` (vision-language) and carries
the ``mtp_*`` settings, and the weights gain top-level ``mtp.*`` (and ``mtp2.*`` ... ``mtp5.*``)
tensors using Qwen3.5-style component names, for either backbone. An inference engine still needs
an LFM3-specific MTP implementation; the prepared checkpoint is not by itself sufficient for vLLM
speculative decoding.

Modes:
  heads       frozen backbone, MTP loss only (the served model's outputs do not change)
  joint       main loss + ``loss_weight`` * MTP loss, backbone trains too
  add-module  append the next sequential module (up to 5), initialised from the last one,
              and train only that module (backbone and earlier modules stay frozen)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import shutil
from dataclasses import dataclass

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

REMOTE_CODE_DIR = pathlib.Path(__file__).parent / "remote_code"


@dataclass(frozen=True)
class _Backbone:
    """Where an LFM3 backbone keeps its text decoder, and how it is auto-loaded."""

    auto_class: str
    mtp_class: str
    remote_code: tuple[str, ...]
    layers_prefix: str
    final_norm_key: str
    nested_text_config: bool = False

    def text_config(self, config: dict) -> dict:
        return config["text_config"] if self.nested_text_config else config


_BACKBONES = {
    "lfm3": _Backbone(
        auto_class="AutoModelForCausalLM",
        mtp_class="modeling_lfm3_mtp.Lfm3ForCausalLMWithMTP",
        remote_code=("modeling_lfm3_mtp.py",),
        layers_prefix="model.layers.",
        final_norm_key="model.embedding_norm.weight",
    ),
    "lfm3_vl": _Backbone(
        auto_class="AutoModelForImageTextToText",
        mtp_class="modeling_lfm3_vl_mtp.Lfm3VlForConditionalGenerationWithMTP",
        remote_code=("modeling_lfm3_mtp.py", "modeling_lfm3_vl_mtp.py"),
        layers_prefix="model.language_model.layers.",
        final_norm_key="model.language_model.embedding_norm.weight",
        nested_text_config=True,
    ),
}
MODES = ("heads", "joint", "add-module")
MAX_DRAFT_TOKENS = 5
MANIFEST_NAME = "mtp_manifest.json"
_MODULE_PREFIX = {
    d: ("mtp." if d == 1 else f"mtp{d}.") for d in range(1, MAX_DRAFT_TOKENS + 1)
}


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
    """Write all weights as one ``model.safetensors``; the old shards go only once it is complete."""
    tmp = model_dir / "model.safetensors.tmp"
    save_file(
        {k: v.contiguous() for k, v in state.items()},
        str(tmp),
        metadata={"format": "pt"},
    )
    for stale in list(model_dir.glob("model*.safetensors")) + list(
        model_dir.glob("model.safetensors.index.json")
    ):
        stale.unlink()
    os.replace(tmp, model_dir / "model.safetensors")


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
    state: dict[str, torch.Tensor],
    text_config: dict,
    source_layer: int,
    backbone: _Backbone,
) -> int:
    """``mtp.*`` from decoder layer ``source_layer``; fc averages the two inputs."""
    hidden = text_config["hidden_size"]
    prefix = f"{backbone.layers_prefix}{source_layer}."
    layer = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not layer:
        raise ValueError(f"No tensors under {prefix}")
    dtype = next(iter(layer.values())).dtype
    new = {f"mtp.layers.0.{k}": v.clone() for k, v in layer.items()}
    eye = torch.eye(hidden, dtype=torch.float32)
    new["mtp.fc.weight"] = (0.5 * torch.cat([eye, eye], dim=1)).to(dtype)
    new["mtp.pre_fc_norm_embedding.weight"] = torch.ones(hidden, dtype=dtype)
    new["mtp.pre_fc_norm_hidden.weight"] = torch.ones(hidden, dtype=dtype)
    new["mtp.norm.weight"] = state[backbone.final_norm_key].clone()
    state.update(new)
    return len(new)


def _copy_module(state: dict[str, torch.Tensor], src: int, dst: int) -> int:
    """Initialise module ``dst`` as a copy of module ``src``."""
    old, new = _MODULE_PREFIX[src], _MODULE_PREFIX[dst]
    copied = {
        new + k[len(old) :]: v.clone() for k, v in state.items() if k.startswith(old)
    }
    state.update(copied)
    return len(copied)


def prepare_lfm3_mtp_checkpoint(
    source: str,
    output_dir: str,
    *,
    mode: str = "heads",
    loss_weight: float = 0.3,
    init_from_layer: int | None = None,
    objective: str = "hard",
    num_draft_tokens: int | None = None,
    revision: str | None = None,
) -> dict:
    """Write an MTP training start point for ``source`` to ``output_dir``.

    ``num_draft_tokens`` (1-5) is the number of sequential draft modules: module d
    drafts the token d positions after the verifier's next token. A checkpoint with
    fewer modules is topped up with copies of its last module and only the new ones
    train; a fresh checkpoint gets all of them (module 1 from ``init_from_layer``).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if objective not in ("hard", "soft"):
        raise ValueError(f"objective must be 'hard' or 'soft', got {objective!r}")
    if loss_weight < 0:
        raise ValueError(f"loss_weight must be non-negative, got {loss_weight}")
    if num_draft_tokens is not None and not 1 <= num_draft_tokens <= MAX_DRAFT_TOKENS:
        raise ValueError(
            f"num_draft_tokens must be 1-{MAX_DRAFT_TOKENS}, got {num_draft_tokens}"
        )
    if mode == "add-module" and num_draft_tokens is not None:
        raise ValueError(
            "add-module appends exactly one module; leave num_draft_tokens unset"
        )

    source_dir = _resolve_source(source, revision)
    out = pathlib.Path(output_dir).expanduser()
    if out.resolve().is_relative_to(source_dir.resolve()):
        raise ValueError("output_dir must not be inside the source checkpoint")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    request = {
        "source": source,
        "revision": revision,
        "mode": mode,
        "objective": objective,
        "num_draft_tokens": num_draft_tokens,
        "loss_weight": loss_weight,
        "init_from_layer": init_from_layer,
    }
    try:
        return _build_start_point(source_dir, out, request)
    except BaseException:
        # `out` was empty or absent, so nothing of the caller's is lost; a half-written start
        # point must not look like a finished one on retry.
        shutil.rmtree(out, ignore_errors=True)
        raise


def _build_start_point(
    source_dir: pathlib.Path, out: pathlib.Path, request: dict
) -> dict:
    mode, objective = request["mode"], request["objective"]
    num_draft_tokens, init_from_layer = (
        request["num_draft_tokens"],
        request["init_from_layer"],
    )
    model_type = json.loads((source_dir / "config.json").read_text()).get("model_type")
    backbone = _BACKBONES.get(model_type)
    if backbone is None:
        raise ValueError(f"Expected an LFM3 or LFM3-VL checkpoint, got {model_type!r}")
    shutil.copytree(source_dir, out, dirs_exist_ok=True)
    for filename in backbone.remote_code:
        shutil.copy2(REMOTE_CODE_DIR / filename, out / filename)

    config_path = out / "config.json"
    config = json.loads(config_path.read_text())
    text_config = backbone.text_config(config)
    layer = (
        _default_init_layer(text_config) if init_from_layer is None else init_from_layer
    )
    layer_types = text_config["layer_types"]
    if not 0 <= layer < len(layer_types):
        raise ValueError(
            f"init_from_layer must be between 0 and {len(layer_types) - 1}, got {layer}"
        )
    if layer_types[layer] not in ("full_attention", "sliding_attention"):
        raise ValueError(f"init_from_layer {layer} is a {layer_types[layer]} layer")

    state = _load_weights(out)
    present = _present_modules(state)
    have = max(present, default=0)
    if mode == "add-module":
        if not present:
            raise ValueError(
                "add-module needs a checkpoint with a trained mtp.* module"
            )
        num_modules = have + 1
        if num_modules > MAX_DRAFT_TOKENS:
            raise ValueError(
                f"The checkpoint already has {have} MTP modules; at most "
                f"{MAX_DRAFT_TOKENS} are supported"
            )
    else:
        num_modules = num_draft_tokens if num_draft_tokens is not None else (have or 1)
    if num_modules < have:
        raise ValueError(
            f"The checkpoint already has {have} MTP modules; asked for {num_modules}"
        )
    if mode == "joint" and num_modules > 1:
        raise ValueError(
            "joint mode trains a single MTP module (num_draft_tokens=1); "
            f"got {num_modules}"
        )

    added = 0
    if not present:
        added += _init_first_module(state, text_config, layer, backbone)
    first_new = max(have, 1) + 1
    for depth in range(first_new, num_modules + 1):
        added += _copy_module(state, depth - 1, depth)
    # Topping up a trained checkpoint trains only the new modules.
    train_modules = (
        list(range(have + 1, num_modules + 1)) if have and num_modules > have else None
    )

    config.setdefault("auto_map", {})[backbone.auto_class] = backbone.mtp_class
    config["architectures"] = [backbone.mtp_class.split(".")[1]]
    config.update(
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        mtp_init_from_layer=config.get("mtp_init_from_layer", layer),
        mtp_loss_weight=request["loss_weight"],
        mtp_train_only=mode != "joint",
        mtp_soft=objective == "soft",
        mtp_num_modules=num_modules,
    )
    if train_modules is not None:
        config["mtp_train_modules"] = train_modules
    else:
        config.pop("mtp_train_modules", None)
    # Weights first, then the config and the manifest that mark the start point as complete.
    if added:
        _save_weights(out, state)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (out / MANIFEST_NAME).write_text(
        json.dumps({"request": request, "source_dir": str(source_dir)}, indent=2) + "\n"
    )

    summary = {
        "output_dir": str(out),
        "mode": mode,
        "objective": objective,
        "mtp_modules": num_modules,
        "added_tensors": added,
        "init_from_layer": config["mtp_init_from_layer"],
        "backbone": model_type,
    }
    logger.info("Prepared LFM3 MTP checkpoint: %s", summary)
    return summary


def draft_count_arg(value: str) -> int:
    count = int(value)
    if not 1 <= count <= MAX_DRAFT_TOKENS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_DRAFT_TOKENS}, got {count}"
        )
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "source", help="LFM3 / LFM3-VL checkpoint directory or Hub model id"
    )
    parser.add_argument("output_dir", help="New directory for the MTP start point")
    parser.add_argument("--mode", choices=MODES, default="heads")
    parser.add_argument("--loss-weight", type=float, default=0.3)
    parser.add_argument("--init-from-layer", type=int, default=None)
    parser.add_argument("--objective", choices=("hard", "soft"), default="hard")
    parser.add_argument(
        "--num-draft-tokens",
        type=draft_count_arg,
        default=None,
        help=f"Draft modules to set up (1-{MAX_DRAFT_TOKENS}); "
        "default: keep the checkpoint's, or 1",
    )
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
                num_draft_tokens=args.num_draft_tokens,
                revision=args.revision,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
