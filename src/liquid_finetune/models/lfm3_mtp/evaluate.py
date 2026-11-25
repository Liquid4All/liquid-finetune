"""Speculative-decoding acceptance of LFM3 / LFM3-VL MTP draft modules (``liquid-eval-mtp``).

For each prompt the model (the verifier) generates a greedy continuation; the draft
modules then run as at inference. At a verifier step whose last accepted token is
x_t, the verifier emits x_{t+1} and module d drafts x_{t+d+1} from module d-1's state
and the embedding of x_{t+d}. Greedy verification accepts the longest correct draft
prefix, so stepping through the greedy sequence gives the exact tokens per verifier
step for 1..N drafts.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import torch

from liquid_finetune.models.lfm3_mtp.prepare import draft_count_arg

logger = logging.getLogger(__name__)


def default_device() -> str:
    """The accelerator if there is one (CUDA and ROCm builds both report ``cuda``), otherwise the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(checkpoint: str, device: str):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    is_vlm = "AutoModelForImageTextToText" in (getattr(config, "auto_map", None) or {})
    if is_vlm:
        from transformers import AutoModelForImageTextToText as auto_model
        from transformers import AutoProcessor as auto_processor
    else:
        from transformers import AutoModelForCausalLM as auto_model
        from transformers import AutoTokenizer as auto_processor
    model = auto_model.from_pretrained(
        checkpoint, trust_remote_code=True, dtype=torch.bfloat16
    )
    if not hasattr(model, "mtp_draft_states") or not model._present_modules():
        raise ValueError(f"{checkpoint} has no MTP draft modules")
    processor = auto_processor.from_pretrained(checkpoint, trust_remote_code=True)
    return model.to(device).eval(), processor, is_vlm


def _prompts(
    dataset: str, subset: str | None, split: str, limit: int, is_vlm: bool
) -> list[list[dict]]:
    """Conversations up to (not including) the first assistant turn."""
    from datasets import load_dataset

    path = pathlib.Path(dataset).expanduser()
    if path.is_file():
        kind = "parquet" if path.suffix in (".parquet", ".pq") else "json"
        rows = load_dataset(kind, data_files=str(path), split="train")
    else:
        rows = load_dataset(dataset, subset, split=split)
    prompts = []
    for row in rows:
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        turns = []
        for message in messages:
            if message["role"] == "assistant":
                break
            content = message["content"]
            if isinstance(content, list):
                content = [_content_item(item, is_vlm) for item in content]
                content = [item for item in content if item is not None]
            turns.append({"role": message["role"], "content": content})
        if turns:
            prompts.append(turns)
        if len(prompts) >= limit:
            break
    return prompts


def _content_item(item, is_vlm: bool):
    if isinstance(item, str):
        item = json.loads(item)
    item = {k: v for k, v in item.items() if v is not None}
    if item.get("type") == "image" and not is_vlm:
        return None
    return item


def _open_images(turns: list[dict]) -> list[dict]:
    """Load the images of one prompt (they are kept as paths until then)."""
    from PIL import Image

    opened = []
    for turn in turns:
        content = turn["content"]
        if isinstance(content, list):
            content = [
                {**item, "image": Image.open(item["image"]).convert("RGB")}
                if item.get("type") == "image"
                else item
                for item in content
            ]
        opened.append({**turn, "content": content})
    return opened


def _encode(processor, turns, is_vlm: bool, device: str) -> dict:
    if is_vlm:
        turns = _open_images(turns)
    if not is_vlm and all(isinstance(t["content"], list) for t in turns):
        turns = [
            {
                "role": t["role"],
                "content": "".join(c.get("text", "") for c in t["content"]),
            }
            for t in turns
        ]
    inputs = processor.apply_chat_template(
        turns,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}


@torch.no_grad()
def _drafts(model, inputs: dict, max_new_tokens: int, num_drafts: int):
    """Greedy continuation and every module's draft at every position: drafts[d-1][t] drafts x_{t+d+1}."""
    prompt_len = inputs["input_ids"].shape[1]
    sequence = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    positions = torch.arange(sequence.shape[1], device=sequence.device)[None]
    extra = {
        k: v
        for k, v in inputs.items()
        if k not in ("input_ids", "attention_mask", "token_type_ids")
    }
    if "pixel_values" in extra:
        extra["pixel_values"] = extra["pixel_values"].to(torch.bfloat16)
    hidden, embeds = model._mtp_backbone(sequence, None, positions, extra)
    states = model.mtp_draft_states(hidden, embeds, positions, num_drafts)
    drafts = [model.lm_head(state).argmax(-1)[0].tolist() for state in states]
    return sequence[0].tolist(), drafts, prompt_len


def _accepted_per_step(tokens, drafts, prompt_len: int, n: int) -> list[int]:
    """Greedy speculative decoding with the first ``n`` modules: accepted drafts per verifier step."""
    lengths, t = [], prompt_len - 1  # the verifier's next step emits x_{t+1}
    while t + 1 < len(tokens):
        accepted = 0
        for depth in range(1, n + 1):
            target = t + depth + 1
            if target >= len(tokens) or drafts[depth - 1][t] != tokens[target]:
                break
            accepted = depth
        lengths.append(accepted)
        t += 1 + accepted
    return lengths


def evaluate_mtp(
    checkpoint: str,
    dataset: str,
    *,
    subset: str | None = None,
    split: str = "test",
    num_samples: int = 64,
    max_new_tokens: int = 256,
    num_draft_tokens: int | None = None,
    device: str | None = None,
) -> dict:
    device = device or default_device()
    model, processor, is_vlm = _load(checkpoint, device)
    available = len(model._present_modules())
    if num_draft_tokens is not None and not 1 <= num_draft_tokens <= available:
        raise ValueError(
            f"num_draft_tokens must be between 1 and the checkpoint's {available} "
            f"modules, got {num_draft_tokens}"
        )
    num_drafts = num_draft_tokens if num_draft_tokens is not None else available
    runs = []
    for turns in _prompts(dataset, subset, split, num_samples, is_vlm):
        inputs = _encode(processor, turns, is_vlm, device)
        runs.append(_drafts(model, inputs, max_new_tokens, num_drafts))
    report = {"checkpoint": checkpoint, "prompts": len(runs), "drafts": {}}
    for n in range(1, num_drafts + 1):
        lengths = [a for run in runs for a in _accepted_per_step(*run, n)]
        if not lengths:
            raise ValueError("No tokens were generated")
        reached = [
            sum(a >= d for a in lengths) for d in range(n + 1)
        ]  # reached[0] = steps
        report["drafts"][n] = {
            "verifier_steps": len(lengths),
            "mean_accepted_drafts": sum(lengths) / len(lengths),
            "tokens_per_step": 1 + sum(lengths) / len(lengths),
            # P(draft d accepted | drafts 1..d-1 accepted)
            "acceptance_by_depth": [
                reached[d] / reached[d - 1] if reached[d - 1] else 0.0
                for d in range(1, n + 1)
            ],
        }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("checkpoint", help="MTP checkpoint directory or Hub model id")
    parser.add_argument(
        "dataset", help="Hub dataset or local jsonl/parquet with `messages`"
    )
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-draft-tokens", type=draft_count_arg, default=None)
    parser.add_argument("--output", default=None, help="Write the report as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = evaluate_mtp(
        args.checkpoint,
        args.dataset,
        subset=args.subset,
        split=args.split,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        num_draft_tokens=args.num_draft_tokens,
    )
    print(
        f"{'drafts':>6} {'tokens/step':>12} {'accepted/step':>14}  acceptance by depth"
    )
    for n, row in report["drafts"].items():
        depth = " ".join(f"{a:.3f}" for a in row["acceptance_by_depth"])
        print(
            f"{n:>6} {row['tokens_per_step']:>12.3f} {row['mean_accepted_drafts']:>14.3f}  {depth}"
        )
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
