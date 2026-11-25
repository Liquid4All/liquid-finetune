"""Sample model completions with vLLM into an SFT ``messages`` dataset.

Each output row is a prompt conversation followed by one sampled assistant
turn, so the file can be trained on directly with ``training_type: distill``.
Sampling from the current student and distilling on the result with
``beta: 1.0`` and ``lmbda: 0.0`` is one round of offline on-policy
distillation.
"""

import argparse
import json
import logging
import pathlib
from collections.abc import Iterable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_LOCAL_DATASET_BUILDERS = {
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample completions from a model with vLLM and write them as an SFT "
            "messages dataset (JSONL)."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to sample from: short name, Hub id, or local checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Prompts: a local .jsonl/.json/.parquet/.csv file, a dataset "
            "directory, or a Hub dataset id. Rows need a `prompt` column "
            "(string or messages) or a `messages` conversation."
        ),
    )
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    parser.add_argument("--subset", default=None, help="Hub dataset config name.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None, help="Use the first N rows.")
    parser.add_argument(
        "--chat-template-path",
        default=None,
        help="Chat template used to render prompts; match the training config.",
    )
    parser.add_argument("--revision", default=None, help="Model revision to pin.")
    parser.add_argument(
        "--num-samples", type=int, default=1, help="Completions per prompt."
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep-truncated",
        action="store_true",
        help="Keep completions that hit --max-tokens (dropped by default).",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Let vLLM load custom model code instead of its native registration.",
    )
    return parser


def load_prompts(
    dataset: str,
    *,
    subset: str | None = None,
    split: str = "train",
    limit: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Load prompt conversations from a local file or a Hub dataset."""
    from datasets import load_dataset

    from liquid_finetune.training.utils.distillation import conversation_prompt

    path = pathlib.Path(dataset).expanduser()
    if path.is_file():
        builder = _LOCAL_DATASET_BUILDERS.get(path.suffix.lower())
        if builder is None:
            raise ValueError(f"Unsupported dataset file type: {path.suffix}")
        rows = load_dataset(builder, data_files=str(path), split=split)
    else:
        rows = load_dataset(dataset, subset, split=split)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    return [conversation_prompt(row) for row in rows]


def messages_rows(
    prompts: Sequence[list[dict[str, Any]]],
    outputs: Iterable[Any],
    *,
    keep_truncated: bool = False,
) -> list[dict[str, Any]]:
    """Turn vLLM request outputs into SFT rows, one per sampled completion."""
    rows = []
    for prompt, request_output in zip(prompts, outputs, strict=True):
        for completion in request_output.outputs:
            if completion.finish_reason == "length" and not keep_truncated:
                continue
            assistant_turn = {"role": "assistant", "content": completion.text}
            rows.append({"messages": [*prompt, assistant_turn]})
    return rows


def _prompt_token_ids(tokenizer, prompt: list[dict[str, Any]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True
    )
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def write_jsonl(rows: Iterable[dict[str, Any]], output: pathlib.Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "Sampling completions requires vLLM. Install the CUDA dependency "
            "group with `uv sync`."
        ) from exc

    from liquid_finetune.checkpointing.model_loading import (
        _resolve_model_id,
        load_tokenizer,
    )

    prompts = load_prompts(
        args.dataset, subset=args.subset, split=args.split, limit=args.limit
    )
    tokenizer = load_tokenizer(
        args.model,
        chat_template_path=args.chat_template_path,
        revision=args.revision,
    )
    prompt_ids = [_prompt_token_ids(tokenizer, prompt) for prompt in prompts]
    if args.max_model_len is not None:
        kept = [
            index
            for index, ids in enumerate(prompt_ids)
            if len(ids) < args.max_model_len
        ]
        if len(kept) < len(prompts):
            logger.warning(
                "Skipping %d prompt(s) that do not fit max_model_len=%d",
                len(prompts) - len(kept),
                args.max_model_len,
            )
        prompts = [prompts[index] for index in kept]
        prompt_ids = [prompt_ids[index] for index in kept]

    llm_kwargs: dict[str, Any] = {
        "model": _resolve_model_id(args.model),
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
    }
    if args.revision:
        llm_kwargs["revision"] = args.revision
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        n=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_ids], sampling_params
    )

    rows = messages_rows(prompts, outputs, keep_truncated=args.keep_truncated)
    count = write_jsonl(rows, args.output)
    sampled = len(prompts) * args.num_samples
    print(f"Wrote {count} of {sampled} sampled completions to {args.output}")


if __name__ == "__main__":
    main()
