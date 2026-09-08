import copy
import logging

import ray
import ray.data
import torch
from datasets import Dataset
import pyarrow as pa
from rich.console import Console
from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt
from trl.data_utils import pack_dataset

from leap_finetune.data_loading.image_loader import load_image
from leap_finetune.data_loading.validate_tool_format import (
    normalize_messages_for_chat_template,
    normalize_row_for_chat_template,
)

logger = logging.getLogger(__name__)
console = Console()


# === VLM Collate ===


def _find_template(seq, template):
    """Yield start indices where template occurs in seq."""
    tlen = len(template)
    for i in range(len(seq) - tlen + 1):
        if seq[i : i + tlen] == template:
            yield i


def create_vlm_collate_fn(processor):
    """Create a collate function with assistant-only label masking.

    Only assistant content + <|im_end|> contribute to loss.
    Images are loaded as PIL and passed to the processor for resize/tiling.
    Bad samples are skipped with a warning instead of crashing the batch.
    """
    tokenizer = processor.tokenizer

    # ChatML: <|im_start|>assistant\n{content}<|im_end|>\n
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _build_labels(input_ids):
        """Mask everything except assistant content + <|im_end|> with -100."""
        labels = torch.full_like(input_ids, -100)
        batch_size, seq_len = input_ids.shape

        for b in range(batch_size):
            ids = input_ids[b].tolist()
            for tmpl_start in _find_template(ids, response_template_ids):
                content_start = tmpl_start + len(response_template_ids)
                j = content_start
                while j < seq_len and ids[j] != end_token_id:
                    j += 1
                if j >= seq_len:
                    # Truncated turn — no <|im_end|> found, skip to avoid
                    # unmasking garbage (padding / partial next turn)
                    continue
                # Unmask content + <|im_end|>
                content_end = j + 1
                labels[b, content_start:content_end] = input_ids[
                    b, content_start:content_end
                ]

        return labels

    def collate_fn(samples):
        valid_samples = []
        all_loaded_images = []
        skip_count = 0

        for raw in samples:
            # Trainer's dataloader yields {"messages": [...]} dicts from HF Dataset
            conversation = raw["messages"] if isinstance(raw, dict) else raw
            sample_copy = copy.deepcopy(conversation)
            loaded_images = []
            try:
                for message in sample_copy:
                    for content in message["content"]:
                        if content["type"] == "image" and isinstance(
                            content["image"], str
                        ):
                            img = load_image(content["image"])
                            content["image"] = img
                            loaded_images.append(img)
                valid_samples.append(normalize_messages_for_chat_template(sample_copy))
                all_loaded_images.extend(loaded_images)
            except Exception as e:
                skip_count += 1
                logger.warning(f"Skipping sample in collate: {e}")
                for img in loaded_images:
                    if hasattr(img, "close"):
                        img.close()

        if skip_count > 0:
            logger.info(f"Collate skipped {skip_count}/{len(samples)} samples")

        if len(valid_samples) == 0:
            raise RuntimeError(
                f"Entire batch failed: all {len(samples)} samples had errors"
            )

        try:
            batch = processor.apply_chat_template(
                valid_samples,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            batch["labels"] = _build_labels(batch["input_ids"])
            return batch

        finally:
            for img in all_loaded_images:
                if hasattr(img, "close"):
                    img.close()
            all_loaded_images.clear()

    return collate_fn


# === SFT Tokenization ===


def _final_assistant_span_mask(assistant_masks: list[int]) -> list[int]:
    """Keep only the last contiguous assistant span."""
    if not assistant_masks:
        return assistant_masks

    end = None
    for idx in range(len(assistant_masks) - 1, -1, -1):
        if assistant_masks[idx]:
            end = idx
            break

    if end is None:
        return [0] * len(assistant_masks)

    start = end
    while start > 0 and assistant_masks[start - 1]:
        start -= 1

    output = [0] * len(assistant_masks)
    for idx in range(start, end + 1):
        output[idx] = 1
    return output


def tokenize_sft(
    row: dict,
    tokenizer,
    max_length: int,
    assistant_only_loss: bool = False,
    completion_only_loss: bool = False,
    truncate: bool = True,
) -> dict:
    """
    Tokenize a single SFT row for use in ray_ds.map().

    Handles two formats:
      - Conversational: row has "messages" → apply_chat_template
      - Plain text: row has "text" → tokenizer()
    """
    if "messages" in row:
        need_masks = assistant_only_loss or completion_only_loss
        messages = normalize_messages_for_chat_template(row["messages"])
        result = tokenizer.apply_chat_template(
            messages,
            tools=row.get("tools") or None,
            tokenize=True,
            truncation=truncate,
            max_length=max_length if truncate else None,
            return_dict=need_masks,
            return_assistant_tokens_mask=need_masks,
        )
        # apply_chat_template returns BatchEncoding (Mapping, not dict)
        input_ids = result["input_ids"] if hasattr(result, "keys") else result
    elif "text" in row:
        if assistant_only_loss or completion_only_loss:
            raise ValueError(
                "assistant_only_loss/completion_only_loss require conversational "
                "SFT rows with a 'messages' column"
            )
        input_ids = tokenizer(
            row["text"],
            truncation=truncate,
            max_length=max_length if truncate else None,
        )["input_ids"]
    else:
        raise ValueError(
            f"Row must have 'messages' or 'text' column, got: {list(row.keys())}"
        )

    output = {"input_ids": list(input_ids), "length": len(input_ids)}
    if "messages" in row and (assistant_only_loss or completion_only_loss):
        assistant_masks = list(result["assistant_masks"])
        if len(assistant_masks) != len(output["input_ids"]):
            raise ValueError(
                "assistant mask length mismatch after chat template tokenization"
            )
        if not any(assistant_masks):
            # A chat template without {% generation %} markers makes
            # apply_chat_template return an all-zero assistant mask: it warns but
            # does not raise, so the run trains on zero supervised tokens and the
            # loss curve still looks plausible. Truncation can legitimately cut
            # the assistant span off a long row, so only reject rows that fit.
            truncated = (
                truncate and max_length is not None and len(input_ids) >= max_length
            )
            if not truncated:
                raise ValueError(
                    "assistant mask is all zeros for a row that was not truncated. "
                    "Check that the chat template has {% generation %} markers, "
                    "that the row contains assistant content, and that the "
                    "assistant span is not fully truncated. Otherwise, use a "
                    "template that marks the assistant span or turn the flag off."
                )
        if assistant_only_loss:
            output["assistant_masks"] = assistant_masks
        if completion_only_loss:
            output["completion_mask"] = _final_assistant_span_mask(assistant_masks)

    return output


def tokenize_and_pack_sft(
    ds: ray.data.Dataset,
    tokenizer,
    max_length: int,
    packing: bool = False,
    assistant_only_loss: bool = False,
    completion_only_loss: bool = False,
    drop_overlength: bool = False,
) -> ray.data.Dataset:
    """
    Tokenize and optionally pack an SFT dataset.

    Pipeline:
      1. Distributed tokenization via ray_ds.map()
      2. If packing: materialize to HF Dataset → pack_dataset (BFD) → back to Ray
         If not packing: return directly (tokenizer already truncated)
    """
    # === 1. Distributed tokenization ===
    ds = ds.map(
        tokenize_sft,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "assistant_only_loss": assistant_only_loss,
            "completion_only_loss": completion_only_loss,
            "truncate": not drop_overlength,
        },
    )

    if drop_overlength:
        # For long-context SFT we often prefilter complete conversations. Do not
        # silently turn a complete example into a partial supervised target if
        # the active tokenizer/template now renders it over the configured limit.
        ds = ds.filter(lambda row: row["length"] <= max_length)

    # === 2. Pack or truncate ===
    if packing:
        # Packing requires full materialization into an HF Dataset. Keep the
        # materialization Arrow-native; building a Python row list duplicates
        # every tokenized sequence and creates a large transient peak.
        if hasattr(ds, "to_arrow_refs"):
            tables = ray.get(ds.to_arrow_refs())
            if not tables:
                return ds
            table = pa.concat_tables(tables)
            columns = [
                name
                for name in ("input_ids", "assistant_masks", "completion_mask")
                if name in table.column_names
            ]
            hf_ds = Dataset(table.select(columns))
        else:
            # The fallback keeps tests and non-Ray dataset adapters working
            # without collecting rows into Python first.
            hf_ds = Dataset.from_generator(ds.iter_rows)
            columns = [
                name
                for name in ("input_ids", "assistant_masks", "completion_mask")
                if name in hf_ds.column_names
            ]
            hf_ds = hf_ds.select_columns(columns)
        console.print(f"[dim]Tokenized {len(hf_ds):,} rows[/dim]")
        console.print(f"[dim]Packing sequences (BFD, max_length={max_length})...[/dim]")
        hf_ds = pack_dataset(hf_ds, seq_length=max_length, strategy="bfd")
        hf_ds = hf_ds.map(lambda row: {"length": len(row["input_ids"])})
        console.print(f"[dim]Packed into {len(hf_ds):,} rows[/dim]")
        return ray.data.from_arrow(hf_ds.data.table)

    # Non-packing: tokenizer already truncated to max_length or overlength rows
    # were explicitly dropped above.
    return ds


# === DPO Tokenization ===


def tokenize_dpo(
    row: dict,
    tokenizer,
    max_prompt_length: int | None,
    max_completion_length: int | None,
) -> dict:
    """
    Tokenize a single DPO row for use in ray_ds.map().

    Replicates DPOTrainer's pipeline:
      1. maybe_extract_prompt — extract shared prompt from chosen/rejected
      2. maybe_apply_chat_template — convert messages → strings
      3. tokenize_row — tokenize + truncate + append eos

    Produces: prompt_input_ids, chosen_input_ids, rejected_input_ids
    """
    row = normalize_row_for_chat_template(row)

    # Extract prompt if not already present
    row = maybe_extract_prompt(row)

    # Apply chat template (converts conversational → strings, no-op for strings)
    row = maybe_apply_chat_template(row, tokenizer)

    # Tokenize the 3 string sequences
    prompt_input_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    chosen_input_ids = tokenizer(row["chosen"], add_special_tokens=False)["input_ids"]
    rejected_input_ids = tokenizer(row["rejected"], add_special_tokens=False)[
        "input_ids"
    ]

    # Append eos to completions (matches DPOTrainer.tokenize_row)
    chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
    rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

    # Truncate: prompt from the left, completions from the right
    if max_prompt_length is not None:
        prompt_input_ids = prompt_input_ids[-max_prompt_length:]
    if max_completion_length is not None:
        chosen_input_ids = chosen_input_ids[:max_completion_length]
        rejected_input_ids = rejected_input_ids[:max_completion_length]

    # Column names must match TRL v1's DPO data collator:
    # prompt_ids, chosen_ids, rejected_ids (changed from *_input_ids in TRL 0.x)
    # Include the actual collated sequence length so length grouping works for DPO.
    return {
        "prompt_ids": list(prompt_input_ids),
        "chosen_ids": list(chosen_input_ids),
        "rejected_ids": list(rejected_input_ids),
        "length": max(
            len(prompt_input_ids) + len(chosen_input_ids),
            len(prompt_input_ids) + len(rejected_input_ids),
        ),
    }


def tokenize_dpo_dataset(
    ds: ray.data.Dataset,
    tokenizer,
    max_prompt_length: int | None = None,
    max_completion_length: int | None = None,
) -> ray.data.Dataset:
    """
    Tokenize a DPO dataset via Ray .map().

    Returns a Ray Dataset with columns:
      prompt_ids, chosen_ids, rejected_ids (TRL v1 column names)
    """
    ds = ds.map(
        tokenize_dpo,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
        },
    )
    # Keep tokenization lazy in Ray. The driver must not gather all tokenized
    # DPO rows with ray.get(); Ray Train shards the dataset first, and each
    # worker materializes only its local shard for the HF trainer.
    return ds
