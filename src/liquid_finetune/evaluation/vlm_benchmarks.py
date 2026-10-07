import copy
import logging
from collections.abc import Callable

import numpy as np
import torch

from liquid_finetune.data_loading.image_loader import load_image
from liquid_finetune.evaluation.backend import (
    GenerateRequest,
    InferenceBackend,
    LogprobRequest,
)
from liquid_finetune.evaluation.base import Benchmark, BenchmarkResult
from liquid_finetune.evaluation.data_loaders import load_benchmark_samples
from liquid_finetune.evaluation.metrics import compute_metric

logger = logging.getLogger(__name__)


class VLMGenerationBenchmark(Benchmark):
    """Generate text with a VLM and score against ground truth.

    The last conversation turn (assistant) is treated as ground truth;
    prior turns form the prompt.
    """

    def __init__(
        self,
        name: str,
        path: str,
        processor,
        metric: str | Callable = "short_answer",
        max_new_tokens: int = 128,
        match_mode: str = "contains",
        limit: int | None = None,
        format: str | None = None,
        image_root: str | None = None,
        **metric_kwargs,
    ):
        super().__init__(name)
        self.path = path
        self.processor = processor
        self.metric = metric
        self.max_new_tokens = max_new_tokens
        self.match_mode = match_mode
        self.limit = limit
        self.format = format
        self.image_root = image_root
        self.metric_kwargs = metric_kwargs

    def load_samples(self) -> list[dict]:
        return load_benchmark_samples(
            self.path, self.limit, self.format, self.image_root
        )

    def score_sample(self, model, sample: dict, device) -> float:
        """Generate a response from prompt turns (with images) and score against the last (GT) turn."""
        messages = sample["messages"]
        ground_truth = _extract_text(messages[-1]["content"])

        prompt, loaded = _prepare_messages(messages[:-1])
        try:
            inputs = self.processor.apply_chat_template(
                [prompt],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                output_ids = model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )

            prompt_len = inputs["input_ids"].shape[1]
            prediction = self.processor.tokenizer.decode(
                output_ids[0, prompt_len:], skip_special_tokens=True
            ).strip()

            if callable(self.metric):
                return self.metric(prediction, ground_truth, **self.metric_kwargs)
            return compute_metric(
                self.metric,
                prediction,
                ground_truth,
                match_mode=self.match_mode,
                **self.metric_kwargs,
            )
        finally:
            _close_images(loaded)

    def evaluate_with_backend(
        self, backend: InferenceBackend, samples: list
    ) -> BenchmarkResult:
        """Build batched VLM generation requests with PIL images, dispatch, score.

        Loaded PIL images are closed after the backend call returns.
        """
        requests: list[GenerateRequest] = []
        ground_truths: list[str] = []
        all_loaded: list = []
        kept_samples: list = []
        skipped_bad_image = 0
        for sample in samples:
            messages = sample["messages"]
            try:
                prompt, loaded = _prepare_messages(messages[:-1])
            except Exception:
                skipped_bad_image += 1
                continue
            kept_samples.append(sample)
            ground_truths.append(_extract_text(messages[-1]["content"]))
            all_loaded.extend(loaded)
            requests.append(
                GenerateRequest(
                    messages=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,
                    images=loaded if loaded else None,
                )
            )

        if skipped_bad_image:
            import logging

            logging.getLogger(__name__).warning(
                "[%s] skipped %d sample(s) with unreadable images",
                getattr(self, "name", "<benchmark>"),
                skipped_bad_image,
            )

        try:
            results = backend.generate(requests)
        finally:
            _close_images(all_loaded)

        total_score = 0.0
        count = 0
        # Zip over kept_samples (not samples) so sample↔result↔gt indices align
        # after the unreadable-image filter above.
        for sample, result, gt in zip(kept_samples, results, ground_truths):
            try:
                if callable(self.metric):
                    score = self.metric(result.text, gt, **self.metric_kwargs)
                else:
                    score = compute_metric(
                        self.metric,
                        result.text,
                        gt,
                        match_mode=self.match_mode,
                        **self.metric_kwargs,
                    )
                total_score += score
                count += 1
            except Exception:
                logger.warning(
                    "[%s] Scoring failed on sample %s",
                    self.name,
                    sample.get("id", count),
                    exc_info=True,
                )
        return BenchmarkResult(metrics={"score": total_score}, count=count)


class VLMLogprobBenchmark(Benchmark):
    """Zero-shot MCQ classification — picks the highest-logprob answer option.

    Runs one forward pass per option, sums the log-probabilities of the answer
    tokens, and selects the option with the highest total.
    """

    def __init__(
        self,
        name: str,
        path: str,
        processor,
        limit: int | None = None,
        format: str | None = None,
        image_root: str | None = None,
    ):
        super().__init__(name)
        self.path = path
        self.processor = processor
        self.limit = limit
        self.format = format
        self.image_root = image_root

    def load_samples(self) -> list[dict]:
        return load_benchmark_samples(
            self.path, self.limit, self.format, self.image_root
        )

    def score_sample(self, model, sample: dict, device) -> float:
        """Score each option by length-normalized log-prob; 1.0 if argmax matches answer_id."""
        options = sample["options"]
        answer_id = int(sample["answer_id"])

        prompt, loaded = _prepare_messages(sample["messages"])
        try:
            prompt_inputs = self.processor.apply_chat_template(
                [prompt],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]

            option_scores = []
            for option in options:
                full_conv = prompt + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": option.strip()}],
                    }
                ]
                full_inputs = self.processor.apply_chat_template(
                    [full_conv],
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                full_inputs = {
                    k: v.to(device) for k, v in full_inputs.items() if v is not None
                }

                with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                    logits = model(**full_inputs).logits

                log_probs = logits[0].log_softmax(dim=-1)
                input_ids = full_inputs["input_ids"][0]
                num_tokens = len(input_ids) - prompt_len
                total_logprob = sum(
                    log_probs[i - 1, input_ids[i].item()].item()
                    for i in range(prompt_len, len(input_ids))
                )
                option_scores.append(
                    total_logprob / num_tokens if num_tokens > 0 else 0.0
                )

            return 1.0 if int(np.argmax(option_scores)) == answer_id else 0.0
        finally:
            _close_images(loaded)

    def evaluate_with_backend(
        self, backend: InferenceBackend, samples: list
    ) -> BenchmarkResult:
        """Build batched VLM logprob requests, dispatch to backend, argmax.

        Note: vLLM's multimodal logprob support is more constrained than text;
        callers (async runners) should be prepared to fall back to HFBackend
        if a particular vLLM build doesn't support image+logprob requests.
        """
        requests: list[LogprobRequest] = []
        answer_ids: list[int] = []
        all_loaded: list = []
        kept_samples: list = []
        skipped_bad_image = 0
        for sample in samples:
            # Mirror the generation path's image-error handling: skip
            # unreadable images instead of aborting the whole benchmark.
            try:
                prompt, loaded = _prepare_messages(sample["messages"])
            except Exception:
                skipped_bad_image += 1
                continue
            kept_samples.append(sample)
            all_loaded.extend(loaded)
            requests.append(
                LogprobRequest(
                    messages=prompt,
                    continuations=list(sample["options"]),
                    images=loaded if loaded else None,
                )
            )
            answer_ids.append(int(sample["answer_id"]))

        if skipped_bad_image:
            logger.warning(
                "[%s] skipped %d sample(s) with unreadable images",
                getattr(self, "name", "<benchmark>"),
                skipped_bad_image,
            )

        try:
            results = backend.logprobs(requests)
        finally:
            _close_images(all_loaded)

        total_score = 0.0
        count = 0
        for sample, result, ans_id in zip(kept_samples, results, answer_ids):
            try:
                if not result.logprobs:
                    raise ValueError("backend returned empty logprobs")
                score = 1.0 if int(np.argmax(result.logprobs)) == ans_id else 0.0
                total_score += score
                count += 1
            except Exception:
                logger.warning(
                    "[%s] Scoring failed on sample %s",
                    self.name,
                    sample.get("id", count),
                    exc_info=True,
                )
        return BenchmarkResult(metrics={"score": total_score}, count=count)


# === Helpers ===


def _prepare_messages(messages: list[dict]) -> tuple[list[dict], list]:
    """Deep-copy messages and load images from string paths into PIL objects.

    Same pattern as the training collate_fn in tokenize_data.py — replaces
    path strings with PIL images so the processor can encode them.
    """
    messages = copy.deepcopy(messages)
    loaded = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image" and isinstance(item.get("image"), str):
                img = load_image(item["image"])
                item["image"] = img
                loaded.append(img)
    return messages, loaded


def _extract_text(content) -> str:
    """Extract plain text from message content (string or structured list)."""
    if isinstance(content, str):
        return content
    return " ".join(item["text"] for item in content if item.get("type") == "text")


def _close_images(images: list) -> None:
    """Close any PIL images opened during evaluation to free memory."""
    for img in images:
        if hasattr(img, "close"):
            img.close()
