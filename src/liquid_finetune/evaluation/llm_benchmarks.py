import logging
from collections.abc import Callable

import numpy as np
import torch

from liquid_finetune.evaluation.backend import (
    GenerateRequest,
    InferenceBackend,
    LogprobRequest,
)
from liquid_finetune.evaluation.base import Benchmark, BenchmarkResult
from liquid_finetune.evaluation.data_loaders import load_benchmark_samples
from liquid_finetune.evaluation.metrics import compute_metric

logger = logging.getLogger(__name__)


class LLMGenerationBenchmark(Benchmark):
    """Generate text with an LLM and score against ground truth.

    The last conversation turn (assistant) is treated as ground truth;
    prior turns form the prompt.
    """

    def __init__(
        self,
        name: str,
        path: str,
        tokenizer,
        metric: str | Callable = "short_answer",
        max_new_tokens: int = 128,
        match_mode: str = "contains",
        limit: int | None = None,
        format: str | None = None,
        **metric_kwargs,
    ):
        super().__init__(name)
        self.path = path
        self.tokenizer = tokenizer
        self.metric = metric
        self.max_new_tokens = max_new_tokens
        self.match_mode = match_mode
        self.limit = limit
        self.format = format
        self.metric_kwargs = metric_kwargs

    def load_samples(self) -> list[dict]:
        return load_benchmark_samples(self.path, self.limit, self.format)

    def score_sample(self, model, sample: dict, device) -> float:
        """Generate a response from prompt turns and score against the last (GT) turn."""
        messages = sample["messages"]
        ground_truth = messages[-1]["content"]
        if isinstance(ground_truth, list):
            ground_truth = " ".join(
                item["text"] for item in ground_truth if item.get("type") == "text"
            )

        prompt_messages = messages[:-1]
        inputs = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            output_ids = model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )

        prompt_len = inputs["input_ids"].shape[1]
        prediction = self.tokenizer.decode(
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

    def evaluate_with_backend(
        self, backend: InferenceBackend, samples: list
    ) -> BenchmarkResult:
        """Build batched generation requests, dispatch to backend, score responses."""
        requests: list[GenerateRequest] = []
        ground_truths: list[str] = []
        for sample in samples:
            messages = sample["messages"]
            gt = messages[-1]["content"]
            if isinstance(gt, list):
                gt = " ".join(item["text"] for item in gt if item.get("type") == "text")
            ground_truths.append(gt)
            requests.append(
                GenerateRequest(
                    messages=messages[:-1],
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,
                )
            )

        results = backend.generate(requests)

        total_score = 0.0
        count = 0
        for sample, result, gt in zip(samples, results, ground_truths):
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


class LLMLogprobBenchmark(Benchmark):
    """Zero-shot MCQ classification — picks the highest-logprob answer option.

    Runs one forward pass per option, sums the log-probabilities of the answer
    tokens, and selects the option with the highest total.
    """

    def __init__(
        self,
        name: str,
        path: str,
        tokenizer,
        limit: int | None = None,
        format: str | None = None,
    ):
        super().__init__(name)
        self.path = path
        self.tokenizer = tokenizer
        self.limit = limit
        self.format = format

    def load_samples(self) -> list[dict]:
        return load_benchmark_samples(self.path, self.limit, self.format)

    def score_sample(self, model, sample: dict, device) -> float:
        """Score each option by length-normalized log-prob; 1.0 if argmax matches answer_id."""
        options = sample["options"]
        answer_id = int(sample["answer_id"])

        prompt_messages = sample["messages"]
        prompt_inputs = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        option_scores = []
        for option in options:
            full_conv = prompt_messages + [
                {"role": "assistant", "content": option.strip()}
            ]
            full_inputs = self.tokenizer.apply_chat_template(
                full_conv,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                logits = model(**full_inputs).logits

            log_probs = logits[0].log_softmax(dim=-1)
            input_ids = full_inputs["input_ids"][0]
            num_tokens = len(input_ids) - prompt_len
            total_logprob = sum(
                log_probs[i - 1, input_ids[i].item()].item()
                for i in range(prompt_len, len(input_ids))
            )
            option_scores.append(total_logprob / num_tokens if num_tokens > 0 else 0.0)

        return 1.0 if int(np.argmax(option_scores)) == answer_id else 0.0

    def evaluate_with_backend(
        self, backend: InferenceBackend, samples: list
    ) -> BenchmarkResult:
        """Build batched logprob requests (one per sample, all options together),
        dispatch to backend, then argmax."""
        requests: list[LogprobRequest] = []
        answer_ids: list[int] = []
        for sample in samples:
            requests.append(
                LogprobRequest(
                    messages=sample["messages"],
                    continuations=list(sample["options"]),
                )
            )
            answer_ids.append(int(sample["answer_id"]))

        results = backend.logprobs(requests)

        total_score = 0.0
        count = 0
        for sample, result, ans_id in zip(samples, results, answer_ids):
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
