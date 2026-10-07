# Evaluation Benchmarks

Run benchmarks automatically during training at every `eval_steps`, or run the
same suite directly with `leap-finetune <eval_config>` without starting a
training job.
Works with SFT, DPO, GRPO, and VLM variants. Results are logged to wandb during
training and printed to stdout.

Benchmarks live under `evals:`. The legacy `benchmarks:` alias still parses,
but new configs should use `evals:` so training and standalone eval configs
share the same `EvalSuiteConfig` shape.

Standalone evals can run without a training job:

```bash
uv run leap-finetune job_configs/eval_standalone_example.yaml
```

Standalone configs require `model_name` or `checkpoint`, an `evals:` suite, and
an optional `backend:` block. They deliberately do not include `dataset`,
`training_type`, `training_config`, or `async_eval`:

```yaml
model_name: "LFM2-1.2B"

evals:
  benchmarks:
    - name: "tiny_qa"
      path: "/data/tiny_qa.jsonl"
      metric: "short_answer"

backend:
  type: "hf" # hf | vllm | llama_cpp
```

Set `backend.quantization: fp8` for online vLLM FP8 or `quark` when an
artifact requires explicit AMD Quark selection.

Use `leap-finetune eval <eval_config> --output results.json` when you want the
CLI to write metrics to a JSON file.

Text standalone evals default to `modality: text`. Set `modality: vlm` only
when running standalone VLM evals outside a VLM training job.

For GGUF evaluation, use `type: llama_cpp` and point `checkpoint` at the GGUF
file. The evaluator launches `llama-server`, waits for `/health`, runs the
generation benchmarks over its OpenAI-compatible API, and stops the server.
Set `backend.mmproj` for VLM GGUF exports. Alternatively set
`backend.base_url` to connect to a caller-managed server. Continuation-logprob
benchmarks are not yet supported by this backend, so use generation-based MCQ
metrics for the GGUF quality matrix.

Python uses the same config:

```python
from leap_finetune import run_config

metrics = run_config("job_configs/eval_standalone_example.yaml")
```

Async training evals use the same `evals:` suite. The `async_eval:` block only
chooses execution mode (`sync`, `sidecar`, or `reserved`) and vLLM/SLURM
placement settings.

## Quick Setup

Add an `evals` section to your job config.

### VLM (vision-language) benchmarks

```yaml
training_config:
  eval_strategy: "steps"
  eval_steps: 2000

evals:
  max_new_tokens: 128 # default for all generation benchmarks
  image_root: "/data/images" # optional, prepended to relative image paths
  benchmarks:
    - name: "mmmu_val"
      path: "/data/mmmu_val.jsonl"
      metric: "short_answer"
      max_new_tokens: 50 # override per benchmark

    - name: "imagenette"
      path: "/data/imagenette_eval.jsonl"
      metric: "logprob_zero_shot"
```

### LLM (text-only) benchmarks

```yaml
training_config:
  eval_strategy: "steps"
  eval_steps: 2000

evals:
  max_new_tokens: 128
  benchmarks:
    - name: "reading_comprehension"
      path: "/data/reading_eval.jsonl"
      metric: "short_answer"

    - name: "phd_mcq"
      path: "/data/phd_logprob.jsonl"
      metric: "logprob_zero_shot"
```

The callback loads data once on the first eval step and caches it for all subsequent runs.

## Data Format

Benchmark data uses the **same format as training data** (HF messages schema). Supported file formats: JSONL, JSON, Parquet, CSV.

### Generation benchmarks

The last assistant turn is the ground truth. Everything before it is the prompt.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "image", "image": "/path/to/img.jpg" },
        { "type": "text", "text": "What is shown in this image?" }
      ]
    },
    {
      "role": "assistant",
      "content": [{ "type": "text", "text": "A dog sitting on a beach." }]
    }
  ]
}
```

### Logprob MCQ benchmarks

No assistant turn needed. The model scores each option by log-probability and picks the highest.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "image", "image": "/path/to/img.jpg" },
        { "type": "text", "text": "What animal is in this image?" }
      ]
    }
  ],
  "options": ["cat", "dog", "bird"],
  "answer_id": 1
}
```

### Text-only (no images)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [{ "type": "text", "text": "Capital of France?" }]
    },
    { "role": "assistant", "content": [{ "type": "text", "text": "Paris" }] }
  ]
}
```

## Available Metrics

| Metric              | Type       | Supported | Description                                                                                                               |
| ------------------- | ---------- | --------- | ------------------------------------------------------------------------------------------------------------------------- |
| `short_answer`      | generation | LLM, VLM  | Case-insensitive substring match. Set `match_mode: "any_in_array"` if ground truth is a JSON array of acceptable answers. |
| `mcq_gen`           | generation | LLM, VLM  | Extracts MCQ letter (A-F) from generated text and compares to ground truth.                                               |
| `grounding_iou`     | generation | VLM only  | Bounding-box IoU. Set `iou_threshold` (default 0.5).                                                                      |
| `logprob_zero_shot` | logprob    | LLM, VLM  | Zero-shot MCQ via per-option log-probability comparison. No generation needed.                                            |

## YAML Reference

Per-benchmark fields:

| Field            | Required | Default      | Description                                                   |
| ---------------- | -------- | ------------ | ------------------------------------------------------------- |
| `name`           | yes      | —            | Benchmark name (used in wandb keys: `benchmark/{name}/score`) |
| `path`           | yes      | —            | Path to data file (JSONL, JSON, Parquet, CSV)                 |
| `metric`         | yes      | —            | One of the metrics above                                      |
| `max_new_tokens` | no       | 128          | Max tokens to generate (generation metrics only)              |
| `match_mode`     | no       | `"contains"` | For `short_answer`: `"contains"` or `"any_in_array"`          |
| `iou_threshold`  | no       | 0.5          | For `grounding_iou`                                           |
| `limit`          | no       | all          | Cap number of eval samples                                    |
| `format`         | no       | auto-detect  | Force file format: `jsonl`, `json`, `parquet`, `csv`          |
| `image_root`     | no       | —            | Prepend to relative image paths                               |

## Adding a New Metric

1. Add a scoring function to `metrics.py`:

   ```python
   def score_my_metric(prediction: str, ground_truth: str, **_) -> float:
       # Return a float score for this sample
       ...
   ```

2. Register it in `_METRIC_DISPATCH` in the same file:

   ```python
   _METRIC_DISPATCH = {
       ...
       "my_metric": score_my_metric,
   }
   ```

3. Add the metric name to `GENERATION_METRICS` or `LOGPROB_METRICS` in the relevant config factory (`vlm_config.py` for VLM, `llm_config.py` for LLM).

## How It Works

1. The trainer reads `evals:` from your YAML config
2. The appropriate factory creates benchmark objects based on training type:
   - **VLM SFT**: `create_vlm_benchmarks_from_config()` — uses the VLM processor for image+text inputs
   - **SFT / DPO**: `create_llm_benchmarks_from_config()` — uses the tokenizer for text-only inputs
3. A `BenchmarkEvalCallback` is added to the HuggingFace Trainer
4. At every `eval_steps`, the callback:
   - Loads and normalizes data on first run (cached for subsequent evals)
   - Shards samples across GPUs
   - Runs evaluation (generation or logprob)
   - All-reduces scores across ranks
   - Logs averaged metrics to wandb
