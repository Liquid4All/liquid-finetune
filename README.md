<div align="center">
  <img
    src="./banner.png"
    alt="leap-finetune"
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> -
    <a href="https://docs.liquid.ai/lfm"><strong>Documentation</strong></a> -
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a>
  </div>
  <br/>
  <a href="https://discord.com/invite/liquid-ai"><img src="https://img.shields.io/discord/1385439864920739850?style=for-the-badge&logo=discord&logoColor=white&label=Discord&color=5865F2" alt="Join Discord"></a>
</div>

<p align="center">
<a href="#setup">Setup</a> -
<a href="#quickstart">Quickstart</a> -
<a href="#cli-and-python-usage">CLI</a> -
<a href="#execution-backends">Backends</a> -
<a href="#datasets">Datasets</a> -
<a href="#grpo">GRPO</a> -
<a href="#evaluation">Evaluation</a> -
<a href="#quantization--gguf-export">GGUF</a> -
<a href="#contributing">Contributing</a>
</p>

LEAP-Finetune is a minimal fine-tuning repo for LFM2. It handles dataset
formatting, validation, distributed orchestration, checkpointing, and export
for local GPU nodes, SLURM clusters, Modal, and Kubernetes/KubeRay.

For feature requests or custom infrastructure support, reach out to
[support@liquid.ai](mailto:support@liquid.ai) with your setup.

## Setup

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo:

```bash
git clone <repository-url>
cd leap_finetune
```

### Backend Install Profiles

CUDA / NVIDIA clusters use the root project and default dependency groups:

```bash
uv sync
```

AMD / ROCm clusters use the ROCm project. Set `UV_PROJECT` once in your shell,
module, or direnv config:

```bash
export UV_PROJECT=envs/rocm
uv sync
```

After `UV_PROJECT` is set, normal commands stay the same: `uv sync`,
`uv run ...`, and the SLURM helpers all use the selected backend project.

```bash
source .venv/bin/activate
leap-finetune job_configs/sft_example_with_slurm.yaml

# ROCm, after `export UV_PROJECT=envs/rocm`:
uv run leap-finetune job_configs/sft_example_with_slurm.yaml
```

<details>
<summary>Backend install details</summary>

The environment variable is `UV_PROJECT`, not `UV_EXPORT`.

The default install paths both try pinned accelerator wheels:

- CUDA: `uv sync` installs the default CUDA lock, including `vllm==0.22.0` and
  the pinned CUDA FlashAttention 2 wheel for the Torch 2.11 / CUDA 13 stack.
- ROCm: `UV_PROJECT=envs/rocm uv sync` installs the ROCm lock from
  [`envs/rocm`](./envs/rocm/), including direct URLs for the validated
  `torch`, `torchvision`, `torchaudio`, `triton`, `flash-attn`, and `vllm==0.22.0+rocm722` wheel set.

Ray is pinned to `2.51.1` for both profiles. The pinned accelerator wheels are
Python 3.12 Linux x86_64 wheels, so use the repo's `.python-version` when
creating GPU environments.

The CUDA project is the repository root, so it does not require `UV_PROJECT`.
Select ROCm explicitly with `export UV_PROJECT=envs/rocm`.

No hardware-specific environment variables are required for installation. On
clusters where the default uv cache is slow, quota-limited, or backed by
node-local scratch, you can prefix either install command with
`UV_CACHE_DIR=.uv-cache` to keep uv's package cache in the repo.

</details>

### FlashAttention 2

The default CUDA and ROCm install paths try to install pinned FA2 wheels. If FA2
installs but does not import or cannot be selected at runtime, training emits an
explicit warning and falls back to SDPA:

```text
FlashAttention 2 not available (...); falling back to SDPA.
```

To inspect the active environment:

```bash
# CUDA, after `uv sync`
uv run leap-finetune env fa2-status

# ROCm, after `export UV_PROJECT=envs/rocm && uv sync`
uv run leap-finetune env fa2-status

# Fail if FA2 is not usable
uv run leap-finetune env fa2-status --require
```

`fa2-status` reports the detected backend, Python tag, platform, Torch version,
CUDA/HIP version, accelerator visibility, installed `flash-attn` version,
selected attention implementation, and the reason FA2 is or is not usable.

<details>
<summary>Install without FA2 or repair FA2 separately</summary>

If a pinned FA2 wheel cannot be resolved or installed, `uv sync` fails at install
time. Install without the pinned FA2 group first, then repair FA2 separately:

```bash
# CUDA without pinned FA2
uv sync --no-group flash-attn
uv run leap-finetune env install-fa2

# ROCm HF training without pinned FA2/vLLM
UV_PROJECT=envs/rocm uv sync --no-group rocm-fa2 --no-group rocm-vllm
UV_PROJECT=envs/rocm uv run leap-finetune env install-fa2
```

`install-fa2` tries, in order:

1. A matching pinned wheel for the detected CUDA or ROCm runtime.
2. Binary-only public resolution for `flash-attn==2.8.3`.
3. Source build, only when explicitly requested.

Source builds are not part of the normal install path. Use them only as an
explicit escape hatch on a machine with the matching CUDA or ROCm toolchain and
enough build memory:

```bash
uv run leap-finetune env install-fa2 --allow-source-build
```

ROCm GRPO/vLLM support requires the full `rocm-vllm` profile and therefore a
compatible ROCm FA2 stack. If that stack cannot resolve on a target cluster, use
the SDPA fallback for non-vLLM training until a matching vLLM/FA2 wheel set is
available.

</details>

## Quickstart

Create a YAML config file or copy one from [`job_configs/`](./job_configs/):

```yaml
project_name: "my_sft_project"
model_name: "LFM2-1.2B"
training_type: "sft"

dataset:
  path: "HuggingFaceTB/smoltalk"
  type: "sft"
  limit: 1000
  test_size: 0.2
  subset: "all"

training_config:
  extends: "DEFAULT_SFT"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 2e-5

peft_config:
  extends: "DEFAULT_LORA"
  use_peft: true
```

`training_config.extends` inherits from a base config such as `DEFAULT_SFT`,
`DEFAULT_DPO`, or `DEFAULT_VLM_SFT`; fields in your YAML override the base.
`peft_config.extends` works the same way for LoRA defaults such as
`DEFAULT_LORA` and `DEFAULT_VLM_LORA`.

Launch training:

```bash
uv run leap-finetune job_configs/sft_example.yaml
```

Training uses Ray Train and Accelerate for distributed execution. SFT, DPO, VLM
SFT, and VLM DPO automatically use the native Hugging Face Trainer when exactly
one GPU is visible. Text and VLM GRPO use the native Trainer only for one-GPU
`vllm_mode: colocate` runs. Set `LEAP_LAUNCHER=ray` to force Ray. Unless
`output_dir` is set, results are written to
`outputs/{project_name}/{run_name}/`. Each run gets a unique name based on the
model, dataset, learning rate, and timestamp.

Useful starter configs:

| Mode                    | Config                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| SFT                     | [`job_configs/sft_example.yaml`](./job_configs/sft_example.yaml)                               |
| SFT + LoRA              | [`job_configs/sft_with_lora_example.yaml`](./job_configs/sft_with_lora_example.yaml)           |
| DPO                     | [`job_configs/dpo_example.yaml`](./job_configs/dpo_example.yaml)                               |
| Dense embedding         | [`job_configs/embedding_example.yaml`](./job_configs/embedding_example.yaml)                   |
| ColBERT                 | [`job_configs/colbert_example.yaml`](./job_configs/colbert_example.yaml)                       |
| VLM SFT                 | [`job_configs/vlm_sft_example.yaml`](./job_configs/vlm_sft_example.yaml)                       |
| VLM DPO                 | [`job_configs/vlm_dpo_example.yaml`](./job_configs/vlm_dpo_example.yaml)                       |
| GRPO                    | [`job_configs/grpo_example.yaml`](./job_configs/grpo_example.yaml)                             |
| VLM GRPO                | [`job_configs/vlm_grpo_grounding_example.yaml`](./job_configs/vlm_grpo_grounding_example.yaml) |
| MoE SFT                 | [`job_configs/moe_sft_example.yaml`](./job_configs/moe_sft_example.yaml)                       |
| MoE DPO                 | [`job_configs/moe_dpo_example.yaml`](./job_configs/moe_dpo_example.yaml)                       |
| Expert-parallel MoE SFT | [`job_configs/moe_ep_sft_example.yaml`](./job_configs/moe_ep_sft_example.yaml)                 |
| Standalone eval         | [`job_configs/eval_standalone_example.yaml`](./job_configs/eval_standalone_example.yaml)       |

## CLI and Python Usage

During development, prefer the repo environment so you get the lockfile-managed
CUDA/vLLM stack:

```bash
uv run leap-finetune job_configs/sft_example.yaml
uv run leap-finetune run job_configs/sft_example.yaml
uv run leap-finetune job_configs/eval_standalone_example.yaml
uv run leap-finetune eval job_configs/eval_standalone_example.yaml --output results.json
```

Install the command as a reusable tool from a checkout:

```bash
uv tool install --editable . --force
leap-finetune /absolute/path/to/config.yaml
leap-finetune slurm /absolute/path/to/config.yaml --output-dir /absolute/path/to/slurms
leap-finetune /absolute/path/to/eval_config.yaml
leap-finetune eval /absolute/path/to/eval_config.yaml --output /absolute/path/to/results.json
```

`uv tool install` creates an isolated tool environment. Use explicit config
paths when invoking the command outside the repo; bare names like
`sft_example.yaml` resolve from the current directory's `job_configs/` first,
then from the installed package's `LEAP_FINETUNE_DIR`.

For one-off execution without installing the command:

```bash
uvx --from . leap-finetune /absolute/path/to/config.yaml
```

You can also start a run from Python. This uses the same backend dispatch as
the CLI: configs with `slurm`, `modal`, or `kuberay` submit remotely; other
configs run local training and require visible CUDA devices. SFT, DPO, VLM SFT,
and VLM DPO use the native Trainer on one GPU; GRPO uses it only for one-GPU
colocated vLLM; other modes use Ray.

```python
from leap_finetune import run_config

run_config("/absolute/path/to/config.yaml")
```

Standalone evals use the same entry point:

```python
from leap_finetune import run_config

metrics = run_config("/absolute/path/to/eval_config.yaml")
```

Run that file inside an environment where `leap-finetune` is installed:

```bash
uv run --with-editable . python launch_training.py
```

## Execution Backends

Configs without a remote backend section run in the current environment through
Ray Train. Add one of the backend blocks below to submit the same config to a
remote runtime.

### Modal

Modal lets you run training jobs on serverless GPUs from a laptop or Mac. No
local GPU is required.

One-time setup:

```bash
huggingface-cli login
modal setup
```

Add a `modal:` section:

```yaml
modal:
  gpu: "H100:4"
  timeout: 86400
  output_volume: "leap-finetune"
  output_dir: "/outputs"
  detach: false
```

Run:

```bash
uv run leap-finetune job_configs/sft_example_modal.yaml
```

In attached mode (`detach: false`), the CLI builds the container image,
auto-creates a `huggingface-secret` from your local HF token, streams logs, and
saves checkpoints to a Modal Volume.

Retrieve checkpoints:

```bash
modal volume ls leap-finetune
modal volume get leap-finetune <checkpoint-name> ./local-outputs
```

Set `detach: true` to submit and disconnect. The CLI prints the Modal app ID
plus commands to monitor or stop it:

```bash
modal app logs ap-...
modal app stop ap-...
```

Use the printed `ap-...` app ID for detached logs. See
[`job_configs/sft_example_modal.yaml`](./job_configs/sft_example_modal.yaml).

### SLURM

If your config includes a `slurm:` section, `leap-finetune` auto-generates and
submits a SLURM script:

```bash
uv run leap-finetune job_configs/sft_example_with_slurm.yaml
```

Generate a SLURM script without submitting it:

```bash
uv run leap-finetune slurm <path_to_config.yaml>
```

Monitor your SLURM jobs in a TUI:

```bash
uv run turm --me
```

### Kubernetes / KubeRay

If your config includes a `kuberay:` section, `leap-finetune` submits a KubeRay
`RayJob` instead of launching local training. You need a configured Kubernetes
context, KubeRay CRDs installed, and a container image that already contains
this repo plus its Python environment.

```yaml
kuberay:
  image: "registry.example.com/leap-finetune:latest"
  namespace: "training"
  worker_replicas: 2
  gpus_per_worker: 4
  output_dir: "/outputs"
  output_pvc: "leap-finetune-outputs"
  env:
    HF_HOME: "/outputs/hf-cache"
```

Run the same command as local training:

```bash
uv run leap-finetune path/to/config.yaml
```

The CLI creates a ConfigMap for the training config, submits a RayJob, and
prints `kubectl` commands for status and logs. The product of
`worker_replicas` and `gpus_per_worker` becomes `ray.num_workers` unless you
set it explicitly.

### Experiment Tracking

Add `tracker` to `training_config`:

```yaml
training_config:
  tracker: "trackio" # or "wandb"
```

[Trackio](https://huggingface.co/blog/trackio) logs to a HuggingFace Space.
`trackio_space_id` is auto-created if needed, and Modal injects the HF token
automatically:

```yaml
training_config:
  tracker: "trackio"
  trackio_space_id: "username/my-dashboard"
```

[Weights & Biases](https://wandb.ai) uses `WANDB_API_KEY`:

```yaml
training_config:
  tracker: "wandb"
```

Set the key locally or add it as a Modal secret:

```bash
export WANDB_API_KEY=your_key
```

```bash
modal secret create wandb-secret WANDB_API_KEY=your_key
```

```yaml
modal:
  secrets:
    - "wandb-secret"
```

### Bundle Checkpoints for LEAP

When training is done, bundle your output checkpoint with `leap-bundle` to use
it directly within LEAP. See the
[LEAP bundle quickstart](https://leap.liquid.ai/docs/leap-bundle/quick-start?utm_source=github&utm_medium=link&utm_campaign=LEAP&utm_content=general).

## Datasets

### Loading Data

The `dataset.path` field accepts local files, HuggingFace Hub IDs, and cloud
storage URIs:

| Source          | Example `path`                                 |
| --------------- | ---------------------------------------------- |
| Local file      | `/path/to/data.jsonl`, `/path/to/data.parquet` |
| HuggingFace Hub | `HuggingFaceTB/smoltalk`                       |
| S3              | `s3://bucket/path/to/data.parquet`             |
| GCS             | `gs://bucket/path/to/data.parquet`             |
| Azure           | `az://container/path/to/data.parquet`          |

Cloud storage requires appropriate AWS, GCP, or Azure credentials. Use `subset`
for HuggingFace datasets with multiple configs, `split` for HF split
expressions such as `train+validation`, and `limit` to cap samples for quick
testing.

### SFT

```json
{
  "messages": [
    { "role": "user", "content": "What is the capital of France?" },
    { "role": "assistant", "content": "The capital of France is Paris." }
  ]
}
```

### DPO

DPO uses preference columns instead of a top-level `messages` field. For
single-turn data, `prompt`, `chosen`, and `rejected` can be plain strings:

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "The capital of France is London."
}
```

For multi-turn DPO, make `prompt` the shared conversation history and make
`chosen` / `rejected` the preferred and rejected assistant completions:

```json
{
  "prompt": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "I am planning a trip to France." },
    { "role": "assistant", "content": "What would you like to know?" },
    { "role": "user", "content": "What is the capital?" }
  ],
  "chosen": [
    { "role": "assistant", "content": "The capital of France is Paris." }
  ],
  "rejected": [
    { "role": "assistant", "content": "The capital of France is London." }
  ]
}
```

Rows without `prompt` are also accepted if `chosen` and `rejected` are full
conversations with the same shared prefix; the tokenizer extracts that prefix
as the prompt. Prefer the explicit `prompt` shape above when writing new data.

### KTO

KTO ([Ethayarajh et al., 2024](https://huggingface.co/papers/2402.01306))
learns from _unpaired_ binary feedback: each row is a single completion marked
desirable or undesirable, so no same-prompt preference pairs are needed. Use
`training_type: "kto"` with `prompt`, `completion`, and boolean `label`
columns (`true` = desirable). Like DPO, `prompt` and `completion` can be plain
strings or message lists:

```json
{
  "prompt": [{ "role": "user", "content": "What is the capital of France?" }],
  "completion": [
    { "role": "assistant", "content": "The capital of France is London." }
  ],
  "label": false
}
```

KTO rows are tokenized by TRL's `KTOTrainer` on each worker rather than by
Leap's pretokenization pipeline. `per_device_train_batch_size` must be greater
than 1 because KTO estimates its KL term from mismatched completions within a
batch. Leap preserves that batch order, uses the train batch size for evaluation,
and drops incomplete final batches. For imbalanced labels, tune
`desirable_weight` and `undesirable_weight`; TRL logs a warning when the
weights do not compensate for the label balance.

### Retrieval and Embedding models

Use `training_type: "embedding"` for LFM Embedding models and
`training_type: "colbert"` for LFM ColBERT models. Both accept the same
dataset shape:

```json
{
  "query": "search text",
  "positive": "relevant document",
  "negative": "irrelevant document"
}
```

`negative` is optional for training and required for triplet-accuracy metrics.
Dense training automatically applies `query: ` and `document: ` prompts.
ColBERT uses PyLate's query/document tokenization and MaxSim loss. On multiple
GPUs, both losses gather in-batch negatives across workers; each retrieval
worker aligns its shard to complete batches so collective tensor shapes stay
equal.
Gradient accumulation is intentionally rejected—select a cached loss and set
`mini_batch_size` when a larger effective batch is needed.

```bash
uv run leap-finetune job_configs/embedding_example.yaml
uv run leap-finetune job_configs/colbert_example.yaml
```

### VLM SFT

```json
{
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are an image-based assistant. Answer questions based on the provided image."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        { "type": "image", "image": "/path/to/image.jpg" },
        { "type": "text", "text": "What do you see in this image?" }
      ]
    },
    {
      "role": "assistant",
      "content": [{ "type": "text", "text": "I see a car in the image." }]
    }
  ]
}
```

VLM datasets often store image paths in a separate column. Merge those image
references into each row's `messages` content before training.

### VLM DPO

Use `training_type: "vlm_dpo"` for multimodal preference data. Each row should
provide `prompt`, `chosen`, and `rejected` message lists plus either an `image`
or `images` column:

```json
{
  "image": "images/chart.png",
  "prompt": [
    {
      "role": "user",
      "content": [{ "type": "text", "text": "Which trend is most important?" }]
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": "Revenue grows fastest in Q4."
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": "The chart is inconclusive."
    }
  ]
}
```

Use `images` instead of `image` for multi-image rows. Relative image paths are
resolved against `dataset.image_root` when set. See
[`job_configs/vlm_dpo_example.yaml`](./job_configs/vlm_dpo_example.yaml) for a
complete LoRA config. `freeze_vision_encoder` and
`optimizer_type: "adamw_8bit"` are optional knobs, not defaults.

### GRPO and VLM GRPO

GRPO can reuse the SFT/VLM SFT `messages` format. The loader turns each row
into `prompt` and `solution` for online reward computation, and any extra
columns are forwarded to reward functions. See [GRPO](#grpo) for the full
dataset, reward, and vLLM rollout contract.

## Tool Calling Datasets

Tool calls use LFM bracket notation in the assistant `content` field. Tool
definitions go in the system prompt, and tool responses use `role: "tool"`.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "List of tools: [{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}]"
    },
    { "role": "user", "content": "What's the weather in Boston?" },
    {
      "role": "assistant",
      "content": "<|tool_call_start|>[get_weather(location=\"Boston\")]<|tool_call_end|>"
    },
    {
      "role": "tool",
      "content": "{\"temperature\": 72, \"condition\": \"sunny\"}"
    },
    { "role": "assistant", "content": "It's 72 F and sunny in Boston." }
  ]
}
```

- Tool calls must be pre-baked in `content` using
  `<|tool_call_start|>[func(args)]<|tool_call_end|>` bracket notation.
- Structured `tool_calls` fields in OpenAI format are auto-converted when
  present.
- Foreign formats such as `<tool_call>` XML are rejected with an actionable
  error.
- Do not include `<|tool_response_start|>` or `<|tool_response_end|>` markers
  in `role: "tool"` messages. The LFM2 chat template adds them during
  tokenization.
- LFM2 models expect `<|tool_list_start|>` and `<|tool_list_end|>` around tool
  definitions in the system prompt. Include them for LFM2 and omit them for
  LFM2.5. The pipeline warns on mismatches and auto-strips
  `<|tool_list_start|>` when training LFM2.5.

## Resuming Training

Resume interrupted runs from the last checkpoint with optimizer state, LR
schedule, training step, RNG state, and tracker continuity intact:

```yaml
training_config:
  resume_from_checkpoint: "latest"
```

`latest` finds the most recent run directory under `outputs/{project_name}/`
and resumes from its latest checkpoint. To resume from a specific checkpoint:

```yaml
training_config:
  resume_from_checkpoint: "/path/to/outputs/my_project/run_name/checkpoint-step-8000"
```

To resume a run, `save_only_model` must be `False`. The wandb run ID is saved
to `<run_dir>/.wandb_run_id`; resumed runs reuse the same wandb run, and fresh
runs create a new one.

## GRPO

GRPO runs online RL with TRL v1's `GRPOTrainer`. Use `training_type: "grpo"`
for text models and `training_type: "vlm_grpo"` for vision-language models.
Both modes use the same YAML entrypoint as SFT/DPO, the same Ray Train
launcher, and vLLM rollouts by default.

### Dataset Contract

Text GRPO can reuse the SFT `messages` format: the loader builds `prompt` from
non-assistant turns and `solution` from the last assistant message. Native
`prompt` / `solution` columns also work. VLM GRPO uses the same multimodal
`messages` shape as VLM SFT. Extra dataset columns are forwarded to reward
functions as keyword arguments.

### Rewards

The `rewards:` block resolves plain Python callables and task recipes from
[`rewards/`](./rewards/README.md). Shipped primitive functions can be
referenced by function name:

```yaml
rewards:
  funcs:
    - "accuracy_reward"
    - "length_reward"
  weights: [1.0, 0.1]
```

Task recipes bundle multiple reward functions and their default weights:

```yaml
rewards:
  recipe: "tasks/vlm_grounding/recipe.py::VLMGroundingIoURecipe"
```

If you combine `recipe:` and `funcs:`, recipe rewards are ordered first. A
`weights:` override must match the expanded reward list. Absolute paths and
`./rewards/file.py::function_name` specs work for custom rewards.

### Judge LLM Reward

Add `rewards.judge` when the reward signal should come from an LLM grader.
Without `base_url`, the driver starts a local `trl vllm-serve` judge server
before Ray initializes and exports the endpoint to workers:

```yaml
rewards:
  judge:
    model: "LFM2-1.2B"
    weight: 1.0
    prompt_template: |
      Prompt:
      {prompt}

      Assistant response:
      {completion}

      Reference answer or rubric:
      {solution}

      Return only JSON: {"score": 0.0}

grpo_rollout:
  judge_gpus: 1
```

For an external judge, set `rewards.judge.base_url` and omit `judge_gpus`.
Judge scores are parsed from JSON or the first number in the response and
normalized from `min_score`/`max_score` to `[0, 1]`.

### vLLM Rollouts

`DEFAULT_GRPO` and `DEFAULT_VLM_GRPO` set `use_vllm: true` and
`vllm_mode: "colocate"`. Colocate mode runs vLLM inside each training worker.
Server mode starts `trl vllm-serve` on driver GPUs before Ray initializes.
Configure GPU counts, not device ids:

```yaml
grpo_rollout:
  server_gpus: 1 # reserve 1 local GPU for vLLM; training gets the rest
  judge_gpus: 1 # optional: reserve 1 local GPU for rewards.judge
  # training_gpus: 3    # or set only this and vLLM gets the remaining GPUs
  tensor_parallel_size: 1
  dtype: "bfloat16"
  gpu_memory_utilization: 0.9

training_config:
  extends: "DEFAULT_GRPO"
  vllm_mode: "server"
  vllm_server_host: "auto"
  vllm_server_port: 8000
```

Local server partitioning is single-node only. For multi-node GRPO, use
colocate mode or point `vllm_server_base_url` at an externally managed vLLM
server without setting `server_gpus` or `training_gpus`.

Example configs:

- [`job_configs/grpo_example.yaml`](./job_configs/grpo_example.yaml): text GRPO
  quickstart with the GSM8K recipe.
- [`job_configs/grpo_server_mode_example.yaml`](./job_configs/grpo_server_mode_example.yaml):
  text GRPO with a local `trl vllm-serve` rollout server.
- [`job_configs/vlm_grpo_grounding_example.yaml`](./job_configs/vlm_grpo_grounding_example.yaml):
  VLM GRPO with the visual-grounding recipe.

Launch the same way as SFT/DPO:

```bash
uv run leap-finetune job_configs/grpo_example.yaml
```

### Agentic Environments

For tasks where the environment state evolves from agent actions, such as
browsing, tool use, game simulators, or stateful multi-turn tasks,
`leap-finetune` supports [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
via an optional `rl_env:` block:

```bash
uv sync --extra rl-env
```

See
[`src/leap_finetune/rl/environments/README.md`](./src/leap_finetune/rl/environments/README.md).
For anything scorable by a pure Python function, prefer the `rewards:` path; it
is simpler and faster.

## Evaluation

Run benchmarks during training at every `eval_steps` by adding an `evals:`
section. The legacy `benchmarks:` alias still parses.

```yaml
evals:
  max_new_tokens: 128
  benchmarks:
    - name: "mmmu_val"
      path: "/data/mmmu_val.jsonl"
      metric: "short_answer"

    - name: "imagenette"
      path: "/data/imagenette_eval.jsonl"
      metric: "logprob_zero_shot"
```

Benchmark data uses the same HF messages schema as training data. Available
metrics include `short_answer`, `grounding_iou`, `mcq_gen`, and
`logprob_zero_shot`. Results are logged to wandb at
`benchmark/{name}/score`.

See the [Evaluation Guide](./src/leap_finetune/evaluation/README.md) for data
format examples, YAML reference, and custom metrics.

Run the same eval suite without training:

```bash
uv run leap-finetune job_configs/eval_standalone_example.yaml
```

Standalone eval configs use `model_name` or `checkpoint`, `evals:`, and an
optional `backend:` block. They do not include `dataset`, `training_type`,
`training_config`, or `async_eval`. Text evals default to `modality: text`;
set `modality: vlm` only for standalone VLM evals.

Use the explicit `eval` subcommand when you want CLI-only eval options such as
writing metrics to JSON:

```bash
uv run leap-finetune eval job_configs/eval_standalone_example.yaml --output results.json
```

The same path is available from Python:

```python
from leap_finetune import run_config

metrics = run_config("job_configs/eval_standalone_example.yaml")
```

### Async Eval (vLLM)

By default, every `eval_steps` blocks training until benchmarks finish. For
large generation suites this dominates wall-clock time. Add an `async_eval`
block to run benchmarks **without blocking training**, using vLLM for the
actual generation. Results are logged to wandb with `benchmark/step` and
`train/global_step` fields so dashboards can align benchmark metrics to the
training step that triggered them.

Three modes (default is `sync` = today's behavior):

| Mode       | Engine          | Pauses training? | GPUs reserved                    | Latency                   | Multi-node training  | Best for                                              |
| ---------- | --------------- | ---------------- | -------------------------------- | ------------------------- | -------------------- | ----------------------------------------------------- |
| `sync`     | HF transformers | Yes              | None                             | Immediate                 | ✓                    | Small/fast eval suites; default                       |
| `sidecar`  | vLLM            | **No**           | None (slurm-scheduled per cycle) | Slurm queue + eval time   | ✓                    | Tight clusters; eval should be free of training cost  |
| `reserved` | vLLM            | **No**           | N throughout the run             | ~30–60s respawn per cycle | **Single-node only** | Spare GPUs on one node, want predictable eval latency |

`reserved` mode carves its GPUs off the same SLURM allocation as
training via the driver's `CUDA_VISIBLE_DEVICES`, which only affects
the head node. Multi-node training will raise `NotImplementedError`
at startup — use `sidecar` instead, which scales to any node count.

Async sidecar mode serves generation through vLLM and falls back to an HF model
inside the sidecar for benchmark types vLLM cannot serve, such as logprob
scoring. Reserved mode keeps a persistent vLLM server and should be used for
generation benchmarks; use `sync` or `sidecar` for logprob suites.

```yaml
# Opt in by adding this block. See job_configs/sft_with_async_eval_example.yaml
async_eval:
  mode: sidecar # sync (default) | sidecar | reserved
  vllm_gpus: 1
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9

  # mode=sidecar: short sbatch job per eval_steps
  sbatch:
    time: "00:30:00"
    # partition / account default to inheriting from the parent job

  # mode=reserved: long-running vllm-serve on dedicated GPUs (single-node only for v1)
  reserved:
    weight_reload: respawn
    server_port: 8100
```

Failures are isolated: if eval crashes or sbatch is rejected, training continues. After `failure.max_consecutive` consecutive failures the callback disables itself for the rest of the run. See [`job_configs/sft_with_async_eval_example.yaml`](./job_configs/sft_with_async_eval_example.yaml) for a full example.

### Post-Training Evaluation with lmms-eval

For standard VLM benchmarks such as MMMU, OCRBench, RefCOCO, and POPE, use an
environment that includes the private Liquid4All `lmms-eval` fork with LFM2
model support.

Evaluate a fine-tuned checkpoint:

```bash
python -m lmms_eval \
  --model lfm2_vl \
  --model_args pretrained=/path/to/checkpoint \
  --tasks mmmu_val,ocrbench,pope \
  --batch_size 1
```

Multi-GPU:

```bash
torchrun --nproc-per-node=4 -m lmms_eval \
  --model lfm2_vl \
  --model_args pretrained=/path/to/checkpoint \
  --tasks mmmu_val,ocrbench,pope \
  --batch_size 1
```

For faster evaluation with the vLLM backend:

```bash
python -m lmms_eval \
  --model lfm2_vl_vllm \
  --model_args pretrained=/path/to/checkpoint,tensor_parallel_size=1,gpu_memory_utilization=0.85 \
  --tasks mmmu_val,ocrbench,pope \
  --batch_size 64
```

The `lmms-eval` and vLLM packages are sourced from private Liquid4All forks
with LFM2 model support, so SSH access to those repos is required.

## Quantization / GGUF Export

Export a HuggingFace checkpoint or PEFT adapter to GGUF with
`leap-export-gguf`:

```bash
uv run leap-export-gguf /path/to/checkpoint --quant F16 --output-dir ./outputs/gguf
```

Repeat `--quant` to produce multiple outputs:

```bash
uv run leap-export-gguf /path/to/checkpoint \
  --quant F16 \
  --quant Q4_K_M \
  --output-dir ./outputs/gguf \
  --llama-cpp-dir /path/to/llama.cpp
```

`F16`, `BF16`, `F32`, and `Q8_0` are exported directly with the bundled
llama.cpp conversion scripts. K-quants such as `Q4_K_M`, `Q5_K_M`, and `Q6_K`
require a built llama.cpp checkout containing `build/bin/llama-quantize`; pass
`--llama-cpp-dir` or set `LLAMA_CPP_DIR`.

PEFT adapter directories can be exported with `F16`, `BF16`, `F32`, or `Q8_0`:

```bash
uv run leap-export-gguf /path/to/adapter \
  --base-model-path /path/to/base-model \
  --quant F16 \
  --output-dir ./outputs/gguf
```

For adapter K-quants, merge the adapter into the base model first, then export
the merged checkpoint.

### Quantization-aware training

QAT is a shared model-preparation layer used by the existing runners; there are
no QAT-specific entrypoints. Dense and vision models support SFT, DPO, and
GRPO; MoE supports SFT and DPO.

```yaml
training_config:
  qat:
    type: gguf_q4_0
```

Profiles are `gguf_q4_0`, `gguf_q8_0`, `mlx_q4`, `mlx_q8`, `vllm_fp8`,
`vllm_mxfp4`, `noise_q4`, and `noise_q8`. QAT starts on the first forward and
keeps checkpoints as standard floating-point Hugging Face models with a
`qat_config.json` sidecar. DPO quantizes its reference by default. QAT GRPO
requires `use_vllm: false` so rollout generation and optimization share the
same fake-quantized policy.

For vLLM FP8, `target: cuda` or `target: rocm_mi300` selects deployment math
independently of the training host; `auto` resolves once and is saved in the
checkpoint metadata. Architecture targeting, profile math, exclusions, resume
behavior, and hardware limitations are documented in the
[QAT implementation notes](./src/leap_finetune/quantization/qat/README.md).

Use [`job_configs/eval_gguf_example.yaml`](./job_configs/eval_gguf_example.yaml)
for llama.cpp evaluation. Reproducible training, conversion, and ROCm execution
recipes live under [`manifests/`](./manifests/); all large paths must point to
persistent shared storage.

## Advanced Configuration

Default base configs live in
[`src/leap_finetune/training/default_configs/`](./src/leap_finetune/training/default_configs/)
and are auto-discovered. New configs added to those files are immediately
available via `extends` in YAML.

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is installed by the
default CUDA group. Enable it with `use_liger_kernel: true` in
`training_config`.

LoRA is configured through `peft_config`. Continued LoRA training can load an
existing adapter with `adapter_path`; VLM configs can optionally freeze the
vision encoder with `freeze_vision_encoder` and use bitsandbytes 8-bit AdamW
with `optimizer_type: "adamw_8bit"`.

Pinned chat templates live in
[`job_configs/chat_templates/`](./job_configs/chat_templates/). Non-local
LiquidAI LFM2.5 and LFM2-24B models select the pinned LFM2.5 template by
default.

For conversational SFT, set `assistant_only_loss: true` to supervise only
assistant turns, or `completion_only_loss: true` to supervise only the final
assistant completion. These options require a chat template with
`{% generation %}` and `{% endgeneration %}` markers around assistant output;
the bundled LFM templates include them. They are not used for plain-text SFT
rows.

## Contributing

### Testing

The test suite is intentionally scoped to config validation, mathematical
correctness, and full-service e2e coverage. See [`tests/README.md`](./tests/README.md) for the current layout.
E2E fixtures and SLURM launchers live under [`tests/e2e/`](./tests/e2e/).

Run the normal tests:

```bash
uv run pytest tests/config tests/numerics -q
```

GPU e2e tests require an appropriate GPU or cluster backend; see
[`tests/e2e/`](./tests/e2e/) for launchers and fixtures.

### Pull Requests

Install pre-commit hooks:

```bash
uv run pre-commit install
```

Run hooks manually:

```bash
uv run pre-commit run --all-files
```

Open a PR with a clear description of the behavior changed, tests run, and any
known limitations.
