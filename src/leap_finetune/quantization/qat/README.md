# Universal quantization-aware training

QAT is a model-preparation option, not a separate runner. It works with dense
SFT/DPO/GRPO, vision SFT/DPO/GRPO, and MoE SFT/DPO.

```yaml
training_config:
  qat:
    type: gguf_q4_0
```

Supported types are `gguf_q4_0`, `gguf_q8_0`, `mlx_q4`, `mlx_q8`,
`vllm_fp8`, `vllm_mxfp4`, `noise_q4`, and `noise_q8`. The optional advanced
DPO setting `quantize_reference: false` leaves the reference model in floating
point; it defaults to `true`.

GGUF profiles use exact Q4_0/Q8_0 block-32 fake quantization. MLX profiles use
affine group-32 weight quantization. vLLM FP8 follows its platform-native
online path (E4M3FN/per-token activations on CUDA and per-tensor activations
on ROCm, with E4M3FNUZ on gfx94x). MXFP4 uses native round-up E8M0 scales
and E2M1 ties.

For cross-platform vLLM training, select the intended deployment contract:

```yaml
training_config:
  qat:
    type: vllm_fp8
    target: rocm_mi300 # or cuda; auto resolves once during preparation
```

The resolved target is saved in `qat_config.json` and validated on resume.
Noise profiles resample uniform blockwise perturbations on each forward.

QAT GRPO must set `use_vllm: false`; using a separate vLLM rollout engine would
make the behavior policy differ from the fake-quantized policy being optimized.
The vision tower, routers, embeddings, normalization layers, and tied output
heads stay floating point. GGUF also keeps the multimodal projector floating
point, matching the current GGUF deployment boundary.

Checkpoints remain ordinary Hugging Face checkpoints. `qat_config.json` records
the profile, and resume fails if it does not match the requested profile.
Loading without a `qat:` training option produces the normal floating-point
model.

## Quality matrix

Expand the baseline plus eight QAT profiles for every supported model/trainer
cell:

```bash
leap-qat-matrix manifests/qat_quality_matrix.yaml -o generated_qat_jobs
```

Run any generated config with `leap-finetune`. The manifest fixes seed 42,
training budgets, representative models, and compact evaluation sizes. Record
the resolved dataset revision and selected row IDs when launching a run.
Hardware-specific conversion commands are in
`manifests/qat_conversion_manifest.yaml`. Native FP8 is available on MI325X.
The current ROCm vLLM wheel has no dense W4A4 MXFP4 linear kernel for gfx942,
so that evaluation cell remains pending MI355X hardware.

Build an honest report from a JSON list of result rows:

```bash
leap-qat-report results.json -o qat_report
```

The report emits JSON, CSV, and Markdown, calculates QAT-vs-PTQ and degradation
deltas, and leaves missing hardware cells marked `pending`.
