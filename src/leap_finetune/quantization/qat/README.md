# Universal quantization-aware training

QAT is a model-preparation option, not a separate runner. It works with dense
SFT/DPO/GRPO, vision SFT/DPO/GRPO, and MoE SFT/DPO.

```yaml
training_config:
  qat:
    type: gguf_q4_0
```

Supported types are `gguf_q4_0`, `gguf_q8_0`, `mlx_q4`, `mlx_q8`,
`vllm_fp8`, `vllm_mxfp4`, `vllm_mxfp8`, `vllm_nvfp4`, `noise_q4`, and
`noise_q8`. The optional advanced DPO setting `quantize_reference: false` leaves the reference model in floating
point; it defaults to `true`.

GGUF profiles use exact Q4_0/Q8_0 block-32 fake quantization. MLX profiles use
affine group-32 weight quantization. vLLM FP8 follows its platform-native
online path (E4M3FN/per-token activations on CUDA and per-tensor activations
on ROCm, with E4M3FNUZ on gfx94x). MXFP4 uses E2M1 values and E8M0 group-32
scales; MXFP8 uses E4M3 values and E8M0 group-32 scales. NVFP4 uses E2M1
values, E4M3 group-16 scales, and an FP32 tensor-level scale.

NVFP4 QAT models the deterministic deployment-format quantize/dequantize
operation. Transformer Engine training can additionally use stochastic
rounding and a random Hadamard transform; those training heuristics are not
simulated. Conversion recalibrates the static tensor scale from representative
data.

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

GGUF QAT leaves tied token embeddings/output heads floating point. Preserve that
contract during Q4_0/Q8_0 export:

```bash
leap-export-gguf CHECKPOINT --quant Q4_0 --token-embedding-type F16 \
  --llama-cpp-dir LLAMA_CPP_DIR
```

## Quality matrix

Expand the baseline plus ten QAT profiles for every supported model/trainer
cell:

```bash
leap-qat-matrix manifests/qat_quality_matrix.yaml -o generated_qat_jobs
```

Run any generated config with `leap-finetune`. Each SFT, DPO, and GRPO row is
independent: it starts from its declared model and never consumes another
trainer's checkpoint. The manifest fixes seed 42, 10,000 training examples,
1,000 validation examples, and public evaluation subsets. Pinned subset
materializers under `manifests/quality/` record immutable dataset revisions,
selected row IDs, and artifact hashes for SFT, DPO, GRPO, and vision SFT.

Hardware-specific conversion commands are in
`manifests/qat_conversion_manifest.yaml`. The native matrix uses H100 for FP8,
B200 for FP8/MXFP8/MXFP4/NVFP4, MI325X for FP8, MI350/MI355 for MX formats,
llama.cpp for GGUF, and Apple hardware for MLX. Unsupported cells remain
explicitly pending rather than being reported as native results.

Build an honest report from a JSON list of result rows:

```bash
leap-qat-report results.json -o qat_report
```

The report emits JSON, CSV, and Markdown, calculates QAT-vs-PTQ and degradation
deltas, and leaves missing hardware cells marked `pending`.
