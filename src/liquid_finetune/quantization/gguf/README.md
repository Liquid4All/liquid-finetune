# Bundled GGUF Conversion Scripts

Vendored from [llama.cpp](https://github.com/ggml-org/llama.cpp) for self-contained GGUF export.

Includes:

- `convert_hf_to_gguf.py` -- Convert HuggingFace models to GGUF (F16, BF16, F32, Q8_0)
- `convert_lora_to_gguf.py` -- Convert PEFT LoRA adapters to GGUF
- `gguf-py/` -- Python GGUF library

These are used by `liquid-finetune export` and do not need to be invoked directly.

For K-quant types (Q4_K_M, Q5_K_M, etc.), the `llama-quantize` binary from a built llama.cpp checkout is still required.
