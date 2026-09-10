from __future__ import annotations

import pathlib
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

# ==== Backend Protocol ====


@dataclass
class GenerateRequest:
    """One generation request.

    ``messages`` is the chat-format prompt. ``images`` is a list of PIL
    images for VLM requests, or None for text-only — the backend uses
    its presence to pick the VLM code path.
    """

    messages: list[dict]
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    images: list[Any] | None = None


@dataclass
class GenerateResult:
    text: str


@dataclass
class LogprobRequest:
    """Score each continuation conditioned on ``messages``. Returns one
    length-normalized log-probability per continuation."""

    messages: list[dict]
    continuations: list[str]
    images: list[Any] | None = None


@dataclass
class LogprobResult:
    logprobs: list[float] = field(default_factory=list)


class InferenceBackend(Protocol):
    name: str

    def generate(self, requests: list[GenerateRequest]) -> list[GenerateResult]: ...

    def logprobs(self, requests: list[LogprobRequest]) -> list[LogprobResult]: ...

    def close(self) -> None: ...


# === HF in-process backend ===


class HFBackend:
    """In-process HF model + tokenizer/processor. Used by the sync path
    and as a fallback inside async runners for benchmarks the vLLM
    backends can't serve (e.g. logprob scoring).

    ``modality`` is ``"text"`` or ``"vlm"``. For VLM, ``processor`` must
    be the full multimodal processor; for text it can be the tokenizer.
    """

    name = "hf"

    def __init__(self, model, processor, device, modality: str = "text"):
        self.model = model
        self.processor = processor
        self.device = device
        self.modality = modality

    def _tokenizer(self):
        # VLM processors expose .tokenizer; text-only processors ARE the tokenizer.
        return getattr(self.processor, "tokenizer", self.processor)

    def generate(self, requests: list[GenerateRequest]) -> list[GenerateResult]:
        import torch

        results: list[GenerateResult] = []
        for req in requests:
            inputs = self.processor.apply_chat_template(
                [req.messages] if self.modality == "vlm" else req.messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            if isinstance(inputs, dict):
                inputs = {
                    k: v.to(self.device) for k, v in inputs.items() if v is not None
                }
            else:
                inputs = inputs.to(self.device)

            with torch.amp.autocast(self.device.type, dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    do_sample=req.temperature > 0,
                    temperature=req.temperature if req.temperature > 0 else 1.0,
                    top_p=req.top_p,
                )

            prompt_len = inputs["input_ids"].shape[1]
            text = (
                self._tokenizer()
                .decode(output_ids[0, prompt_len:], skip_special_tokens=True)
                .strip()
            )
            results.append(GenerateResult(text=text))
        return results

    def logprobs(self, requests: list[LogprobRequest]) -> list[LogprobResult]:
        import torch

        results: list[LogprobResult] = []
        for req in requests:
            prompt_inputs = self.processor.apply_chat_template(
                [req.messages] if self.modality == "vlm" else req.messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            if isinstance(prompt_inputs, dict):
                prompt_inputs = {
                    k: v.to(self.device)
                    for k, v in prompt_inputs.items()
                    if v is not None
                }
            else:
                prompt_inputs = prompt_inputs.to(self.device)
            prompt_len = prompt_inputs["input_ids"].shape[1]

            scores: list[float] = []
            for cont in req.continuations:
                if self.modality == "vlm":
                    full_conv = req.messages + [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": cont.strip()}],
                        }
                    ]
                    full_inputs = self.processor.apply_chat_template(
                        [full_conv],
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                else:
                    full_conv = req.messages + [
                        {"role": "assistant", "content": cont.strip()}
                    ]
                    full_inputs = self.processor.apply_chat_template(
                        full_conv,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )

                if isinstance(full_inputs, dict):
                    full_inputs = {
                        k: v.to(self.device)
                        for k, v in full_inputs.items()
                        if v is not None
                    }
                else:
                    full_inputs = full_inputs.to(self.device)

                with torch.amp.autocast(self.device.type, dtype=torch.bfloat16):
                    logits = self.model(**full_inputs).logits

                log_probs = logits[0].log_softmax(dim=-1)
                input_ids = full_inputs["input_ids"][0]
                num_tokens = len(input_ids) - prompt_len
                total_logprob = sum(
                    log_probs[i - 1, input_ids[i].item()].item()
                    for i in range(prompt_len, len(input_ids))
                )
                scores.append(total_logprob / num_tokens if num_tokens > 0 else 0.0)

            results.append(LogprobResult(logprobs=scores))
        return results

    def close(self) -> None:
        # Backend doesn't own the model — caller does.
        pass


# === vLLM in-process backend (sidecar mode) ===


class VLLMInProcessBackend:
    """Wraps ``vllm.LLM`` in the same process. Used by the sidecar runner
    where the eval job owns its own GPU. Logprob scoring is not
    implemented — the runner falls back to ``HFBackend`` for those.
    """

    name = "vllm-in-process"

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        trust_remote_code: bool = True,
        **extra_llm_kwargs,
    ):
        from vllm import LLM

        kwargs: dict = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "seed": 1,
            "enforce_eager": False,
            "enable_chunked_prefill": True,
            # FULL CUDA graphs interact with prefix caching on long
            # multimodal contexts and cause inter-cycle output drift on
            # vLLM 0.19. PIECEWISE keeps speed without the drift.
            "compilation_config": {"cudagraph_mode": "PIECEWISE"},
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        kwargs.update(extra_llm_kwargs)

        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        try:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
        except Exception:
            self.processor = None

    def _format_prompt(self, req: GenerateRequest):
        if req.images:
            stripped: list[dict] = []
            for msg in req.messages:
                content = msg.get("content")
                if isinstance(content, list):
                    new_content = [
                        {"type": "image"} if item.get("type") == "image" else item
                        for item in content
                    ]
                    stripped.append({**msg, "content": new_content})
                else:
                    stripped.append(msg)

            apply = (
                self.processor.apply_chat_template
                if self.processor is not None
                else self.tokenizer.apply_chat_template
            )
            prompt_text = apply(stripped, tokenize=False, add_generation_prompt=True)
            out: dict = {
                "prompt": prompt_text,
                "multi_modal_data": {"image": req.images},
            }
            # vLLM 0.19's LFM2-VL multi-image preprocessor can crash on
            # empty spatial_shapes; force single-tile + no thumbnail.
            if len(req.images) > 1:
                out["mm_processor_kwargs"] = {
                    "do_image_splitting": False,
                    "min_tiles": 1,
                    "max_tiles": 1,
                    "use_thumbnail": False,
                }
            return out

        return self.tokenizer.apply_chat_template(
            req.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, requests: list[GenerateRequest]) -> list[GenerateResult]:
        from vllm import SamplingParams

        if not requests:
            return []

        prompts = [self._format_prompt(r) for r in requests]
        # All requests in one benchmark share sampling params; the first
        # request's settings drive the batch.
        first = requests[0]
        params = SamplingParams(
            max_tokens=first.max_new_tokens,
            temperature=first.temperature,
            top_p=first.top_p,
        )
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return [GenerateResult(text=o.outputs[0].text) for o in outputs]

    def logprobs(self, requests: list[LogprobRequest]) -> list[LogprobResult]:
        raise NotImplementedError(
            "VLLMInProcessBackend.logprobs not implemented; "
            "fall back to HFBackend for logprob benchmarks."
        )

    def close(self) -> None:
        # Best-effort release of vLLM's CUDA resources before process exit.
        try:
            del self.llm
        except Exception:
            pass


# === vLLM server backend (reserved mode) ===


class VLLMServerBackend:
    """HTTP client to the OpenAI-compatible vLLM server used by reserved
    eval mode. VLM requests embed images as base64 data URLs. Logprob
    scoring is not implemented.
    """

    name = "vllm-server"

    def __init__(
        self,
        base_url: str,
        model_id: str = "default",
        timeout: float = 600.0,
    ):
        import requests

        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout
        self.session = requests.Session()

    def _embed_images(self, messages: list[dict], images: list) -> list[dict]:
        # Convert PIL images to base64 data URLs and inject into the last
        # user message as image_url content blocks (OpenAI-compat format).
        import base64
        import copy
        import io

        out = copy.deepcopy(messages)
        target_idx = next(
            (i for i in range(len(out) - 1, -1, -1) if out[i].get("role") == "user"),
            len(out) - 1,
        )

        msg = out[target_idx]
        existing = msg.get("content", "")
        if isinstance(existing, str):
            existing = [{"type": "text", "text": existing}]

        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            existing.insert(
                0,
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            )

        msg["content"] = existing
        out[target_idx] = msg
        return out

    def generate(self, requests: list[GenerateRequest]) -> list[GenerateResult]:
        results: list[GenerateResult] = []
        for req in requests:
            messages = req.messages
            if req.images:
                messages = self._embed_images(messages, req.images)
            payload = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": req.max_new_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
            }
            resp = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
            results.append(GenerateResult(text=text))
        return results

    def logprobs(self, requests: list[LogprobRequest]) -> list[LogprobResult]:
        raise NotImplementedError(
            "VLLMServerBackend.logprobs not implemented; "
            "fall back to HFBackend for logprob benchmarks."
        )

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass


def build_llama_server_command(
    *,
    server_binary: str,
    model_path: str,
    host: str,
    port: int,
    n_gpu_layers: int,
    model_id: str,
    mmproj: str | None = None,
    server_args: list[str] | None = None,
) -> list[str]:
    """Build a llama-server command without invoking a shell."""
    command = [
        server_binary,
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--alias",
        model_id,
    ]
    if mmproj:
        command.extend(["--mmproj", mmproj])
    command.extend(server_args or [])
    return command


class LlamaCppServerBackend(VLLMServerBackend):
    """OpenAI-compatible llama.cpp client with optional server ownership.

    When ``base_url`` is omitted the backend launches ``llama-server`` and
    terminates it on close. Supplying ``base_url`` connects to a server managed
    by Slurm, a container, or the caller.
    """

    name = "llama-cpp-server"

    def __init__(
        self,
        model_path: str,
        *,
        base_url: str | None = None,
        model_id: str | None = None,
        server_binary: str = "llama-server",
        host: str = "127.0.0.1",
        port: int = 8080,
        n_gpu_layers: int = 999,
        mmproj: str | None = None,
        server_args: list[str] | None = None,
        startup_timeout: float = 300.0,
        request_timeout: float = 600.0,
        log_path: str | None = None,
    ):
        self.process: subprocess.Popen | None = None
        self._log_handle = None
        resolved_model_id = model_id or pathlib.Path(model_path).stem
        resolved_base_url = base_url or f"http://{host}:{port}"

        super().__init__(
            base_url=resolved_base_url,
            model_id=resolved_model_id,
            timeout=request_timeout,
        )
        if base_url:
            return

        model = pathlib.Path(model_path).expanduser()
        if not model.is_file():
            self.session.close()
            raise FileNotFoundError(f"GGUF model not found: {model}")
        if mmproj and not pathlib.Path(mmproj).expanduser().is_file():
            self.session.close()
            raise FileNotFoundError(f"GGUF multimodal projector not found: {mmproj}")

        binary = shutil.which(server_binary)
        if binary is None:
            candidate = pathlib.Path(server_binary).expanduser()
            if not candidate.is_file():
                self.session.close()
                raise FileNotFoundError(
                    f"llama-server not found: {server_binary}. Build llama.cpp "
                    "with -DGGML_HIP=ON or set backend.server_binary."
                )
            binary = str(candidate)

        if log_path:
            path = pathlib.Path(log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = path.open("a", encoding="utf-8")
            output = self._log_handle
        else:
            output = subprocess.DEVNULL

        command = build_llama_server_command(
            server_binary=binary,
            model_path=str(model),
            host=host,
            port=port,
            n_gpu_layers=n_gpu_layers,
            model_id=resolved_model_id,
            mmproj=mmproj,
            server_args=server_args,
        )
        self.process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            self._wait_until_ready(startup_timeout)
        except Exception:
            self.close()
            raise

    def _wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited during startup with code "
                    f"{self.process.returncode}"
                )
            try:
                response = self.session.get(
                    f"{self.base_url}/health",
                    timeout=min(5.0, timeout),
                )
                if response.ok:
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(0.25)
        detail = f": {last_error}" if last_error else ""
        raise TimeoutError(f"llama-server did not become healthy in {timeout}s{detail}")

    def close(self) -> None:
        super().close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
