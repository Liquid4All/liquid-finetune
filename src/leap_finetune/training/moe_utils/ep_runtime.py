import logging
import os
import re
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed._functional_collectives import (
    all_to_all_single as _functional_a2a,
    all_to_all_single_autograd,
)

from leap_finetune.quantization.qat.experts import (
    prepare_qat_experts,
)
from leap_finetune.training.moe_utils.memory_trace import write_memory_trace_event
from leap_finetune.training.moe_utils.losses import (
    InjectAuxLoss,
    store_moe_metrics,
    switch_load_balancing_loss,
    z_loss,
)
from leap_finetune.training.moe_utils.ops import (
    grouped_mm,
    permute_tokens,
    route_tokens_to_experts,
    tokens_per_expert_from_routing,
    unpermute_tokens,
)

logger = logging.getLogger(__name__)

_MEM_DEBUG_ENABLED = os.getenv("LEAP_MOE_MEM_DEBUG", "").lower() in {
    "1",
    "true",
    "yes",
}
_MEM_SUMMARY_ENABLED = os.getenv("LEAP_MOE_MEM_SUMMARY", "").lower() in {
    "1",
    "true",
    "yes",
}
_STAGE_TRACE_ENABLED = os.getenv("LEAP_STAGE_TRACE", "").lower() in {
    "1",
    "true",
    "yes",
}
_A2A_TRACE_ENABLED = os.getenv("LEAP_EP_A2A_TRACE", "").lower() in {
    "1",
    "true",
    "yes",
}
_mem_debug_tags_seen: set[tuple[int, str]] = set()


def _log_cuda_memory_once(tag: str, *, summary: bool = False) -> None:
    """Log one-shot CUDA allocator stats for the current rank."""
    if not _MEM_DEBUG_ENABLED or not torch.cuda.is_available():
        return

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    key = (rank, tag)
    if key in _mem_debug_tags_seen:
        return
    _mem_debug_tags_seen.add(key)

    device = torch.cuda.current_device()
    alloc_gb = torch.cuda.memory_allocated(device) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
    max_alloc_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
    logger.info(
        "[mem][rank=%s][%s] alloc=%.2fGB reserved=%.2fGB peak_alloc=%.2fGB "
        "peak_reserved=%.2fGB",
        rank,
        tag,
        alloc_gb,
        reserved_gb,
        max_alloc_gb,
        max_reserved_gb,
    )

    if summary and _MEM_SUMMARY_ENABLED:
        logger.info(
            "[mem][rank=%s][%s] summary:\n%s",
            rank,
            tag,
            torch.cuda.memory_summary(device=device),
        )


def log_cuda_memory(tag: str, *, summary: bool = False) -> None:
    """Public helper for targeted CUDA memory instrumentation."""
    _log_cuda_memory_once(tag, summary=summary)


def trace_stage(tag: str, *, extra: dict | None = None) -> None:
    """Emit detailed stage markers for one-off EP diagnosis runs."""
    if not _STAGE_TRACE_ENABLED:
        return
    write_memory_trace_event(tag, extra=extra)


class EPTokenDispatcher:
    """AlltoAll dispatcher for EP token exchange.

    Flow: 1) local expert permute, 2) EP all-to-all, 3) local expert-major
    sort for grouped_mm, then the exact inverse after expert compute.
    """

    def __init__(
        self,
        n_local_experts: int,
        local_expert_indices: list[int],
        n_experts: int,
        top_k: int,
        ep_size: int,
        ep_group: dist.ProcessGroup,
    ):
        self.n_local_experts = n_local_experts
        self.local_expert_indices = local_expert_indices
        self.n_experts = n_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_group = ep_group

        self.input_splits: list[int] = []
        self.output_splits: list[int] = []
        self.tokens_per_local_expert: torch.Tensor | None = None
        self.tokens_per_local_expert_by_sender: torch.Tensor | None = None
        self._reverse_map: torch.Tensor | None = None
        self._n_tokens: int = 0
        self._hidden_size: int = 0

    def _trace_a2a(
        self,
        phase: str,
        send_splits: list[int],
        recv_splits: list[int],
        buffer: torch.Tensor,
    ) -> None:
        if not _A2A_TRACE_ENABLED:
            return

        rank = dist.get_rank()
        ep_rank = dist.get_rank(self.ep_group)
        logger.info(
            "[ep-a2a][%s][rank=%s ep_rank=%s] send_sum=%s recv_sum=%s "
            "send=%s recv=%s buffer_shape=%s",
            phase,
            rank,
            ep_rank,
            sum(send_splits),
            sum(recv_splits),
            send_splits,
            recv_splits,
            tuple(buffer.shape),
        )

    def preprocess(self, tokens_per_expert: torch.Tensor) -> None:
        """Exchange per-expert token counts across EP ranks."""
        tpe_by_rank = tokens_per_expert.reshape(self.ep_size, self.n_local_experts)

        with torch.no_grad():
            input_splits_t = tpe_by_rank.sum(dim=1).contiguous()
            output_splits_t = _functional_a2a(
                input_splits_t, None, None, group=self.ep_group
            )
            output_splits_t = torch.ops._c10d_functional.wait_tensor(output_splits_t)
            self.input_splits = input_splits_t.tolist()
            self.output_splits = output_splits_t.tolist()

            expert_split = [self.n_local_experts] * self.ep_size
            tpe_flat = tpe_by_rank.flatten().contiguous()
            received_flat = _functional_a2a(
                tpe_flat, expert_split, expert_split, group=self.ep_group
            )
            received_flat = torch.ops._c10d_functional.wait_tensor(received_flat)

        self.tokens_per_local_expert_by_sender = received_flat.reshape(
            self.ep_size, self.n_local_experts
        )
        self.tokens_per_local_expert = self.tokens_per_local_expert_by_sender.sum(dim=0)

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sort tokens by expert, exchange tokens/weights, then sort local experts."""
        self._n_tokens = hidden_states.shape[0]
        self._hidden_size = hidden_states.shape[1]

        # === 1. Local permute: sort by global expert index ===
        permuted, weights, self._reverse_map, _ = permute_tokens(
            hidden_states, selected_experts, routing_weights, self.n_experts
        )

        # === 2. EP A2A: exchange both tokens and routing weights ===
        self._trace_a2a(
            "forward",
            self.input_splits,
            self.output_splits,
            permuted,
        )
        global_tokens = all_to_all_single_autograd(
            permuted,
            self.output_splits,
            self.input_splits,
            self.ep_group,
        )
        global_weights = all_to_all_single_autograd(
            weights,
            self.output_splits,
            self.input_splits,
            self.ep_group,
        )

        # === 3. Local sort: sender-major -> expert-major for grouped_mm ===
        if self.n_local_experts > 1:
            assert self.tokens_per_local_expert_by_sender is not None
            global_tokens = sender_major_to_expert_major(
                global_tokens,
                self.tokens_per_local_expert_by_sender,
            )
            global_weights = sender_major_to_expert_major(
                global_weights,
                self.tokens_per_local_expert_by_sender,
            )

        return global_tokens, global_weights

    def token_unpermutation(self, expert_output: torch.Tensor) -> torch.Tensor:
        """Unsort local experts, reverse AlltoAll, then restore local token order."""
        # === 1. Local unsort: expert-major -> sender-major for reverse A2A ===
        if self.n_local_experts > 1:
            assert self.tokens_per_local_expert_by_sender is not None
            expert_output = expert_major_to_sender_major(
                expert_output,
                self.tokens_per_local_expert_by_sender,
            )

        # === 2. Reverse EP A2A: one collective with inverted split sizes ===
        self._trace_a2a(
            "reverse",
            self.output_splits,
            self.input_splits,
            expert_output,
        )
        local_tokens = all_to_all_single_autograd(
            expert_output,
            self.input_splits,
            self.output_splits,
            self.ep_group,
        )

        # === 3. Local unpermute: scatter expert outputs back to input tokens ===
        assert self._reverse_map is not None
        return unpermute_tokens(
            local_tokens,
            self._reverse_map,
            None,
            self._n_tokens,
            self._hidden_size,
        )


# === Local Expert Compute ===


def _split_sender_major(
    tensor: torch.Tensor,
    tokens_per_local_expert_by_sender: torch.Tensor,
) -> list[list[torch.Tensor]]:
    chunks: list[list[torch.Tensor]] = []
    start = 0
    for sender_counts in tokens_per_local_expert_by_sender.tolist():
        sender_chunks = []
        for count in sender_counts:
            end = start + int(count)
            sender_chunks.append(tensor[start:end])
            start = end
        chunks.append(sender_chunks)
    return chunks


def sender_major_to_expert_major(
    tensor: torch.Tensor,
    tokens_per_local_expert_by_sender: torch.Tensor,
) -> torch.Tensor:
    """Reorder [sender][expert] chunks into [expert][sender] chunks."""
    chunks = _split_sender_major(tensor, tokens_per_local_expert_by_sender)
    ordered = [
        chunks[sender][expert]
        for expert in range(tokens_per_local_expert_by_sender.shape[1])
        for sender in range(tokens_per_local_expert_by_sender.shape[0])
    ]
    return torch.cat(ordered, dim=0) if ordered else tensor


def expert_major_to_sender_major(
    tensor: torch.Tensor,
    tokens_per_local_expert_by_sender: torch.Tensor,
) -> torch.Tensor:
    """Reorder [expert][sender] chunks back into [sender][expert] chunks."""
    counts = tokens_per_local_expert_by_sender
    expert_chunks: list[list[torch.Tensor]] = []
    start = 0
    for expert in range(counts.shape[1]):
        chunks_for_expert = []
        for sender in range(counts.shape[0]):
            end = start + int(counts[sender, expert].item())
            chunks_for_expert.append(tensor[start:end])
            start = end
        expert_chunks.append(chunks_for_expert)

    ordered = [
        expert_chunks[expert][sender]
        for sender in range(counts.shape[0])
        for expert in range(counts.shape[1])
    ]
    return torch.cat(ordered, dim=0) if ordered else tensor


def compute_local_experts(
    experts: nn.Module,
    tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    routing_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run local experts over expert-major tokens with upstream grouped_mm."""
    # ==== Local expert MLP ====
    # Tokens are already sorted by this rank's local experts. `offsets` marks the
    # expert boundaries expected by grouped_mm: gate/up projection, activation,
    # routing weight, then down projection.
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    qat = prepare_qat_experts(experts, tokens)
    gate_up_out = grouped_mm(qat.tokens, qat.gate_up_weight.transpose(-2, -1), offsets)
    gate, up = gate_up_out.chunk(2, dim=-1)
    activated = experts.act_fn(gate) * up
    if routing_weights is not None:
        activated = activated * routing_weights.to(activated.dtype)
    activated = qat.prepare_down_input(activated)
    return grouped_mm(activated, qat.down_weight.transpose(-2, -1), offsets)


def _inject_ep_aux_losses(
    block: nn.Module,
    router_logits: torch.Tensor,
    moe_config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route tokens and inject aux losses into the live routing tensors."""
    n_experts = router_logits.shape[-1]

    if moe_config is not None and moe_config.z_loss_coef > 0:
        zl = z_loss(router_logits, moe_config.z_loss_coef)
        router_logits = InjectAuxLoss.apply(router_logits, zl)
        block._moe_z_loss = zl.detach()

    selected_experts, routing_weights, router_probs = route_tokens_to_experts(
        block, router_logits
    )
    top_k = selected_experts.shape[-1]

    if moe_config is not None and moe_config.aux_loss_coef > 0:
        aux = switch_load_balancing_loss(
            router_probs,
            selected_experts,
            n_experts,
            top_k,
            moe_config.aux_loss_coef,
        )
        routing_weights = InjectAuxLoss.apply(routing_weights, aux)
        block._moe_aux_loss = aux.detach()

    return selected_experts, routing_weights, router_probs


def create_ep_mesh(
    ep_size: int, num_experts: int
) -> tuple[dict, "torch.distributed.device_mesh.DeviceMesh"]:
    """Create EP process groups via a 2D DeviceMesh [dp, ep]."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size % ep_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ep_size ({ep_size}). "
            f"Got {world_size} GPUs with ep_size={ep_size}."
        )
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size}). "
            "Each EP rank must hold an equal number of experts."
        )

    dp_size = world_size // ep_size
    n_local_experts = num_experts // ep_size
    dp_rank = rank // ep_size

    mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))

    ep_group = mesh["ep"].get_group()
    ep_rank = dist.get_rank(ep_group)

    local_expert_indices = list(
        range(ep_rank * n_local_experts, (ep_rank + 1) * n_local_experts)
    )

    logger.info(
        f"Parallelism: {world_size} GPUs = {dp_size} DP x {ep_size} EP "
        f"({n_local_experts} of {num_experts} experts per rank)"
    )
    logger.debug(
        f"Rank {rank}: dp={rank // ep_size} ep={ep_rank}/{ep_size} "
        f"local experts {local_expert_indices}"
    )

    ep_config = {
        "ep_group": ep_group,
        "dp_rank": dp_rank,
        "ep_rank": ep_rank,
        "ep_size": ep_size,
        "n_local_experts": n_local_experts,
        "local_expert_indices": local_expert_indices,
        "num_experts": num_experts,
    }

    return ep_config, mesh


def create_dp_mesh() -> "torch.distributed.device_mesh.DeviceMesh":
    """Create a 1D DP mesh spanning the full world size."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    logger.info(
        "Parallelism: %s GPUs = %s DP x 1 EP (rank=%s)",
        world_size,
        world_size,
        rank,
    )
    return mesh


def shard_experts(model: nn.Module, ep_config: dict) -> None:
    """Remove non-local expert weights to free memory."""
    local_indices = ep_config["local_expert_indices"]
    n_local = ep_config["n_local_experts"]
    sharded = 0

    for module in model.modules():
        class_name = type(module).__name__
        if "Expert" not in class_name:
            continue

        if hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"):
            idx = torch.tensor(local_indices)
            module.gate_up_proj = nn.Parameter(
                module.gate_up_proj.data[idx].contiguous()
            )
            module.down_proj = nn.Parameter(module.down_proj.data[idx].contiguous())
            module.num_experts = n_local
            module._local_expert_offset = local_indices[0]
            module._n_local_experts = n_local
            sharded += 1

        elif hasattr(module, "__getitem__"):
            local_experts = nn.ModuleList([module[i] for i in local_indices])
            module.clear()
            module.extend(local_experts)
            module._local_expert_offset = local_indices[0]
            module._n_local_experts = n_local
            sharded += 1

    logger.info(
        f"Sharded {sharded} expert modules: keeping {n_local} "
        f"of {ep_config['num_experts']} experts"
    )


def apply_fsdp2(
    model: nn.Module,
    dp_mesh: "torch.distributed.device_mesh.DeviceMesh",
    reshard_after_forward: bool = True,
    cpu_offload: bool = False,
    activation_checkpointing: bool = True,
) -> nn.Module:
    """Wrap model with FSDP2 on a DP mesh."""

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    offload_policy = CPUOffloadPolicy() if cpu_offload else OffloadPolicy()

    fsdp_kwargs = {
        "mesh": dp_mesh,
        "mp_policy": mp_policy,
        "reshard_after_forward": reshard_after_forward,
        "offload_policy": offload_policy,
    }

    n_feed_forward = 0
    n_operator = 0
    for name, module in reversed(list(model.named_modules())):
        if re.search(r"layers\.\d+\.feed_forward$", name):
            fully_shard(module, **fsdp_kwargs)
            n_feed_forward += 1
        elif re.search(r"layers\.\d+\.(self_attn|conv)$", name):
            fully_shard(module, **fsdp_kwargs)
            n_operator += 1

    fully_shard(model, **fsdp_kwargs)

    logger.info(
        "FSDP2 wrapped %s feed-forward blocks and %s operator blocks on DP mesh "
        "(size=%s)",
        n_feed_forward,
        n_operator,
        dp_mesh.size(),
    )

    if activation_checkpointing:
        ckpt_fn = partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        n_ckpt = sum(
            1
            for module in model.modules()
            if type(module).__name__ == "Lfm2MoeDecoderLayer"
        )
        if n_ckpt > 0:
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=ckpt_fn,
                check_fn=lambda module: type(module).__name__ == "Lfm2MoeDecoderLayer",
            )
            logger.info("Applied activation checkpointing to %s decoder layers", n_ckpt)
    else:
        logger.info("Skipped activation checkpointing for this FSDP2 wrap")

    return model


def patch_moe_block_for_ep(
    block: nn.Module,
    dispatcher: EPTokenDispatcher,
    moe_config=None,
) -> None:
    """Monkey-patch an Lfm2MoeSparseMoeBlock for EP-aware forward."""
    experts = block.experts

    def ep_moe_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        # ==== EP MoE forward ====
        # 1. Route tokens with the full router.
        # 2. Exchange token counts, tokens, and routing weights across EP ranks.
        # 3. Run only this rank's local experts, then reverse the exchange.
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)
        layer_id = getattr(block, "layer_id", None)
        stage_extra = {
            "layer_id": layer_id,
            "batch_size": batch_size,
            "seq_len": sequence_length,
            "hidden_size": hidden_dim,
        }
        trace_stage("ep_moe_start", extra=stage_extra)

        router_logits = block.gate(x)
        n_experts = router_logits.shape[-1]
        trace_stage(
            "ep_after_router",
            extra={**stage_extra, "num_experts": n_experts},
        )

        selected_experts, routing_weights, _ = _inject_ep_aux_losses(
            block, router_logits, moe_config
        )
        trace_stage(
            "ep_after_route_tokens",
            extra={
                **stage_extra,
                "num_experts": n_experts,
                "top_k": int(selected_experts.shape[-1]),
            },
        )

        tokens_per_expert = tokens_per_expert_from_routing(selected_experts, n_experts)
        dispatcher.preprocess(tokens_per_expert)
        trace_stage(
            "ep_after_preprocess",
            extra={
                **stage_extra,
                "num_experts": n_experts,
                "tokens_per_local_expert_sum": int(
                    dispatcher.tokens_per_local_expert.sum().item()
                ),
                "input_splits_sum": int(sum(dispatcher.input_splits)),
                "output_splits_sum": int(sum(dispatcher.output_splits)),
            },
        )
        global_tokens, global_weights = dispatcher.token_permutation(
            x,
            selected_experts,
            routing_weights,
        )
        trace_stage(
            "ep_after_token_permutation",
            extra={
                **stage_extra,
                "received_tokens": int(global_tokens.shape[0]),
            },
        )

        expert_output = compute_local_experts(
            experts,
            global_tokens,
            dispatcher.tokens_per_local_expert,
            global_weights,
        )
        trace_stage(
            "ep_after_local_experts",
            extra={
                **stage_extra,
                "received_tokens": int(global_tokens.shape[0]),
            },
        )

        output = dispatcher.token_unpermutation(expert_output).view(
            batch_size, sequence_length, hidden_dim
        )
        trace_stage("ep_after_token_unpermutation", extra=stage_extra)

        if moe_config is not None:
            store_moe_metrics(block, selected_experts, router_logits, n_experts)

        return output

    block.forward = ep_moe_forward


def apply_ep_to_model(model: nn.Module, ep_config: dict, moe_config=None) -> None:
    """Apply EP dispatching to all MoE blocks in the model."""
    patched = 0

    for module in model.modules():
        if type(module).__name__ != "Lfm2MoeSparseMoeBlock":
            continue

        top_k = getattr(module, "top_k", 2)

        dispatcher = EPTokenDispatcher(
            n_local_experts=ep_config["n_local_experts"],
            local_expert_indices=ep_config["local_expert_indices"],
            n_experts=ep_config["num_experts"],
            top_k=top_k,
            ep_size=ep_config["ep_size"],
            ep_group=ep_config["ep_group"],
        )

        patch_moe_block_for_ep(module, dispatcher, moe_config=moe_config)
        patched += 1

    logger.info("Applied EP dispatch to %s MoE blocks", patched)
