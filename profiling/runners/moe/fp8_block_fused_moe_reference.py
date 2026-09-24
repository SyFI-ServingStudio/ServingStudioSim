"""Independent Torch reference for the DeepSeek-FP8 block-scale fused MoE.

This is the math the FlashInfer TRT-LLM ``trtllm_fp8_block_scale_moe`` call
computes for vLLM's GLM-5.3-Flash layer, written from the model definition
rather than from the kernel: DeepSeekV3 ``noaux_tc`` routing (sigmoid scores,
bias only for selection, renormalized top-k, routed scale applied to the
weights), block-dequantized per-expert gate/up and down projections, the
SwiGLU clamp (``gate <= limit``, ``-limit <= up <= limit``), and the FP8
re-quantization of the intermediate activation that the DeepSeek-FP8 recipe
performs between the two GEMMs. Only experts in
``[local_offset, local_offset + num_local)`` contribute, as on one EP rank.

Weights here are in vLLM's checkpoint order ``w13 = [gate; up]``; the runner
converts to FlashInfer's ``[up; gate]`` BlockMajorK layout separately, so a
layout mistake in that conversion shows up as a mismatch.
"""

from __future__ import annotations

from typing import Any

BLOCK = 128
FP8_E4M3_MAX = 448.0


def deepseek_v3_routing(
    torch: Any,
    logits: Any,
    bias: Any,
    *,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[Any, Any]:
    """Return ``(topk_ids, topk_weights)`` for ungrouped (n_group == 1) routing."""

    scores = torch.sigmoid(logits.float())
    choice = scores + bias.float()
    ids = torch.topk(choice, top_k, dim=-1).indices
    weights = torch.gather(scores, 1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids, weights * routed_scaling_factor


def dequantize_blocks(torch: Any, values: Any, scales: Any) -> Any:
    """Dequantize ``[..., M, K]`` FP8 with ``[..., M/128, K/128]`` FP32 scales."""

    expanded = scales.float().repeat_interleave(BLOCK, dim=-2).repeat_interleave(BLOCK, dim=-1)
    return values.float() * expanded


def dequantize_token_groups(torch: Any, values: Any, scales: Any) -> Any:
    """Dequantize ``[T, K]`` FP8 with per-token-group ``[T, K/128]`` scales."""

    return values.float() * scales.float().repeat_interleave(BLOCK, dim=-1)


def quantize_token_groups(torch: Any, values: Any) -> Any:
    """Round-trip ``[T, K]`` through FP8 E4M3 with per-token-group-128 scales."""

    rows, cols = values.shape
    grouped = values.float().reshape(rows, cols // BLOCK, BLOCK)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / FP8_E4M3_MAX
    quantized = (grouped / scale).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape(rows, cols)


def fp8_block_fused_moe_reference(
    torch: Any,
    *,
    hidden: Any,
    hidden_scale: Any,
    w13: Any,
    w13_scale: Any,
    w2: Any,
    w2_scale: Any,
    routing_logits: Any,
    routing_bias: Any,
    top_k: int,
    routed_scaling_factor: float,
    clamp_limit: float | None,
    local_offset: int,
    num_local: int,
) -> Any:
    """Return the FP32 ``[T, H]`` output of this rank's routed experts."""

    x = dequantize_token_groups(torch, hidden, hidden_scale)
    ids, weights = deepseek_v3_routing(
        torch,
        routing_logits,
        routing_bias,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
    )
    intermediate = w13.shape[1] // 2
    output = torch.zeros_like(x)
    for local in range(num_local):
        rows, slots = torch.nonzero(ids == local_offset + local, as_tuple=True)
        if rows.numel() == 0:
            continue
        gate_up = x[rows] @ dequantize_blocks(torch, w13[local], w13_scale[local]).t()
        gate, up = gate_up[:, :intermediate], gate_up[:, intermediate:]
        if clamp_limit is not None:
            gate = torch.clamp(gate, max=clamp_limit)
            up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
        act = quantize_token_groups(torch, torch.nn.functional.silu(gate) * up)
        down = act @ dequantize_blocks(torch, w2[local], w2_scale[local]).t()
        output.index_add_(0, rows, down * weights[rows, slots, None])
    return output


__all__ = [
    "deepseek_v3_routing",
    "dequantize_blocks",
    "dequantize_token_groups",
    "fp8_block_fused_moe_reference",
    "quantize_token_groups",
]
