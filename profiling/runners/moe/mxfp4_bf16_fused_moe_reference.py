"""Torch reference for the MXFP4-weight / BF16-activation monolithic MoE.

Independent of FlashInfer. The call routes inside the kernel, so the reference
routes too, with the FP8 block-scale backend's ungrouped DeepSeek-V3 reference
(sigmoid of the logits plus the correction bias picks the top-k; the weights
are the unbiased selected sigmoids, renormalized, times the routed scale;
FlashInfer's 1e-20 denominator floor only matters when every selected sigmoid
underflows, which forced routing never produces). The BF16 hidden
states reach GEMM1 unquantized. The experts are the shared MXFP4 core of
``mxfp4_fused_moe_reference`` with plain (unclamped) SwiGLU.

``bf16_boundaries`` stores the route weights and the SwiGLU output in BF16
before they are read again, as FlashInfer's routing and GEMM1 epilogue do
(``trtllm_fused_moe_routing_runner.cu``: BF16 expert weights;
``trtllm_fused_moe_kernel_launcher.cu``: BF16 GEMM1 output buffer). Accumulation
order, approximate SiLU and the GEMM2 output buffer are not reproduced.
"""

from __future__ import annotations

from typing import Any

from profiling.runners.moe.fp8_block_fused_moe_reference import deepseek_v3_routing
from profiling.runners.moe.mxfp4_fused_moe_reference import mxfp4_routed_experts


def mxfp4_bf16_fused_moe_reference(
    torch: Any,
    *,
    hidden: Any,
    logits: Any,
    bias: Any,
    w13: Any,
    w13_scale: Any,
    w2: Any,
    w2_scale: Any,
    top_k: int,
    routed_scale: float,
    local_offset: int,
    num_local: int,
    bf16_boundaries: bool,
) -> Any:
    """FP32 ``[T, H]``: this EP rank's finalized share of the routed MoE."""

    ids, weights = deepseek_v3_routing(
        torch, logits, bias, top_k=top_k, routed_scaling_factor=routed_scale
    )

    def to_bf16(value: Any) -> Any:
        return value.to(torch.bfloat16).float()

    return mxfp4_routed_experts(
        torch,
        x=hidden.float(),
        w13=w13,
        w13_scale=w13_scale,
        w2=w2,
        w2_scale=w2_scale,
        topk_ids=ids,
        topk_weights=to_bf16(weights) if bf16_boundaries else weights,
        local_offset=local_offset,
        num_local=num_local,
        clamp_limit=None,
        round_intermediate=to_bf16 if bf16_boundaries else None,
    )


__all__ = ["mxfp4_bf16_fused_moe_reference"]
