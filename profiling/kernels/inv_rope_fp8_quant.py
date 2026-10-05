"""Fused inverse RoPE and grouped FP8 quantization of attention outputs."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "inv_rope_fp8_quant"


@dataclass(frozen=True)
class InvRopeFp8QuantArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Attention output rows transformed together.")


DOC = KernelDoc(
    title="Inverse RoPE and FP8 quantization",
    summary="Undo RoPE on attention outputs and quantize them in groups for the output projection.",
    description=(
        "When keys and values share one latent whose trailing elements carry "
        "RoPE, the attention output carries the rotation too and has to be "
        "rotated back before a grouped output projection. This Triton call "
        "undoes RoPE on the trailing 64 of each head's 512 elements and "
        "quantizes the result to FP8 in 128-element groups, with 64 heads "
        "arranged as eight groups of eight for the projection. The measurement"
        " uses position 0 and an identity RoPE table, so the rotation leaves "
        "the values unchanged."
    ),
    category="Quantization",
    formula=(
        "x = inverse_RoPE(attention output)",
        "FP8 group = clamp(x / scale, −448, 448), with 128 elements per scale",
        "GB/s = (2 · num_tokens · 64 · 512 + 8 · num_tokens + "
        "4 · num_tokens · 64 + num_tokens · 64 · 512 + "
        "4 · num_tokens · 8 · (8 · 512 / 128)) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(f"{CUPTI_METHOD} Three warm-up calls run first; every launch of the call is counted."),
    caveats=(
        "The identity RoPE table keeps the rotation arithmetic but not real angles.",
        "GB/s counts one RoPE-table row per token, although all tokens share one row here.",
    ),
    # The runner checks quantization at identity RoPE, not a separate full reference.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.inv_rope_fp8_quant_vllm_triton"),
            function_name="profile_inv_rope_fp8_quant_vllm_triton",
        ),
        table_name=KIND,
        args_schema=InvRopeFp8QuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # FP8 e4m3 conversion needs SM89+.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 9),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's fused_inv_rope_fp8_quant Triton call combines inverse "
                "RoPE with grouped FP8 output."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py",
        ),
    )
)

__all__ = ["KIND"]
