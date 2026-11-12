"""Residual-stream sum + RMSNorm with a selectable output set, as Inductor fuses it."""

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

KIND = "multi_add_rms_norm"


@dataclass(frozen=True)
class MultiAddRmsNormArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows normalized together.")
    hidden_dim: int = arg(unit="elements", doc="Width of each row.")
    num_addends: int = arg(
        unit="tensors",
        doc=(
            "Row-aligned tensors summed before the norm: 1 is a plain RMSNorm, 2 a "
            "residual add, 3 a residual add that also recomputes the previous one."
        ),
    )
    store_sum: bool = arg(doc="Whether the sum is written back as the new residual.")
    store_normed: bool = arg(doc="Whether the normalized rows are written in dtype.")
    store_fp8: bool = arg(
        doc="Whether the normalized rows are written as static per-tensor FP8 E4M3."
    )
    dtype: DType = arg(doc="Element type of the addends, the weight and the stored rows.")


DOC = KernelDoc(
    title="Summed RMSNorm with fused outputs",
    summary=(
        "Sum one to three residual-stream tensors, RMS-normalize, and write any of "
        "the sum, the normalized rows and their FP8 quantization in one kernel."
    ),
    description=(
        "Under torch.compile vLLM's native fused_add_rms_norm becomes part of one "
        "Inductor reduction kernel together with its neighbours. Inductor does not "
        "store a residual sum that only feeds the next layer's norm: that kernel "
        "reads the previous addends again and re-adds them. A static FP8 input "
        "quantization of the following linear layer fuses in as a third output "
        "plane. The output set is part of the kernel: each plane is a full write."
    ),
    category="Normalization",
    subcategory="RMSNorm",
    formula=(
        "s = x₁ + … + x_k, accumulated in fp32",
        "n = s / √(mean(s²) + ε) · weight",
        "outputs: s (store_sum), n (store_normed), clamp(n / scale, ±448) as E4M3 (store_fp8)",
        "GB/s = (k·m·h·b + h·b + m·h·(b·store_sum + b·store_normed + store_fp8)) / time,"
        " with m = num_tokens, h = hidden_dim, b = bytes per element",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} The block is compiled once with a symbolic token count, "
        "and its first call, at 8192 tokens or at the row's token count when larger, "
        "picks the launch config the row then uses, as vLLM's profile run at "
        "max_num_batched_tokens does. Every launch of the compiled block is counted."
    ),
    caveats=(
        "GB/s counts each input once. The kernel reads the addends twice, once for "
        "the mean square and once for the outputs; the second read mostly hits L2.",
        "The weight is a plain multiplier; a (1 + w) weight adds one addition per "
        "element in registers and leaves the traffic unchanged.",
    ),
    reference="profiling.runners.norm.multi_add_rms_norm_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_inductor",
        # Inductor emits a portable Triton reduction; Triton lowers bf16 only on SM80+
        # (an FP8 output plane needs SM89, checked per spec by the runner).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.norm.multi_add_rms_norm_vllm_inductor",
            function_name="profile_multi_add_rms_norm_vllm_inductor",
        ),
        table_name=KIND,
        args_schema=MultiAddRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's native fused_add_rms_norm and QuantFP8 compiled by Inductor "
                "into one Triton reduction kernel."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/ir/ops/layernorm.py",
        ),
    )
)

__all__ = ["KIND", "MultiAddRmsNormArgs"]
