"""Static per-tensor FP8 quantization of an activation, optionally output-gated first."""

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

KIND = "static_fp8_quant"


@dataclass(frozen=True)
class StaticFp8QuantArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows quantized together.")
    hidden_dim: int = arg(unit="elements", doc="Width of each row.")
    gated: bool = arg(
        doc=(
            "Whether each row is first multiplied by the sigmoid of a gate row of the "
            "same width, as an attention output gate does."
        )
    )
    dtype: DType = arg(doc="Element type of the input (and of the gate).")


DOC = KernelDoc(
    title="Static FP8 quantization",
    summary="Scale an activation by a fixed per-tensor scale and round it to FP8 E4M3.",
    description=(
        "A linear layer with static per-tensor FP8 activation scales quantizes its "
        "input before the GEMM. Under torch.compile vLLM's native QuantFP8 becomes "
        "one Inductor pointwise kernel, and an attention output gate in front of "
        "the projection fuses into it. Inputs are seeded random bf16 rows; the "
        "scale is a scalar fp32 tensor read on the device."
    ),
    category="Quantization",
    formula=(
        "y = clamp(x / scale, ±448) as E4M3, or clamp(x · σ(g) / scale, ±448) when gated",
        "GB/s = (2 + 2·gated + 1) · num_tokens · hidden_dim / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} vllm_inductor: the block is compiled once with a symbolic "
        "token count, and its first call, at 8192 tokens or at the row's token count "
        "when larger, picks the launch config the row then uses, as vLLM's profile run "
        "at max_num_batched_tokens does. Every launch of the compiled block is counted. "
        "tensorfold_cuda counts its one quant8_kernel launch."
    ),
    caveats=(
        "GB/s counts the input, the gate and the output once each.",
        "tensorfold_cuda multiplies by the fp32 reciprocal of the scale and writes each "
        "16-byte group in its FP8 GEMM's fragment order (a permutation within the "
        "group); it has no gated form, since TensorFold applies the attention gate "
        "in a separate launch.",
    ),
    reference="profiling.runners.elementwise.static_fp8_quant_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_inductor",
        # A portable Triton pointwise kernel; Triton lowers FP8 E4M3 only on SM89+.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 9),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.static_fp8_quant_vllm_inductor",
            function_name="profile_static_fp8_quant_vllm_inductor",
        ),
        table_name=KIND,
        args_schema=StaticFp8QuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's QuantFP8.forward_native (static, per-tensor) compiled by "
                "Inductor into one Triton kernel."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/input_quant_fp8.py",
        ),
    )
)

# TensorFold perf-study 6469730a, tensorfold/cuda/nvfp4/checkpoint.py
# `quant8(x, act)`: one quant8_kernel launch (act.cu), e4m3(x · (1/act))
# saturating, under the projection's static per-tensor input scale. It lives in
# the extension that also builds the SM12x FP4 GEMMs, and feeds the
# single_gemm tensorfold_* rows, so it declares their SM12x family.
TENSORFOLD_CUDA = "tensorfold_cuda"

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend=TENSORFOLD_CUDA,
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_120f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.tensorfold_quant",
            function_name="profile_static_fp8_quant_tensorfold_cuda",
        ),
        table_name=KIND,
        args_schema=StaticFp8QuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's checkpoint.quant8: quant8_kernel, e4m3 under the static "
                "per-tensor input scale in its FP8 GEMM's fragment order, ungated. SM12x."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)

__all__ = ["KIND", "StaticFp8QuantArgs"]
