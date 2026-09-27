"""Dense-linear BF16-to-FP8 per-token-group input quantization kind.

The vLLM DeepGEMM dense path quantizes each activation row in independent
``group_size`` chunks and writes UE8M0 FP32 scales in column-major layout before
QKV and output-projection GEMMs.  Group size and scale layout remain explicit
cache axes because both select kernel semantics and launch/storage behavior,
even though the first exact backend supports only the production pair below.
"""

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

KIND: str = "fp8_per_token_group_quant"


@dataclass(frozen=True)
class Fp8PerTokenGroupQuantArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Activation rows to quantize.")
    hidden_size: int = arg(unit="elements", doc="Elements in each activation row.")
    group_size: int = arg(unit="elements", doc="Elements sharing one quantization scale.")
    input_dtype: DType = arg(doc="Element type of the input activations.")
    scale_format: str = arg(doc="Encoding and layout of the output scales.")


DOC = KernelDoc(
    title="FP8 per-token-group quantization",
    summary="Quantize BF16 activation rows to FP8 E4M3 with a scale per 128-element group.",
    description=(
        "vLLM's blockwise FP8 path quantizes activations before dense "
        "projections and before the MoE gate/up GEMM. Each row is split into "
        "group_size-element groups, and each group gets one scale rounded up to"
        " a power of two (UE8M0), stored as FP32 in a column-major [token, "
        "group] layout that the FP8 GEMM reads."
    ),
    category="Quantization",
    formula=(
        "GB/s = (num_tokens · hidden_size · (BF16 bytes + FP8 bytes) + "
        "num_tokens · hidden_size / group_size · 4 scale bytes) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(f"{CUPTI_METHOD} Only per_token_group_quant_8bit_kernel launches are counted."),
    caveats=(
        "Only group_size = 128 with UE8M0 column-major scales is measured, on H100 and H200.",
        "GB/s counts logical BF16 input, FP8 output and FP32 scale bytes, not "
        "physical memory transactions.",
    ),
    # No separate PyTorch reference implementation exists for this kind.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H100", "NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.fp8_per_token_group_quant",
            function_name="profile_fp8_per_token_group_quant_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=Fp8PerTokenGroupQuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's per_token_group_fp8_quant CUDA op with UE8M0 scales in column-major layout."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu",
        ),
    )
)
