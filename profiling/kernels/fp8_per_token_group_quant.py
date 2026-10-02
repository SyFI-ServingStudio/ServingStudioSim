"""Dense-linear BF16-to-FP8 per-token-group input quantization kind.

The vLLM FP8 block-scaled paths quantize each activation row in independent
``group_size`` chunks and write UE8M0 scales before the GEMM/MoE that consumes
them.  Group size and scale layout remain explicit cache axes because both
select kernel semantics and launch/storage behavior.  The runner lists the
accepted ``scale_format`` values and pairs each with its verified GPUs.
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
        " a power of two (UE8M0). scale_format picks the layout the consuming "
        "kernel reads: ue8m0_column_major stores FP32 scales in a column-major "
        "[token, group] layout for Hopper DeepGEMM, ue8m0_row_major stores them"
        " row-major for the Blackwell FP8 block-scale MoE, and "
        "ue8m0_packed_int32 packs four one-byte scales per int32 for Blackwell "
        "DeepGEMM."
    ),
    category="Quantization",
    formula=(
        "G = hidden_size / group_size; ue8m0_packed_int32: G = ⌈hidden_size / group_size / 4⌉",
        "GB/s = (num_tokens · hidden_size · (BF16 bytes + FP8 bytes) + "
        "num_tokens · G · 4 scale bytes) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Only the quantization kernel is counted: "
        "per_token_group_quant_8bit_packed_register_kernel for "
        "ue8m0_packed_int32, per_token_group_quant_8bit_kernel otherwise."
    ),
    caveats=(
        "The runner accepts only group_size = 128, with ue8m0_column_major "
        "scales on H100 and H200 and ue8m0_row_major or ue8m0_packed_int32 "
        "scales on B200.",
        "GB/s counts logical BF16 input, FP8 output and scale bytes, not "
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
            gpus=frozenset({"NVIDIA H100", "NVIDIA H200", "NVIDIA B200"}),
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
                "vLLM's per_token_group_fp8_quant CUDA op with UE8M0 scales, or "
                "per_token_group_fp8_quant_packed for the packed int32 layout."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu",
        ),
    )
)


# MI300X elementwise byte-placeholder floor (GLM-5.3-Flash port, decision #37
# "mechanism B"): this kind carries a negligible predicted share of iteration
# time and has no MI300X-native backend yet, so instead of leaving it pinned to
# an NVIDIA-only backend -- which a real MI300X ``timing-predict`` rejects at
# ``BackendSupport.allows`` -- its MI300X cost is floored onto the *measured*
# ``elementwise`` ``torch_rocm`` byte-mover: the runner derives this kind's
# memory-bound byte footprint from its shape args and times that many bytes
# (``profiling.runners.elementwise.floor``). MI300X-gated and compute-agnostic,
# so every NVIDIA target -- B200 included -- stays byte-identical.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="elementwise_floor",
        supports=BackendSupport(compute=None, gpus=frozenset({"MI300X"})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.floor",
            function_name="profile_fp8_per_token_group_quant_floor",
        ),
        table_name=KIND,
        args_schema=Fp8PerTokenGroupQuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_rocm_env",
        doc=BackendDoc(
            summary=(
                "Elementwise byte-placeholder floor: times the measured MI300X "
                "``elementwise`` ``torch_rocm`` byte-mover for this kind's "
                "shape-derived memory-bound footprint. A negligible-share floor "
                "for the GLM-5.3-Flash MI300X port, not a native kernel."
            ),
        ),
    )
)
