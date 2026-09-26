"""BF16-to-NVFP4 activation quantization selected on Blackwell."""

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

KIND = "nvfp4_quant"


@dataclass(frozen=True)
class Nvfp4QuantArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Activation rows to quantize.")
    hidden_size: int = arg(unit="elements", doc="Elements in each activation row.")
    group_size: int = arg(unit="elements", doc="Elements sharing one FP4 scale.")
    input_dtype: DType = arg(doc="Element type of the input activations.")
    scale_format: str = arg(doc="Encoding and layout of the output scales.")


DOC = KernelDoc(
    title="NVFP4 activation quantization",
    summary="Quantize BF16 activation rows to packed NVFP4 values and E4M3 group scales.",
    description=(
        "Before an NVFP4 MoE GEMM on B200, each activation row is quantized in "
        "16-element groups to packed FP4 values with one E4M3 scale per group. "
        "Both backends write the group scales in the linear row-major layout "
        "that the TensorRT-LLM MoE GEMM reads; NVFP4 linear layers use a "
        "swizzled layout instead. vLLM uses one global scale of 1; FlashInfer, "
        "on SGLang's path, also writes a per-token FP32 scale."
    ),
    category="Quantization",
    formula=(
        "GB/s = num_tokens · hidden_size · (2 input + 1/2 packed FP4 + "
        "1/16 E4M3 scale) bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "vllm_cuda counts cvt_fp16_to_fp4_sf_major; flashinfer_cutedsl counts "
        "nvfp4_quantize, after one call that builds the kernel."
    ),
    caveats=(
        "The two backends scale differently, so their outputs are not interchangeable.",
        "GB/s counts BF16 input, packed FP4 output and E4M3 group scales; "
        "global and per-token scales are excluded.",
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
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.nvfp4_quant",
            function_name="profile_nvfp4_quant_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=Nvfp4QuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM's scaled_fp4_quant CUDA op with unswizzled E4M3 scales.",
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_cutedsl",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.nvfp4_quant",
            function_name="profile_nvfp4_quant_flashinfer_cutedsl",
        ),
        table_name=KIND,
        args_schema=Nvfp4QuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary=(
                "FlashInfer nvfp4_quantize, CuTe DSL backend with per-token activation "
                "scales, as SGLang calls it."
            ),
        ),
    )
)
