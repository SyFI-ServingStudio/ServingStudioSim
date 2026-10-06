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
    scale_format: str = arg(
        doc="Encoding and layout of the output scales: linear_e4m3 or swizzled_e4m3."
    )


DOC = KernelDoc(
    title="NVFP4 activation quantization",
    summary="Quantize BF16 activation rows to packed NVFP4 values and E4M3 group scales.",
    description=(
        "Before an NVFP4 GEMM on B200, each activation row is quantized in "
        "16-element groups to packed FP4 values with one E4M3 scale per group. "
        "scale_format selects where the group scales go. linear_e4m3 writes "
        "them row-major, as the TensorRT-LLM NVFP4 MoE GEMM reads them; both "
        "backends support it. swizzled_e4m3, vllm_cuda only, writes them in the "
        "128-row by 4-group tile layout that vLLM's default B200 NVFP4 linear "
        "GEMM reads, padded with zero scales to whole tiles. vLLM's MoE path "
        "uses a global scale of 1 and its linear path the calibrated input "
        "scale; FlashInfer, on SGLang's path, also writes a per-token FP32 scale."
    ),
    category="Quantization",
    formula=(
        "GB/s = (num_tokens · hidden_size · (2 input + 1/2 packed FP4) + scale bytes) / time",
        "scale bytes = num_tokens · hidden_size/16 for linear_e4m3; "
        "ceil(num_tokens/128)·128 · ceil(hidden_size/64)·4 for swizzled_e4m3",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "vllm_cuda counts cvt_fp16_to_fp4_sf_major for linear_e4m3 and "
        "cvt_fp16_to_fp4 (not _sf_major) for swizzled_e4m3; flashinfer_cutedsl counts "
        "nvfp4_quantize, after one call that builds the kernel."
    ),
    caveats=(
        "The two backends scale differently, so their outputs are not interchangeable.",
        "GB/s counts BF16 input, packed FP4 output and the E4M3 scale tensor, "
        "including the zero-filled tile padding of swizzled_e4m3; global and "
        "per-token scales are excluded.",
        "swizzled_e4m3 walks ceil(num_tokens/128)·128 rows, so below 128 tokens "
        "the padded rows still cost scale writes.",
    ),
    # No separate PyTorch reference implementation exists for this kind.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        # vLLM's NVFP4 quant entry accepts SM100-SM129 (nvfp4_quant_entry.cu,
        # nvfp4_quant_sm_supported); FP4 conversion has no SM90 path.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(10, 0),
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
            summary=(
                "vLLM's scaled_fp4_quant CUDA op, with row-major E4M3 scales for the "
                "TensorRT-LLM MoE or swizzled ones for the NVFP4 linear GEMM."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_cutedsl",
        # FlashInfer's CuTe DSL nvfp4_quantize requires SM100+.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(10, 0),
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
