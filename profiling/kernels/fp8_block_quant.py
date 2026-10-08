"""BF16-to-FP8 1x128 dynamic block quantization kernel kind.

The upper layer has already resolved routing and EP ownership before it reaches
L1. Consequently ``num_tokens`` is the final local-rank row count. The static
``num_problems`` (local expert count) is retained because it changes the grouped
kernel's launch and scale-storage layout; the full per-expert batch vector is
intentionally not part of this wire schema. The runner synthesizes uniform
boundaries with the same problem count while preserving the final row count.

The output contract is fixed for the first backend: FP8 E4M3 values plus FP32
inverse/dequant scales, with BF16 input and a 128-element block size.  Those
fixed choices do not become cache-key fields; ``input_dtype`` remains explicit
because dtype is a capability and future backend-selection axis.
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

KIND: str = "fp8_block_quant"


@dataclass(frozen=True)
class Fp8BlockQuantArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Routed token rows on this GPU.")
    hidden_size: int = arg(unit="elements", doc="Elements in each token row.")
    num_problems: int = arg(unit="experts", doc="Local expert segments in the grouped input.")
    input_dtype: DType = arg(doc="Element type of the input rows.")


DOC = KernelDoc(
    title="FP8 block quantization",
    summary="Quantize BF16 expert inputs to FP8 E4M3 in 128-element blocks.",
    description=(
        "The input step of the FP8 blockscale grouped GEMM "
        "(fp8_blockscale_grouped_gemm): each routed row is split into "
        "128-element blocks, and each block is scaled to FP8 E4M3 with one FP32"
        " scale. The kernel walks the rows expert by expert, so the measurement"
        " splits num_tokens into num_problems contiguous segments of near-equal"
        " size."
    ),
    category="Quantization",
    formula=(
        "scale = max(abs(block)) / 448, or 1 for an all-zero block",
        "FP8 block = FP8_E4M3(block / scale)",
        "GB/s = num_tokens · (hidden_size · (BF16 bytes + FP8 bytes) + "
        "hidden_size / 128 · FP32 bytes) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Only the grouped scale_1x128_kernel launch is counted. FlashInfer does"
        " not expose this launch on its own, so a local binding launches the "
        "vendored TensorRT-LLM template directly."
    ),
    caveats=(
        "Segments are near-equal; real routing gives uneven segments for the "
        "same num_tokens and num_problems.",
        "GB/s counts logical input, output and scale bytes, without the padding"
        " of the grouped scale layout.",
    ),
    # No separate PyTorch reference implementation exists for this kind.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_90a"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.fp8_block_quant",
            function_name="profile_fp8_block_quant_flashinfer_trtllm",
        ),
        table_name=KIND,
        args_schema=Fp8BlockQuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="flashinfer_pip_env",
        doc=BackendDoc(
            summary=(
                "TensorRT-LLM's grouped scale_1x128_kernel from the FlashInfer wheel, "
                "launched alone."
            ),
        ),
    )
)
