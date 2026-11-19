"""Hadamard rotation of each row followed by per-row power-of-two FP8 quantization.

A sparse-attention indexer can rotate each query head by a Walsh-Hadamard
transform before quantizing it to FP8, so the outlier energy of a few channels
spreads over the whole head and one absmax scale per head loses less. The
rotation and the quantization run as one launch: rows of ``head_dim`` BF16
elements in, FP8 E4M3 rows and one FP32 power-of-two (UE8M0-valued) scale per
row out.
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

KIND: str = "hadamard_fp8_quant"


@dataclass(frozen=True)
class HadamardFp8QuantArgs(KernelArgs):
    num_rows: int = arg(
        unit="rows", doc="Rows rotated and quantized together (tokens times heads)."
    )
    head_dim: int = arg(
        unit="elements", doc="Elements in each row; also the Hadamard transform size."
    )
    input_dtype: DType = arg(doc="Element type of the input rows.")


DOC = KernelDoc(
    title="Hadamard rotation and FP8 quantization",
    summary=(
        "Rotate each row by a Hadamard transform, then quantize it to FP8 with one "
        "power-of-two scale."
    ),
    description=(
        "Before a sparse-attention indexer scores cached keys, each query head "
        "is multiplied by the normalized Walsh-Hadamard matrix of its width and "
        "quantized to FP8 E4M3. The rotated row is rounded to BF16 first, as an "
        "unfused rotation would store it; its absolute maximum, floored at 1e-4, "
        "sets one scale per row rounded up to a power of two (UE8M0), and the "
        "row divided by that scale is clamped to ±448. The FP8 rows and the FP32"
        " scales are written in one launch."
    ),
    category="Quantization",
    formula=(
        "y = BF16(x · H_head_dim / √head_dim)",
        "scale = 2^⌈log2(max(|y|, 1e-4) / 448)⌉ per row",
        "q = clamp(y / scale, −448, 448) as FP8 E4M3",
        "GB/s = num_rows · (head_dim · (BF16 bytes + FP8 bytes) + 4) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} The call is one Triton launch; output allocation is "
        "outside the counted kernel time."
    ),
    caveats=(
        "The vLLM-fork callable takes only head_dim = 128.",
        "GB/s counts logical input, FP8 output and scale bytes, not physical memory transactions.",
    ),
    # The runner checks the callable against a dense Hadamard matmul in float64.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        # The kernel stores tl.float8e4nv, which Triton lowers only on SM89+.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 9),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.hadamard_fp8_quant_vllm_triton",
            function_name="profile_hadamard_fp8_quant_vllm_triton",
        ),
        table_name=KIND,
        args_schema=HadamardFp8QuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "The vLLM fork's fwht128_quant_fp8 Triton call: a 128-point "
                "Hadamard butterfly and per-row UE8M0 FP8 quantization in one "
                "launch, 32 rows per program."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/models/glm5next/nvidia/ops/kpool_compress.py",
        ),
    )
)

__all__ = ["KIND", "HadamardFp8QuantArgs"]
