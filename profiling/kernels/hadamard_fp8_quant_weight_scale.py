"""Hadamard-rotate and FP8-quantize each indexer query head read in place from a wider GEMM row,
and scale the head weights by each head's quant scale, in one launch.

When a sparse-attention indexer's query projection runs as part of one GEMM over
concatenated weights, its heads are a column span of each output row. This
kind reads that span in place (no copy), rotates each head by the normalized
Walsh-Hadamard transform, quantizes it to FP8 E4M3 with one power-of-two scale
per head, and writes the head weights multiplied by those scales, which the
indexer's logits kernel reads. Unfused, that is ``hadamard_fp8_quant`` on a
contiguous projection output plus a separate weight-scale launch.
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

KIND: str = "hadamard_fp8_quant_weight_scale"


@dataclass(frozen=True)
class HadamardFp8QuantWeightScaleArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Input rows, one per token.")
    num_heads: int = arg(unit="heads", doc="Indexer query heads per token.")
    head_dim: int = arg(unit="elements", doc="Elements per head; also the Hadamard transform size.")
    lead_cols: int = arg(
        unit="elements",
        doc=(
            "Columns of each input row before the query heads (the other projections' "
            "outputs of the concatenated GEMM); the row holds lead_cols + "
            "num_heads·head_dim columns."
        ),
    )
    input_dtype: DType = arg(doc="Element type of the input rows.")


DOC = KernelDoc(
    title="Hadamard FP8 quantization with head-weight scaling",
    summary=(
        "Rotate each query head of a wider GEMM row by a Hadamard transform, quantize it to "
        "FP8 with one power-of-two scale, and scale the head weights by it, in one launch."
    ),
    description=(
        "A DSA indexer rotates each query head by the normalized Walsh-Hadamard "
        "matrix and quantizes it to FP8 E4M3 before scoring cached keys, and scales "
        "each head's weight by that head's quant scale. When the query projection "
        "is one column span of a concatenated GEMM's output, this kind reads the "
        "span in place (row stride lead_cols + num_heads·head_dim). The rotated head "
        "is rounded to BF16 first; its absolute maximum, floored at 1e-4, sets a "
        "scale rounded up to a power of two (UE8M0); the head divided by it is "
        "clamped to ±448. The FP32 head weights (one per token and head) are "
        "multiplied by the scale and by head_dim^-0.5 · num_heads^-0.5. FP8 heads, "
        "FP32 scales and FP32 scaled weights are written in one launch."
    ),
    category="Quantization",
    formula=(
        "y = BF16(x · H_head_dim / √head_dim) per head",
        "scale = 2^⌈log2(max(|y|, 1e-4) / 448)⌉ per head",
        "q = clamp(y / scale, −448, 448) as FP8 E4M3",
        "w' = w · scale · head_dim^-0.5 · num_heads^-0.5",
        "GB/s = num_tokens · num_heads · (head_dim · (BF16 + FP8 bytes) + 3 · 4) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} One launch per call; three warm-up calls run first. Before "
        "timing, the outputs are checked byte for byte against the vLLM fork's "
        "fwht128_quant_fp8 on a contiguous copy of the span plus the weight scale, "
        "and the FP8 heads against a float64 dense-Hadamard reference."
    ),
    caveats=(
        "The triton backend has its head geometry compiled in (32 heads of 128) and "
        "refuses any other; lead_cols is free.",
        "The triton backend forms input offsets in int32, so num_tokens times the row "
        "width (lead_cols + num_heads·head_dim) must stay below 2^31: about 104,000 "
        "tokens at a 20,480-column row. Larger calls are refused.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.hadamard_fp8_quant_weight_scale_triton",
            function_name="profile_hadamard_fp8_quant_weight_scale_triton",
        ),
        table_name=KIND,
        args_schema=HadamardFp8QuantWeightScaleArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # The kernel stores tl.float8e4nv, which Triton lowers only on SM89+.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 9),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "A repo-vendored Triton kernel, 32 heads per program: the vLLM fork's "
                "fwht128_quant_fp8 butterflies on strided loads, plus the weight scale."
            ),
            url=None,
        ),
    )
)

__all__ = ["HadamardFp8QuantWeightScaleArgs", "KIND"]
