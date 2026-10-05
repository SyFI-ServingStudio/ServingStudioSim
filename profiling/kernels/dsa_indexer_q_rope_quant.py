"""SGLang fused DSA indexer-query RoPE and FP8 quantization kernel kind."""

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

KIND: str = "dsa_indexer_q_rope_quant"


@dataclass(frozen=True)
class DsaIndexerQRopeQuantArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Indexer query tokens in the batch.")
    num_heads: int = arg(unit="heads", doc="Indexer query heads per token.")
    head_dim: int = arg(unit="elements", doc="Elements in each indexer query head.")
    rope_dim: int = arg(unit="elements", doc="Leading query dimensions rotated by RoPE.")
    # Both fields select compile-time CUDA instantiations.
    rope_layout: str = arg(doc="Placement of RoPE dimensions within each query head.")
    hadamard: bool = arg(doc="Whether the query path applies a Hadamard transform.")
    input_dtype: DType = arg(doc="Element type of the input queries and gate weights.")
    q_output_dtype: DType = arg(doc="Element type of the quantized indexer queries.")
    weight_output_dtype: DType = arg(doc="Element type of the output indexer weights.")


DOC = KernelDoc(
    title="Indexer query RoPE and FP8 quantization",
    summary="Rotate indexer queries and quantize them to FP8 while producing per-head weights.",
    description=(
        "The first step of the DSA indexer on SGLang: before the indexer "
        "scores cached keys, each query head has RoPE applied to its first "
        "rope_dim elements and is quantized to FP8, and the per-head weights "
        "are written as FP32, all in one call. The weights are read as a "
        "strided slice of the key-and-weight projection output, as in serving. "
        "Positions are random rows of a 131,072-row RoPE table."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "q_fp8 = FP8(RoPE(q) on the first rope_dim elements)",
        "rows = num_tokens · num_heads",
        "TFLOPS = rows · (⌊rope_dim / 2⌋ · 6 + head_dim) / time",
        "GB/s = [rows · (3 · head_dim + 2 + 4) + num_tokens · (4 · rope_dim + 8)] / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "One call runs before the capture. The wrapper allocates its outputs on"
        " each call; CUPTI counts kernel time only, so the allocation is not "
        "included."
    ),
    caveats=(
        "The runner takes only 128-element heads with 64 RoPE elements first "
        "and no Hadamard transform.",
        "GB/s counts logical input and output bytes, including the FP8 queries "
        "and FP32 weights, not physical memory transactions.",
    ),
    # No separate PyTorch reference implementation exists for this kind.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_indexer_q_rope_quant",
            function_name="profile_dsa_indexer_q_rope_quant_sglang_cuda",
        ),
        table_name=KIND,
        args_schema=DsaIndexerQRopeQuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary=(
                "SGLang fused_q_indexer_rope_first_quant: RoPE, FP8 query quantization "
                "and FP32 weights in one call."
            ),
            url="https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/dsv4/elementwise.py",
        ),
    )
)

__all__ = ["KIND", "DsaIndexerQRopeQuantArgs"]
