"""GDN chunk-local triangular-solve kernel kind.

The initial ``torch`` backend measures a multi-launch semantic implementation.
It is a correctness/performance baseline, not the production fused Triton
launch, and must not be selected for production simulation after the vLLM
backend is registered.
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

KIND: str = "gdn_chunk_solve_tril"


@dataclass(frozen=True)
class GdnChunkSolveTrilArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens across the input sequences.")
    num_chunks: int = arg(unit="chunks", doc="Sequence-local chunks whose matrices are inverted.")
    max_chunk_tokens: int = arg(unit="tokens", doc="Largest valid chunk length, at most 64.")
    num_heads: int = arg(unit="heads", doc="Heads with separate triangular matrices.")
    dtype: DType = arg(doc="Element type of the inverse; only bf16 is measured.")


DOC = KernelDoc(
    title="Chunk-local triangular inverse",
    summary="Invert each chunk's unit lower-triangular key-product matrix.",
    description=(
        "A launch of the split, FLA Triton form of chunked Gated DeltaNet "
        "prefill: it inverts each chunk's I + A, where A is the "
        "lower-triangular key-product matrix from the previous step, for the W "
        "and U recomputation that follows. The measurement builds num_chunks "
        "nonempty chunks, the longest exactly max_chunk_tokens long. vLLM on "
        "H200 runs Gated DeltaNet prefill as one fused FlashInfer kernel "
        "(gdn_chunk_delta_rule), so this kind is kept for deployments that "
        "select the Triton path."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "M = (I + A)⁻¹ for each valid chunk block",
        "FLOPs = num_heads · Σchunks p(p − 1)(2p − 1)/6, where p is the valid chunk length",
        "bytes = 6·num_tokens·num_heads·64",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of its inverse; vllm_triton counts only "
        "merge_16x16_to_64x64_inverse_kernel. Inputs are built before the "
        "capture."
    ),
    caveats=(
        "FLOPs count the valid triangular recurrence; bytes count an FP32 input"
        " and a BF16 output in 64-column storage. Neither is physical work.",
    ),
    reference="profiling.runners.attention.gdn_chunk_solve_tril_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_solve_tril_torch",
            function_name="profile_gdn_chunk_solve_tril",
        ),
        table_name=KIND,
        args_schema=GdnChunkSolveTrilArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="PyTorch applies the triangular recurrence across separate launches."
        ),
        subprocess_env="default_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton",
            function_name="profile_gdn_chunk_solve_tril_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnChunkSolveTrilArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="vLLM's solve_tril selects the 64-column Triton inverse kernel.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/solve_tril.py",
        ),
        subprocess_env="vllm_env",
    )
)
