"""Qwen GDN chunk-local cumulative-decay kernel kind.

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

KIND: str = "gdn_chunk_local_cumsum"


@dataclass(frozen=True)
class GdnChunkLocalCumsumArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens in the ragged prefill batch.")
    num_chunks: int = arg(
        unit="chunks", doc="Balanced chunks in the measured partition, one per sequence."
    )
    num_heads: int = arg(unit="heads", doc="Decay values accumulated per token.")
    dtype: DType = arg(doc="Element type of decay inputs and outputs.")


DOC = KernelDoc(
    title="Chunk-local cumulative decay",
    summary="Compute an inclusive cumulative sum of decay values within each prefill chunk.",
    description=(
        "The first launch of the split, FLA Triton form of chunked Gated "
        "DeltaNet prefill: for every head, the running sum of the log decay "
        "within each chunk of at most 64 tokens, restarting at every chunk "
        "boundary. The measurement splits num_tokens into num_chunks sequences "
        "of near-equal length, each short enough to fit in one chunk. vLLM on "
        "H200 runs Gated DeltaNet prefill as one fused FlashInfer kernel "
        "(gdn_chunk_delta_rule), so this kind is kept for deployments that "
        "select the Triton path."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "out[t, h] = Σᵢ₌chunk_startᵗ g[i, h]",
        "FLOPs = num_heads · (num_tokens − num_chunks)",
        "bytes = 8 · num_tokens · num_heads",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "torch launches one cumsum per chunk and counts them all; vllm_triton "
        "counts only chunk_local_cumsum_scalar_kernel. Partitioning, metadata "
        "copies and the vLLM output check run before timing."
    ),
    caveats=(
        "The vLLM wrapper allocates a fresh output on each call; CUPTI counts "
        "only the selected kernel, so the allocation is not included.",
        "The reference module runs on the CPU; the torch backend computes the "
        "same chunked sum on the GPU.",
        "GB/s excludes chunk metadata; FLOPs count the additions of the "
        "operation, not instructions.",
    ),
    reference="profiling.runners.attention.gdn_chunk_local_cumsum_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.FP32})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_local_cumsum_torch",
            function_name="profile_gdn_chunk_local_cumsum",
        ),
        table_name=KIND,
        args_schema=GdnChunkLocalCumsumArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="Preallocated PyTorch cumsum calls, one per balanced chunk in the batch."
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton"),
            function_name="profile_gdn_chunk_local_cumsum_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnChunkLocalCumsumArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM's chunk_local_cumsum Triton call uses a single scalar kernel for the "
                "ragged batch."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/cumsum.py",
        ),
        subprocess_env="vllm_env",
    )
)
