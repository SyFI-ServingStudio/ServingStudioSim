"""GDN chunk-local scaled-dot KKT construction kernel kind.

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

KIND: str = "gdn_chunk_scaled_dot_kkt"


@dataclass(frozen=True)
class GdnChunkScaledDotKktArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens across the input sequences.")
    num_chunks: int = arg(unit="chunks", doc="Sequence-local chunks of at most 64 tokens.")
    num_key_heads: int = arg(unit="heads", doc="Key heads shared by the output heads.")
    num_heads: int = arg(unit="heads", doc="Output heads with separate gates and beta values.")
    key_head_dim: int = arg(unit="elements", doc="Features in each key head.")
    dtype: DType = arg(doc="Element type of the keys; only bf16 is measured.")


DOC = KernelDoc(
    title="Chunk-local scaled key products",
    summary="Build the gated, beta-scaled lower-triangular key-product matrix for each chunk.",
    description=(
        "A launch of the split, FLA Triton form of chunked Gated DeltaNet "
        "prefill. Within each chunk it builds the strictly lower-triangular "
        "matrix of key products, scaled by beta and by the decay between the "
        "two positions; the next step inverts it. Output heads share key heads "
        "but have their own beta and decay. The measurement splits num_tokens "
        "into num_chunks sequences of at most 64 tokens. vLLM on H200 runs "
        "Gated DeltaNet prefill as one fused FlashInfer kernel "
        "(gdn_chunk_delta_rule), so this kind is kept for deployments that "
        "select the Triton path."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "Aᵢⱼ = βᵢ · (kᵢ · kⱼᵀ) · exp(gᵢ − gⱼ), for j < i in one chunk; otherwise 0",
        "P = Σchunks p(p − 1)/2, where p is the valid chunk length",
        "FLOPs = num_heads · [num_tokens · key_head_dim + P · (2·key_head_dim + 2)]",
        "bytes = 2·num_tokens·num_key_heads·key_head_dim + "
        "8·num_tokens·num_heads + 4·num_tokens·num_heads·64",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of its calculation; vllm_triton counts only "
        "chunk_scaled_dot_kkt_fwd_kernel. Inputs are built before the capture."
    ),
    caveats=(
        "FLOPs count only the valid lower-triangle pairs, while bytes count the"
        " full 64-column output; neither is physical work.",
    ),
    reference="profiling.runners.attention.gdn_chunk_scaled_dot_kkt_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_scaled_dot_kkt_torch",
            function_name="profile_gdn_chunk_scaled_dot_kkt",
        ),
        table_name=KIND,
        args_schema=GdnChunkScaledDotKktArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "PyTorch tensor operations compute the complete chunk-local matrix "
                "across separate launches."
            )
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
            module_name=("profiling.runners.attention.gdn_chunk_scaled_dot_kkt_vllm_triton"),
            function_name="profile_gdn_chunk_scaled_dot_kkt_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnChunkScaledDotKktArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM's chunk_scaled_dot_kkt_fwd Triton call builds the matrix "
                "in one kernel launch."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py",
        ),
        subprocess_env="vllm_env",
    )
)
