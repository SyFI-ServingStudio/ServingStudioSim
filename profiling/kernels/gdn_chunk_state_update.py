"""GDN fused chunk-state-update kernel kind.

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

KIND: str = "gdn_chunk_state_update"


@dataclass(frozen=True)
class GdnChunkStateUpdateArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens across the input sequences.")
    num_chunks: int = arg(unit="chunks", doc="Chunks across all input sequences.")
    num_sequences: int = arg(
        unit="sequences", doc="Independent sequences with separate recurrent states."
    )
    max_chunks_per_sequence: int = arg(unit="chunks", doc="Chunks in the longest sequence.")
    num_key_heads: int = arg(unit="heads", doc="Key heads shared by the output heads.")
    num_heads: int = arg(unit="heads", doc="Output heads with separate recurrent states.")
    key_head_dim: int = arg(unit="elements", doc="Key features in each recurrent state.")
    value_head_dim: int = arg(unit="elements", doc="Value features in each recurrent state.")
    dtype: DType = arg(doc="Element type of token inputs and outputs; only bf16 is measured.")


DOC = KernelDoc(
    title="Chunk state update",
    summary="Advance the gated delta-rule state across chunks and produce updated values.",
    description=(
        "A launch of the split, FLA Triton form of chunked Gated DeltaNet "
        "prefill: W and U advance each sequence's matrix state chunk by chunk, "
        "and every chunk emits the state it started from plus corrected values "
        "for the output step. The measurement builds num_sequences sequences "
        "from num_tokens and num_chunks, the longest with "
        "max_chunks_per_sequence chunks. vLLM on H200 runs Gated DeltaNet "
        "prefill as one fused FlashInfer kernel (gdn_chunk_delta_rule), so this"
        " kind is kept for deployments that select the Triton path."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "v_new = U − W·Hᵀ; H_next = exp(g_end)·H + [exp(g_end − g)·v_new]ᵀ·K",
        "FLOPs = num_heads · Σchunks [4p·key_head_dim·value_head_dim + "
        "p·value_head_dim + key_head_dim·value_head_dim + 2p + 1], "
        "where p is the valid chunk length",
        "bytes = 2·num_tokens·num_key_heads·key_head_dim + "
        "2·num_tokens·num_heads·key_head_dim + "
        "4·num_tokens·num_heads·value_head_dim + 4·num_tokens·num_heads + "
        "8·num_sequences·num_heads·value_head_dim·key_head_dim + "
        "2·num_chunks·num_heads·value_head_dim·key_head_dim",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch, including its state reset; vllm_triton "
        "counts only chunk_gated_delta_rule_fwd_kernel_h_blockdim64. Sequence "
        "metadata is built before the capture."
    ),
    caveats=(
        "FLOPs use the valid chunk lengths; bytes count tensor inputs and "
        "outputs, not padding or temporaries.",
    ),
    reference="profiling.runners.attention.gdn_chunk_state_update_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_state_update_torch",
            function_name="profile_gdn_chunk_state_update",
        ),
        table_name=KIND,
        args_schema=GdnChunkStateUpdateArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(summary="PyTorch updates each sequence's state across separate launches."),
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
            module_name="profiling.runners.attention.gdn_chunk_state_update_vllm_triton",
            function_name="profile_gdn_chunk_state_update_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnChunkStateUpdateArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM's chunk_gated_delta_rule_fwd_h Triton call fuses each chunk's state update."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py",
        ),
        subprocess_env="vllm_env",
    )
)
