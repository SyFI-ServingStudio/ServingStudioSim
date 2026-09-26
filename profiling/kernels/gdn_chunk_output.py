"""Qwen GDN fused chunk-output kernel kind.

The Torch backend is a multi-launch semantic baseline. Production simulation
must select the fused vLLM Triton backend.
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

KIND: str = "gdn_chunk_output"


@dataclass(frozen=True)
class GdnChunkOutputArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens across the input sequences.")
    num_chunks: int = arg(unit="chunks", doc="Sequence-local chunks producing output.")
    num_key_heads: int = arg(unit="heads", doc="Key and query heads shared by the output heads.")
    num_heads: int = arg(unit="heads", doc="Output heads with separate values and states.")
    key_head_dim: int = arg(unit="elements", doc="Features in each key and query head.")
    value_head_dim: int = arg(unit="elements", doc="Features in each value and output head.")
    dtype: DType = arg(
        doc="Element type of the query, key, value and output; only bf16 is measured."
    )


DOC = KernelDoc(
    title="Chunked delta-rule output",
    summary="Combine the incoming state with gated causal key-value products for each token.",
    description=(
        "The last launch of the split, FLA Triton form of chunked Gated "
        "DeltaNet prefill: each query reads the state at its chunk's start plus"
        " the decayed contributions of earlier positions in the same chunk. The"
        " measurement splits num_tokens into num_chunks sequences of at most 64"
        " tokens. vLLM on H200 runs Gated DeltaNet prefill as one fused "
        "FlashInfer kernel (gdn_chunk_delta_rule), so this kind is kept for "
        "deployments that select the Triton path."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "oᵢ = [exp(gᵢ)·H·qᵢ + Σⱼ≤ᵢ exp(gᵢ − gⱼ)·(qᵢ·kⱼ)·v_new[j]] / √key_head_dim",
        "FLOPs = num_heads · Σchunks [2p·key_head_dim·value_head_dim + "
        "2p²·key_head_dim + p + 3p·value_head_dim + "
        "p(p + 1)·(1 + value_head_dim)], where p is the valid chunk length",
        "bytes = 4·num_tokens·num_key_heads·key_head_dim + "
        "4·num_tokens·num_heads·value_head_dim + "
        "2·num_chunks·num_heads·value_head_dim·key_head_dim + "
        "4·num_tokens·num_heads",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of its calculation; vllm_triton counts only "
        "chunk_fwd_kernel_o. Inputs are built before the capture."
    ),
    caveats=(
        "FLOPs count the valid causal pairs; bytes count tensor inputs and "
        "outputs, not padding or temporaries.",
    ),
    reference="profiling.runners.attention.gdn_chunk_output_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_output_torch",
            function_name="profile_gdn_chunk_output",
        ),
        table_name=KIND,
        args_schema=GdnChunkOutputArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(summary="PyTorch combines state and causal terms across separate launches."),
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
            module_name="profiling.runners.attention.gdn_chunk_output_vllm_triton",
            function_name="profile_gdn_chunk_output_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnChunkOutputArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="vLLM's chunk_fwd_o Triton call combines state and causal terms in one launch.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/chunk_o.py",
        ),
        subprocess_env="vllm_env",
    )
)
