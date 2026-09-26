"""Qwen GDN chunk-local WY recomputation kernel kind.

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

KIND: str = "gdn_chunk_recompute_w_u"


@dataclass(frozen=True)
class GdnChunkRecomputeWUArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens across the input sequences.")
    num_chunks: int = arg(unit="chunks", doc="Sequence-local chunks whose factors are recomputed.")
    num_key_heads: int = arg(unit="heads", doc="Key heads shared by the output heads.")
    num_heads: int = arg(unit="heads", doc="Output heads with separate W and U factors.")
    key_head_dim: int = arg(unit="elements", doc="Features in each key head and W row.")
    value_head_dim: int = arg(unit="elements", doc="Features in each value head and U row.")
    dtype: DType = arg(doc="Element type of the keys, values and factors; only bf16 is measured.")


DOC = KernelDoc(
    title="Chunk-local W and U recomputation",
    summary="Apply the solved triangular matrix to gated keys and beta-scaled values.",
    description=(
        "A launch of the split, FLA Triton form of chunked Gated DeltaNet "
        "prefill: the inverted chunk matrix is applied to the decayed, "
        "beta-scaled keys and to the beta-scaled values, giving the W and U "
        "factors of the state update. Output heads share key heads but have "
        "their own values and gates. The measurement splits num_tokens into "
        "num_chunks sequences of at most 64 tokens. vLLM on H200 runs Gated "
        "DeltaNet prefill as one fused FlashInfer kernel "
        "(gdn_chunk_delta_rule), so this kind is kept for deployments that "
        "select the Triton path."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "U = A · (β·V); W = A · (β·exp(g)·K) within each chunk",
        "S = Σchunks p², where p is the valid chunk length",
        "FLOPs = num_heads·(key_head_dim + value_head_dim)·S + "
        "num_heads·num_tokens·(value_head_dim + 2·key_head_dim + 1)",
        "bytes = 2·num_tokens·num_key_heads·key_head_dim + "
        "2·num_tokens·num_heads·value_head_dim + 8·num_tokens·num_heads + "
        "2·num_tokens·num_heads·64 + 2·num_tokens·num_heads·key_head_dim + "
        "2·num_tokens·num_heads·value_head_dim",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of its calculation; vllm_triton counts only "
        "recompute_w_u_fwd_kernel. Inputs are built before the capture."
    ),
    caveats=(
        "FLOPs count one unit per multiply-add term, not instructions; bytes "
        "count tensor inputs and outputs, not temporaries or metadata.",
    ),
    reference="profiling.runners.attention.gdn_chunk_recompute_w_u_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_recompute_w_u_torch",
            function_name="profile_gdn_chunk_recompute_w_u",
        ),
        table_name=KIND,
        args_schema=GdnChunkRecomputeWUArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(summary="PyTorch computes the W and U factors across separate launches."),
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
            module_name=("profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton"),
            function_name="profile_gdn_chunk_recompute_w_u_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnChunkRecomputeWUArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="vLLM's recompute_w_u_fwd Triton call computes both factors in one launch.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/wy_fast.py",
        ),
        subprocess_env="vllm_env",
    )
)
