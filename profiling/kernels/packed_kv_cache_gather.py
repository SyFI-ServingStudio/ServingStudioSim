"""Dequantize-and-gather from a packed FP8/BF16 paged KV cache."""

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

KIND = "packed_kv_cache_gather"


@dataclass(frozen=True)
class PackedKvCacheGatherArgs(KernelArgs):
    seq_lens: tuple[int, ...] = arg(unit="tokens", doc="Cache sequence length per request.")
    gather_lens: tuple[int, ...] = arg(
        unit="tokens", doc="Suffix length per request; an empty tuple gathers full sequences."
    )
    workspace_rows: int = arg(unit="rows", doc="Output rows allocated per request.")
    block_table_width: int = arg(unit="blocks", doc="Block-table entries allocated per request.")
    block_size: int = arg(unit="tokens", doc="Cache tokens in each physical block.")
    offset: int = arg(unit="rows", doc="First output row written within each request's workspace.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads in the packed cache.")
    head_dim: int = arg(unit="elements", doc="Output elements per gathered key.")
    fp8_dim: int = arg(unit="elements", doc="FP8 elements in each packed key.")
    quant_group_size: int = arg(unit="elements", doc="FP8 elements sharing one scale.")
    cache_dtype: str = arg(doc="Packed cache storage format.")
    output_dtype: DType = arg(doc="Element type of the gathered keys.")
    cache_layout: str = arg(doc="Ordering of packed cache data and scales within a block.")
    scale_format: str = arg(doc="Encoding of the packed FP8 scales.")


DOC = KernelDoc(
    title="Packed KV cache gather",
    summary="Gather and dequantize paged FP8 keys into a BF16 prefill workspace.",
    description=(
        "Before compressed sparse MLA prefill, vLLM gathers keys from the "
        "paged, packed FP8 cache into a contiguous BF16 workspace, dequantizing"
        " as it copies: either the compressed keys or a sliding-window suffix. "
        "seq_lens gives each request's cache length; gather_lens, when given, "
        "limits the copy to the trailing rows. The measurement reverses the "
        "physical block order and fills the cache with patterned FP8 and BF16 "
        "values, so page lookup and dequantization are both exercised."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=(
        "G = Σi (gather_lens[i] if supplied else seq_lens[i]); R = len(seq_lens)",
        "TFLOPS = G·448 / time",
        "GB/s = [G·(584 + 4 + 512·2) + 4·R·(2 if gather_lens else 1)] / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "One call with an output check and three warm-up calls run before the "
        "capture; every launch of the gather call is counted."
    ),
    caveats=(
        "TFLOPS counts 448 dequantization operations per gathered row; there is"
        " no matrix multiply.",
    ),
    # The measured gather has no separate PyTorch reference module.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cutedsl",
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.packed_kv_cache_gather_cutedsl"),
            function_name="profile_packed_kv_cache_gather_cutedsl",
        ),
        table_name=KIND,
        args_schema=PackedKvCacheGatherArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            min_compute_capability=(8, 9),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's dequantize_and_gather_k_cache, dispatched to its CuTe DSL "
                "kernel for the packed FP8 cache."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py",
        ),
    )
)

__all__ = ["PackedKvCacheGatherArgs", "KIND"]
