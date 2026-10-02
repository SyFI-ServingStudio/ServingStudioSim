"""GLM-5.2 request-local to global sparse-index remap kernel kind."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "dsa_sparse_index_remap"


@dataclass(frozen=True)
class DsaSparseIndexRemapArgs(KernelArgs):
    num_queries: int = arg(unit="tokens", doc="Query rows with selected indices.")
    num_requests: int = arg(unit="requests", doc="Requests represented by the query rows.")
    selected_k: int = arg(unit="tokens", doc="Index slots scanned per query row.")
    block_size: int = arg(unit="tokens", doc="Tokens in each KV cache block.")
    max_blocks_per_request: int = arg(
        unit="blocks", doc="Block-table entries reserved for each request."
    )
    request_row_counts: str = arg(doc="Encoded query-row count for each request.")
    local_span_lengths: str = arg(doc="Encoded local token span of each query row.")
    valid_counts: str = arg(doc="Encoded number of valid index slots in each query row.")
    index_distribution: str = arg(doc="Pattern used to construct local selected indices.")
    page_table_mapping: str = arg(doc="Pattern used to assign physical cache blocks.")
    workspace_partition: str = arg(doc="Decode and prefill request split for workspace mapping.")
    return_valid_counts: bool = arg(doc="Whether to return valid index counts with mapped indices.")
    index_dtype: str = arg(doc="Element type of local and mapped indices.")


DOC = KernelDoc(
    title="Sparse index remap",
    summary="Map DSA-selected request-local token indices to cache or prefill-workspace positions.",
    description=(
        "The DSA indexer selects positions local to each request; sparse MLA "
        "attention needs them as slots in the paged KV cache or, for prefill, "
        "in a workspace. Decode rows go through the block table; prefill rows "
        "add their local position to the request's workspace start. Invalid "
        "slots stay -1; when valid counts are returned, each row's valid slots "
        "are packed to the front in no fixed order. The measurement builds the "
        "rows, indices and page table from the encoded counts and distribution "
        "patterns in the arguments."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "global = block_table[request, ⌊local / block_size⌋]·block_size + local mod block_size",
        "workspace = workspace_start + local; invalid slots remain -1",
        "Q = num_queries; K = selected_k; T = Q·K/128; G = valid decode slots; W = workspace rows",
        "I_w = 1 if workspace exists, else 0; I_c = 1 if counts are returned, else 0",
        "logical bytes = 8·Q·K + 4·T + 4·G + I_w·(4·T + 4·W·K/128) + I_c·(4·Q + 8·T)",
        "GB/s = logical bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of its row-chunked composite; vllm_triton "
        "counts the union of GPU busy intervals, so overlapping launches count "
        "once. Operand construction and output checks run before timing."
    ),
    caveats=(
        "GB/s assumes 128-slot tiles and includes count-buffer initialization "
        "and atomic updates when counts are returned; the single-tile native "
        "path skips both.",
        "Page tables and local indices follow the chosen patterns; they are not"
        " captured from serving.",
        "Only 64-token blocks are accepted, with selected_k = 2048 for torch "
        "and 2048 or 2176 for vllm_triton.",
    ),
    reference="profiling.runners.attention.dsa_sparse_index_remap_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=None,
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_index_remap",
            function_name="profile_dsa_sparse_index_remap_torch",
        ),
        table_name=KIND,
        args_schema=DsaSparseIndexRemapArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "A row-chunked PyTorch composite that writes mapped indices into "
                "preallocated output."
            )
        ),
    )
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=None,
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_index_remap",
            function_name="profile_dsa_sparse_index_remap_vllm_triton",
        ),
        table_name=KIND,
        args_schema=DsaSparseIndexRemapArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's triton_convert_req_index_to_global_index wrapper, with "
                "optional valid-count output."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/sparse_utils.py",
        ),
    )
)


# MI300X elementwise byte-placeholder floor (GLM-5.3-Flash port, decision #37
# "mechanism B"): this kind carries a negligible predicted share of iteration
# time and has no MI300X-native backend yet, so instead of leaving it pinned to
# an NVIDIA-only backend -- which a real MI300X ``timing-predict`` rejects at
# ``BackendSupport.allows`` -- its MI300X cost is a closed-form analytic memory
# roofline (decision #49): the runner derives this kind's memory-bound byte
# footprint from its shape args and converts it to a time arithmetically,
# t = launch_latency + (read+write bytes) / effective MI300X HBM bandwidth
# (``profiling.runners.elementwise.floor``) -- no allocation, no rocprofv3, so a
# cell whose footprint exceeds 192 GB HBM yields a finite time instead of OOM.
# MI300X-gated and compute-agnostic,
# so every NVIDIA target -- B200 included -- stays byte-identical.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="elementwise_floor",
        supports=BackendSupport(compute=None, gpus=frozenset({"MI300X"})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.floor",
            function_name="profile_dsa_sparse_index_remap_floor",
        ),
        table_name=KIND,
        args_schema=DsaSparseIndexRemapArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_rocm_env",
        doc=BackendDoc(
            summary=(
                "Analytic memory-roofline floor: t = launch_latency + "
                "(read+write bytes) / effective MI300X HBM bandwidth, over this "
                "kind's shape-derived footprint. A derived negligible-share floor "
                "for the GLM-5.3-Flash MI300X port, not a measured native kernel."
            ),
        ),
    )
)
