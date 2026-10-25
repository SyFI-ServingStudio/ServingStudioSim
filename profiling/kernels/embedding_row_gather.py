"""Raw row gather from a large per-head-partitioned embedding table.

Source: alignment fork, ``vllm/models/qwen4_exp/nvidia/ngram_embedding.py``.
With ``EngramConfig.cpu_offload=True`` (the default, ``config/engram.py:38``)
the n-gram (PLE) table is allocated in pinned host memory
(``Qwen4ExpPLEPinnedHostEmbedding.allocate_embedding_weight``) and
``start_prefetch`` launches ``_lookup_ple_embedding_from_pinned_kernel``
(``:357-384, :445-479``) on a side stream: one program per (token, head) id,
each copying ``head_dim`` raw table elements through the table's UVA device
view into a device staging buffer. Nothing is dequantized here: the FP8 rows
keep the table dtype, and the per-tensor scale is applied later by separate
elementwise launches (``Qwen4ExpPLEFp8EmbeddingMethod.dequantize``).

Not the ``engram_lookup`` kind: that kernel reads a second UE8M0 scale table,
dequantizes per 32-value block and writes BF16 rows with a half-SM persistent
grid. This one copies raw table bytes with one program per row, so the logical
I/O differs.

Args:

- ``num_tokens``: rows of the [tokens, heads] id tensor (scheduled tokens).
- ``num_heads``: ids per token, each into its own disjoint table range
  (``(ngram_size - 1) * heads_per_ngram``).
- ``head_dim``: table elements per row.
- ``table_rows``: table length; with uniform ids it sets the TLB and page
  footprint of the random gather, so it is an exact identity.
- ``residency``: ``host_uva`` (pinned host memory read over UVA, the
  ``cpu_offload=True`` path) or ``device`` (HBM, ``cpu_offload=False``, where
  production calls ``F.embedding`` synchronously instead).
- ``weight_dtype``: table element dtype (``fp8_e4m3``; ``bf16`` for ``torch``,
  e.g. a token embedding table).
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
from profiling.kernels.neuron_common import neuron_spec

KIND = "embedding_row_gather"


@dataclass(frozen=True)
class EmbeddingRowGatherArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows of the [tokens, heads] id tensor.")
    num_heads: int = arg(unit="heads", doc="Ids per token, each in its own table range.")
    head_dim: int = arg(unit="elements", doc="Table elements in one row.")
    table_rows: int = arg(unit="rows", doc="Rows of the table (sets the gather footprint).")
    residency: str = arg(
        doc="Where the table lives: host_uva (pinned host memory over UVA) or device (HBM)."
    )
    weight_dtype: DType = arg(doc="Element type of the table rows.")


DOC = KernelDoc(
    title="Embedding row gather",
    summary=(
        "Copy num_tokens × num_heads randomly addressed rows of a large table, "
        "kept in pinned host memory or HBM, into a device buffer."
    ),
    description=(
        "Per-layer n-gram embeddings hash each token's preceding n-grams into "
        "num_heads ids, one per disjoint range of a very large table. The gather "
        "copies those rows unchanged (no dequantization) into a device staging "
        "buffer. With the table in pinned host memory the rows cross the host "
        "link one small read at a time, so the time follows the link's random-"
        "read rate and the table footprint, not HBM bandwidth."
    ),
    category="Other",
    formula=(
        "out[t, h, :] = table[id[t, h], :]",
        "GB/s = num_tokens · num_heads · (8 + 2 · head_dim · bytes(weight_dtype)) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Each timed call reads the next of 4 to 4096 id sets, as many as bring "
        "the pool to about four million rows (fewer for calls under 1024 ids), "
        "so consecutive calls do not reuse rows left in L2 or the TLBs. Before "
        "timing, the output must equal a PyTorch index_select bit for bit."
    ),
    caveats=(
        "Ids are uniform within each head's equal share of the table; real "
        "n-gram hashes repeat frequent n-grams and may hit fewer distinct pages.",
        "The table is pinned in the profiling process's host memory; NUMA "
        "placement relative to the GPU follows the worker's CPU binding.",
        "TFLOPS is not computed.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.embedding.embedding_row_gather",
            function_name="profile_embedding_row_gather_vllm_triton",
        ),
        table_name=KIND,
        args_schema=EmbeddingRowGatherArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # The kernel is a plain byte copy (tl.load/tl.store of the table dtype);
        # FP8 pointers lower on SM89+ in Triton.
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            min_compute_capability=(8, 9),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "The fork's Qwen4ExpPLEPinnedHostEmbedding._lookup "
                "(_lookup_ple_embedding_from_pinned_kernel) on a pinned host table "
                "through its UVA view; host_uva only."
            ),
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        runner_ref=RunnerRef(
            module_name="profiling.runners.embedding.embedding_row_gather",
            function_name="profile_embedding_row_gather_torch",
        ),
        table_name=KIND,
        args_schema=EmbeddingRowGatherArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # F.embedding is a raw row index_select; no capability rule.
        supports=BackendSupport(compute=frozenset({DType.FP8_E4M3, DType.BF16})),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "torch.nn.functional.embedding on an HBM table, the "
                "cpu_offload=False path (Qwen4ExpPLEDeviceEmbedding); device only."
            ),
        ),
    )
)


# The Flash-Next Trainium engine's masked local gathers, one region, NRT device time:
# 16 x 160 -> the PLE gather from the rank's real table slice
# (`flashnext/neuron/ops/ple.py::ple_gather_torch`), 1 x 2560 -> the vocab-parallel token
# embed (`flashnext/neuron/ops/head.py::embed_torch`). Device-resident tables only.
register(
    neuron_spec(
        KIND,
        "torch",
        EmbeddingRowGatherArgs,
        "The Flash-Next Trainium engine's masked index_select of local table rows, one region.",
    )
)

__all__ = ["EmbeddingRowGatherArgs", "KIND"]
