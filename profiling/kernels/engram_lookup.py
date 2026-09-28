"""Engram n-gram embedding row lookup (FP8 rows + UE8M0 scales -> BF16).

Source: alignment fork, ``vllm/models/deepseek_v41/common/engram.py``
``ParallelEngramEmbedding.lookup`` -> ``_engram_lookup_kernel`` (one Triton
launch per Engram layer per iteration). With ``EngramConfig.cpu_offload=True``
(the default) the tables live in pinned host memory and the kernel reads them
through a UVA device view (``nvidia/engram.py`` ``_storage``); the launch is
issued on a per-layer side stream right after the n-gram hash kernel at the
start of the forward (``nvidia/model.py`` ``prepare_embeddings`` loop), with
``background=True`` (grid capped at half the SMs). The consumer layer waits on
that stream (``_finish_prefetch``) just before its ``wkv`` GEMM.

Each launch gathers ``num_tokens * local_heads`` rows. A row is ``head_dim``
FP8 E4M3 bytes plus ``head_dim / quant_block_size`` UE8M0 scale bytes in a
separate scale table; the kernel dequantizes and writes ``head_dim`` BF16
values per row to a device staging buffer.

Args:

- ``num_tokens``: rows of the hash-id tensor handed to this launch (the
  iteration's scheduled tokens; the DP-gathered token slot under Engram DP).
- ``local_heads``: hash columns this rank owns (``part_n_hash_cols`` =
  ceil((max_ngram - 1) * n_heads / shards); 24 columns on V4.1-Flash, so 6 at
  TP4). Rows gathered per launch = ``num_tokens * local_heads``.
- ``head_dim``: FP8 values per row (256).
- ``quant_block_size``: values per UE8M0 scale byte (32), so a row costs
  ``head_dim + head_dim / quant_block_size`` bytes (264).
- ``table_rows``: rows of this rank's table slice (``part_num_embeddings``;
  ~96.0M on V4.1-Flash TP4, 23.60 GiB). Sets the TLB footprint of the random
  gather.
- ``residency``: ``host_uva`` (pinned host memory read over UVA, launched with
  the production half-SM background grid) or ``device`` (HBM, full-SM grid,
  the ``cpu_offload=False`` path).
- ``weight_dtype``: table element dtype (``fp8_e4m3``).
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

KIND = "engram_lookup"


@dataclass(frozen=True)
class EngramLookupArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows of hashed n-gram ids handed to one call.")
    local_heads: int = arg(unit="heads", doc="Hash columns (heads) this GPU owns.")
    head_dim: int = arg(unit="elements", doc="FP8 values in one table row.")
    quant_block_size: int = arg(unit="elements", doc="Row values sharing one UE8M0 scale.")
    table_rows: int = arg(unit="rows", doc="Rows of this GPU's table slice.")
    residency: str = arg(
        doc=(
            "Where the table lives: host_uva (pinned host memory read over UVA, "
            "half-SM background grid) or device (HBM, full-SM grid)."
        )
    )
    weight_dtype: DType = arg(doc="Element type of the table rows.")


DOC = KernelDoc(
    title="Engram n-gram embedding lookup",
    summary=(
        "Gather hashed n-gram rows from a large FP8 table, dequantize them and "
        "write BF16 rows for the layers that consume them."
    ),
    description=(
        "An Engram layer adds embeddings looked up by hashed token n-grams. One "
        "Triton call gathers num_tokens × local_heads rows from this GPU's slice "
        "of the table. Each row is head_dim FP8 E4M3 values plus one UE8M0 scale "
        "per quant_block_size values in a separate scale table. The call "
        "dequantizes the row to BF16 and writes it to a device staging buffer. "
        "Tables usually stay in pinned host memory and are read over UVA on a "
        "side stream with half the SMs; table_rows sets the TLB footprint of the "
        "random gather."
    ),
    category="Other",
    formula=(
        "out[t, h] = bf16(table[id[t, h]] · 2^(scale[id[t, h]] − 127)), per 32-value block",
        "GB/s = num_tokens · local_heads · [4 + head_dim + head_dim / quant_block_size "
        "+ 2 · head_dim] / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Only the lookup kernel's launches are counted. Each timed launch reads "
        "another id set from a pool of at least about a million distinct rows, "
        "so repeats find no rows left in L2 or the TLBs by an earlier launch. "
        "Before timing, the output must equal a PyTorch gather and dequantize "
        "bit for bit."
    ),
    caveats=(
        "Only head_dim 256, quant_block_size 32, FP8 tables, up to 24 local "
        "heads, 65,536 tokens and 200 million table rows are measured, on B200.",
        "Ids are uniform within each head's slice of the table; real n-gram "
        "hashes may hit some rows more often and cache better.",
        "TFLOPS is not computed.",
    ),
    # The runner's bit-exact check is its own PyTorch gather; no separate module.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.embedding.engram_lookup",
            function_name="profile_engram_lookup_vllm_triton",
        ),
        table_name=KIND,
        args_schema=EngramLookupArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        subprocess_env="vllm_upstream_fork_env",
        doc=BackendDoc(
            summary=(
                "vLLM's Triton _engram_lookup_kernel through ParallelEngramEmbedding.lookup, "
                "with the module's own grid rule."
            ),
        ),
    )
)

__all__ = ["EngramLookupArgs", "KIND"]
