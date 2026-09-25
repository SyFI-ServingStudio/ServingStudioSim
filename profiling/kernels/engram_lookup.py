"""DeepSeek V4.1 Engram n-gram row lookup (FP8 rows + UE8M0 scales -> BF16).

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

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
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
    num_tokens: int
    local_heads: int
    head_dim: int
    quant_block_size: int
    table_rows: int
    residency: str
    weight_dtype: DType


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
        subprocess_env="vllm_fork_env",
    )
)

__all__ = ["EngramLookupArgs", "KIND"]
