"""ShardQ token/head exchange around TP-sharded sparse-MLA prefill (vLLM)."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)
from profiling.runners.comm._multinode import local_rank_count

KIND = "sparse_mla_shard_q"
DIRECTIONS = ("shard_q", "unshard_out")


@dataclass(frozen=True)
class SparseMlaShardQArgs(KernelArgs):
    num_gpus: int = arg(unit="GPUs", doc="GPUs in the tensor-parallel group.")
    num_tokens: int = arg(
        unit="tokens", doc="Prefill tokens of the step, split zigzag across the group."
    )
    num_heads: int = arg(unit="heads", doc="Attention heads one GPU holds under TP.")
    head_dim: int = arg(
        unit="elements",
        doc="Per-head width moved: the query width for shard_q, the value width for unshard_out.",
    )
    dtype: DType = arg(doc="Element type moved (fp8 query, bf16 output).")
    direction: str = arg(doc="shard_q (head shards to token shards) or unshard_out (back).")


DOC = KernelDoc(
    title="Sparse MLA ShardQ exchange",
    summary="All-to-all that moves sparse-MLA prefill between head shards and token shards.",
    description=(
        "Under tensor parallelism vLLM's sparse-MLA prefill (ShardQ) gives each GPU a zigzag "
        "slice of the step's prefill tokens instead of a slice of the heads. shard_q moves the "
        "query from [T, H/tp, D] to [T/tp, H, D] before attention; unshard_out moves the output "
        "from [T/tp, H, D_v] back to [T, H/tp, D_v]. Each call is one Triton kernel that "
        "stores rows directly into the peers' symmetric-memory buffers, between two group "
        "barriers."
    ),
    category="Communication",
    formula=(
        "moved bytes per GPU = num_tokens · (tp − 1)/tp · num_heads · head_dim · itemsize",
        "algbw = busbw = moved bytes / time",
    ),
    default_metric="time_ms",
    method=(
        "vLLM's own shard_q / unshard_out (barrier, push kernel, barrier) on a ShardQ workspace "
        "in symmetric memory; N back-to-back calls timed with CUDA events on each GPU, the "
        "slowest GPU kept."
    ),
    caveats=(
        "The leading barrier also waits for the slowest peer to reach the exchange; measured "
        "alone, that wait is near zero, while in serving it absorbs cross-GPU skew.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_symm_mem",
        supports=BackendSupport(compute=frozenset({DType.BF16, DType.FP8_E4M3})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.sparse_mla_shard_q_vllm",
            function_name="profile_sparse_mla_shard_q_batch",
        ),
        table_name=KIND,
        args_schema=SparseMlaShardQArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        imex_channel=True,
        # A group larger than one node (TP16 over GB200 trays) reserves its share per node.
        gpu_count_fn=lambda spec: local_rank_count(int(spec["num_gpus"]), "nvlink"),
        list_native=True,
        doc=BackendDoc(
            summary=(
                "vLLM ShardQ over torch symmetric memory: one Triton push kernel per exchange "
                "between two symmetric-memory barriers."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/shardq.py",
        ),
    )
)
