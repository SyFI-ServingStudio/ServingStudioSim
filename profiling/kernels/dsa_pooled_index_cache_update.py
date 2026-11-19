"""DSA pooled index-key decode/verify cache update kernel kind."""

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

KIND: str = "dsa_pooled_index_cache_update"


@dataclass(frozen=True)
class DsaPooledIndexCacheUpdateArgs(KernelArgs):
    num_requests: int = arg(unit="requests", doc="Requests decoded together.")
    next_n: int = arg(unit="tokens", doc="New tokens per request in this step.")
    pool_size: int = arg(unit="tokens", doc="Tokens compressed into one cached index key.")
    tail_ring: int = arg(unit="tokens", doc="Slots in each request's raw-token tail ring.")
    index_dim: int = arg(unit="elements", doc="Elements in each index key.")
    block_size: int = arg(unit="pools", doc="Pooled keys stored in each cache page.")
    input_dtype: DType = arg(doc="Element type of the new keys, gate scores and tail ring.")
    cache_dtype: DType = arg(doc="Element type of the cached pooled keys.")


DOC = KernelDoc(
    title="Pooled indexer key cache update",
    summary=(
        "Stash each decode/verify token's index key in its request's tail ring and "
        "compress every pool it completes into the paged FP8 index cache."
    ),
    description=(
        "A pooled DSA indexer caches one key per pool_size consecutive tokens. "
        "During decode and speculative verify, one program per request walks its "
        "next_n new tokens in position order. Every token's BF16 key and gate score "
        "go into the request's tail ring; a token that closes a pool first combines "
        "the pool's keys with a per-element softmax over gate score plus a learned "
        "per-slot bias, rotates the result with a 128-point Hadamard transform and "
        "stores it as FP8 with one power-of-two scale in a page-planar cache page. "
        "Request r's first new token sits at pool phase pool_size - 1 - (r mod "
        "pool_size), so request 0 always completes a pool and the requests cover "
        "every phase."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "pooled = Σ_slot softmax_slot(score + ape) · key; cache = FP8(H128 · pooled / s), "
        "s = 2^ceil(log2(max|H128 · pooled| / 448))",
        "C = completed pools; R = num_requests · next_n",
        "B = R · 8 · index_dim (BF16 key + score in, and into the ring) + C · (pool_size − 1) "
        "· 4 · index_dim (ring reads) + C · (index_dim + 4) + R · 12",
        "GB/s = B / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} The call is one Triton launch, "
        "_kpool_decode_update_batched_kernel; operand construction and the check "
        "against the PyTorch reference run before timing."
    ),
    caveats=(
        "Time is set by the longest program, so it follows next_n and the pool "
        "phase; with next_n < pool_size the rows assume a request that completes a "
        "pool in this step (one in pool_size steps for a lone request).",
        "On SM100-family GPUs the scale is stored as four equal UE8M0 bytes, "
        "elsewhere as an FP32; the device picks it, so it is not a key field.",
        "GB/s counts logical key, score, ring, cache and index bytes, not physical transactions.",
    ),
    reference="profiling.runners.attention.dsa_pooled_index_cache_update_reference",
)


# Triton lowers the tl.float8e4nv cache store only on SM89+; the fork imports the
# module from vllm.models.glm5next.nvidia.ops.kpool_compress.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_fork_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            min_compute_capability=(8, 9),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_pooled_index_cache_update",
            function_name="profile_dsa_pooled_index_cache_update_vllm_fork_triton",
        ),
        table_name=KIND,
        args_schema=DsaPooledIndexCacheUpdateArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "The vLLM fork's kpool_decode_update_and_maybe_write_cache_batched: "
                "one Triton program per request over its next_n tokens."
            ),
        ),
    )
)

__all__ = ["DsaPooledIndexCacheUpdateArgs", "KIND"]
