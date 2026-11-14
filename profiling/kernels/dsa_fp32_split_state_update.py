"""DSA decode state update from an fp32 split-GEMM row, in one launch."""

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

KIND = "dsa_fp32_split_state_update"


@dataclass(frozen=True)
class DsaFp32SplitStateUpdateArgs(KernelArgs):
    num_requests: int = arg(unit="requests", doc="Decode requests in the step (one program each).")
    next_n: int = arg(unit="tokens", doc="Rows per request in this step (1 for plain decode).")
    q_lora_cols: int = arg(
        unit="elements", doc="Leading query-latent columns, RMSNormed into bf16."
    )
    kv_lora_cols: int = arg(
        unit="elements", doc="KV-latent columns, RMSNormed and appended to the FP8 MLA cache."
    )
    index_dim: int = arg(
        unit="elements",
        doc="Indexer key columns (LayerNormed) and as many gate-score columns after them.",
    )
    fp32_tail_cols: int = arg(
        unit="elements", doc="Last columns of each row, left in place as fp32 (not read)."
    )
    pool_size: int = arg(unit="tokens", doc="Tokens compressed into one cached index key.")
    tail_ring: int = arg(unit="tokens", doc="Slots in each request's raw-token tail ring.")
    block_size: int = arg(unit="pools", doc="Pooled keys stored in each index-cache page.")


DOC = KernelDoc(
    title="DSA state update from an fp32 split row",
    summary=(
        "From one fp32 GEMM row per token: q/kv RMSNorm, FP8 MLA cache append, indexer "
        "k LayerNorm and the pooled index-cache update, in one launch."
    ),
    description=(
        "When a DSA layer's hidden-state projections run as one GEMM with an fp32 "
        "output (columns [q latent | kv latent | indexer k | gate score | fp32 head "
        "weights]), this one launch writes every decode state that depends on that row. "
        "Row programs round the q latent to bf16 and RMSNorm it into bf16, and round "
        "the kv latent to bf16, RMSNorm it and store it as FP8 (divided by the static "
        "kv scale) at the token's MLA cache slot. One program per request rounds its "
        "next_n indexer-k and gate-score rows to bf16, LayerNorms k (fp32 weight and "
        "bias), writes both, stashes each token in the request's tail ring and "
        "compresses every pool a token completes (softmax over gate score plus a "
        "per-slot bias, Hadamard-128, FP8 with a power-of-two scale) into the "
        "page-planar index cache. Positions are read as int64. Request r's first row "
        "sits at pool phase pool_size - 1 - (r mod pool_size)."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "qn = RMSNorm(bf16(y[:, :Q])); cache[slot] = FP8(bf16(RMSNorm(bf16(y[:, Q:Q+KV]))) / s)",
        "k = bf16(LayerNorm(bf16(y[:, Q+KV:Q+KV+D]))); gs = bf16(y[:, Q+KV+D:Q+KV+2D]); "
        "pooled index update as dsa_pooled_index_cache_update",
        "R = num_requests · next_n; C = completed pools",
        "B = R · [4(Q + KV + 2D) + 2Q + KV + 8D + 20] + C · [(pool_size − 1) · 4D + D + 4]",
        "GB/s = B / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} One launch per call. Before timing, the launch is checked "
        "against the incumbent chain it replaces (the vendored row-split post_a, the "
        "vLLM fork's fused_q_kv_rmsnorm, concat_and_cache_mla in fp8 and the fork's "
        "kpool_decode_update_and_maybe_write_cache_batched on int32 positions): the "
        "normed q, gate scores, MLA cache and index cache / tail ring bit-identical "
        "(the fork kpool fed this launch's k), and k within 1 bf16 ulp of post_a."
    ),
    caveats=(
        "The triton backend has its row geometry compiled in (1536 q, 512 kv, 128 "
        "indexer columns, 8 warps) and refuses any other.",
        "Decode rows only: every request has next_n rows and owns a pooled-cache tail "
        "block; prefill rows keep the separate launches.",
        "On SM100-family GPUs the index-cache scale is four equal UE8M0 bytes, elsewhere "
        "an FP32; the device picks it, so it is not a key field.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_fp32_split_state_update_triton",
            function_name="profile_dsa_fp32_split_state_update_triton",
        ),
        table_name=KIND,
        args_schema=DsaFp32SplitStateUpdateArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # Fixed dtypes (fp32 row in, bf16 outputs, FP8 caches), so no dtype
        # axis. Triton lowers the tl.float8e4nv cache stores only on SM89+.
        supports=BackendSupport(compute=None, min_compute_capability=(8, 9)),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "A repo-vendored Triton kernel (post_plus): 2·rows row programs for the "
                "q/kv norms and cache append, then one program per request for k, gate "
                "and the pooled index update; request programs launch first."
            ),
            url=None,
        ),
    )
)

__all__ = ["DsaFp32SplitStateUpdateArgs", "KIND"]
