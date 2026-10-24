"""Speculative-tree verify attention kernel kind.

All Python-side per-kernel knowledge for ``tree_verify_attention`` lives here:
the wire string ``KIND``, the ``TreeVerifyAttentionArgs`` schema, and the
``register(...)`` call that wires the kind into ``profiling.db.registry``.

Source: TensorFold (perf-study 6469730a) verifying DFlash2 draft trees.
``families/qwen3_5/cuda/forward.py`` ``multi_tree_forward`` puts every decoding
stream's window (a best-first draft tree, or a copy chain) into ONE forward;
each softmax-attention layer makes one call of
``tensorfold.cuda.kernels.attention.attention(q, k_nodes, v_nodes, offs, plan,
scale=)``. Query row ``r`` of stream ``s`` attends to

- all ``committed_len`` keys of stream ``s``'s own contiguous bf16 cache, and
- the window keys on its own root-to-row path (its ancestors and itself), which
  sit in ``k_nodes`` / ``v_nodes`` and are not yet in the cache.

The call issues three Triton launches over all streams together: ``_shared``
(committed keys in 512-key chunks; four chunks fold into one program when that
stream's whole 2048-key groups times its rows reach 64), ``_tail`` (each row's
last committed chunk plus its own path) and ``_merge`` (folds every partial in
a fixed order). The whole call is the L1 unit. The plan (``plan`` /
``plan_host``: host work plus one ``_paths`` launch, shared by every attention
layer of the forward) is not part of it.

``query_context_pairs`` is the exact launch: ``(window rows, committed keys)``
per stream, in stream order. Unlike ``fa4_varlen_attention`` the second number
excludes the window, since the window's keys are read from the node tensors,
not the cache. The tree's parent structure is not a field: it only changes
which ``_tail`` key loads are masked, and on GB10 a chain, a star, a binary
tree and a 4-wide tree of the same rows measure within repeat noise (1-16
streams, 4-128 rows, 1k-60k keys). The runner times chains.

Reuse verdict (new kind):

- ``flashinfer_attn_decode`` / ``paged_attn_batch_decode``: one query token per
  request over a paged cache, keyed by batch totals. No multi-row window, no
  per-row mask inside the window.
- ``flashinfer_attn_rect``: one request, every query sees every key. No
  committed prefix plus tree mask, one stream only.
- ``fa4_varlen_attention``: several requests in one call, but each query row
  attends causally (or in a sliding window) over a paged cache that already
  holds its own row's key. A tree row sees its ancestors, not every earlier
  window row, and the window keys come from separate node tensors.

None of these can carry "committed prefix plus own tree path, across several
streams in one launch" without changing what its fields mean.
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

KIND = "tree_verify_attention"


@dataclass(frozen=True)
class TreeVerifyAttentionArgs(KernelArgs):
    num_qo_heads: int = arg(unit="heads", doc="Query and output heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads on this GPU.")
    head_dim: int = arg(unit="elements", doc="Dimension of each query, key and value head.")
    q_dtype: DType = arg(doc="Query element type.")
    kv_dtype: DType = arg(doc="Key and value element type, cache and window alike.")
    o_dtype: DType = arg(doc="Output element type.")
    query_context_pairs: tuple[tuple[int, int], ...] = arg(
        doc=(
            "(window rows, committed keys) per stream in the one call; the "
            "committed keys exclude the window."
        )
    )


DOC = KernelDoc(
    title="Speculative-tree verify attention",
    summary=(
        "GQA attention of several streams' draft-tree windows in one call: each "
        "window row sees its stream's committed keys and its own root-to-row path."
    ),
    description=(
        "One call verifies the draft windows of every decoding stream. A window "
        "row attends to all committed keys of its stream's contiguous cache and "
        "to the window keys of its ancestors and itself. TensorFold splits the "
        "committed keys into 512-key chunks shared by all of a stream's rows and "
        "heads of one KV head, attends each row's last chunk and own path "
        "separately, and merges the partial results in a fixed order."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "O_r = softmax(Q_r · [K_committed(s); K_path(r)]ᵀ · scale) · [V_committed(s); V_path(r)]",
        "TFLOPS = 4 · num_qo_heads · head_dim · Σ_s rows_s · (committed_s + mean path) / time",
        "GB/s = (Q + O + window K and V + each stream's committed K and V once) bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "The time is the sum of the call's three launches (_shared, _tail, "
        "_merge). The plan is built before timing, as the forward builds it "
        "once for all its attention layers. Each stream's window is a chain. "
        "Before timing, the output is checked against a PyTorch reference with "
        "an explicit tree mask on a small shape."
    ),
    caveats=(
        "Queries, keys and values are random; each stream has its own cache.",
        "Tree structure is not a field: same-row trees measure within noise.",
        "The plan's host work and _paths launch are charged elsewhere.",
    ),
    reference="profiling.runners.attention.tree_verify_attention_reference",
)


# The three kernels are Triton (bf16 tl.dot), launched straight from
# tensorfold/cuda/kernels/attention.py without TensorFold's CUDA extension
# loader, so the floor is bf16 MMA (sm_80). On sm_12x the module picks its
# GB10 tiles (QUERY_TILE 32, 2 stages); elsewhere 0.6.6's (16, 1 stage).
# attention() takes head_dim 128 or 256 and bf16 q / node keys / caches only.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tensorfold_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.tree_verify_attention_tensorfold",
            function_name="profile_tree_verify_attention_tensorfold_triton",
        ),
        table_name=KIND,
        args_schema=TreeVerifyAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's Triton tree attention (_shared + _tail + _merge), "
                "tensorfold.cuda.kernels.attention.attention as "
                "multi_tree_forward calls it."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)

__all__ = ["KIND", "TreeVerifyAttentionArgs"]
