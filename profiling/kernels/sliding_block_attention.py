"""Block attention over a sliding-window context, several streams per launch.

All Python-side per-kernel knowledge for ``sliding_block_attention`` lives
here: the wire string ``KIND``, the ``SlidingBlockAttentionArgs`` schema, and
the ``register(...)`` call that wires the kind into ``profiling.db.registry``.

Source: TensorFold (perf-study 6469730a) drafting with DFlash2.
``families/qwen3_5/cuda/dflash2.py`` ``_layer_fast`` runs every drafter layer
over all live streams' blocks at once (``launch_blocks``: ``block_rows`` rows a
stream, the pending token then mask tokens) and makes ONE call of
``draft_attention.block_attention(q, k, v, keys, values, length, window, scale,
causal)``. Query row ``r`` of stream ``s`` attends to

- the keys of stream ``s``'s own contiguous context ``(num_kv_heads, n_s,
  head_dim)`` (the last ``window`` committed taps, at most), except the first
  ones a row's window drops (``n_s + r - key < window + 1``), and
- every key of its own block (or the earlier ones when ``causal``; the
  checkpoint's ``is_causal`` is false).

One ``_block_attention`` Triton launch, grid (streams, kv heads): each program
holds a KV head's query group x block rows and walks the stream's context in
64-key tiles, then the block.

Cost on GB10 (32 q / 8 kv heads x 128, bf16, 16 rows, window 2047), one call:
linear in the context (one stream 7.5 us at 16 keys to 89 us at 2,047) and
rising with streams past the 48 SMs (8 streams 323 us, 16 streams 605 us at
2,047 keys). A ragged batch (half 2,047, half 256 keys) measured within 1% of
the even batch at the mean context, so the key is the mean.

Reuse verdict (new kind):

- ``flashinfer_attn_rect``: one request, contiguous K/V holding every key, no
  sliding-window mask, no streams in one launch.
- ``tree_verify_attention``: a row sees its committed keys and its own tree
  path (ancestors); here a row sees its whole block and a window-clipped
  context held in a separate tensor per stream. The mask and the key storage
  both differ.
- ``fa4_varlen_attention`` / ``flashinfer_attn_prefill``: causal or windowed
  attention over a paged cache that already holds the query rows' keys.
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

KIND = "sliding_block_attention"


@dataclass(frozen=True)
class SlidingBlockAttentionArgs(KernelArgs):
    num_qo_heads: int = arg(unit="heads", doc="Query and output heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads on this GPU.")
    head_dim: int = arg(unit="elements", doc="Dimension of each query, key and value head.")
    dtype: DType = arg(doc="Element type of queries, keys, values and output.")
    block_rows: int = arg(unit="tokens", doc="Query rows of each stream's block.")
    window: int = arg(unit="tokens", doc="Context keys a query row may see at most.")
    causal: bool = arg(doc="Whether a block row sees only the block rows up to itself.")
    batch_size: int = arg(unit="streams", doc="Streams whose blocks one launch attends.")
    context_len: int = arg(
        unit="tokens", doc="Context keys of each stream (at most window, at least 1)."
    )


DOC = KernelDoc(
    title="Sliding block attention",
    summary=(
        "GQA attention of several streams' query blocks in one launch: each block "
        "row sees its stream's sliding-window context and its own block."
    ),
    description=(
        "A block-parallel drafter (DFlash2) drafts a block of positions per stream "
        "at once. Each block row attends to its stream's context keys, held in a "
        "separate contiguous tensor per stream and clipped to the window, and to "
        "the keys of every row of its block (or the earlier rows when causal). The "
        "measurement gives every stream the same context length."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "O_r = softmax(Q_r · [K_ctx(s); K_block(s)]ᵀ · scale) · [V_ctx(s); V_block(s)]",
        "TFLOPS = 4 · batch_size · block_rows · num_qo_heads · head_dim · "
        "(context_len + block_rows) / time",
        "GB/s = (Q + O + block K and V + each stream's context K and V once) bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "The time is the call's one _block_attention launch; the call's small "
        "pointer/length table copy is a memcpy. Inputs are random. Before timing, a "
        "ragged guard batch is checked against a PyTorch masked-softmax reference."
    ),
    caveats=(
        "Every stream has the same context length; a ragged batch is priced at its "
        "mean, which measured within 1% on GB10.",
    ),
    reference="profiling.runners.attention.sliding_block_attention_reference",
)


# _block_attention is Triton with bf16 tl.dot (sm_80 floor), launched from
# tensorfold.families.qwen3_5.cuda.draft_attention without TensorFold's CUDA
# extension loader. block_attention() refuses non-bf16 contexts; tl.arange needs
# a power-of-two head_dim.
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
            module_name="profiling.runners.attention.sliding_block_attention_tensorfold",
            function_name="profile_sliding_block_attention_tensorfold_triton",
        ),
        table_name=KIND,
        args_schema=SlidingBlockAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's Triton _block_attention (families/qwen3_5/cuda/"
                "draft_attention.py block_attention), one launch for every stream, as "
                "the DFlash2 drafter's layers call it."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)

__all__ = ["KIND", "SlidingBlockAttentionArgs"]
