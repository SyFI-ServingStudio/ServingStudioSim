"""Gated DeltaNet state replay: commit accepted rows into the recurrent states.

All Python-side per-kernel knowledge for ``gdn_state_replay`` lives here: the
wire string ``KIND``, the ``GdnStateReplayArgs`` schema, and the
``register(...)`` call that wires the kind into ``profiling.db.registry``.

Source: TensorFold (perf-study 6469730a) committing a verify round.
``families/qwen3_5/cuda/forward.py`` ``_commit`` (via ``commit_streams`` from
``multi.py`` and ``commit`` from ``decode.py``) makes ONE call of
``tensorfold.cuda.kernels.gdn.replay(table, layers, streams, rows, counts, k0,
v0, in_place=True)`` for every GDN layer and every stream. The verify forward
saved each window row's ``k`` (bf16, L2-normalized), ``v`` (bf16), ``g`` (the
exponentiated decay, fp32) and ``beta`` (fp32) per layer; it never wrote the
committed states. ``replay_kernel`` (gdn.cu) re-runs the gated delta rule from
each stream's committed fp32 state over that stream's accepted path, the rows
gathered by index, and writes the state back. No query, no output. The grid is
(value rows / 32, value heads, layers x streams): every (stream, layer) state is
read once and written once, and each block walks the accepted rows serially.

Cost on GB10 (Qwen3.8-27B GDN, 48 layers):

- linear in layers and in streams: the launch is just layers x streams
  independent state sets (0.036 / 0.44 / 0.88 / 1.75 ms at 1 / 12 / 24 / 48
  layers, one stream of 16 rows);
- a stream of 1-4 rows reads and writes its 48 states (302 MB) at about
  225 GB/s (1.32 ms); past that the serial walk adds about 0.035 ms a row a
  stream (16 rows 1.72 ms, 128 rows 5.9 ms);
- a ragged batch measures within 0-3% of a uniform batch with the same streams
  and total rows (2-16 streams, rows 1-16 mixed), so the key is the totals
  (``batch_size``, ``num_tokens``), not the per-stream lengths.

``num_layers`` is an arg because the one launch spans every GDN layer the GPU
holds; the time scales with it.

Reuse verdict (new kind):

- ``gdn_recurrent_decode``: one layer per call, reads a query and writes one
  output token per request, ``num_tokens`` equal per request. The replay has no
  query or output, gathers rows by index from a saved window, and one launch
  spans every layer.
- ``gdn_chunk_delta_rule``: a prompt chunk, one layer, outputs per row.
- ``indexed_row_copy`` / ``elementwise``: copies or pointwise maps; the replay
  is a serial recurrence over a 128 x 128 state per value head.
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

KIND = "gdn_state_replay"


@dataclass(frozen=True)
class GdnStateReplayArgs(KernelArgs):
    num_layers: int = arg(unit="layers", doc="Gated DeltaNet layers the one launch commits.")
    batch_size: int = arg(
        unit="streams", doc="Streams committed together, each from its own states."
    )
    num_qk_heads: int = arg(unit="heads", doc="Key heads per row.")
    num_value_heads: int = arg(unit="heads", doc="Value heads and recurrent states per layer.")
    key_head_dim: int = arg(unit="elements", doc="Features in each key head.")
    value_head_dim: int = arg(unit="elements", doc="Features in each value head.")
    dtype: DType = arg(doc="Element type of the saved keys and values.")
    state_dtype: DType = arg(doc="Element type of the recurrent state and gates.")
    num_tokens: int = arg(
        unit="tokens",
        doc=(
            "Accepted rows replayed across all streams (each stream at least one); "
            "a ragged split times like an even one."
        ),
    )


DOC = KernelDoc(
    title="State replay",
    summary=(
        "Fold each stream's accepted rows into its Gated DeltaNet states, every layer "
        "and stream in one launch."
    ),
    description=(
        "After a speculative verify round, each stream's accepted path is committed "
        "into its recurrent states. The verify pass saved every window row's key, "
        "value, decay and beta for each layer; the replay re-runs the gated "
        "delta-rule update from the committed FP32 state over the accepted rows, "
        "gathered by index, and writes the state back. It reads no query and "
        "produces no output."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        (
            "for each layer, stream and accepted row t: "
            "S = g_t·S; S = S + k_t·(beta_t·(v_t − kᵀ_t S))ᵀ"
        ),
        "E₁ = num_value_heads·key_head_dim·value_head_dim",
        "FLOPs = 6·num_layers·num_tokens·E₁",
        (
            "bytes = num_layers·(2·batch_size·E₁·bytes(state_dtype) + "
            "num_tokens·(num_qk_heads·key_head_dim + "
            "num_value_heads·value_head_dim)·bytes(dtype) + "
            "2·num_tokens·num_value_heads·bytes(state_dtype))"
        ),
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "The time is the one replay_kernel launch of a gdn.replay(..., in_place=True) "
        "call. The pointer table, row indices and counts are on the device before "
        "timing, as _commit builds them. Rows split as evenly as possible across "
        "streams; each stream's window holds max(16, rows) rows and its path is the "
        "first rows of it. Before timing, a ragged guard batch is checked against an "
        "FP32 PyTorch delta-rule reference."
    ),
    caveats=(
        "Every timed call overwrites the states in place; the update is contractive, "
        "so they stay finite.",
        "The conv-window and attention KV copies of the same commit are separate launches.",
        "Ragged batches measure within 3% of even ones; the per-stream split is not a field.",
    ),
    reference="profiling.runners.attention.gdn_state_replay_reference",
)


# gdn.cu replay_kernel: plain fp32 CUDA with bf16 loads, built by
# tensorfold.cuda.build for the GPU present, which refuses anything under its
# MIN_CAPABILITY (8, 9). gdn.cpp replay and gdn.py replay_table take fp32
# (Hv, Dv, 128) states (DK = 128 hard-coded); the keys are bf16 in production
# (glue.gdn_pre), fp32 keys are a debug path not measured here.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tensorfold_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 9),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_state_replay_tensorfold",
            function_name="profile_gdn_state_replay_tensorfold_cuda",
        ),
        table_name=KIND,
        args_schema=GdnStateReplayArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's CUDA replay_kernel (tensorfold.cuda.kernels.gdn.replay), "
                "one launch for every GDN layer and stream, as _commit calls it."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)

__all__ = ["KIND", "GdnStateReplayArgs"]
