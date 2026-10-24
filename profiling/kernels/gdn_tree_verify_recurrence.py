"""Gated DeltaNet recurrence over speculative draft trees (verify).

All Python-side per-kernel knowledge for ``gdn_tree_verify_recurrence`` lives here:
the wire string ``KIND``, the ``GdnTreeVerifyRecurrenceArgs`` schema, and the
``register(...)`` call.

Source: TensorFold (perf-study 6469730a) verifying DFlash2 draft trees.
``families/qwen3_5/cuda/forward.py`` ``multi_tree_forward`` (and the single-stream
``tree_forward``) puts every decoding stream's window into one forward; each GDN
layer makes one call of ``tensorfold.cuda.kernels.gdn.tree(q, k, v, g, beta, plan,
table=...)``, one ``tree_kernel`` launch (``gdn.cu``) over all streams (grid z =
streams). For every window row of every stream it steps the fp32 gated delta rule
along the row's root-to-row path, starting from the stream's committed state (read
through a device pointer table, never written), and writes the row's bf16 read-out.
q, k, v, g and beta come from the separate Triton ``_gdn_pre`` launch (conv, SiLU,
L2 norm, gates); the output goes to the separate ``_gated_norm``.

Slot class
----------
The kernel's cost depends on the tree shapes, not only on rows: TensorFold's
``schedule`` (``gdn.py``) needs some number of shared-memory states ("slots") per
stream, and ``dispatch_tree`` compiles one variant from the largest count in the
launch: 0 (every window a chain) runs 8 value rows a warp; 1-2 slots 4 rows; 3-4,
5-8 two rows with 4 warps; 9-16 two warps; 17-32 one warp. Same rows, same streams,
GB10: a 1-2 slot launch costs 16-35% more than chains, a 3-4 slot launch about 80%
more; shapes within one class measure within about 6%. ``slot_class`` is the class
of the launch's largest slot count, so one bushy stream sets it for the whole launch.
The rule from a parents list to the slot count is written out, and implemented
without Torch, in ``profiling.runners.attention.gdn_tree_verify_recurrence_reference``
(``slot_count_for_parents`` / ``slot_class_for_parents``). On sm_120 GPUs with at
least 96 SMs, ``dispatch_tree`` splits ``slots_1_2`` into two variants for several
streams; GB10 (sm_121) never takes that branch.

Rows
----
``num_tokens`` counts window rows over all streams; the runner splits them as evenly
as possible (the first ``num_tokens % batch_size`` streams take one row more); when
no stream would then hold the class's smallest tree, one stream takes that tree and
the rest split the remainder. On GB10 a ragged launch costs what a uniform one with
the same total rows does (8 streams: 4 x 16 + 4 x 4 rows 200 us, 8 x 10 rows 201 us;
one 16-row tree + 7 x 4 rows within 5% of the uniform line through 80 and 128 rows).

Reuse verdict (new kind):

- ``gdn_recurrent_decode``: its call includes the L2 norm and gate computation that
  TensorFold does in ``_gdn_pre``; a verify step (``num_tokens``) there writes one
  state checkpoint per token, while this call writes no state; and its rows form
  equal chains, so it cannot say what a tree costs.
- ``gdn_chunk_delta_rule``: a prefill chain (or chunked scan) over sequences that
  writes the final state; no tree paths, no per-stream committed state table.
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

KIND = "gdn_tree_verify_recurrence"


@dataclass(frozen=True)
class GdnTreeVerifyRecurrenceArgs(KernelArgs):
    batch_size: int = arg(unit="streams", doc="Streams verified in the one launch.")
    num_tokens: int = arg(
        unit="tokens", doc="Window rows (tree nodes) over all streams, split as evenly as possible."
    )
    num_qk_heads: int = arg(unit="heads", doc="Query and key heads per row.")
    num_value_heads: int = arg(unit="heads", doc="Value heads and recurrent states per stream.")
    key_head_dim: int = arg(unit="elements", doc="Features in each query or key head.")
    value_head_dim: int = arg(unit="elements", doc="Features in each value head.")
    dtype: DType = arg(doc="Element type of q, k, v and the output.")
    state_dtype: DType = arg(doc="Element type of the committed states and the gates.")
    slot_class: str = arg(
        doc=(
            "Kernel variant from the largest slot count of any stream's tree: chain "
            "(0: every window a chain), slots_1_2, slots_3_4, slots_5_8, slots_9_16, "
            "slots_17_32."
        )
    )


DOC = KernelDoc(
    title="Tree verify recurrence",
    summary=(
        "Step the gated delta rule along every draft-tree row's path from its "
        "stream's committed state and read one output per row, all streams in one launch."
    ),
    description=(
        "During speculative verify, each stream's window is a draft tree. Every row "
        "starts from its parent's state (the root from the stream's committed FP32 "
        "state), decays it, writes its key-value pair with strength beta, and reads "
        "the result with its query. The committed states are only read; the commit "
        "step replays the accepted path later. Query and key heads are already "
        "L2-normalized and the gates already computed by a separate launch."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "S_r = g_r·S_parent(r) + k_r·(beta_r·(v_r − k_rᵀ(g_r·S_parent(r))))ᵀ; y_r = q_rᵀS_r",
        "T = num_tokens, E₁ = num_value_heads·key_head_dim·value_head_dim",
        "FLOPs = 7·T·E₁",
        (
            "bytes = T·(2·num_qk_heads·key_head_dim + 2·num_value_heads·value_head_dim)"
            "·bytes(dtype) + (2·T·num_value_heads + batch_size·E₁)·bytes(state_dtype)"
        ),
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Only tree_kernel is timed. Each stream's tree is the class's canonical tree "
        "(a seed tree that needs the class's lowest slot count, then a chain tail), a "
        "chain for streams with fewer rows than the seed; the plan and pointer table "
        "are built before timing, as the forward builds them once for all its layers. "
        "After the L2 flush, each timed call rewrites q, k, v, g and beta (the writes "
        "of _gdn_pre, which runs right before in the forward) so they are L2-hot while "
        "the committed states come from DRAM; a fully cold L2 prices the launch 7-60% "
        "above captured verify rounds. "
        "Before timing, a small multi-stream launch of the same class is checked "
        "against a sequential FP32 PyTorch reference."
    ),
    caveats=(
        "Inputs are random with _gdn_pre's ranges (unit q and k, g in (0, 1), beta a sigmoid).",
        "Within a slot class, shapes measure within about 6% (a star's rows read "
        "their parent from shared memory, a chain's from registers).",
        "Ragged streams are priced by total rows; GB10 measures them so.",
        "The plan's host work and its copy to the device are charged elsewhere.",
    ),
    reference="profiling.runners.attention.gdn_tree_verify_recurrence_reference",
)


# gdn.cu's tree_kernel is plain CUDA (bf16 loads, warp shuffles, fp32 math), built
# by tensorfold.cuda.build.load, which refuses a GPU below compute capability 8.9
# (MIN_CAPABILITY). gdn.cpp `tree` takes bf16 (or fp32) q/k, bf16 v, fp32 g/beta and
# states, and 128-wide key heads; the runner measures the bf16 path production uses.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tensorfold_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 9),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_tree_verify_recurrence_tensorfold",
            function_name="profile_gdn_tree_verify_recurrence_tensorfold_cuda",
        ),
        table_name=KIND,
        args_schema=GdnTreeVerifyRecurrenceArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's CUDA tree_kernel, tensorfold.cuda.kernels.gdn.tree with a "
                "state pointer table, as multi_tree_forward calls it."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)

__all__ = ["KIND", "GdnTreeVerifyRecurrenceArgs"]
