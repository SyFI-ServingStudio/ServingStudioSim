"""FlashInfer MNNVL two-sided MoE all-to-all kernel kind.

Wire string: ``"moe_alltoall"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/moe_alltoall.rs``.

This is the transfer vLLM actually runs for expert parallelism when started with
``--all2all-backend=flashinfer_nvlink_two_sided``: ``MnnvlMoe.mnnvl_moe_alltoallv``
on the way out and ``mnnvl_moe_alltoallv_combine`` on the way back, one
``moeAllToAllKernel`` launch each. It is NOT a point-to-point send: every rank
sends and receives simultaneously, the rows it moves are gathered by index out
of a workspace staged in fabric memory, and the achieved bandwidth is
correspondingly well below a two-rank NCCL ``send``/``recv``. Pricing it off the
``p2p_intra`` curve under-predicted a measured 8k-token GLM-5.2 prefill layer by
~1.4x even after the payload width was corrected, which is why this kind exists.

The measured call is ``moe_comm`` itself, not the Python wrapper: combine's
wrapper also zero-fills a ``token_count x top_k`` staging buffer and reduces
over the top-k axis, and both are token-count work that would have forced a
third sweep axis. The arch prices those as elementwise leaves instead.

The two sweep axes are the busiest rank's send and receive row counts. They are
independent — the load on a rank comes from both sides at once, and dispatch and
combine transpose them — but their ratio is bounded by ``ep_size`` because every
row sent is a row received. The Rust ``infeasible_mask`` strips the corner that
bound forbids.

``direction`` is a row key rather than a separate kind because dispatch and
combine are the same kernel; what differs is the read-side reuse (dispatch reads
one token row once per destination it reaches, combine reads each expert-output
row exactly once), so one table keeps the pair comparable side by side.

``metric_family=COMM`` (algbw/busbw, no tflops); ``gpu_count_fn`` reserves the
whole EP group, as ``all_reduce`` does for ``num_gpus``. The runner is referenced
lazily so the main process never eager-imports torch or flashinfer.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import KernelArgs
from profiling.db.doc import BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "moe_alltoall"


@dataclass(frozen=True)
class MoeAlltoallArgs(KernelArgs):
    # Ranks in the expert-parallel group. Every one of them is a real GPU the
    # launcher must reserve, and the kernel asserts the workspace spans them all.
    ep_size: int = arg(unit="GPUs", doc="GPUs in the expert-parallel group.")
    # Rows leaving the busiest sender, and arriving at the busiest receiver. A
    # collective ends when its slowest rank does, and a rank is loaded from both
    # sides at once; `sum(send) == sum(recv)` bounds their ratio by `ep_size`.
    max_send_rows: int = arg(unit="rows", doc="Rows sent by the busiest GPU.")
    max_recv_rows: int = arg(unit="rows", doc="Rows received by the busiest GPU.")
    # Experts each token selects, and the global slot count they are drawn from.
    # Together they fix the fan-out: how many distinct ranks a token reaches,
    # which is what the runner reproduces in the send indices' reuse pattern.
    top_k: int = arg(unit="experts", doc="Expert slots selected per token.")
    slot_count: int = arg(unit="experts", doc="Expert slots across the group.")
    # One row's payload. bf16 hidden for the vLLM block-scale path, which defers
    # activation quantisation until after the transfer.
    hidden_bytes: int = arg(unit="bytes", doc="Bytes in one transferred row.")
    # "dispatch" or "combine".
    direction: str = arg(doc="Transfer direction: dispatch or combine.")
    # Fabric token (matches Rust `Fabric` serde wire form). A row/cache key only:
    # MNNVL is NVLink by construction.
    fabric: str = arg(doc="Interconnect label used to identify the measurement.")


DOC = KernelDoc(
    title="MoE all-to-all transfer",
    summary="Move token rows to expert GPUs or return expert outputs to their source GPUs.",
    description=(
        "In expert-parallel MoE, dispatch sends each token's row to the GPUs that "
        "own its selected experts, and combine sends the expert outputs back. vLLM "
        "runs both legs as this one transfer under "
        "--all2all-backend=flashinfer_nvlink_two_sided. The transfer ends when its "
        "slowest GPU ends, so the two row counts are those of the busiest sender "
        "and the busiest receiver. The legs differ in how they read their input: "
        "dispatch reads a token row once for every GPU it goes to, and combine "
        "reads each expert-output row once."
    ),
    category="Communication",
    formula=(
        "payload bytes = max_send_rows · hidden_bytes",
        "algbw = payload bytes / time",
        "busbw = algbw",
    ),
    default_metric="time_ms",
    method=(
        "Mean time per call from CUDA events around 20 replays of a CUDA graph "
        "that captures 4 to 100 moe_comm calls, fewer for larger transfers. Ten "
        "eager calls and two replays warm up first. Rank 0's time is kept."
    ),
    caveats=(
        "Combine's zero fill and top-k sum, which vLLM runs around the transfer, "
        "are not included; the simulator prices them as separate elementwise "
        "kernels.",
        "The transfer indices are constructed from the requested row counts, "
        "rather than sampled from a router's expert choices.",
        "Bandwidth counts rank 0's sent payload, not traffic across all GPUs.",
    ),
    # No separate PyTorch reference implements this fabric-memory collective.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_mnnvl",
        # Comm is size-keyed; the payload width is carried explicitly as
        # `hidden_bytes` rather than as a dtype.
        supports=BackendSupport(
            compute=None,
            gpus=frozenset({"NVIDIA H100", "NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.flashinfer_mnnvl_alltoall",
            function_name="profile_moe_alltoall_batch",
        ),
        table_name=KIND,
        args_schema=MoeAlltoallArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        gpu_count_fn=lambda spec: int(spec["ep_size"]),
        # list_native: one rank-group spawn (plus one fabric-memory workspace
        # allocation, which is the expensive part) serves the whole chunk.
        list_native=True,
        doc=BackendDoc(
            summary=(
                "FlashInfer's MNNVL moe_comm: every GPU sends and receives indexed "
                "rows through a workspace in fabric memory."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/trtllm_alltoall.py",
        ),
    )
)
