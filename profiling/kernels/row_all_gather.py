"""Variable-shard all-gather of bf16 rows within a GPU group."""

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

KIND = "row_all_gather"


@dataclass(frozen=True)
class RowAllGatherArgs(KernelArgs):
    num_gpus: int = arg(unit="GPUs", doc="GPUs in the group.")
    per_rank_tokens: tuple[int, ...] = arg(
        unit="tokens", doc="Rows each GPU contributes, in rank order."
    )
    hidden_size: int = arg(unit="elements", doc="Elements in each row.")
    dtype: DType = arg(doc="Element type of the rows; bf16 is measured.")
    fabric: str = arg(doc="Interconnect label; these backends require nvlink.")


DOC = KernelDoc(
    title="Row all-gather",
    summary="Every GPU of a group receives every GPU's rows.",
    description=(
        "GPU r contributes per_rank_tokens[r] rows of hidden_size elements and every GPU "
        "receives all sum(per_rank_tokens) rows in rank order (all-gatherv semantics). A "
        "data-parallel group that runs one op sharded over its members (for example a "
        "tensor-parallel attention block inside data-parallel attention) gathers its "
        "members' inputs this way before the op."
    ),
    category="Communication",
    formula=(
        "row bytes = hidden_size · 2 (bf16)",
        "received bytes = max over GPUs of (sum(per_rank_tokens) − own rows) · row bytes",
        "algbw = busbw = max(received bytes, sent bytes) / time",
    ),
    default_metric="time_ms",
    method=(
        "N calls captured in one CUDA graph and replayed back to back; CUDA-event time per "
        "call on each GPU, the slowest GPU kept. Every shape is first checked bit for bit "
        "against a Torch concatenation of the GPUs' rows."
    ),
    caveats=(
        "Each group is measured alone on num_gpus GPUs; in serving, the other groups of the "
        "node run their gathers at the same time.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="mnnvl_lamport",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.moe_ep_mnnvl_lamport",
            function_name="profile_all_gather_batch",
        ),
        table_name=KIND,
        args_schema=RowAllGatherArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        imex_channel=True,
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
        doc=BackendDoc(
            summary=(
                "One kernel launch: every GPU stores its rows directly into every peer's landing "
                "rows in fabric (MNNVL) memory; each GPU copies the arrived rows out "
                "(Lamport-style arrival, one round marker per peer, no separate barrier)."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/comm/moe_alltoall_lamport.cuh",
        ),
    )
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="mnnvl_pull",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.moe_ep_mnnvl_lamport",
            function_name="profile_all_gather_pull_batch",
        ),
        table_name=KIND,
        args_schema=RowAllGatherArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        imex_channel=True,
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
        doc=BackendDoc(
            summary=(
                "One kernel launch, reader side only: every GPU's rows already sit in its own "
                "fabric-mapped buffer (the producer wrote them there); after a ready flag per "
                "peer, each GPU reads every row from its owner. The producer's write into the "
                "mapped buffer is not part of this time."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/comm/moe_alltoall_lamport.cuh",
        ),
    )
)
