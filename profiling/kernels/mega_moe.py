"""DeepGEMM's fused expert-parallel MoE layer (``deep_gemm.fp8_fp4_mega_moe``).

Wire string: ``"mega_moe"`` -- matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/mega_moe.rs``.

One launch of ``sm100_fp8_fp4_mega_moe_impl`` does the whole routed MoE of one
expert-parallel rank: it pulls the rows routed to its local expert slots from
every peer's symmetric buffer (dispatch), runs the gate/up GEMM, the clamped
SwiGLU and the down GEMM, writes the results back to the source ranks and sums
the top-k rows (combine). DeepInfer fuses the shared expert into the same launch
(``shared_l1_weights``/``shared_l2_weights``). Nothing of it can be priced as
separate kernels: dispatch, both GEMMs and combine overlap inside one persistent
kernel, and the ranks synchronize through the symmetric buffer.

Why a new kind: the nearest kind, ``nvfp4_fused_moe``, is one GPU's routed MoE
over pre-gathered rows. This callable also moves the rows between the ranks and
adds the shared expert, so its logical boundary is the whole EP MoE layer.

Key: model geometry and dtypes only, plus ``ep_size`` and the per-rank token
count. The symmetric-buffer capacity (``num_max_tokens_per_rank``, a template
argument of the kernel) is not a key: on 4 B200s, at 140, 560, 4096 and 16384
tokens per rank, capacities from 1920 to 67200 moved the time by at most 0.5%,
less than the 0.9% between two runs at one capacity. The tile size
(``get_block_m_for_mega_moe``) is a function of ``ep_size``, ``num_experts``,
``top_k`` and ``num_tokens`` only, all of which are in the key.

Routing is an input: ``per_slot_rows`` gives the rows every physical slot of
the group receives (rank ``r`` owns slots ``[r * L, (r + 1) * L)``, ``L =
num_experts / ep_size``). DeepGEMM's scheduler (``scheduler/mega_moe.cuh``)
gives each rank ``sum(ceil(rows / block_m))`` M blocks over its slots plus the
shared expert's blocks on its own tokens, the dispatch pulls exactly those rows
over NVLink, and the combine waits for every peer, so the slot histogram -- not
the token count alone -- sets the time. The simulator draws it from the
deployment's routing and expert placement (EPLB replicas included); the runner
builds top-k ids that realize it exactly.
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

KIND = "mega_moe"


@dataclass(frozen=True)
class MegaMoeArgs(KernelArgs):
    ep_size: int = arg(
        unit="GPUs",
        doc=(
            "GPUs in the expert-parallel group; 1 runs every expert on one GPU (no NVLink traffic)."
        ),
    )
    num_tokens: int = arg(
        unit="tokens",
        doc="Tokens entering the MoE layer on each GPU (every GPU has the same count).",
    )
    hidden_size: int = arg(unit="elements", doc="Width of each token's hidden state.")
    intermediate_size: int = arg(
        unit="elements", doc="Width of each routed expert's intermediate state."
    )
    num_experts: int = arg(
        unit="experts",
        doc=(
            "Physical routed expert slots across all GPUs, redundant (EPLB) slots "
            "included; each GPU holds num_experts / ep_size of them."
        ),
    )
    num_shared_experts: int = arg(
        unit="experts", doc="Shared experts fused into the same launch, run on every token."
    )
    top_k: int = arg(unit="experts", doc="Distinct expert slots selected for each token.")
    activation_dtype: DType = arg(
        doc="Tensor-core precision of the activations: mxfp8_e4m3 (one UE8M0 scale per 32)."
    )
    weight_dtype: DType = arg(
        doc=(
            "Routed expert weight format: mxfp4_e2m1 (one UE8M0 scale per 32). The "
            "shared expert's weights are mxfp8_e4m3."
        )
    )
    per_slot_rows: tuple[int, ...] = arg(
        unit="rows",
        doc=(
            "Rows each physical slot receives over the whole group, rank-major "
            "(num_experts entries summing to ep_size * num_tokens * top_k)."
        ),
    )


DOC = KernelDoc(
    title="Fused expert-parallel MoE layer",
    summary=(
        "One kernel per GPU that dispatches tokens to expert GPUs, runs both expert "
        "GEMMs and the shared expert, and combines the results."
    ),
    description=(
        "DeepGEMM's mega MoE runs a whole expert-parallel MoE layer as one persistent "
        "kernel on each GPU. Every GPU writes its tokens and their top-k expert slots "
        "into a symmetric buffer; the kernel pulls the rows routed to its local slots "
        "from every GPU over NVLink, runs the gate/up GEMM, a clamped SwiGLU and the "
        "down GEMM, sends each result row back to its source GPU and sums the top-k "
        "rows there. The shared expert runs on the GPU's own tokens inside the same "
        "launch. Activations are MXFP8, routed weights MXFP4, the output bf16. "
        "The rows each slot receives are an input, drawn from the deployment's "
        "routing and expert placement, so a skewed router or an unbalanced "
        "placement loads the busiest GPU as it does in serving."
    ),
    category="MoE",
    subcategory="Expert compute",
    formula=(
        "routed rows of GPU r = sum(per_slot_rows[r·L:(r+1)·L]), L = num_experts / ep_size",
        "FLOPs per GPU = 6 · hidden_size · intermediate_size · "
        "(routed rows + num_tokens · num_shared_experts)",
        "bytes per GPU = touched local expert weights + shared weights + routed "
        "activations (fp8 in, fp8 intermediate, bf16 out) + bf16 output",
    ),
    default_metric="time_ms",
    method=(
        CUPTI_METHOD + " All GPUs launch together: each launch follows a GPU sleep "
        "and a one-element NCCL all-reduce, so no GPU waits on another's late "
        "launch. The time is the median over launches of the slowest GPU's kernel "
        "time in that launch."
    ),
    caveats=(
        "Which source GPU a routed row comes from is not an input: the runner "
        "spreads each slot's rows over the source GPUs round-robin.",
        "Every GPU runs the same token count; unequal per-GPU counts are not measured.",
        "Energy is the mean per-GPU energy of back-to-back launches.",
    ),
    reference="profiling.runners.moe.mega_moe_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_sm100",
        # deep_gemm/csrc/apis/mega_moe.hpp dispatches only `arch_major == 10`
        # (sm100_fp8_fp4_mega_moe), else DG_HOST_UNREACHABLE.
        supports=BackendSupport(
            compute=frozenset({DType.MXFP4_E2M1}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.mega_moe_deepgemm",
            function_name="profile_mega_moe_batch",
        ),
        table_name=KIND,
        args_schema=MegaMoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        gpu_count_fn=lambda spec: int(spec["ep_size"]),
        # One rank-group spawn, NCCL init and symmetric-memory rendezvous serve
        # the whole chunk.
        list_native=True,
        doc=BackendDoc(
            summary=(
                "DeepGEMM fp8_fp4_mega_moe as DeepInfer calls it: shared expert fused, "
                "SwiGLU clamp 10, fast math, MXFP8 x MXFP4 MMA."
            ),
            url="https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/mega/__init__.py",
        ),
    )
)
