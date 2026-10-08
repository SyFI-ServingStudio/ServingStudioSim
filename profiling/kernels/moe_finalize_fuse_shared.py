"""SGLang deferred-MoE finalize with an optional fused shared-expert add."""

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

KIND = "moe_finalize_fuse_shared"


@dataclass(frozen=True)
class MoeFinalizeFuseSharedArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Original tokens whose expert outputs are combined.")
    top_k: int = arg(unit="experts", doc="Selected expert outputs per token.")
    hidden_dim: int = arg(unit="elements", doc="Features in each output row.")
    dtype: DType = arg(doc="Element type of expert and shared outputs.")
    fuse_shared_output: bool = arg(doc="Whether to add the shared-expert output.")


DOC = KernelDoc(
    title="MoE finalization with shared output",
    summary="Gather and weight routed expert outputs, optionally adding the shared-expert row.",
    description=(
        "SGLang's last MoE step when the fused MoE defers its combine: follow "
        "the permutation map, sum each token's top_k expert rows with their "
        "router weights, and optionally add the shared expert's output in the "
        "same pass. Expert rows, permutation and weights are random."
    ),
    category="MoE",
    subcategory="Routing and combine",
    formula=(
        "output[token, h] = Σₖ weight[token, k] · expert_row[token, k, h] "
        "+ shared[token, h] when enabled",
        "TFLOPS = num_tokens·hidden_dim·(2·top_k + shared_flag) / time",
        "GB/s = (2·num_tokens·hidden_dim·(top_k + 1 + shared_flag) "
        "+ 8·num_tokens·top_k) / time; shared_flag = 1 when enabled, else 0",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Five warm-up calls run first, and every launch of the call is counted;"
        " the first call builds the JIT kernel."
    ),
    caveats=(
        "The call allocates a fresh output each time; CUPTI counts only kernel "
        "time, so the allocation is not included.",
    ),
    # No separate PyTorch reference implementation exists for this backend.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_finalize_fuse_shared",
            function_name="profile_moe_finalize_fuse_shared_sglang",
        ),
        table_name=KIND,
        args_schema=MoeFinalizeFuseSharedArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary=(
                "SGLang moe_finalize_fuse_shared JIT CUDA kernel with optional shared-row addition."
            )
        ),
    )
)
