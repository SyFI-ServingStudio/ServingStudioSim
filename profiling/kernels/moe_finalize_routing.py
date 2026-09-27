"""FlashInfer/TensorRT-LLM MoE finalize-routing kernel kind.

The kernel unpermutes EP-local expert outputs, applies router scales, and
reduces the local top-k contributions into one BF16 row per original token.
The following EP all-reduce is a separate communication kernel.
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

KIND: str = "moe_finalize_routing"


@dataclass(frozen=True)
class MoeFinalizeRoutingArgs(KernelArgs):
    token_count: int = arg(unit="tokens", doc="Original tokens in the batch.")
    hidden_size: int = arg(unit="elements", doc="Features in each expert output row.")
    top_k: int = arg(unit="experts", doc="Experts selected per original token.")
    num_experts_per_rank: int = arg(unit="experts", doc="Experts owned by this rank.")
    local_routed_token_count: int = arg(
        unit="tokens", doc="Selected expert rows resident on this rank."
    )
    dtype: DType = arg(doc="Element type of the expert outputs and reduced result.")


DOC = KernelDoc(
    title="MoE routing finalization",
    summary=("Unpermute, scale and sum local expert outputs into one row per original token."),
    description=(
        "With expert parallelism, each GPU computes only the rows routed to its"
        " own experts. This TensorRT-LLM kernel, vendored in FlashInfer, "
        "gathers those rows back into token order, scales each by its router "
        "weight and sums them per token. token_count is all tokens; "
        "local_routed_token_count is the rows this GPU computed. The rows are "
        "grouped by local expert, as the upstream permutation leaves them."
    ),
    category="MoE",
    subcategory="Routing and combine",
    formula=(
        "output[token, h] = Σₖ local_scale[token, k] · expert_row[token, k, h]",
        "TFLOPS = 2·local_routed_token_count·hidden_size / time",
        "GB/s = (2·local_routed_token_count·hidden_size + 2·token_count·hidden_size "
        "+ 8·local_routed_token_count + 4·token_count·top_k) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Only finalizeMoeRoutingKernel launches are counted. The JIT build and "
        "one launch run before timing."
    ),
    caveats=(
        "Rows routed elsewhere all point at one expert outside this GPU's "
        "range, which the kernel skips; the cross-GPU reduction that follows is"
        " a separate step.",
        "GB/s counts the local rows read and the output written, not the whole "
        "expanded-row buffer.",
    ),
    reference="profiling.runners.moe.flashinfer_trtllm_finalize",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H100", "NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.flashinfer_trtllm_finalize",
            function_name="profile_moe_finalize_routing_flashinfer_trtllm",
        ),
        table_name=KIND,
        args_schema=MoeFinalizeRoutingArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # Compile against the same FlashInfer/Torch stack as the measured vLLM
        # run; this kernel's vendored source changed across FlashInfer releases.
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "FlashInfer's TensorRT-LLM finalizeMoeRoutingKernel through "
                "a direct BF16 SM90 launcher."
            )
        ),
    )
)
