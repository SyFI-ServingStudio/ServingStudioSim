"""Learned/hash sqrt-softplus routing as one physical CUDA kind."""

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

KIND = "moe_topk_softplus_sqrt"


@dataclass(frozen=True)
class MoeTopkSoftplusSqrtArgs(KernelArgs):
    selection_mode: str = arg(doc="Routing mode: learned scores or token hash lookup.")
    num_tokens: int = arg(unit="tokens", doc="Tokens routed together.")
    num_experts: int = arg(unit="experts", doc="Experts available to each token.")
    top_k: int = arg(unit="experts", doc="Experts selected per token.")
    hash_vocab_size: int = arg(unit="tokens", doc="Token IDs addressable by the hash table.")
    logits_dtype: DType = arg(doc="Element type of the router logits.")


DOC = KernelDoc(
    title="Softplus-sqrt MoE routing",
    summary=(
        "Select experts with sqrt-softplus scores or a token hash table and produce scaled weights."
    ),
    description=(
        "This MoE router scores each expert as √softplus(logit). In "
        "learned mode it selects the top_k experts by score plus a correction "
        "bias; in hash mode the experts come from a table indexed by token ID. "
        "Both modes renormalize the selected scores and scale the weights by "
        "1.5, in one vLLM CUDA kernel. Logits are random fp32 values."
    ),
    category="MoE",
    subcategory="Routing and combine",
    formula=(
        "score = √softplus(logit)",
        "weight = 1.5 · selected score / sum(selected scores)",
        "GB/s = 4·(num_tokens·num_experts + 3·num_tokens·top_k + mode_vector) / time; "
        "mode_vector = num_experts (learned) or num_tokens (hash)",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Only topkGatingSoftplusSqrt launches are counted. Outputs are checked "
        "before timing."
    ),
    caveats=(
        "Hash mode uses a 129,280-entry table.",
        "The correction bias, the hash table and the token IDs are synthetic.",
        "GB/s counts logical inputs and outputs, not the whole hash table.",
    ),
    # The PyTorch check is inline in the measured runner, not a separate reference module.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_topk_softplus_sqrt_vllm_cuda",
            function_name="profile_moe_topk_softplus_sqrt_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=MoeTopkSoftplusSqrtArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # No capability rule: a vLLM _moe_C CUDA op built for every arch the wheel targets.
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM topk_hash_softplus_sqrt, with learned and hash selection in one CUDA kernel."
            )
        ),
    )
)

__all__ = ["KIND"]
