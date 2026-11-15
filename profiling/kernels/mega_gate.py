"""DeepGEMM's fused MoE router (``deep_gemm.bf16_mega_gate``).

Wire string: ``"mega_gate"`` -- matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/mega_gate.rs``.

One ``sm100_bf16_mega_gate_impl`` launch per MoE layer and GPU computes the
router logits (bf16 hidden states times the bf16 gate weight, fp32
accumulation), applies the scoring function and the score-correction bias,
selects the top-k logical experts, maps each choice to a physical (EPLB) slot
and writes the int32 slot indices and fp32 routing weights. DeepInfer calls it
from ``v4_modules/mlp.py`` (``use_mega_gate``) before the mega MoE.

Why a new kind: ``gemm_fp32_output`` prices only the logits GEMM and
``moe_fused_topk`` only the selection; this callable is both plus the
logical-to-physical slot mapping in one launch, so its boundary is the whole
router.

Key: model geometry and dtype plus the per-GPU token count. The redundant
slots enter through ``num_physical_experts`` (the slot map's width).
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

KIND = "mega_gate"


@dataclass(frozen=True)
class MegaGateArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens routed on this GPU.")
    hidden_size: int = arg(unit="elements", doc="Width of each token's hidden state.")
    num_experts: int = arg(unit="experts", doc="Logical routed experts the router scores.")
    num_physical_experts: int = arg(
        unit="experts",
        doc="Physical expert slots, redundant (EPLB) slots included, that choices map to.",
    )
    top_k: int = arg(unit="experts", doc="Experts selected per token.")
    scoring_func: str = arg(doc="Router scoring function, e.g. sqrtsoftplus or sigmoid.")
    dtype: DType = arg(doc="Element type of the hidden states and gate weight (bf16).")


DOC = KernelDoc(
    title="Fused MoE router",
    summary=(
        "One kernel that computes router logits, scores and biases them, picks the "
        "top-k experts and maps them to physical expert slots."
    ),
    description=(
        "DeepGEMM's mega gate multiplies the bf16 hidden states by the bf16 router "
        "weight with fp32 accumulation, applies the scoring function and the "
        "score-correction bias, selects the top-k experts per token, scales their "
        "weights by the routed scaling factor and maps each choice to one of the "
        "expert's physical slots. It writes int32 slot indices and fp32 weights."
    ),
    category="MoE",
    subcategory="Routing and combine",
    formula=(
        "FLOPs = 2 · num_tokens · num_experts · hidden_size",
        "bytes = 2 · num_tokens · hidden_size + 2 · num_experts · hidden_size + "
        "8 · num_tokens · top_k",
    ),
    default_metric="time_ms",
    method=CUPTI_METHOD,
    caveats=(
        "Every redundant slot of an expert is equally likely; the slot map is "
        "DeepInfer's initial EPLB layout.",
    ),
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_sm100",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.mega_gate_deepgemm",
            function_name="profile_mega_gate_deepgemm",
        ),
        table_name=KIND,
        args_schema=MegaGateArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepGEMM bf16_mega_gate as DeepInfer calls it: score-correction bias, "
                "routed scaling, logical-to-physical slot map, no shared-as-routed."
            ),
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)
