"""Gated DeltaNet recurrent-decode kernel kind.

The initial ``torch`` backend measures the complete multi-launch semantic
reference. It is a correctness/performance baseline, not the production fused
decode launch, and must not be selected for production simulation after the
vLLM backend is registered.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg, optional_arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "gdn_recurrent_decode"


@dataclass(frozen=True)
class GdnRecurrentDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded together, one token each.")
    num_qk_heads: int = arg(unit="heads", doc="Query and key heads per token.")
    num_value_heads: int = arg(unit="heads", doc="Value heads and recurrent states per token.")
    key_head_dim: int = arg(unit="elements", doc="Features in each query or key head.")
    value_head_dim: int = arg(unit="elements", doc="Features in each value head.")
    dtype: DType = arg(doc="Element type of activations and output.")
    state_dtype: DType = arg(doc="Element type of the recurrent state and gate parameters.")
    num_tokens: int | None = optional_arg(
        unit="tokens",
        same_as="batch_size",
        doc=(
            "Tokens across the batch, an equal number per request: a speculative "
            "(MTP verify) step's accepted token plus drafts. Absent when each request "
            "decodes one token."
        ),
    )


DOC = KernelDoc(
    title="Recurrent decode",
    summary="Update the gated delta-rule state and read one output token per request.",
    description=(
        "The core of Gated DeltaNet decode. Query and key heads are "
        "L2-normalized and repeated to match the value heads. For each value "
        "head, a decay gate shrinks the matrix state, a delta-rule update "
        "writes the new key-value pair with strength beta, and the query reads "
        "the updated state. Each request has its own FP32 state; activations "
        "are BF16."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        (
            "q, k = L2Norm(q, k); q = q·key_head_dim⁻¹ᐟ²; "
            "decay = exp(−exp(A_log)·softplus(a + dt_bias))"
        ),
        "beta = sigmoid(b) in FP32",
        "S′ = decay·S + k·(beta·(v − kᵀ(decay·S)))ᵀ; y = qᵀS′",
        "T = num_tokens (batch_size when absent); E₁ = num_value_heads·key_head_dim·value_head_dim",
        (
            "FLOPs = 2·T·num_qk_heads·(3·key_head_dim + 1) + "
            "T·num_value_heads·key_head_dim + 4·T·num_value_heads + "
            "num_value_heads + 7·T·E₁"
        ),
        (
            "bytes = (2·T·num_qk_heads·key_head_dim + "
            "2·T·num_value_heads·value_head_dim + "
            "2·T·num_value_heads)·bytes(dtype) + (2·num_value_heads + "
            "(batch_size + T)·E₁)·bytes(state_dtype)"
        ),
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of its recurrent reference; vllm_triton "
        "counts only fused_recurrent_gated_delta_rule_packed_decode_kernel and "
        "vllm_triton_sigmoid_gating only fused_sigmoid_gating_delta_rule_update_kernel. "
        "Input packing and the vLLM output check run before timing."
    ),
    caveats=(
        "num_tokens > batch_size is a speculative verify step, priced by "
        "vllm_triton_sigmoid_gating only (the call vLLM makes for the verify requests "
        "of a step that also carries a prefill): each request starts from the "
        "checkpoint of its last accepted token and writes one checkpoint per token "
        "(gdn_spec_decode_reference).",
        "torch and vllm_triton_sigmoid_gating take separate Q, K and V tensors; "
        "vllm_triton takes packed QKV. Both vLLM backends index into a state pool.",
        "vLLM uses vllm_triton for decode-only steps and vllm_triton_sigmoid_gating "
        "for the decode requests of a step that also carries a prefill.",
        "All backends update the states in place on every timed call.",
        "FLOPs and bytes count the operation itself, not the torch backend's intermediates.",
    ),
    reference="profiling.runners.attention.gdn_recurrent_decode_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_recurrent_decode_torch"),
            function_name="profile_gdn_recurrent_decode",
        ),
        table_name=KIND,
        args_schema=GdnRecurrentDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "The PyTorch reference, with separate normalization, gate and "
                "state-update launches."
            ),
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_recurrent_decode_vllm_triton"),
            function_name="profile_gdn_recurrent_decode_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnRecurrentDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM's fused_recurrent_gated_delta_rule_packed_decode Triton call uses packed "
                "QKV and indexed FP32 state."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py",
        ),
        subprocess_env="vllm_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton_sigmoid_gating",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name=(
                "profiling.runners.attention.gdn_recurrent_decode_vllm_triton_sigmoid_gating"
            ),
            function_name="profile_gdn_recurrent_decode_vllm_triton_sigmoid_gating",
        ),
        table_name=KIND,
        args_schema=GdnRecurrentDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM's fused_sigmoid_gating_delta_rule_update Triton call, which "
                "updates the decode requests of a step that also carries a prefill: "
                "separate Q/K/V, one-token varlen batch, indexed FP32 state."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py",
        ),
        subprocess_env="vllm_env",
    )
)
