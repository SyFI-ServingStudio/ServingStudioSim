"""Gated DeltaNet speculative-decode recurrence with the output norm fused.

vLLM's fused CUDA GDN decode (``VLLM_GDN_DECODE_KERNEL=cuda``, the default for
BF16 models with K = V = 128) runs a pure MTP verify step's recurrence and the
gated RMSNorm in one launch, ``fused_gdn_decode_post_conv_mtp``
(``qwen_gdn_linear_attn.py`` ``_forward_core_decode_spec_post_conv_fused_norm``).
Its logical boundary includes the norm, so it is its own kind rather than a
``gdn_recurrent_decode`` backend: a step priced with it runs no separate
``gdn_gated_rms_norm``.
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

KIND: str = "gdn_recurrent_decode_gated_norm"


@dataclass(frozen=True)
class GdnRecurrentDecodeGatedNormArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests verified together.")
    num_tokens: int = arg(
        unit="tokens",
        doc=(
            "Tokens across the batch, an equal number per request: the accepted "
            "token plus the drafts."
        ),
    )
    num_qk_heads: int = arg(unit="heads", doc="Query and key heads per token.")
    num_value_heads: int = arg(unit="heads", doc="Value heads and recurrent states per token.")
    key_head_dim: int = arg(unit="elements", doc="Features in each query or key head.")
    value_head_dim: int = arg(unit="elements", doc="Features in each value head.")
    dtype: DType = arg(doc="Element type of activations, gate, norm weight and output.")
    state_dtype: DType = arg(doc="Element type of the recurrent state and gate parameters.")


DOC = KernelDoc(
    title="Verify recurrence with output norm",
    summary=(
        "Run the gated delta rule over each request's verify tokens and apply the "
        "gated RMSNorm, in one launch."
    ),
    description=(
        "A speculative (MTP) verify step carries q = num_tokens / batch_size "
        "tokens per request. Each request starts from the FP32 state checkpoint of "
        "its last accepted token, runs the Gated DeltaNet decode recurrence over "
        "its q tokens (L2-normalized Q/K, decay and beta gates in FP32), and "
        "writes one checkpoint per token. Each value head's BF16 output row is "
        "then RMS-normalized, scaled by a learned weight and multiplied by the "
        "output gate."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "per token: S′ = decay·S + k·(beta·(v − kᵀ(decay·S)))ᵀ; o = BF16(qᵀS′)",
        "y = BF16(o / √(mean(o²) + eps) · weight · gate(z)), gate = sigmoid or SiLU",
        "T = num_tokens; E₁ = num_value_heads·key_head_dim·value_head_dim",
        (
            "FLOPs = 2·T·num_qk_heads·(3·key_head_dim + 1) + "
            "T·num_value_heads·key_head_dim + 4·T·num_value_heads + "
            "num_value_heads + 7·T·E₁ + 7·T·num_value_heads·value_head_dim"
        ),
        (
            "bytes = (2·T·num_qk_heads·key_head_dim + "
            "3·T·num_value_heads·value_head_dim + 2·T·num_value_heads + "
            "value_head_dim)·bytes(dtype) + (2·num_value_heads + "
            "(batch_size + T)·E₁)·bytes(state_dtype)"
        ),
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "vllm_cuda counts only gdn_decode_post_conv_mtp_kernel. Operands come from "
        "one strided qkvz and one ba projection as in vLLM; the reference check "
        "and its state restoration run before timing."
    ),
    caveats=(
        "The gate activation is not a key: the kernel's sigmoid and SiLU builds "
        "differ by one multiply per element. Rows are measured with sigmoid.",
        "Requests' accepted counts cycle through 1..q; the kernel reads one "
        "checkpoint whatever the count.",
        "Every timed call advances the checkpoints in place.",
    ),
    reference="profiling.runners.attention.gdn_spec_decode_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        # csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu (upstream #51674,
        # #52539) uses cp.async and BF16 intrinsics; vLLM enables it on compute
        # capability 8.0+ (qwen_gdn_linear_attn.py _fused_gdn_decode_unsupported_reason).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_recurrent_decode_gated_norm_vllm_cuda",
            function_name="profile_gdn_recurrent_decode_gated_norm_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=GdnRecurrentDecodeGatedNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM's fused_gdn_decode_post_conv_mtp CUDA op: verify recurrence, "
                "per-token FP32 checkpoints and gated RMSNorm in one kernel "
                "(K = V = 128, num_value_heads / num_qk_heads in 1, 2, 3, 4, 8, "
                "at most 8 tokens per request)."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu",
        ),
        subprocess_env="vllm_env",
    )
)
