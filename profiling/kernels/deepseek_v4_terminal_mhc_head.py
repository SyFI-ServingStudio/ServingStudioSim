"""DeepSeek V4 terminal MHC-post, hc-head, and final RMSNorm compound."""

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

KIND = "deepseek_v4_terminal_mhc_head"


@dataclass(frozen=True)
class DeepseekV4TerminalMhcHeadArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Hidden-state rows entering the terminal head.")
    hidden_size: int = arg(unit="elements", doc="Elements in each hidden-state channel.")
    hc_mult: int = arg(unit="channels", doc="Residual channels combined by the head.")
    hidden_dtype: DType = arg(doc="Element type of the hidden states.")


DOC = KernelDoc(
    title="DeepSeek V4 terminal mHC head",
    summary=(
        "Collapse the final mHC residual channels into one hidden state and "
        "RMS-normalize it for the LM head."
    ),
    description=(
        "After DeepSeek V4's last layer, the mHC post-block merges the layer "
        "output into the hc_mult residual channels. The head then weights each "
        "channel by a learned sigmoid gate, sums them into one hidden state and"
        " applies the final RMSNorm before the LM head. The measured call also "
        "copies the post-block result into the buffer that multi-token "
        "prediction reads."
    ),
    category="Normalization",
    formula=(
        "post[j] = Σi comb_mix[i, j]·residual[i] + post_mix[j]·x",
        "w = sigmoid((post_flat · head_fnᵀ) · rsqrt(mean(post_flat²) + 1e−6) "
        "· head_scale + head_base) + 1e−6",
        "y = RMSNorm(Σj post[j]·w[j], ε = 1e−6)",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} "
        "Three warm-up calls run first. Every launch is counted: the mHC "
        "post-block, the buffer copy, the fused channel head and RMSNorm."
    ),
    caveats=(
        "The copy into the multi-token prediction buffer is always included; "
        "serving copies only when that buffer exists.",
        "The final RMSNorm weight is all ones.",
        "TFLOPS is not computed. GB/s counts the external inputs read once and "
        "the prediction buffer and output written once; the post-block result "
        "passed between launches is not counted.",
    ),
    # The PyTorch correctness calculation is local to the measured runner.
    reference=None,
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_tilelang",
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.deepseek_v4_terminal_mhc_head_vllm_tilelang",
            function_name="profile_deepseek_v4_terminal_mhc_head_vllm_tilelang",
        ),
        table_name=KIND,
        args_schema=DeepseekV4TerminalMhcHeadArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's TileLang mhc_post_tilelang and hc_head_fused_kernel_tilelang "
                "run with a buffer copy and RMSNorm."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/mhc/tilelang.py",
        ),
    )
)

__all__ = ["DeepseekV4TerminalMhcHeadArgs", "KIND"]
