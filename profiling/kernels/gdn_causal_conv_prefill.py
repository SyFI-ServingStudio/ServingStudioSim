"""GDN causal-convolution prefill kernel kind.

The initial ``torch`` backend measures the complete multi-launch semantic
reference. It is a correctness/performance baseline, not the production fused
prefill launch, and must not be selected for production simulation after the
vLLM backend is registered.
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

KIND: str = "gdn_causal_conv_prefill"


@dataclass(frozen=True)
class GdnCausalConvPrefillArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Fresh sequences in the batch.")
    sequence_length: int = arg(unit="tokens", doc="Tokens in each sequence.")
    channels: int = arg(unit="channels", doc="Channels in each token and depthwise filter.")
    kernel_size: int = arg(unit="tokens", doc="Width of each causal filter.")
    dtype: DType = arg(doc="Element type of the input, filter and output.")
    state_dtype: DType = arg(doc="Element type of the written convolution state.")


DOC = KernelDoc(
    title="Prefill causal convolution",
    summary=(
        "Apply a depthwise causal convolution and SiLU to fresh sequences and save "
        "their final input samples as state."
    ),
    description=(
        "In Gated DeltaNet prefill, each new sequence runs a "
        "depthwise causal convolution from a zero history: every channel has "
        "its own kernel_size-wide filter, followed by SiLU. The sequence's last"
        " kernel_size − 1 input samples, zero-padded on the left if it is "
        "shorter, are saved as the request's convolution state. Sequences have "
        "equal length and bounded BF16 inputs; vllm_triton gets packed tokens "
        "and sequence metadata prepared before timing."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "y[b, t, c] = SiLU(Σᵢ padded_x[b, t + i, c] · weight[c, i])",
        "FLOPs = batch_size · sequence_length · channels · (2·kernel_size + 1)",
        (
            "bytes = (2·batch_size·sequence_length·channels + "
            "channels·kernel_size)·bytes(dtype) + batch_size·channels·(kernel_size − "
            "1)·bytes(state_dtype) + 4·batch_size"
        ),
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of the reference; vllm_triton counts only "
        "_causal_conv1d_fwd_kernel. Packing, metadata and the vLLM output check"
        " run before timing."
    ),
    caveats=(
        "Prior state is ignored; repeated calls rewrite the same state rows "
        "with the same input tails.",
        "The torch reference checks request indices on the GPU inside each timed call.",
        "FLOPs and bytes count the operation itself, not the torch backend's intermediate tensors.",
    ),
    reference="profiling.runners.attention.gdn_causal_conv_prefill_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_causal_conv_prefill_torch"),
            function_name="profile_gdn_causal_conv_prefill",
        ),
        table_name=KIND,
        args_schema=GdnCausalConvPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "The PyTorch reference, with separate convolution, SiLU and state-write launches."
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
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton"),
            function_name="profile_gdn_causal_conv_prefill_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnCausalConvPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM's causal_conv1d_fn Triton call computes fresh prefill with packed tokens "
                "and indexed state writes."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
        ),
        subprocess_env="vllm_env",
    )
)
