"""GDN causal-convolution decode kernel kind.

The initial ``torch`` backend measures the complete multi-launch semantic
reference. It is a correctness/performance baseline, not the production fused
decode launch, and must not be selected for production simulation after the
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

KIND: str = "gdn_causal_conv_decode"


@dataclass(frozen=True)
class GdnCausalConvDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded together, one token each.")
    channels: int = arg(unit="channels", doc="Channels in each token and depthwise filter.")
    kernel_size: int = arg(unit="tokens", doc="Width of each causal filter.")
    dtype: DType = arg(doc="Element type of the input, filter and output.")
    state_dtype: DType = arg(doc="Element type of the convolution state.")


DOC = KernelDoc(
    title="Decode causal convolution",
    summary=(
        "Apply a depthwise causal convolution and SiLU to one token per request, "
        "updating the convolution state."
    ),
    description=(
        "In Gated DeltaNet and KDA layers, decode runs a short "
        "depthwise causal convolution over each channel: the request's last kernel_size −"
        " 1 samples plus the new token, weighted by that channel's filter, "
        "summed in FP32, then SiLU. The new token is shifted into the "
        "convolution state. Each request has its own state row; inputs are "
        "bounded BF16 values, and repeated calls keep advancing the states."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "y[b, c] = SiLU(Σᵢ window[b, c, i] · weight[c, i]); state shifts by one sample",
        "FLOPs = batch_size · channels · (2·kernel_size + 1)",
        (
            "bytes = (2·batch_size·channels + channels·kernel_size)·bytes(dtype) + "
            "2·batch_size·channels·(kernel_size − 1)·bytes(state_dtype) + 4·batch_size"
        ),
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of the reference, which updates the state in"
        " place; vllm_triton counts only _causal_conv1d_update_kernel. "
        "Allocation and the vLLM output check run before timing."
    ),
    caveats=(
        "torch writes a new output tensor, while vllm_triton overwrites the "
        "input token; both update the state in place.",
        "The torch reference checks request indices on the GPU inside each timed call.",
        "FLOPs and bytes count the operation itself, not the torch backend's "
        "extra tensors and launches.",
    ),
    reference="profiling.runners.attention.gdn_causal_conv_decode_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_causal_conv_decode_torch"),
            function_name="profile_gdn_causal_conv_decode",
        ),
        table_name=KIND,
        args_schema=GdnCausalConvDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="The PyTorch reference, including its separate state update and SiLU launches."
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
            module_name=("profiling.runners.attention.gdn_causal_conv_decode_vllm_triton"),
            function_name="profile_gdn_causal_conv_decode_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnCausalConvDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM's causal_conv1d_update Triton call: convolution, SiLU and "
                "in-place state update in one kernel."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
        ),
        subprocess_env="vllm_env",
    )
)


# The AMD/ROCm path. GLM-5.3-Flash's glm5next plugin imports causal_conv1d_update
# straight from the generic mamba ops with no is_rocm()/aiter branch (common/
# kda.py:554), so the ROCm decode conv is the SAME Triton kernel as the NVIDIA
# vllm_triton backend, only built for CDNA3. Measured on MI300X in the
# vllm_rocm_env image, timed kernel-only via rocprofv3 filtered to
# _causal_conv1d_update_kernel. NVIDIA rows are untouched.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_rocm",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            arch_targets=frozenset({"CDNA3"}),
        ),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_causal_conv_decode_torch_rocm"),
            function_name="profile_gdn_causal_conv_decode_torch_rocm",
        ),
        table_name=KIND,
        args_schema=GdnCausalConvDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_rocm_env",
        doc=BackendDoc(
            summary=(
                "AMD/ROCm: vLLM's generic Triton causal_conv1d_update "
                "(_causal_conv1d_update_kernel) on CDNA3, the same fused "
                "convolution, SiLU and in-place state update as the NVIDIA path."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
        ),
    )
)
