"""Qwen GDN fused-prefill post-convolution preparation kernel kind.

The initial ``torch`` backend measures the complete multi-launch semantic
reference. It is a correctness/performance baseline, not the production fused
Triton launch, and must not be selected for production simulation after the
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

KIND: str = "gdn_prefill_post_conv"


@dataclass(frozen=True)
class GdnPrefillPostConvArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens in the prefill batch.")
    num_qk_heads: int = arg(unit="heads", doc="Query and key heads per token.")
    num_value_heads: int = arg(unit="heads", doc="Value heads and gate values per token.")
    key_head_dim: int = arg(unit="elements", doc="Features in each query or key head.")
    value_head_dim: int = arg(unit="elements", doc="Features in each value head.")
    dtype: DType = arg(doc="Element type of the packed convolution output and raw gates.")


DOC = KernelDoc(
    title="Prefill post-convolution split",
    summary=(
        "Split the convolution output into normalized queries and keys, values,"
        " the log decay and beta."
    ),
    description=(
        "After the causal convolution in Qwen3.6's Gated DeltaNet prefill, the "
        "packed output is split into queries, keys and values. Queries and keys"
        " are L2-normalized per head, values are copied, and two raw gates are "
        "combined with FP32 parameters into the log decay g and the update "
        "strength beta. Activations are bounded BF16 values."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "P = 2·num_qk_heads·key_head_dim + num_value_heads·value_head_dim",
        ("q, k = packed q, k / √(sum(features²) + 1e−6); v = packed v"),
        "g = −exp(A_log)·softplus(a + dt_bias); beta = sigmoid(b)",
        (
            "FLOPs = 2·num_tokens·num_qk_heads·(3·key_head_dim + 1) + "
            "4·num_tokens·num_value_heads + num_value_heads"
        ),
        (
            "bytes = (2·num_tokens·P + 2·num_tokens·num_value_heads)·bytes(dtype) + "
            "(2·num_value_heads + 2·num_tokens·num_value_heads)·4"
        ),
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of the reference; vllm_triton counts only "
        "_fused_post_conv_kernel. Input construction and the vLLM output check "
        "run before timing."
    ),
    caveats=(
        "Queries and keys are normalized in FP32 and rounded to BF16; g and beta stay FP32.",
        "FLOPs and bytes count the operation itself, not the torch backend's temporary tensors.",
    ),
    reference="profiling.runners.attention.gdn_prefill_post_conv_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_prefill_post_conv_torch",
            function_name="profile_gdn_prefill_post_conv",
        ),
        table_name=KIND,
        args_schema=GdnPrefillPostConvArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "The PyTorch reference, timed across its separate normalization and gate launches."
            )
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_prefill_post_conv_vllm_triton"),
            function_name="profile_gdn_prefill_post_conv_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnPrefillPostConvArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM's fused_post_conv_prep Triton call: Q, K, V, log decay and beta "
                "in one launch."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py",
        ),
        subprocess_env="vllm_env",
    )
)
