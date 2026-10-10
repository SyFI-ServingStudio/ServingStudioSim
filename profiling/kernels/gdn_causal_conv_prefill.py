"""GDN causal-convolution prefill kernel kind.

The initial ``torch`` backend measures the complete multi-launch semantic
reference. It is a correctness/performance baseline, not the production fused
prefill launch, and must not be selected for production simulation after the
vLLM backend is registered. ``vllm_triton`` is vLLM's one-launch varlen call;
``dao_channellast`` is Dao-AILab causal-conv1d's channel-last kernel, one launch
per sequence.
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
        "In Gated DeltaNet and KDA prefill, each new sequence runs a "
        "depthwise causal convolution from a zero history: every channel has "
        "its own kernel_size-wide filter, followed by SiLU. The sequence's last"
        " kernel_size − 1 input samples, zero-padded on the left if it is "
        "shorter, are saved as the request's convolution state. Sequences have "
        "equal length and bounded BF16 inputs; vllm_triton gets packed tokens "
        "and sequence metadata prepared before timing. vllm_triton covers the "
        "whole batch in one launch; dao_channellast launches once per sequence, "
        "so its batch_size-B row is B launches."
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
        "_causal_conv1d_fwd_kernel; dao_channellast sums every "
        "causal_conv1d_channellast_fwd_kernel launch of the batch. Packing, "
        "metadata and the output checks run before timing."
    ),
    caveats=(
        "Prior state is ignored; repeated calls rewrite the same state rows "
        "with the same input tails.",
        "The torch reference checks request indices on the GPU inside each timed call.",
        "FLOPs and bytes count the operation itself, not the torch backend's intermediate tensors.",
        "dao_channellast passes no initial state. A continuation chunk would read "
        "the slot as initial_states and so needs its final state in a separate "
        "buffer plus a small copy back, which these rows leave out.",
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

# Dao-AILab causal-conv1d 1.7.0's release wheel ships SASS for sm_75, sm_80,
# sm_87, sm_90, sm_100 and sm_120 and no PTX (cuobjdump --list-elf), so a
# device older than 7.5 has no kernel to load.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="dao_channellast",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(7, 5),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_causal_conv_prefill_dao_channellast",
            function_name="profile_gdn_causal_conv_prefill_dao_channellast",
        ),
        table_name=KIND,
        args_schema=GdnCausalConvPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        row_provenance_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_causal_conv_prefill_dao_channellast",
            function_name="row_provenance",
        ),
        doc=BackendDoc(
            summary=(
                "Dao-AILab causal-conv1d's channel-last kernel, one launch per sequence, "
                "writing each final state straight into its cache slot."
            ),
            url="https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_fwd.cu",
        ),
        subprocess_env="causal_conv1d_env",
    )
)
