"""Fused reduce-scatter + short conv + all-gather + residual + RMSNorm.

The model replaces every tensor-parallel all-reduce with one collective call,
``LamportRSConv.rs_sconv_ag_add_norm`` (vLLM ``models/inkling/nvidia/ops/
lamport.py``): the TP-partial sublayer output (plus the MoE sink partial) is
reduce-scattered over the hidden dim, each rank runs the causal depthwise
k=4 short conv with skip on its shard using the paged conv-state cache, the
shards are all-gathered, added to the residual and RMS-normalized with the next
sublayer's norm. It is four PDL-chained Triton launches whose cross-GPU
spin-waits couple them, so the kind times the whole call as one unit. No
all-reduce kind fits: the comm pattern, the conv-state cache semantics and the
launch boundary all differ.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "rs_sconv_ag_residual_rms_norm"


@dataclass(frozen=True)
class RsSconvAgResidualRmsNormArgs(KernelArgs):
    num_gpus: int = arg(unit="GPUs", doc="GPUs in the tensor-parallel group.")
    num_tokens: int = arg(
        unit="tokens",
        doc="Token rows on every GPU (the padded CUDA-graph batch); one token runs 8-way split.",
    )
    hidden_dim: int = arg(
        unit="elements", doc="Model hidden size; each GPU owns hidden_dim / num_gpus."
    )
    conv_kernel_size: int = arg(unit="tokens", doc="Short-conv window; the kernel requires 4.")
    has_shared: bool = arg(
        doc="Whether a second (MoE sink) partial is summed into the reduce-scatter input."
    )
    dtype: DType = arg(doc="Element type of the partials, residual, cache and outputs.")
    fabric: str = arg(
        doc=(
            "Interconnect label used to identify the measurement; the run uses the "
            "reserved GPUs' links."
        )
    )


DOC = KernelDoc(
    title="Reduce-scatter + short conv + all-gather + residual + RMSNorm",
    summary=(
        "Sum sublayer partials across GPUs, run a causal short convolution on each "
        "GPU's hidden shard, gather it back, add the residual and RMS-normalize."
    ),
    description=(
        "The TP collective after every attention and MLP sublayer. Each GPU "
        "publishes its partial (plus the MoE sink partial when has_shared) to the "
        "shard owners through symmetric memory; each owner sums them, writes the "
        "result into the paged conv-state cache, applies the causal depthwise "
        "k=4 convolution with skip over that cache, adds the residual and "
        "publishes its shard to every GPU, which gathers the full row and applies "
        "RMSNorm. Inputs are random BF16 partials; the conv-state cache uses "
        "the fork's paged layout (block size 4) with one decode token per request "
        "and its prior taps already cached."
    ),
    category="Communication",
    formula=(
        "x = Σᵣ (partialᵣ + sharedᵣ)  (reduce-scatter over hidden_dim)",
        "y = x + Σᵢ conv_weightᵢ · x[pos − 3 + i]",
        "h = residual + y;  normed = h / √(mean(h²) + 1e-6) · weight",
        "algbw = num_tokens·hidden_dim·bytes per element / time",
        "busbw = algbw · 2(num_gpus − 1) / num_gpus",
    ),
    default_metric="time_ms",
    method=(
        "CUDA-event time of graph replays (vLLM runs the call inside a graph) "
        "of 30 calls, each after an L2 clear, minus replays of the 30 clears "
        "alone. After a reference check and 50 direct calls, a host barrier "
        "aligns the GPUs before each of three timed trials; the median trial "
        "per GPU and the slowest GPU are kept."
    ),
    caveats=(
        "Every GPU arrives together, so the time excludes the cross-GPU skew "
        "that the spin-waits absorb in serving (for example uneven expert load "
        "before an MoE call).",
        "The clear before each call leaves its input cold too; in serving the "
        "GEMM that writes it runs just before.",
        "algbw and busbw divide the reduce-scatter input size by the whole call's "
        "time, so they include the conv, residual and norm work.",
    ),
    reference="profiling.runners.comm.rs_sconv_ag_residual_rms_norm_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_lamport_triton",
        # BF16 only: rs_sconv_ag_add_norm rejects other dtypes. The Triton
        # kernels do BF16 arithmetic and bit packing (sm_80+); PDL is enabled
        # only from compute capability 9 (lamport.py LamportRSConv.__init__).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.vllm_lamport_triton",
            function_name="profile_rs_sconv_ag_residual_rms_norm_batch",
        ),
        table_name=KIND,
        args_schema=RsSconvAgResidualRmsNormArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        # LamportRSConv exists only in the rebased alignment checkout.
        subprocess_env="vllm_upstream_fork_env",
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
        doc=BackendDoc(
            summary=(
                "The alignment fork's LamportRSConv.rs_sconv_ag_add_norm: four PDL-chained "
                "Triton launches over torch symmetric memory, run in a real vLLM TP group."
            ),
        ),
    )
)
