"""Two-tap grouped dynamic causal convolution (DFlash2 drafter), optionally plus a residual."""

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

KIND = "dynamic_causal_conv"


@dataclass(frozen=True)
class DynamicCausalConvArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows convolved (every stream's block, back to back).")
    hidden: int = arg(unit="elements", doc="Channels of each row.")
    group_size: int = arg(
        unit="elements",
        doc="Channels that share one dynamic tap pair (hidden / group_size groups).",
    )
    residual: bool = arg(doc="Whether a residual row is read and added to the result.")
    dtype: DType = arg(doc="Element type of the rows, taps, residual and result.")


DOC = KernelDoc(
    title="Dynamic causal convolution",
    summary=(
        "Mix each row with the row before it in its block, with per-row, per-channel-group "
        "taps, and optionally add a residual row."
    ),
    description=(
        "The DFlash2 drafter wraps its attention and its MLP in a two-tap causal "
        "convolution over each stream's block of draft rows. The taps are a static base "
        "kernel plus a dynamic part the layer projects from the normed row, one pair per "
        "group of group_size channels: k0 = base0 + dyn0, k1 = base1 + dyn1 (each rounded "
        "to bf16), y = x · k0 + x_prev · k1, where x_prev is the previous row of the same "
        "block (zero for a block's first row). The output-side call adds the layer's "
        "residual row to y. Each layer runs it four times: before and after attention, "
        "before and after the MLP, the after-calls with the residual."
    ),
    category="Other",
    formula=(
        "y[r, c] = bf16(x[r, c] · k0[r, c / g] + x[r - 1, c] · k1[r, c / g]) (+ residual[r, c])",
        "GB/s = num_tokens · hidden · 2 · (2 + residual) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} A guard call is compared with an independent PyTorch reference "
        "before capture; only the _dconv_kernel launch is counted."
    ),
    caveats=(
        "Blocks are 16 rows, TensorFold's drafting block on GB10; the block length only "
        "masks the previous-row load and does not change the bytes.",
        "GB/s counts x, the result and the residual; the dynamic taps (hidden / group_size "
        "· 4 values a row) and the base kernel are not counted.",
    ),
    reference="profiling.runners.elementwise.dynamic_causal_conv_reference",
)


# TensorFold perf-study 6469730a, tensorfold/families/qwen3_5/cuda/dflash2.py
# `_dconv(x, dyn, base, branch, group_size, residual, seg)`: the Triton
# `_dconv_kernel` on grid (rows, cdiv(hidden, 1024)), four warps, bf16 stores.
# Plain Triton, so Triton's sm_80 floor for bf16.
TENSORFOLD_TRITON = "tensorfold_triton"

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend=TENSORFOLD_TRITON,
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.dynamic_causal_conv_tensorfold_triton",
            function_name="profile_dynamic_causal_conv_tensorfold_triton",
        ),
        table_name=KIND,
        args_schema=DynamicCausalConvArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's Triton _dconv_kernel (DFlash2 _dconv): one launch, a program per "
                "row and 1024 channels, the residual add fused."
            ),
        ),
    )
)

__all__ = ["DynamicCausalConvArgs", "KIND"]
