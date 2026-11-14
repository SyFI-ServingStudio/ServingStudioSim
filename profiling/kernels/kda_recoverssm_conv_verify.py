"""KDA RecoverSSM speculative verify with the short convolution fused in, as one public call.

Wire string: ``"kda_recoverssm_conv_verify"``. The facade stem is
``get_kda_recoverssm_conv_verify_times`` / ``count_missing_kda_recoverssm_conv_verify``.

The measured boundary is one ``CuteRecoverSSMVerify.__call__`` from the vLLM
fork's ``vllm.models.kimi_k3.nvidia.recoverssm.kernels``: the RecoverSSM verify
a KDA layer runs on SM10x when ``can_use_cute_recoverssm`` holds (head_dim 128,
a width-4 convolution without bias, a window of 1..8 tokens, a bounded gate,
and bf16 activations, conv state and SSM state). One persistent CuTe launch per
layer takes the raw ``q|k|v`` projection rows (before the convolution), runs
the short causal convolution + SiLU over each request's history and window,
writes the rolled conv window back to the request's page, verifies the window
against the request's checkpoint without writing it, and writes the outputs
and the per-token fp32 correction and key/gate records the commit
(``kda_recoverssm_commit``) folds in.

It replaces the two calls of the Triton fallback, ``causal_conv1d_update``
(``gdn_causal_conv_decode``) followed by ``kda_recoverssm_verify``; the
boundary includes the convolution, so it is its own kind.

``conv_weight_dtype`` is the conv taps' element type (the fork accepts bf16
or fp32 taps and compiles a variant per dtype). The checkpoint pool is bf16
by the support rule above, so it is not an axis.

Backends differ only in the conv state layout the deployment stores
(``VLLM_SSM_CONV_STATE_LAYOUT``), which picks the kernel:

- ``vllm_cute``: the default layout, ``[state_len, dim]`` per page (SD); the
  fork runs ``cutedsl/verify.py``.
- ``vllm_cute_dim_first``: ``[dim, state_len]`` per page (DS); the fork runs
  the warp-specialized ``cutedsl/verify_ws.py`` with bulk copies.

``cuda_persistent_wy_mma`` is not the fork's kernel: a vendored persistent CUDA
launch (one CTA per request and head) that does the same conv + verify with
the verify in WY form on tensor cores, writing the same conv window, outputs
and records. It runs on the default SD pages with fp32 taps, for windows 8 (the
vendored source) and 4 (our window-4 build of it, pending the author's review).
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

KIND: str = "kda_recoverssm_conv_verify"


@dataclass(frozen=True)
class KdaRecoverSSMConvVerifyArgs(KernelArgs):
    batch_size: int = arg(
        unit="requests", doc="Requests verified together, each with a full window."
    )
    num_heads: int = arg(unit="heads", doc="Local KDA heads, each with its own state.")
    head_dim: int = arg(unit="elements", doc="Features in each query, key and value head.")
    spec_query_len: int = arg(
        unit="tokens", doc="Window per request: the target token plus the drafts (k + 1)."
    )
    dtype: DType = arg(
        doc="Element type of the activations, conv state, checkpoint, key/gate record and output."
    )
    conv_weight_dtype: DType = arg(doc="Element type of the short convolution's taps.")


DOC = KernelDoc(
    title="KDA RecoverSSM conv + verify",
    summary=(
        "The short causal convolution and the RecoverSSM verify of each request's "
        "speculative window in one tensor-core launch, recording per-token corrections "
        "for a later commit."
    ),
    description=(
        "The speculative-verify step of KDA with RecoverSSM, short convolution "
        "included. One persistent CuTe kernel per layer reads the raw q/k/v "
        "projection rows, convolves them with each request's conv history "
        "(width 4, SiLU), writes the rolled conv window back, then runs the "
        "window through the gated delta rule from the request's BF16 "
        "checkpoint on tensor cores without writing the checkpoint. It writes "
        "one output per token and the FP32 correction and raw key/gate records "
        "the commit consumes."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "x̃_t = silu(Σ_j w_j·x_{t−3+j}) over the history and window, per channel",
        "decay = exp(−5·sigmoid(exp(A_log)·(raw_g + dt_bias))), per head and key channel",
        "per token: S ← diag(decay)·S; c = sigmoid(b)·(v − S·k̂); S ← S + c·k̂ᵀ; y = S·q̂",
        "FLOPs = 7·batch_size·spec_query_len·num_heads·head_dim² + 2·4·T·3·num_heads·head_dim",
        "bytes = 2·T·3HK (raw q/k/v) + 2·T·HK (gate) + 2·T·H (beta) + 2·T·HK (out) + "
        "8·T·HK (records) + 2·batch_size·H·head_dim² (checkpoint) + "
        "2·batch_size·3HK·(3 + state_len) (conv history in, window out), "
        "T = batch_size·spec_query_len, HK = num_heads·head_dim",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Five warm-up calls run first. Every request has a full "
        "window and an accepted count from 1 to spec_query_len for its conv "
        "history. Before timing, one call is checked against an fp32 PyTorch "
        "convolution and the step-by-step RecoverSSM reference on the first 16 "
        "requests: outputs, correction and key/gate records, the rolled conv "
        "window, and that the checkpoint is unchanged."
    ),
    caveats=(
        "Every request verifies a full spec_query_len window.",
        "The accepted-state commit is a separate kind, kda_recoverssm_commit.",
        "The fork falls back to the Triton conv + verify when an activation does "
        "not start on a 16-byte boundary; the measured operands are aligned.",
    ),
    reference="profiling.runners.attention.kda_recoverssm_reference",
)


def _register(
    backend: str,
    function_name: str,
    summary: str,
    module_name: str = "profiling.runners.attention.kda_recoverssm_vllm_cute",
) -> None:
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=backend,
            # can_use_cute_recoverssm: SM10x (is_device_capability_family(100)),
            # bf16 activations, conv state and SSM state.
            supports=BackendSupport(
                compute=frozenset({DType.BF16}),
                sm_targets=frozenset({"sm_100f"}),
            ),
            runner_ref=RunnerRef(
                module_name=module_name,
                function_name=function_name,
            ),
            table_name=KIND,
            args_schema=KdaRecoverSSMConvVerifyArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            # The CuTe RecoverSSM kernels ship in the vllm_env image's upstream base (5629b26).
            subprocess_env="vllm_env",
            doc=BackendDoc(
                summary=summary,
                url="https://github.com/vllm-project/vllm",
            ),
        )
    )


_register(
    "vllm_cute",
    "profile_kda_recoverssm_conv_verify_vllm_cute",
    "vLLM fork's CuteRecoverSSMVerify on the default [state_len, dim] conv pages "
    "(cutedsl/verify.py): conv + verify + records in one persistent launch.",
)
_register(
    "vllm_cute_dim_first",
    "profile_kda_recoverssm_conv_verify_vllm_cute_dim_first",
    "vLLM fork's CuteRecoverSSMVerify on [dim, state_len] conv pages "
    "(VLLM_SSM_CONV_STATE_LAYOUT=DS): the warp-specialized cutedsl/verify_ws.py.",
)
_register(
    "cuda_persistent_wy_mma",
    "profile_kda_recoverssm_conv_verify_cuda_persistent_wy_mma",
    "Vendored persistent CUDA conv + verify on the default [state_len, dim] conv pages: "
    "the window's recurrence in WY form on bf16 hi/lo tensor-core GEMMs, records in the "
    "fork's request-row layout (fp32 taps; window 8 vendored, window 4 our build of it).",
    module_name="profiling.runners.attention.kda_recoverssm_conv_verify_fused",
)
