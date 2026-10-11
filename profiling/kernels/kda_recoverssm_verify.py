"""KDA (Kimi Delta Attention) RecoverSSM speculative verify as one public call.

Wire string: ``"kda_recoverssm_verify"``. The facade stem is
``get_kda_recoverssm_verify_times`` / ``count_missing_kda_recoverssm_verify``.

The measured boundary is one ``kda_recoverssm_verify`` call from the vLLM
fork's ``vllm.models.kimi_k3.nvidia.ops.recoverssm``, made as
``KimiK3DeltaAttention._forward`` makes it on a speculative-decode step with
``--use-kda-recoverssm``. It is one Triton launch per layer. For every request
it reads the fp32 checkpoint state once, runs the ``spec_query_len`` window
tokens through the KDA gated delta rule in registers, and writes the output
tokens. It does NOT write any per-position state. Instead it writes small
per-token records into the request's cache page: the fp32 correction
``[H, S, V]`` and the raw key and gate ``[H, S, 2K]`` (activation dtype). A
separate commit (kind ``kda_recoverssm_commit``) later folds the accepted
records into the checkpoint.

This is a separate kind, not a backend of ``kda_recurrent_decode``: that call
updates the state in place for one token, copies q/k/v/beta first, and writes
no records. Here the operands are strided views taken directly (no copies), the
state is only read, and the window length is a launch constant.

``causal_conv1d_update`` on the window runs before the call and is costed
elsewhere. The state is fp32 (``kda_state_dtype``), ``lower_bound=-5.0``,
``scale=K**-0.5``, BV=32, 4 warps and 2 stages are fixed by the production
path.
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

KIND: str = "kda_recoverssm_verify"


@dataclass(frozen=True)
class KdaRecoverSSMVerifyArgs(KernelArgs):
    batch_size: int = arg(
        unit="requests", doc="Requests verified together, each with a full window."
    )
    num_heads: int = arg(unit="heads", doc="Local KDA heads, each with its own state.")
    head_dim: int = arg(unit="elements", doc="Features in each query, key and value head.")
    spec_query_len: int = arg(
        unit="tokens", doc="Window per request: the target token plus the drafts (k + 1)."
    )
    dtype: DType = arg(doc="Element type of the activations, key/gate record and output.")


DOC = KernelDoc(
    title="KDA RecoverSSM verify",
    summary=(
        "Verify each request's speculative window against its KDA checkpoint without "
        "writing any state, recording per-token corrections for a later commit."
    ),
    description=(
        "The speculative-verify step of KDA with RecoverSSM. One Triton kernel "
        "reads each request's FP32 checkpoint state once and runs the target and "
        "draft tokens through the gated delta rule in registers, writing one "
        "output per token. Instead of writing a state per draft position, it "
        "stores each token's FP32 correction vector and its raw key and gate in "
        "the request's cache page, so the accepted prefix can be folded into "
        "the checkpoint later by kda_recoverssm_commit. Activations are BF16."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "decay = exp(−5·sigmoid(exp(A_log)·(raw_g + dt_bias))), per head and key channel",
        "per token: S ← diag(decay)·S; c = sigmoid(b)·(v − S·k̂); S ← S + c·k̂ᵀ; y = S·q̂",
        "FLOPs = 7·batch_size·spec_query_len·num_heads·head_dim²",
        "bytes = 10·T·num_heads·head_dim + 2·T·num_heads + 8·T·num_heads·head_dim + "
        "4·batch_size·num_heads·head_dim², T = batch_size·spec_query_len",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Five warm-up calls run first. Every request has a full "
        "window. Before timing, one call is checked against a step-by-step "
        "PyTorch reference on the first 16 requests: outputs, correction "
        "records and key/gate records, and that the checkpoint is unchanged."
    ),
    caveats=(
        "Every request verifies a full spec_query_len window; shorter windows "
        "run the same masked loop.",
        "The short causal convolution before the call is not included.",
        "The accepted-state commit is a separate kind, kda_recoverssm_commit.",
    ),
    reference="profiling.runners.attention.kda_recoverssm_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        # A plain @triton.jit kernel with no arch-specific instructions; the
        # fork serves it in bf16 (the kg record must match the activation dtype).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recoverssm_vllm_triton",
            function_name="profile_kda_recoverssm_verify_vllm_triton",
        ),
        table_name=KIND,
        args_schema=KdaRecoverSSMVerifyArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # Kimi-K3 RecoverSSM exists only in the alignment checkout rebased onto
        # upstream vLLM (892da08), not in the vllm_env image (3f667d7).
        subprocess_env="vllm_upstream_fork_env",
        doc=BackendDoc(
            summary=(
                "vLLM fork's kda_recoverssm_verify (Kimi-K3): one Triton kernel that "
                "reads the checkpoint, verifies the window and writes the records."
            ),
            url="https://github.com/vllm-project/vllm",
        ),
    )
)
