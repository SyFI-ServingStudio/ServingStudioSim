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
elsewhere. ``lower_bound=-5.0``, ``scale=K**-0.5``, BV=32, 4 warps and 2
stages are fixed by the production path.

``state_dtype`` is the checkpoint pool's element type (vLLM
``--mamba-ssm-cache-dtype``). The fork's Triton op loads the checkpoint at its
pool dtype and runs the recurrence in fp32, so it serves fp32 and bf16 pools.

Backend ``flashinfer_cute`` is FlashInfer's drop-in for the same op:
``recurrent_kda(..., disable_state_update=True, correction_cache=...,
kg_cache=...)``, which reaches ``kda_decode_wy_output_only.kda_recoverssm_verify``.
It evaluates the window in the chunk-parallel WY form with tensor-core MMAs
instead of a per-token recurrence, reads the checkpoint once and writes the
same fp32 correction and raw key/gate records RecoverSSM's commit consumes.
It needs a bf16 pool and ``K == V == 128``.
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
    state_dtype: DType = arg(doc="Element type of the checkpoint state pool the call reads.")


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
        # It loads the checkpoint at the pool's dtype and upcasts to fp32.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP32, DType.BF16}),
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


# FlashInfer's frozen verify. kda_decode_wy_output_only.py: "Requires SM90+
# (TMA + mbarrier), validated on SM100a (B200); K == V == 128; bf16 I/O and
# state"; recurrent_kda rejects disable_state_update without a bf16 pool.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_cute",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            min_compute_capability=(9, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recoverssm_flashinfer",
            function_name="profile_kda_recoverssm_verify_flashinfer_cute",
        ),
        table_name=KIND,
        args_schema=KdaRecoverSSMVerifyArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # FlashInfer main after 0.7.1 with CuTe DSL >= 4.7 (see env.py).
        subprocess_env="flashinfer_kda_env",
        doc=BackendDoc(
            summary=(
                "FlashInfer recurrent_kda frozen verify (disable_state_update=True with "
                "RecoverSSM correction and key/gate caches): the WY-parallel tensor-core "
                "kernel over the window, reading a bf16 checkpoint once."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/kda_decode.py",
        ),
        row_provenance_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recoverssm_flashinfer",
            function_name="row_provenance_flashinfer_cute",
        ),
    )
)
