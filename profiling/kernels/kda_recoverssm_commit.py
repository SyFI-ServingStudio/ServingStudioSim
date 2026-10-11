"""KDA (Kimi Delta Attention) RecoverSSM accepted-state commit as one public call.

Wire string: ``"kda_recoverssm_commit"``. The facade stem is
``get_kda_recoverssm_commit_times`` / ``count_missing_kda_recoverssm_commit``.

The measured boundary is one ``KDARecoverSSMCommitContext.commit`` call from
the vLLM fork's ``vllm.models.kimi_k3.nvidia.ops.recoverssm``. vLLM makes it
once per KDA KV-cache group after sampling (``RecoverSSMState.commit_step``),
and one call covers every layer of the group. It launches three kernels:

- ``_prepare_commit_plan_kernel`` clamps each request's accepted count to its
  window and plans the destination slots;
- ``_compact_conv_state_kernel`` shifts each layer's short-conv window so it
  ends at the last accepted token;
- ``_commit_kda_state_kernel`` reads each layer's fp32 checkpoint once, folds
  the first ``commit_len`` correction and key/gate records into it, and writes
  it once.

``num_layers`` is the number of KDA layers in the group, a static property of
the deployment: vLLM's hybrid allocator splits the KDA layers into groups the
size of the smallest attention type. ``commit_len`` is the accepted count of
every request (1..spec_query_len; spec decode always accepts the target token).

The non-align cache mode is measured (no prefix caching), so the committed
state goes back to the checkpoint's own slot. ``state_dtype`` is the
checkpoint pool's element type (vLLM ``--mamba-ssm-cache-dtype``); the fold
runs in fp32 and rounds once on the store.

Backend ``tilelang`` is this repository's own kernel for the same call
(``profiling.runners.attention.kda_recoverssm_commit_tilelang``), shipped by
no engine: a Triton prep forms each row's decayed key, then a TileLang fold
adds the accepted rows in closed form with one tensor-core product per state
tile, so the state is read and written once whatever the accepted length;
the conv compaction is the fork's. It runs in the main environment, not the
vLLM fork's.
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

KIND: str = "kda_recoverssm_commit"


@dataclass(frozen=True)
class KdaRecoverSSMCommitArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests whose accepted tokens are committed.")
    commit_len: int = arg(
        unit="tokens", doc="Accepted tokens folded into every request's state (1..spec_query_len)."
    )
    num_heads: int = arg(unit="heads", doc="Local KDA heads per layer.")
    head_dim: int = arg(unit="elements", doc="Features in each key and value head.")
    spec_query_len: int = arg(
        unit="tokens", doc="Verify window per request (k + 1), the record depth."
    )
    num_layers: int = arg(unit="layers", doc="KDA layers committed by the one call.")
    dtype: DType = arg(doc="Element type of the activations and key/gate record.")
    state_dtype: DType = arg(doc="Element type of the checkpoint state pool the call folds into.")


DOC = KernelDoc(
    title="KDA RecoverSSM commit",
    summary=(
        "Fold each request's accepted tokens into its KDA checkpoint and conv window, "
        "for every layer of a KV-cache group in one call."
    ),
    description=(
        "The commit step of KDA with RecoverSSM speculative decoding. After "
        "sampling, one call per KDA cache group plans each request's accepted "
        "length, shifts every layer's short-conv window to end at the last "
        "accepted token, and folds the accepted tokens' correction and key/gate "
        "records into each layer's FP32 checkpoint: one read and one write of "
        "the state per request, head and layer. Three Triton launches."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "S ← S·Π_t diag(decay_t) + Σ_t (c_t·k̂_tᵀ)·Π_{u<t} diag(decay_u), t < commit_len",
        "FLOPs = num_layers·batch_size·num_heads·(3·commit_len + 2)·head_dim²",
        "bytes = num_layers·batch_size·(8·num_heads·head_dim² + "
        "8·commit_len·num_heads·head_dim + 4·3·num_heads·head_dim·3)",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Five warm-up calls run first; all three launches are "
        "counted. Every call advances the states and conv windows in place, as "
        "vLLM does. Before timing, one call is checked against a PyTorch "
        "reference on the first 16 requests of every layer."
    ),
    caveats=(
        "Every request commits the same commit_len.",
        "Measured without prefix caching (no align-mode boundary state).",
        "The cost grows with num_layers; a deployment calls it once per KDA cache group.",
    ),
    reference="profiling.runners.attention.kda_recoverssm_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        # Plain @triton.jit kernels with no arch-specific instructions; the
        # fork serves them in bf16.
        # The fold loads the checkpoint at the pool's dtype, accumulates in
        # fp32 and stores back at the pool's dtype, so fp32 and bf16 pools run.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP32, DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recoverssm_vllm_triton",
            function_name="profile_kda_recoverssm_commit_vllm_triton",
        ),
        table_name=KIND,
        args_schema=KdaRecoverSSMCommitArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # Kimi-K3 RecoverSSM exists only in the alignment checkout rebased onto
        # upstream vLLM (892da08), not in the vllm_env image (3f667d7).
        subprocess_env="vllm_upstream_fork_env",
        doc=BackendDoc(
            summary=(
                "vLLM fork's KDARecoverSSMCommitContext.commit (Kimi-K3): plan, conv "
                "compaction and the fused all-layer state fold."
            ),
            url="https://github.com/vllm-project/vllm",
        ),
    )
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tilelang",
        # T.gemm on fp32 operands runs on Ampere+ tensor cores; fp32 and bf16
        # pools both load into an fp32 fold and store back at the pool's dtype.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP32, DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recoverssm_commit_tilelang",
            function_name="profile_kda_recoverssm_commit_tilelang",
        ),
        table_name=KIND,
        args_schema=KdaRecoverSSMCommitArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "This repository's commit (no engine ships it): a Triton prep, a "
                "TileLang closed-form fold of the accepted rows with one tensor-core "
                "product per state tile, and the conv compaction; three launches."
            ),
        ),
    )
)
