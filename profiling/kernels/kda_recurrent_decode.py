"""KDA (Kimi Delta Attention) recurrent decode as one public call.

Wire string: ``"kda_recurrent_decode"``. The facade stem is
``get_kda_recurrent_decode_times`` / ``count_missing_kda_recurrent_decode``.

The measured boundary is one call of the vendored
``vllm.models.glm5next.nvidia.ops.third_party.kda.fused_recurrent_kda``, made
exactly as ``Glm5NextLinearAttention._forward`` makes it on a plain decode
step (no prefill, no speculative decoding). The call launches five kernels:

- the ``.contiguous()`` copies of q, k, v and beta. All four are row-strided
  views into the merged ``in_proj_qkvbfg_a`` output, because the short conv
  updates its q|k|v slice in place;
- one ``fused_recurrent_gated_delta_rule_fwd_kernel`` in KDA mode. It computes
  the per-channel gate (``COMPUTE_GATE``), the beta sigmoid, and the q/k l2norm
  in the kernel, and updates the fp32 state in place through
  ``ssm_state_indices``.

``_causal_conv1d_update`` runs before the call and is costed elsewhere.

This is a separate kind, not a backend of ``gdn_recurrent_decode``. That kind
times vLLM's ``fused_recurrent_gated_delta_rule_packed_decode``, whose gate is
a scalar per value head (``-exp(A_log[h]) * softplus(a + dt_bias[h])``, from
``a[B,HV]``). Its operands are the packed ``mixed_qkv`` plus ``a``, ``b``,
``A_log[HV]`` and ``dt_bias[HV]``. KDA's gate is per key channel,
``lower_bound * sigmoid(exp(A_log[h]) * (raw_g + dt_bias[h,:]))``, and is
computed from a bf16 ``raw_g[B,H,K]`` and an fp32 ``dt_bias[H*K]``. KDA has no
separate q/k and value head counts, and a different entry point that includes
four copy launches.

The state is fp32 ``[slots, H, V, K]``. ``kda_state_dtype`` hard-codes the
recurrent state to fp32 whatever ``mamba_cache_dtype`` is, so state dtype is
not an arg. ``lower_bound=-5.0``, ``scale=K**-0.5``, BV=8 and one warp are
fixed by the production path.
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

KIND: str = "kda_recurrent_decode"


@dataclass(frozen=True)
class KdaRecurrentDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded together, one token each.")
    num_heads: int = arg(unit="heads", doc="Heads, each with its own query, key, value and state.")
    head_dim: int = arg(unit="elements", doc="Features in each query, key and value head.")
    dtype: DType = arg(doc="Element type of the activations, gate input and output.")


DOC = KernelDoc(
    title="KDA recurrent decode",
    summary="Update each request's KDA state, gated per key channel, and read one output token.",
    description=(
        "The core of KDA decode. KDA is a gated delta rule like Gated DeltaNet, "
        "but its decay gate has one value per key channel instead of one per "
        "head. vLLM's fused_recurrent_kda call first copies q, k, v and beta out"
        " of the merged input projection, as four copy launches. One Triton "
        "kernel then computes the gate from the raw projection, applies the "
        "beta sigmoid and the q and k L2 norms, updates each request's FP32 "
        "state in place in a slot-indexed state pool, and writes the output. "
        "Activations are BF16."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "decay = exp(−5·sigmoid(exp(A_log)·(raw_g + dt_bias))), per head and key channel",
        "q, k = L2Norm(q, k); q = q·head_dim⁻¹ᐟ²; beta = sigmoid(b)",
        "S′ = diag(decay)·S + k·(beta·(v − kᵀ(diag(decay)·S)))ᵀ; y = qᵀS′",
        "FLOPs = 7·batch_size·num_heads·head_dim²",
        "bytes = 10·batch_size·num_heads·head_dim + 2·batch_size·num_heads + "
        "8·batch_size·num_heads·head_dim²",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Five warm-up calls run first; the four copies and the "
        "recurrent kernel are all counted. Every call advances the states in "
        "place, as vLLM does. Before timing, one call is checked against a "
        "single-step PyTorch reference and the states are restored."
    ),
    caveats=(
        "The kernel takes any head_dim.",
        "Only a plain decode step is measured, one token per request and no speculative decoding.",
        "The short causal convolution before the call is a separate kind, gdn_causal_conv_decode.",
        "GB/s counts each input once and leaves out the copies' extra reads and writes.",
    ),
    reference="profiling.runners.attention.kda_recurrent_decode_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recurrent_decode_vllm_triton",
            function_name="profile_kda_recurrent_decode_vllm_triton",
        ),
        table_name=KIND,
        args_schema=KdaRecurrentDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's fused_recurrent_kda: four input copies, then one Triton "
                "kernel that gates, normalizes and updates the indexed FP32 states."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py",
        ),
        # vLLM defaults TRITON_CACHE_AUTOTUNING=1, which would let the first
        # shape tuned in a TRITON_CACHE_DIR fix the configs of every later row
        # and process. The runner instead tunes at a documented anchor per
        # worker and records it through row_provenance.
        worker_env=(("TRITON_CACHE_AUTOTUNING", "0"),),
        row_provenance_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recurrent_decode_vllm_triton",
            function_name="row_provenance",
        ),
    )
)


# The AMD/ROCm path. common/kda.py's ``is_rocm()`` branch dispatches the same
# ``fused_recurrent_kda`` call to the glm5next ``amd`` subtree (AMD Triton
# kernels); the callable's signature is identical to the NVIDIA one, so the
# operand layout and keyword set are unchanged. Measured on MI300X in the
# vllm_rocm_env image, timed kernel-only via rocprofv3 (the ROCm counterpart of
# the CUPTI path the NVIDIA backend uses). NVIDIA B200 rows are untouched.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_rocm",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            arch_targets=frozenset({"CDNA3"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recurrent_decode_torch_rocm",
            function_name="profile_kda_recurrent_decode_torch_rocm",
        ),
        table_name=KIND,
        args_schema=KdaRecurrentDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_rocm_env",
        doc=BackendDoc(
            summary=(
                "AMD glm5next fused_recurrent_kda (is_rocm() dispatch): the same "
                "four input copies and recurrent kernel as the NVIDIA backend, "
                "run as AMD Triton kernels on CDNA3."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/models/glm5next/amd/ops/third_party/kda/kernels.py",
        ),
    )
)
