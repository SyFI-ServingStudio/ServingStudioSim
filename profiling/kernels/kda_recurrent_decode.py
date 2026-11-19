"""KDA (Kimi Delta Attention) recurrent decode as one public call.

Wire string: ``"kda_recurrent_decode"``. The facade stem is
``get_kda_recurrent_decode_times`` / ``count_missing_kda_recurrent_decode``.

The measured boundary is one call of the vendored
``vllm.models.glm5next.nvidia.ops.third_party.kda.fused_recurrent_kda``, made
exactly as ``Glm5NextLinearAttention._forward`` makes it on a plain decode
step (no prefill, no speculative decoding), or, with ``num_tokens`` set, on a
speculative verify step (the spec branch: ``q = num_tokens / batch_size`` tokens
per request, ``ssm_state_indices`` ``[B, q]`` and ``num_accepted_tokens``; the
kernel reads each request's last accepted state once and stores the state after
every token into that token's slot). The call launches five kernels:

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

The state pool is ``[slots, H, V, K]`` at vLLM's ``mamba_ssm_cache_dtype``:
fp32 by default (``auto``), bf16 when selected (``kda_state_dtype``). Rows
measured before ``state_dtype`` existed are fp32 and keep their keys (absent =
fp32). ``lower_bound=-5.0``, ``scale=K**-0.5``, BV=8 and one warp are fixed by
the production path.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg, optional_arg
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
    num_tokens: int | None = optional_arg(
        unit="tokens",
        same_as="batch_size",
        doc=(
            "Tokens across the batch, an equal number per request: a speculative verify "
            "step's accepted token plus drafts, each with its own state slot. Absent when "
            "each request decodes one token."
        ),
    )
    state_dtype: DType | None = optional_arg(
        absent=DType.FP32,
        doc="Element type of the recurrent state pool. Absent means fp32.",
    )


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
        "T = num_tokens (batch_size when absent); s = bytes(state_dtype), fp32 when absent",
        "FLOPs = 7·T·num_heads·head_dim²",
        "bytes = 10·T·num_heads·head_dim + 2·T·num_heads + s·(batch_size + T)·num_heads·head_dim²",
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
        "num_tokens > batch_size is a speculative verify step: each request reads the state "
        "at its last accepted slot and writes a full state after every one of its "
        "num_tokens / batch_size tokens; bytes then count one state read plus one write per "
        "token.",
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
        tunes_every_worker=True,
        row_provenance_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recurrent_decode_vllm_triton",
            function_name="row_provenance",
        ),
    )
)
