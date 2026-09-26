"""GLM-5.3-Flash KDA (Kimi Delta Attention) recurrent decode as one public call.

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
    batch_size: int
    num_heads: int
    head_dim: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recurrent_decode_vllm_triton",
            function_name="profile_kda_recurrent_decode_vllm_triton",
        ),
        table_name=KIND,
        args_schema=KdaRecurrentDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_fork_env",
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
