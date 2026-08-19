"""vLLM's Triton ``fused_moe_kernel`` -- gather + blocked GEMM in one launch.

Deliberately a separate kind from ``fp8_blockscale_grouped_gemm`` and
``grouped_gemm``. Those model the *permute-then-group* realization: tokens are
physically reordered into per-expert contiguous blocks and each group is handed
to a dense GEMM. This kernel never materializes that permutation. It walks a
padded ``sorted_token_ids`` list produced by ``moe_align_block_size``, and each
Triton program gathers its own ``BLOCK_SIZE_M`` rows before multiplying.

The cost consequence is the reason this needs its own table: work is quantized
to ``BLOCK_SIZE_M``-row blocks *per expert*, so an expert holding 2 rows costs a
whole block. Against a measured vLLM Qwen3.6-35B-A3B-FP8 EP1 run -- 64 decode
tokens, top-8, 256 experts, i.e. ~2 rows per expert -- costing these launches on
the TRT-LLM grouped-GEMM curve over-predicted the down projection by 45.8% and
the gate/up projection by 11.2%. Neither is a cache-fidelity error; they are
different algorithms.

vLLM issues exactly two launches per MoE layer and they are not symmetric, which
``launch_role`` encodes:

* ``w13`` -- gate/up. ``A`` is one row per token, the kernel's ``top_k`` is the
  router's, and routed weights are NOT applied.
* ``w2`` -- down. ``A`` is one row per (token, selected expert), the kernel's
  ``top_k`` is 1, and routed weights ARE applied.

Both consume the *same* ``sorted_token_ids``; only the launch role differs. The
pair is one field rather than two independent flags because vLLM never mixes
them, and two flags would admit two combinations that cannot occur.
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

KIND: str = "vllm_fused_moe"

#: Accepted ``launch_role`` values, mirroring vLLM's own w13/w2 naming.
LAUNCH_ROLE_GATE_UP: str = "w13"
LAUNCH_ROLE_DOWN: str = "w2"


@dataclass(frozen=True)
class VllmFusedMoeArgs(KernelArgs):
    """Cache identity of one ``fused_moe_kernel`` launch.

    ``per_group_batches`` is not redundant with ``num_tokens`` even though it
    sums to ``num_tokens * experts_per_token``: the launch grid is driven by the
    *padded* block count, which is ``sum(cdiv(count, BLOCK_SIZE_M))`` over
    experts. Two distributions with the same total therefore cost differently,
    and a uniform assumption systematically under-counts blocks. This is the
    same reason ``fp8_blockscale_grouped_gemm`` keeps the vector.

    ``block_size`` is the FP8 activation/weight scale granularity (128 for the
    block-scale recipe), not the Triton ``BLOCK_SIZE_M``. The latter is chosen by
    vLLM's own tuned-config lookup from (E, N, K, num_tokens, dtype) and is
    therefore backend behavior, not an independent axis -- the runner lets vLLM
    pick it exactly as production does.
    """

    n: int
    k: int
    dtype: DType
    num_local_experts: int
    num_tokens: int
    experts_per_token: int
    launch_role: str
    block_size: int
    per_group_batches: tuple[int, ...]


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.vllm_fused_moe_triton",
            function_name="profile_vllm_fused_moe_triton",
        ),
        table_name=KIND,
        args_schema=VllmFusedMoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
