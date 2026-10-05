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
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
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

    n: int = arg(unit="elements", doc="Output features of this expert projection.")
    k: int = arg(unit="elements", doc="Input features of this expert projection.")
    dtype: DType = arg(doc="Element type of the FP8 activation and weight.")
    num_local_experts: int = arg(unit="experts", doc="Experts whose weights reside on this GPU.")
    num_tokens: int = arg(unit="tokens", doc="Tokens entering the routed MoE layer.")
    experts_per_token: int = arg(unit="experts", doc="Experts selected for each token.")
    launch_role: str = arg(doc="Projection role: w13 for gate/up or w2 for down.")
    block_size: int = arg(unit="elements", doc="FP8 activation and weight scale granularity.")
    per_group_batches: tuple[int, ...] = arg(
        unit="tokens", doc="Selected token count for each local expert, in expert order."
    )


DOC = KernelDoc(
    title="vLLM fused MoE GEMM",
    summary=(
        "Gather selected token rows and multiply them by FP8 expert weights in one Triton launch."
    ),
    description=(
        "A routed MoE layer runs its expert projections as two launches of "
        "vLLM's Triton fused_moe_kernel with FP8 block scales. The gate-up "
        "launch (w13) reads one activation row per token; the down launch (w2) "
        "reads one row per token-expert pair and multiplies by the router "
        "weight. Each measured launch uses the tile sizes from vLLM's tuned "
        "tables and the expert alignment vLLM would choose for the given "
        "per-expert counts."
    ),
    category="MoE",
    subcategory="Expert compute",
    formula=(
        "rows = sum(per_group_batches)",
        "active_experts = count(per_group_batches > 0)",
        "TFLOPS = 2·rows·n·k / time",
        "logical_bytes = rows·k + 4·rows·⌈k/block_size⌉ "
        "+ active_experts·n·k + 4·active_experts·⌈n/block_size⌉·⌈k/block_size⌉ "
        "+ 2·rows·n",
        "GB/s = logical_bytes / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "Only the Triton launch is timed; expert alignment and one compiling "
        "call run before it."
    ),
    caveats=(
        "TFLOPS and GB/s count routed rows and active experts without the padded "
        "tile work performed by the kernel.",
        "Routing, alignment, activation quantization and final expert reduction "
        "are outside the timed launch.",
    ),
    # No separate PyTorch reference module exists for this launch.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        # FP8 e4m3 conversion needs SM89+.
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            min_compute_capability=(8, 9),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.vllm_fused_moe_triton",
            function_name="profile_vllm_fused_moe_triton",
        ),
        table_name=KIND,
        args_schema=VllmFusedMoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM invoke_fused_moe_triton_kernel with FP8 block scales "
                "and its tuned expert alignment."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py",
        ),
        subprocess_env="vllm_env",
    )
)
