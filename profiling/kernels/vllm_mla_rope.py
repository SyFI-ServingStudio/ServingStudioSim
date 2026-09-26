"""vLLM MLA query RoPE, as the inductor fusion actually emits it.

This is not a rope-slice-sized kernel. In ``DeepseekV2MLAAttention.forward``
(vllm ``model_executor/models/deepseek_v2.py``) the rope call is followed by

    q[..., qk_nope_head_dim:] = q_pe

and the mutated ``q`` is then consumed by ``self.attn(q, k, v)``, so the
functionalized inductor graph materialises a whole new ``q`` rather than
scattering into its rope slice. The generated triton kernel iterates the full
``[num_tokens, num_heads, qk_nope + rope]`` space, loads and stores every
column, and applies ``tl.where(col >= qk_nope, roped, original)`` -- reading and
writing 100% of q to change ``rope_dim / (qk_nope + rope_dim)`` of it.

That is why it sustains ~0.9 TB/s where a streaming pointwise kernel reaches
~4.3 TB/s on the same GPU, and why pricing it as an elementwise leaf over the
rope bytes under-predicted the GLM-5.2 prefill slot by 74%.

The measured anchor is the GLM-5.2 DP8+EP8 nsys capture: at 8192 tokens the
traced ``triton_poi_fused_add_copy_index_select_mul_slice_split_stack_sub_unsqueeze_view_*``
runs 600.21 us, and this kind reproduces 558.75 us (-6.9%).

KNOWN LIMIT: at decode-scale token counts the reproduction is SLOWER than the
served run (5.72 us at 48 tokens against 2.69 us measured at 43), because vLLM
compiles shape-specialised variants for its CUDA-graph capture sizes that a
single compiled callable does not reproduce. Trust this curve in the
prefill-scale range; the small end is pessimistic.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "vllm_mla_rope"


@dataclass(frozen=True)
class VllmMlaRopeArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Query token rows transformed together.")
    num_heads: int = arg(unit="heads", doc="Query heads on this GPU.")
    qk_nope_head_dim: int = arg(unit="elements", doc="Non-rotary columns in each query head.")
    rope_dim: int = arg(unit="elements", doc="Rotary columns in each query head.")
    # The cos/sin table is gathered per token with an indirect index, so its
    # row count is a real cost axis rather than a value-only detail: it decides
    # whether the gather hits cache. `rope_theta` is deliberately absent -- it
    # only changes the table's values, never its shape or access pattern.
    max_position: int = arg(unit="positions", doc="Rows in the rotary lookup table.")
    is_neox_style: bool = arg(doc="Whether the rotary dimensions use NeoX pairing.")
    input_dtype: DType = arg(doc="Element type of the query and key inputs.")


DOC = KernelDoc(
    title="vLLM MLA query RoPE",
    summary="Rotate the MLA query's positional columns and materialize the full query tensor.",
    description=(
        "In vLLM's MLA attention (GLM-5.2), RoPE changes only the rope_dim "
        "columns of each query head, but the Inductor-compiled kernel reads and"
        " writes the whole query, including the qk_nope_head_dim columns it "
        "leaves unchanged, because vLLM writes the rotated slice back and then "
        "passes the whole query to attention. The measurement compiles the same"
        " block for one fixed shape, with sequential positions, and checks it "
        "against an eager run before timing."
    ),
    category="Attention",
    subcategory="MLA",
    formula=(
        "output query = [q_nope, RoPE(q_rope)]",
        "GB/s = 2 · num_tokens · num_heads · (qk_nope_head_dim + rope_dim) · input bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        "GPU kernel time from CUPTI activity records, averaged over repeated "
        "launches with the L2 cache flushed before each. The triton_poi_fused "
        "launch is counted when Inductor emits it; for shapes where Inductor "
        "declines the fusion, the union of all GPU busy intervals is taken "
        "instead."
    ),
    caveats=(
        "GB/s counts one full read and one full write of the query; the "
        "rotary-table gather and the second read of the rotary columns are "
        "excluded.",
        "At decode-scale token counts the measurement is slower than serving, "
        "where vLLM compiles a variant per CUDA-graph size; trust the prefill "
        "range.",
        "Once a worker process has compiled more shapes than dynamo's cache "
        "limit, later shapes run a slower dynamic kernel, so rows filled in "
        "large batches can read up to 5.5x slower.",
    ),
    # The eager comparison is local to the runner, not a separate reference module.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_inductor",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H100", "NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.vllm_mla_rope",
            function_name="profile_vllm_mla_rope_vllm_inductor",
        ),
        table_name=KIND,
        args_schema=VllmMlaRopeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's native rotary embedding compiled by Inductor into one "
                "full-query Triton kernel."
            ),
        ),
    )
)
