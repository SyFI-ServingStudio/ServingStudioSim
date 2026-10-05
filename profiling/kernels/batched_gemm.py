"""Batched-GEMM kernel kind for MLA's per-head query absorption and value expansion.

Each backend freezes one production storage layout (vLLM's packed Q-absorption
or V-up views) instead of presenting its constants as generic batched GEMM
behavior. Any per-rank head count launches, except that the padded V-up layout
holds at most 64 heads.
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

KIND: str = "batched_gemm"


@dataclass(frozen=True)
class BatchedGemmArgs(KernelArgs):
    num_batches: int = arg(
        unit="heads", doc="Attention heads multiplied independently on this GPU."
    )
    m: int = arg(unit="tokens", doc="Token rows per attention head.")
    n: int = arg(unit="elements", doc="Output features per attention head.")
    k: int = arg(unit="elements", doc="Input features per attention head.")
    dtype: DType = arg(doc="Element type of the inputs and output; the backends use bf16.")


DOC = KernelDoc(
    title="MLA batched GEMM",
    summary="Multiply one activation and weight matrix pair per attention head in MLA.",
    description=(
        "vLLM's MLA runs two per-head batched multiplies. Query absorption "
        "multiplies each head's query by W_UK, mapping it into the compressed KV "
        "space; value expansion multiplies each head's attention output by W_UV, "
        "mapping it back to the value head. num_batches is the heads on this GPU "
        "and m the tokens. Each backend builds the same strided views of the "
        "packed weight, activation and output that vLLM passes to torch.bmm."
    ),
    category="GEMM",
    formula=(
        "C[b, m, n] = A[b, m, k] · B[b, k, n]",
        "TFLOPS = 2·num_batches·m·n·k / time",
        "GB/s = num_batches·(m·k + k·n + m·n)·bytes per element / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} Each backend times only its torch.bmm call; operand "
        "construction and any head-padding copy are outside the timed call."
    ),
    caveats=(
        "GB/s counts the logical operand elements, not the gaps in the packed "
        "storage the strided views skip.",
        "Each backend fixes k and n to its layout: k = 192 or 256, n = 512 for "
        "query absorption and k = 512, n = 256 for value expansion.",
        "torch_mla_v_up reads an attention output padded to 64 heads, so it takes "
        "at most 64 heads.",
    ),
    reference="profiling.runners.gemm.batched_gemm_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_mla_q_absorb",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.batched_gemm",
            function_name="profile_mla_q_absorb",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="torch.bmm for query absorption, on vLLM's packed W_UK view.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_mla_v_up",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.batched_gemm",
            function_name="profile_mla_v_up",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "torch.bmm for value expansion, on vLLM's packed W_UV view and strided output."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py",
        ),
    )
)


# An MLA layout with qk_nope 256 and no RoPE part, v_head 256, kv_lora 512 and an
# unpadded attention output. Same torch.bmm call as the two backends above.
_MLA_URL = "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py"
for _backend, _function, _summary in (
    (
        "torch_mla_q_absorb_no_rope",
        "profile_mla_q_absorb_no_rope",
        "torch.bmm for query absorption when the query head has no RoPE part, "
        "on vLLM's packed W_UK view.",
    ),
    (
        "torch_mla_v_up_unpadded",
        "profile_mla_v_up_unpadded",
        "torch.bmm for value expansion from an attention output with no head "
        "padding, on vLLM's packed W_UV view.",
    ),
):
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=_backend,
            supports=BackendSupport(
                compute=frozenset({DType.BF16}),
            ),
            runner_ref=RunnerRef(
                module_name="profiling.runners.gemm.batched_gemm",
                function_name=_function,
            ),
            table_name=KIND,
            args_schema=BatchedGemmArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="vllm_env",
            doc=BackendDoc(summary=_summary, url=_MLA_URL),
        )
    )
