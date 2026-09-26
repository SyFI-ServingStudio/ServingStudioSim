"""Batched-GEMM kernel kind with GLM production-layout backend identities.

The backend names deliberately freeze the model-specific Q-absorption and V-up
storage layouts instead of presenting their constants as generic batched GEMM
behavior.
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
    summary="Multiply one activation and weight matrix pair per attention head in GLM-5.2 MLA.",
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
    ),
    reference="profiling.runners.gemm.batched_gemm_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_mla_q_absorb_glm52",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.batched_gemm",
            function_name="profile_mla_q_absorb_glm52",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="torch.bmm for query absorption, on vLLM's packed W_UK view (GLM-5.2 shapes).",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_mla_v_up_glm52",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.batched_gemm",
            function_name="profile_mla_v_up_glm52",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "torch.bmm for value expansion, on vLLM's packed W_UV view and "
                "strided output (GLM-5.2 shapes)."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py",
        ),
    )
)
