"""Dense MLP fused by production Neuron serving libraries."""

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

KIND = "neuron_dense_mlp"


@dataclass(frozen=True)
class NeuronDenseMlpArgs(KernelArgs):
    m: int = arg(unit="tokens", doc="Token rows in the fused MLP invocation.")
    hidden: int = arg(unit="elements", doc="Input and output hidden width.")
    intermediate: int = arg(unit="elements", doc="Gate and up projection width.")
    dtype: DType = arg(doc="Compute and weight element type; BF16 initially.")


DOC = KernelDoc(
    title="Neuron dense MLP",
    summary="Fuse gate/up projections, the SiLU product, and the down projection.",
    description=(
        "The public Neuron serving MLP executes all three "
        "dense projections and SiLU multiplication within one compiled invocation. "
        "This specialization has no residual add, normalization, bias or quantization."
    ),
    category="GEMM",
    formula=(
        "y = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down",
        "TFLOPS = 6*m*hidden*intermediate / time",
        "GB/s = 2*(2*m*hidden + 3*hidden*intermediate) / time",
    ),
    default_metric="tflops",
    method=(
        "Median native nc_exec_running interval union across both physical cores "
        "of one LNC2 logical unit. Compilation, validation, copies and warmup "
        "are excluded. Device timestamps are synchronized by the Neuron runtime."
    ),
    caveats=(
        "Times and capacity describe one LNC2 unit, not all units on a chip.",
        "TFLOPS omits SiLU transcendental work; GB/s counts logical external operands.",
        "Energy sampling is unavailable; energy_j is zero.",
        "nki_library: only m=1, hidden=4096, intermediate=14336 has passed validation. "
        "m128 exceeds this SDK's SBUF limits in both CTE and DECODE modes.",
        "nki_library uses default AUTO dispatch with public transpose gate/up layout. "
        "TKG keeps projections and activation FP32, then rounds the SiLU product to BF16.",
        "vllm_neuron covers BF16 hidden=4096, intermediate=3584 at m=1,16,512, "
        "using the stock public MLP with both column-tiling options enabled. "
        "It validates against independent FP32 math with relative L2 <=0.02 and "
        "peak error/reference peak <=0.05. This is a rank-local TP4 shard; "
        "collectives and full-forward additivity are outside this measurement.",
    ),
    reference="profiling.runners.neuron.dense.dense_mlp_reference",
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_neuron",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            device_family="neuron",
            architectures=frozenset({"Trainium2"}),
        ),
        runner_ref=RunnerRef("profiling.runners.neuron.vllm_mlp", "profile_dense_mlp"),
        row_provenance_ref=RunnerRef("profiling.runners.neuron.vllm_mlp", "row_provenance"),
        table_name=KIND,
        args_schema=NeuronDenseMlpArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_neuron_env",
        doc=BackendDoc(
            summary=(
                "vllm_neuron.functional.mlp.mlp, one BF16 rank-local LNC2 invocation; "
                "stock column tiling, no norm/residual/bias or collectives."
            ),
            url=(
                "https://github.com/vllm-project/vllm-neuron/blob/"
                "f8abae640a43824c1dc73aed3cf2f67b83bce507/vllm_neuron/functional/mlp.py"
            ),
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="nki_library",
        # NKI [2] is the Trainium2 physical pair in one LNC2 logical unit.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            device_family="neuron",
            architectures=frozenset({"Trainium2"}),
        ),
        runner_ref=RunnerRef("profiling.runners.neuron.dense", "profile_dense_mlp"),
        table_name=KIND,
        args_schema=NeuronDenseMlpArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="neuron_env",
        doc=BackendDoc(
            summary="nkilib.core.mlp.mlp.mlp[2], BF16 with normalization disabled.",
            url="https://github.com/aws-neuron/nki-library/blob/2.32_release/src/nkilib_src/nkilib/core/mlp/mlp.py",
        ),
    )
)
