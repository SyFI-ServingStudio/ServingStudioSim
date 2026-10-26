"""Standalone NKI QKV projection (prefill): one launch unit of the Llama forward."""

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
from profiling.kernels.neuron_llama_forward import VLLM_NEURON_WORKER_ENV

KIND = "neuron_nki_qkv"

# Shared by the six standalone NKI kinds (the others import these). The method
# and the measurement caveats are one text; each kind adds its own description,
# category, formula and caveats.
NKI_BACKEND = "vllm_neuron_nki"
NKI_METHOD = (
    "One reserved LNC2 logical core inside the pinned vLLM Neuron image. The unit is compiled "
    "through the production path: Dynamo full graph, neuron_libtorch, the serving compiler flags. "
    "Weights and inputs are random BF16 at the per-rank shape. The unit is warmed, then executed "
    "50 times under the NRT system profile. time = the median device execution (nc_exec_running, "
    "union of the core's two physical cores), raw. The empty-NEFF floor (a near-empty NEFF, about "
    "0.026 ms) is measured once per batch and kept in the row's provenance, not subtracted."
)
NKI_CAVEATS = (
    ("Rows exist only for compiled variants; the cache is a direct lookup with no interpolation."),
    (
        "The row is the device time of the unit alone and includes the fixed cost one standalone "
        "execution pays. TP collectives and the other units of the layer are separate rows. At "
        "C512 the raw standalone sums match the in-model layer: prefill kernels within 2% of the "
        "layer's kernel spans, patched decode 13% above the layer span (11% below once the floor "
        "is removed per unit). Charging the floor is the composition's decision. Other shapes are "
        "not verified."
    ),
    (
        "FLOPs and bytes are operand-count estimates for the rate columns, not device counters. "
        "Energy is unreported (zero)."
    ),
)


def nki_leaf_spec(kind, args_schema, backend_summary):
    """The registry entry every standalone NKI kind shares: one reserved LNC2
    logical core in the pinned vLLM Neuron image, the ``nki_bench`` runner
    function named after the kind, BF16 on Trainium2."""
    return KernelProfilerSpec(
        kernel_kind=kind,
        backend=NKI_BACKEND,
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            device_family="neuron",
            architectures=frozenset({"Trainium2"}),
        ),
        runner_ref=RunnerRef("profiling.runners.neuron.nki_bench", f"profile_{kind}_batch"),
        row_provenance_ref=RunnerRef("profiling.runners.neuron.nki_bench", "row_provenance"),
        table_name=kind,
        args_schema=args_schema,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_neuron_env",
        neuron_logical_cores=1,
        list_native=True,
        worker_env=VLLM_NEURON_WORKER_ENV,
        doc=BackendDoc(
            summary=backend_summary,
            url="https://github.com/vllm-project/vllm-neuron/tree/f8abae640a43824c1dc73aed3cf2f67b83bce507",
        ),
    )


@dataclass(frozen=True)
class NeuronNkiQkvArgs(KernelArgs):
    token_bucket: int = arg(unit="tokens", doc="Compiled prefill token bucket T (a stock context).")
    tp_size: int = arg(unit="ranks", doc="Four LNC2 ranks on one Trainium2 chip; per-rank shapes.")
    dtype: DType = arg(doc="Weights, activations and KV dtype; BF16 only.")


DOC = KernelDoc(
    title="Standalone NKI QKV projection (prefill)",
    summary=(
        "One prefill layer's fused QKV projection with RoPE, the nkilib qkv kernel, timed alone "
        "per TP4 rank."
    ),
    description=(
        "`vllm_neuron.functional.qkv_proj`: hidden [1, T, 4096] @ W[4096, 1536] with the Llama 3.1 "
        "RoPE applied to Q and K, as the stock Llama3 prefill issues it. Prefill only: on the "
        "per-head block table decode path the projection is fused into neuron_nki_attention_tkg. "
        "Variants are the prefill token buckets of the stock context ladder."
    ),
    category="GEMM",
    formula=(
        "flops = 2 * T * 4096 * 1536 per rank",
        "bytes = 2 * (T*4096 + 4096*1536 + T*1536)",
    ),
    default_metric="time_ms",
    method=NKI_METHOD,
    caveats=NKI_CAVEATS,
)

register(
    nki_leaf_spec(
        KIND,
        NeuronNkiQkvArgs,
        backend_summary=(
            "The NKI kernel of pinned vLLM Neuron 0.24.0.1.1.0, compiled and timed alone on one "
            "LNC2 logical core at the per-rank shapes of TP4."
        ),
    )
)
