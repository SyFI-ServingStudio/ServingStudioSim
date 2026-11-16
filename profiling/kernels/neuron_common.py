"""Shared registration for the imported-only Neuron backends (AWS Trainium2, LNC2).

Not a kernel kind: a helper the Neuron kind modules call. Every such backend

- is named ``neuron_<impl>``, with ``<impl>`` the engine op module's ``IMPLS`` key (or the bench
  row's ``impl``);
- runs on a Trainium2 logical core (``device_family="neuron"``);
- records NRT device time of one compiled region per call, imported from the engine project's
  benchmark JSONL by ``tools/neuron-bench-import`` (``profiling.runners.neuron.imported`` refuses
  to profile live).

Contract: ``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``.
"""

from __future__ import annotations

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import BackendDoc
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
)

NEURON_PREFIX = "neuron_"
NEURON_GPU_NAME = "AWS Trainium2 LNC2"
RUNNER_MODULE = "profiling.runners.neuron.imported"

# The bench `impl` values each GLM-5.3-Flash kind is registered for.
GLM_IMPLS: dict[str, tuple[str, ...]] = {
    "kda_prefill": ("torch", "torch_barrier"),
    "kda_decode": ("torch",),
    "dsa_prefill": ("radix", "topk"),
    "dsa_decode": ("radix", "topk"),
    "moe_partial": ("auto", "dense_local", "cte_hybrid"),
    "dense_ffn_partial": ("dense",),
    "moe_router": ("fp32",),
    "mhc_pre": ("torch_bf16", "torch_fp32"),
    "mhc_post": ("torch",),
    "lm_head": ("torch",),
    "embed": ("torch",),
}


def neuron_backend(impl: str) -> str:
    return NEURON_PREFIX + impl


def neuron_supports(compute: frozenset[DType] | None = frozenset({DType.BF16})) -> BackendSupport:
    return BackendSupport(
        compute=compute, device_family="neuron", architectures=frozenset({"Trainium2"})
    )


def neuron_spec(
    kind: str,
    impl: str,
    args_schema: type[KernelArgs],
    summary: str,
    *,
    compute: frozenset[DType] | None = frozenset({DType.BF16}),
    metric_family: MetricFamily = MetricFamily.COMPUTE,
) -> KernelProfilerSpec:
    """One ``(kind, neuron_<impl>)`` registry row whose rows are imported, never profiled here."""
    return KernelProfilerSpec(
        kernel_kind=kind,
        backend=neuron_backend(impl),
        supports=neuron_supports(compute),
        runner_ref=RunnerRef(module_name=RUNNER_MODULE, function_name="profile"),
        table_name=kind,
        args_schema=args_schema,
        metric_family=metric_family,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(summary=summary, url=None),
    )
