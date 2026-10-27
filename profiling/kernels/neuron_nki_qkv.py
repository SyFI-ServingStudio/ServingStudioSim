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

KIND = "neuron_nki_qkv"

# Shared by the standalone launch-unit kinds (the others import these). The
# method and the measurement caveats are one text; each kind adds its own
# description, category, formula and caveats.
NKI_BACKEND = "vllm_neuron_nki"
# The decode attention leaf's second backend: the stock call's torch fallback.
NEURON_TORCH_BACKEND = "vllm_neuron_torch"
# The vLLM Neuron release whose functional calls the units reproduce, in the
# `vllm_neuron_native_env` image (profiling/container/images.py).
NKI_SOURCE_URL = "https://github.com/vllm-project/vllm-neuron/tree/release-0.29.0.1.2.0"
# Worker environment of the 0.29 image: the native compile route (the server
# reads the same variable), the NRT root communicator a single-rank process
# still needs to load a NEFF that carries collectives (the lite backend emits
# them even for one rank), no EFA on a single chip, a bounded host pool.
NATIVE_WORKER_ENV = (
    ("VLLM_NEURON_BACKEND", "neuron_native"),
    ("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:61234"),
    ("NEURON_SKIP_EFA_AFFINITY", "1"),
    ("OMP_NUM_THREADS", "4"),
)
NKI_METHOD = (
    "One reserved LNC2 logical core inside the vLLM Neuron 0.29 image (Neuron SDK 2.33). The "
    "unit is one torch.nn.Module: one production functional call, or for the glue and head "
    "kinds the torch ops of that launch. It is compiled through the production path (one "
    "Dynamo full graph on the engine's compile backend with the serving compiler flags), "
    "warmed, then executed 50 times under the NRT "
    "system profile started through the engine's own runtime binding. time = the median device "
    "execution (nc_exec_running, union of the core's two physical cores), raw. The empty-graph "
    "floor is measured once per batch and kept in the row's provenance, not subtracted."
)
NKI_CAVEATS = (
    ("Rows exist only for compiled variants; the cache is a direct lookup with no interpolation."),
    (
        "The row is the device time of the unit alone and includes the fixed cost one standalone "
        "execution pays. TP collectives, the glue and the head are separate rows. The composition "
        "llama3_neuron_ops is gated against the engine's own profile of one iteration "
        "(doc/pr/llama31-8b-trn2/README.md, 'Check 1'): at C512 the prefill units add up within "
        "1.3%; the decode units add up 12-17% high on the kernel path, because inside the "
        "engine's one graph they overlap and fuse. On the torch fallback they are 24% high at B1 "
        "and 23% low at B16, where the K/V gathers overlap the MLP's weight stream inside the "
        "engine's one graph. The decode step "
        "is therefore costed from the neuron_decoder_layer and neuron_decode_fixed leaves by "
        "default; these rows remain "
        "the attribution view. Charging the floor is the composition's decision."
    ),
    (
        "FLOPs and bytes are operand-count estimates for the rate columns, not device counters. "
        "Energy is unreported (zero)."
    ),
)


def nki_leaf_spec(
    kind,
    args_schema,
    backend_summary,
    *,
    backend=NKI_BACKEND,
    runner=None,
    module="profiling.runners.neuron.nki_bench",
    logical_cores=1,
):
    """The registry entry every Neuron launch-unit kind shares: the native-route
    image, BF16 on Trainium2, the runner function named after the kind (or
    ``runner``) in ``module``, on ``logical_cores`` reserved LNC2 cores (one for
    a unit alone, the chip's four for a TP4 forward)."""
    return KernelProfilerSpec(
        kernel_kind=kind,
        backend=backend,
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            device_family="neuron",
            architectures=frozenset({"Trainium2"}),
        ),
        runner_ref=RunnerRef(module, runner or f"profile_{kind}_batch"),
        row_provenance_ref=RunnerRef(module, "row_provenance"),
        table_name=kind,
        args_schema=args_schema,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_neuron_native_env",
        neuron_logical_cores=logical_cores,
        list_native=True,
        worker_env=NATIVE_WORKER_ENV,
        doc=BackendDoc(summary=backend_summary, url=NKI_SOURCE_URL),
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
        "RoPE applied to Q and K, as the stock Llama3 prefill issues it. Prefill only: in decode, "
        "both attention paths fuse the projection into neuron_attention_decode. "
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
            "The launch unit of vLLM Neuron 0.29, compiled as one torch.compile graph and timed "
            "alone on one LNC2 logical core at the per-rank shapes of TP4."
        ),
    )
)
