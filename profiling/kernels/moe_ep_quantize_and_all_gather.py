"""MoE dispatch that NVFP4-quantizes bf16 activations while all-gathering them."""

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

KIND = "moe_ep_quantize_and_all_gather"


@dataclass(frozen=True)
class MoeEpQuantizeAndAllGatherArgs(KernelArgs):
    num_gpus: int = arg(unit="GPUs", doc="GPUs in the data-parallel group.")
    per_rank_tokens: tuple[int, ...] = arg(
        unit="tokens", doc="Token counts for each GPU, in rank order."
    )
    hidden_size: int = arg(unit="elements", doc="Elements in each token's hidden state.")
    top_k: int = arg(unit="experts", doc="Experts selected per token.")
    activation_dtype: DType = arg(
        doc="Element type of the gathered activations; nvfp4_e2m1 is measured."
    )
    fabric: str = arg(doc="Interconnect label; this backend requires nvlink.")


DOC = KernelDoc(
    title="MoE all-gather that quantizes on the way out",
    summary=(
        "Quantize each GPU's bf16 activations to NVFP4 and gather them, with the "
        "top-k routing, to every data-parallel GPU in one launch."
    ),
    description=(
        "The same exchange as moe_ep_quantized_all_gather, but it starts from the "
        "bf16 hidden states: the kernel computes the NVFP4 values and e4m3 block "
        "scales (one per 16 elements, linear layout) and writes them, with the fp32 "
        "top-k weights and int32 expert IDs, straight into every GPU's landing rows. "
        "It replaces a separate NVFP4 quantization kernel plus the gather. GPU r "
        "contributes per_rank_tokens[r] tokens; every GPU ends with all of them in "
        "rank order."
    ),
    category="Communication",
    formula=(
        "token bytes = hidden_size/2 + hidden_size/16 + 8·top_k",
        "total bytes = sum(per_rank_tokens) · token bytes",
        "algbw = (total bytes / num_gpus) / time",
        "busbw = (total bytes − min(per_rank_tokens) · token bytes) / time",
    ),
    default_metric="time_ms",
    method=(
        "N calls captured in one CUDA graph and replayed 20 times after 20 warm-up "
        "calls; CUDA-event time per call; the slowest rank is kept. Every shape is "
        "first checked bit for bit against vLLM's own NVFP4 quantization "
        "(moe_kernel_quantize_input) of the same inputs."
    ),
    caveats=(
        "The scales are produced in the linear layout; the swizzle the grouped GEMM "
        "needs happens after the gather and is not part of this measurement.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="mnnvl_lamport",
        supports=BackendSupport(compute=frozenset({DType.NVFP4_E2M1})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.moe_ep_mnnvl_lamport",
            function_name="profile_moe_ep_quantize_and_all_gather_batch",
        ),
        table_name=KIND,
        args_schema=MoeEpQuantizeAndAllGatherArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        # vLLM's quantization is the per-spec reference; FlashInfer 0.7.3 maps the
        # fabric memory and JIT-builds the kernel.
        subprocess_env="vllm_env",
        imex_channel=True,
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
        doc=BackendDoc(
            summary=(
                "One kernel launch: each thread quantizes 16-byte vectors of its rows "
                "(TRT-LLM's NVFP4 math) and stores them directly into every peer's "
                "landing rows in fabric (MNNVL) memory; Lamport-style arrival, no "
                "separate barrier."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/comm/moe_alltoall_lamport.cuh",
        ),
    )
)
