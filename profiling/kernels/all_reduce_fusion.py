"""FlashInfer shape-aware standalone all-reduce kernel kind.

Backends: ``flashinfer_trtllm`` (TRT-LLM IPC workspace) and ``flashinfer_mnnvl``
(NVLink-multicast workspace, vLLM's default on B200).

Unlike the generic byte-keyed ``all_reduce`` kind, vLLM's FlashInfer path
requires a contiguous ``[num_tokens, hidden_dim]`` tensor and changes its PDL
completion policy at 16 tokens; on MNNVL, FlashInfer also switches from
one-shot to two-shot above ``num_tokens * hidden_dim * num_gpus * 2 = 1 MiB``.
Those two dimensions therefore belong in this kernel's cache identity.
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

KIND: str = "all_reduce_fusion"


@dataclass(frozen=True)
class AllReduceFusionArgs(KernelArgs):
    num_gpus: int = arg(unit="GPUs", doc="GPUs in the tensor-parallel group.")
    num_tokens: int = arg(unit="tokens", doc="Token rows in each GPU's input tensor.")
    hidden_dim: int = arg(unit="elements", doc="Features in each input row.")
    dtype: DType = arg(doc="Element type of the input tensor.")
    fabric: str = arg(
        doc=(
            "Interconnect label used to identify the measurement; the run uses the "
            "reserved GPUs' links."
        )
    )


DOC = KernelDoc(
    title="All-reduce, FlashInfer kernels",
    summary=(
        "Sum a [num_tokens, hidden_dim] tensor across the GPUs of a "
        "tensor-parallel group with FlashInfer's all-reduce."
    ),
    description=(
        "vLLM can end a tensor-parallel block with FlashInfer's all-reduce "
        "instead of NCCL, over a TRT-LLM IPC workspace or, by default on B200, "
        "an NVLink-multicast (MNNVL) workspace. Its cost depends on the "
        "[num_tokens, hidden_dim] shape rather than on the bytes alone, and "
        "vLLM changes its launch at 16 tokens, so it is measured by shape, "
        "unlike the byte-keyed all_reduce. Each GPU contributes a random normal"
        " tensor."
    ),
    category="Communication",
    formula=(
        "output = Σᵣ inputᵣ, on each GPU",
        "algbw = num_tokens·hidden_dim·bytes per element / time",
        "busbw = algbw · 2(num_gpus − 1) / num_gpus",
        "MNNVL: one-shot when num_tokens·hidden_dim·num_gpus·bytes per element ≤ 1 MiB, "
        "otherwise two-shot",
    ),
    default_metric="busbw_gbps",
    method=(
        "Wall-clock time of CUDA graph replays, since vLLM runs this collective"
        " inside a graph. After 50 direct calls and 50 more all-reduces in "
        "graph replays, the timed replays run from a barrier to a device "
        "synchronize, and the slowest rank's mean per all-reduce is kept. "
        "flashinfer_trtllm puts one all-reduce in each graph and times 100; "
        "flashinfer_mnnvl puts ten in each graph and times 1000, after checking"
        " one output against the FP32 sum of every rank's input."
    ),
    caveats=(
        "PDL is on, and completion is signaled at kernel end only above 16 "
        "tokens, as vLLM launches it.",
        "Host time around graph replays includes the replay launch cost, not only kernel time.",
        "The fabric argument labels the row; the run uses whatever links "
        "connect the reserved GPUs.",
        "flashinfer_mnnvl is measured only where vLLM routes to it: fabric "
        "nvlink, a TP size vLLM has a FlashInfer all-reduce budget for, and a "
        "tensor within that budget; vLLM sends larger tensors to another all-reduce.",
    ),
    # No separate PyTorch reference implementation exists for this kind.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.flashinfer_trtllm",
            function_name="profile_all_reduce_fusion_batch",
        ),
        table_name=KIND,
        args_schema=AllReduceFusionArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="flashinfer_pip_env",
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
        doc=BackendDoc(
            summary=(
                "FlashInfer's TRT-LLM allreduce_fusion call with the kAllReduce "
                "pattern and PDL enabled."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/allreduce.py",
        ),
    )
)

# vLLM's default FlashInfer all-reduce backend on single-node B200
# (``VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto`` -> mnnvl). It runs in the vLLM
# fork venv because production pins FlashInfer 0.6.18 there; the project venv
# ships 0.6.11, older than the mnnvl CUDA-graph fix vLLM relies on (>= 0.6.12).
# One-shot vs two-shot is FlashInfer's AUTO rule inside the call, not an arg.
# vLLM sizes the workspace, and bounds the tensors it sends here, by a budget
# keyed by the exact compute capability (SM103 is not SM100); the runner
# reproduces SM100's (``SM100_MAX_SIZE_MB``), so the backend is SM100 only
# until it carries another capability's budget.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_mnnvl",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_100a"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.flashinfer_mnnvl",
            function_name="profile_all_reduce_fusion_batch",
        ),
        table_name=KIND,
        args_schema=AllReduceFusionArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
        doc=BackendDoc(
            summary=(
                "FlashInfer's allreduce_fusion over an NVLink-multicast workspace, "
                "as vLLM calls it on B200; FlashInfer picks one-shot or two-shot by size."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/trtllm_mnnvl_ar.py",
        ),
    )
)

# MI300X analytic Infinity-Fabric all-reduce roofline (GLM-5.3-Flash port,
# decision #42). vLLM-ROCm runs an RCCL / aiter custom all-reduce over Infinity
# Fabric, not FlashInfer MNNVL (NVLink-multicast only); a *measured* RCCL
# all-reduce needs a 4+ GPU Infinity-Fabric group. Where that measurement is
# unavailable, this backend supplies a principled ring all-reduce roofline derived
# from the MI300X Infinity-Fabric bandwidth (``profiling.runners.comm.fabric_roofline``)
# so a real MI300X ``timing-predict`` resolves the kind instead of rejecting the
# NVIDIA-only MNNVL default at ``BackendSupport.allows``. The cost is DERIVED, not
# timed: no rank group (``gpu_count`` defaults to 1), and it is MI300X /
# InfinityFabric-gated, so every NVIDIA target -- B200 included -- keeps
# flashinfer_mnnvl and stays byte-identical.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="rocm_fabric_roofline",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            arch_targets=frozenset({"CDNA3"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.fabric_roofline",
            function_name="profile_all_reduce_fusion_fabric_roofline",
        ),
        table_name=KIND,
        args_schema=AllReduceFusionArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_rocm_env",
        doc=BackendDoc(
            summary=(
                "Analytic Infinity-Fabric all-reduce roofline: the ring "
                "all-reduce data-movement law 2(N-1)/N · bytes / 896 GB/s (the "
                "MI300X Infinity-Fabric bandwidth). A derived fabric cost for the "
                "GLM-5.3-Flash MI300X port, not a measured multi-GPU row."
            ),
        ),
    )
)
