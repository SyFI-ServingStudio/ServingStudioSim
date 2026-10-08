"""RCCL (ROCm Communication Collectives Library) all-reduce runner.

This is the MI300X / Infinity-Fabric twin of :mod:`profiling.runners.comm.nccl`.
On a ROCm PyTorch build the ``torch.distributed`` process-group backend string
``"nccl"`` aliases to RCCL (``librccl``), so the *measurement code is identical*
to the NCCL runner — it spawns ``num_gpus`` ranks via ``TorchMpLauncher``, times
``dist.all_reduce`` on each size in one live group, and returns one
``RunnerResult`` per spec with the ``algbw``/``busbw`` ring formulas.

Why a separate module (not just reusing the ``nccl`` backend): the cache row and
the simulator backend pin must name the real library — RCCL, not NCCL — and this
backend is AMD-arch gated (``arch_targets={"CDNA3"}``) and runs in the ROCm venv
(``subprocess_env="vllm_rocm_env"``), whereas the ``nccl`` backend is the NVIDIA
path. The body delegates to the shared NCCL measurement so there is one
collective-timing implementation, not two.

vLLM-ROCm reaches the large (non-fused) all-reduce through the same pynccl path
as the NVIDIA engine; that path dispatches RCCL on ROCm. This runner reproduces
exactly that call, so the measured curve is the large all-reduce vLLM-ROCm runs.
"""

from __future__ import annotations

from profiling.runners.comm.nccl import profile_all_reduce_batch as _profile_nccl_batch
from profiling.runners.metrics import RunnerResult


def profile_all_reduce_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    """Profile a homogeneous chunk of RCCL all-reduce configs with ONE rank-group
    spawn. Delegates to the shared NCCL measurement: on the ROCm Torch build this
    subprocess uses (``vllm_rocm_env``), the ``"nccl"`` process-group backend is
    RCCL, so the same ``dist.all_reduce`` timing measures RCCL over Infinity
    Fabric. All specs share ``num_gpus`` (the chunk is grouped by gpu_count)."""
    return _profile_nccl_batch(kwargs_list)
