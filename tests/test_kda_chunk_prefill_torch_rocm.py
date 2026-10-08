"""Focused CPU tests for the MI300X ``kda_chunk_prefill`` torch_rocm backend.

No GPU, no DB. The GPU comparison of the AMD callable against the oracle runs
inside the runner on the ROCm host, and its evidence comes from the Slurm step.
Each test names the defect it guards against.
"""

from __future__ import annotations

import inspect
import subprocess
import sys

import torch

from profiling.runners.attention import kda_chunk_prefill_torch_rocm as rocm
from profiling.runners.attention import kda_chunk_prefill_vllm_triton as nvidia


def test_torch_rocm_backend_registered_for_mi300x() -> None:
    # The MI300X AMD path is a separate backend of the same kind, gated to
    # MI300X and routed to the vllm_rocm_env image. The NVIDIA B200 vllm_triton
    # row must be left untouched.
    from profiling.db.registry import find_kernel_profiler_spec

    rocm_spec = find_kernel_profiler_spec("kda_chunk_prefill", "torch_rocm")
    assert rocm_spec.supports.arch_targets == frozenset({"CDNA3"})
    assert rocm_spec.subprocess_env == "vllm_rocm_env"
    assert rocm_spec.runner_ref.function_name == "profile_kda_chunk_prefill_torch_rocm"

    nvidia_spec = find_kernel_profiler_spec("kda_chunk_prefill", "vllm_triton")
    assert nvidia_spec.supports.arch_targets is None
    assert nvidia_spec.subprocess_env == "vllm_env"


def test_rocm_profile_signature_matches_nvidia() -> None:
    # Defect: the ROCm runner drifting from the kind's arg order, so the worker
    # passes positional args the runner reads in a different order.
    rocm_params = list(inspect.signature(rocm.profile_kda_chunk_prefill_torch_rocm).parameters)
    nvidia_params = list(
        inspect.signature(nvidia.profile_kda_chunk_prefill_vllm_triton).parameters
    )
    assert rocm_params == nvidia_params


def test_host_to_device_operands_keep_qkv_as_strided_views() -> None:
    # Defect: building contiguous q/k/v (skips the callable's three copy
    # launches) or a device generator (setup launches kernels the trace counts).
    # On CPU, device="cpu": the only difference from the GPU build is .to().
    shape = rocm.validate_args(
        num_tokens=6,
        max_sequence_length=3,
        num_decode_sequences=0,
        num_heads=2,
        head_dim=128,
        dtype="bf16",
    )
    operands = rocm._build_operands_host_to_device(torch, shape, device=torch.device("cpu"))
    width = 3 * 2 * 128
    assert operands.qkv.shape == (6, width) and operands.qkv.is_contiguous()
    storage = operands.qkv.untyped_storage().data_ptr()
    for name in ("q", "k", "v"):
        view = getattr(operands, name)
        assert view.shape == (1, 6, 2, 128) and view.dtype is torch.bfloat16
        assert not view.is_contiguous()
        assert view.untyped_storage().data_ptr() == storage
    assert operands.raw_g.shape == (1, 6, 2, 128) and operands.raw_g.dtype is torch.bfloat16
    assert operands.beta.shape == (1, 6, 2) and operands.beta.dtype is torch.float32
    assert operands.a_log.shape == (1, 1, 2, 1) and operands.dt_bias.shape == (256,)
    # One 6-token prefill, no decodes -> one sequence, state row zero-filled.
    assert operands.initial_state.shape == (2, 2, 128, 128)
    assert operands.initial_state.dtype is torch.float32
    assert torch.count_nonzero(operands.initial_state) == 0
    assert operands.cu_seqlens.tolist() == [0, 3, 6]


def test_dispatches_per_launch_is_a_positive_constant() -> None:
    # The trailing fold needs a known, positive per-call dispatch count.
    assert isinstance(rocm._DISPATCHES_PER_LAUNCH, int)
    assert rocm._DISPATCHES_PER_LAUNCH >= 1


def test_torch_rocm_runner_is_import_light() -> None:
    # Importing the ROCm runner must not eager-import torch or vllm; both are
    # built lazily only when a measurement runs on the ROCm host.
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import profiling.runners.attention.kda_chunk_prefill_torch_rocm as r; "
            "print('torch' in sys.modules, 'vllm' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False False"
