"""CPU registration tests for the ``batched_gemm`` L1 kind."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.kernels.batched_gemm import KIND, BatchedGemmArgs
from profiling.runners.exceptions import ProfilerNotImplemented

_BACKEND = "torch_mla_q_absorb_glm52"


def test_args_field_order_and_dtype_coercion():
    assert [field.name for field in fields(BatchedGemmArgs)] == [
        "num_batches",
        "m",
        "n",
        "k",
        "dtype",
    ]
    args = coerce_args(
        BatchedGemmArgs,
        {
            "num_batches": 64,
            "m": 128,
            "n": 512,
            "k": 192,
            "dtype": "bfloat16",
        },
    )
    assert args == BatchedGemmArgs(
        num_batches=64,
        m=128,
        n=512,
        k=192,
        dtype=DType.BF16,
    )
    with pytest.raises(Exception):
        args.m = 1


def test_kind_table_backend_and_runner_ref_contract():
    spec = find_kernel_profiler_spec(KIND, _BACKEND)

    assert KIND == "batched_gemm"
    assert known_backends(KIND) == [_BACKEND]
    assert spec.kernel_kind == KIND
    assert spec.table_name == KIND
    assert spec.backend == _BACKEND
    assert spec.args_schema is BatchedGemmArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.subprocess_env is None
    assert spec.runner_ref.module_name == "profiling.runners.gemm.batched_gemm"
    assert spec.runner_ref.function_name == "profile_mla_q_absorb_glm52"


def test_backend_support_is_bf16_h200_only():
    support = find_kernel_profiler_spec(KIND, _BACKEND).supports

    assert support.allows(DType.BF16, gpu="NVIDIA H200")
    assert not support.allows(DType.FP16, gpu="NVIDIA H200")
    assert not support.allows(DType.FP32, gpu="NVIDIA H200")
    assert not support.allows(DType.BF16, gpu="NVIDIA H100")
    assert not support.allows(DType.BF16, gpu="NVIDIA B200")


def test_registry_barrel_import_is_lazy():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('profiling.runners.gemm.batched_gemm' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == ["False", "False"]


def test_runner_ref_resolves_without_importing_torch():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from profiling.db.registry import find_kernel_profiler_spec; "
                "runner = find_kernel_profiler_spec("
                "'batched_gemm', 'torch_mla_q_absorb_glm52').runner_ref.load(); "
                "print(runner.__module__); "
                "print(runner.__name__); "
                "print('torch' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.gemm.batched_gemm",
        "profile_mla_q_absorb_glm52",
        "False",
    ]


@pytest.mark.parametrize(
    ("num_batches", "m", "n", "k"),
    [
        (0, 1, 512, 192),
        (16, 0, 512, 192),
        (16, 1, 0, 192),
        (16, 1, 512, 0),
    ],
)
def test_runner_rejects_nonpositive_dimensions_before_cuda(
    num_batches,
    m,
    n,
    k,
):
    from profiling.runners.gemm.batched_gemm import _validate_args

    with pytest.raises(ValueError, match="must be > 0"):
        _validate_args(num_batches, m, n, k, DType.BF16)


@pytest.mark.parametrize("num_batches", [1, 8, 63, 128])
def test_runner_rejects_unsupported_head_counts(num_batches):
    from profiling.runners.gemm.batched_gemm import _validate_args

    with pytest.raises(ValueError, match=r"num_batches in \{16, 32, 64\}"):
        _validate_args(num_batches, 1, 512, 192, DType.BF16)


@pytest.mark.parametrize(("n", "k"), [(256, 192), (512, 128), (513, 192)])
def test_runner_rejects_wrong_matrix_dimensions(n, k):
    from profiling.runners.gemm.batched_gemm import _validate_args

    with pytest.raises(ValueError, match=r"requires \(k, n\) == \(192, 512\)"):
        _validate_args(64, 1, n, k, DType.BF16)


@pytest.mark.parametrize("dtype", [DType.FP16, DType.FP32, DType.FP8_E4M3])
def test_runner_rejects_non_bf16_dtype(dtype):
    from profiling.runners.gemm.batched_gemm import _validate_args

    with pytest.raises(ValueError, match="supports only bf16"):
        _validate_args(64, 1, 512, 192, dtype)


def test_runner_rejects_missing_cuda_and_unverified_gpu():
    from profiling.runners.gemm.batched_gemm import _validate_cuda_device

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)

    h100 = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA H100",
        )
    )
    with pytest.raises(
        ProfilerNotImplemented,
        match="verified only on NVIDIA H200, got NVIDIA H100",
    ):
        _validate_cuda_device(h100)


@pytest.mark.parametrize("num_batches", [64, 32, 16])
def test_q_absorb_operand_constructor_matches_vllm_layout(num_batches):
    from profiling.runners.gemm.batched_gemm import _build_q_absorb_operands

    m = 2
    operands = _build_q_absorb_operands(
        torch,
        num_batches=num_batches,
        m=m,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    assert operands.q_base.shape == (m, num_batches, 256)
    assert operands.q_base.stride() == (num_batches * 256, 256, 1)
    assert operands.lhs.shape == (num_batches, m, 192)
    assert operands.lhs.stride() == (256, num_batches * 256, 1)
    assert operands.lhs.storage_offset() == 0
    assert operands.lhs.untyped_storage().data_ptr() == (
        operands.q_base.untyped_storage().data_ptr()
    )
    assert not operands.lhs.is_contiguous()

    assert operands.packed_weight.shape == (num_batches * 448, 512)
    assert operands.packed_weight.stride() == (512, 1)
    assert operands.rhs.shape == (num_batches, 192, 512)
    assert operands.rhs.stride() == (448 * 512, 512, 1)
    assert operands.rhs.storage_offset() == 0
    assert operands.rhs.untyped_storage().data_ptr() == (
        operands.packed_weight.untyped_storage().data_ptr()
    )
    assert operands.rhs.stride(0) - 192 * 512 == 256 * 512
    assert not operands.rhs.is_contiguous()

    assert operands.out.shape == (num_batches, m, 512)
    assert operands.out.stride() == (m * 512, 512, 1)
    assert operands.out.is_contiguous()


def test_logical_traffic_excludes_packed_gaps():
    from profiling.runners.gemm.batched_gemm import _logical_elements

    assert _logical_elements(64, 128, 512, 192) == (
        64 * 128 * 192 + 64 * 192 * 512 + 64 * 128 * 512
    )


def test_generated_facades_are_available():
    assert hasattr(perf_api, "get_batched_gemm_times")
    assert hasattr(perf_api, "count_missing_batched_gemm")
