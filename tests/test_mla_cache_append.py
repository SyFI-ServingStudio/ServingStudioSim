"""Registration and CPU runner-helper tests for ``mla_cache_append``."""

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
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.kernels.mla_cache_append import KIND, MlaCacheAppendArgs
from profiling.runners.attention.mla_cache_append_reference import (
    mla_cache_append_reference,
)
from profiling.runners.exceptions import ProfilerNotImplemented

_BACKEND = "torch"


def test_args_field_order_and_dtype_coercion():
    assert [field.name for field in fields(MlaCacheAppendArgs)] == [
        "num_tokens",
        "kv_lora_rank",
        "rope_dim",
        "block_size",
        "input_dtype",
        "kv_dtype",
        "cache_format",
    ]
    args = coerce_args(
        MlaCacheAppendArgs,
        {
            "num_tokens": 128,
            "kv_lora_rank": 512,
            "rope_dim": 64,
            "block_size": 64,
            "input_dtype": "bfloat16",
            "kv_dtype": "bf16",
            "cache_format": "plain",
        },
    )
    assert args == MlaCacheAppendArgs(
        num_tokens=128,
        kv_lora_rank=512,
        rope_dim=64,
        block_size=64,
        input_dtype=DType.BF16,
        kv_dtype=DType.BF16,
        cache_format="plain",
    )


def test_kind_table_backend_runner_and_support_contract():
    spec = find_kernel_profiler_spec(KIND, _BACKEND)

    assert KIND == "mla_cache_append"
    assert known_backends(KIND) == [_BACKEND]
    assert spec.kernel_kind == KIND
    assert spec.table_name == KIND
    assert spec.args_schema is MlaCacheAppendArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env is None
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.mla_cache_append"
    )
    assert spec.runner_ref.function_name == "profile_mla_cache_append_torch"

    assert spec.supports.allows(
        DType.BF16,
        kv_dtype=DType.BF16,
        gpu="NVIDIA H200",
    )
    assert not spec.supports.allows(
        DType.FP16,
        kv_dtype=DType.BF16,
        gpu="NVIDIA H200",
    )
    assert not spec.supports.allows(
        DType.BF16,
        kv_dtype=DType.FP16,
        gpu="NVIDIA H200",
    )
    assert not spec.supports.allows(
        DType.BF16,
        kv_dtype=DType.BF16,
        gpu="NVIDIA H100",
    )


def test_registry_barrel_import_is_lazy():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print("
                "'profiling.runners.attention.mla_cache_append' in sys.modules"
                ")"
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
                "'mla_cache_append', 'torch').runner_ref.load(); "
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
        "profiling.runners.attention.mla_cache_append",
        "profile_mla_cache_append_torch",
        "False",
    ]


@pytest.mark.parametrize(
    (
        "num_tokens",
        "kv_lora_rank",
        "rope_dim",
        "block_size",
    ),
    [
        (0, 512, 64, 64),
        (1, 0, 64, 64),
        (1, 512, 0, 64),
        (1, 512, 64, 0),
    ],
)
def test_runner_rejects_nonpositive_dimensions_before_cuda(
    num_tokens,
    kv_lora_rank,
    rope_dim,
    block_size,
):
    from profiling.runners.attention.mla_cache_append import _validate_args

    with pytest.raises(ValueError, match="must be > 0"):
        _validate_args(
            num_tokens,
            kv_lora_rank,
            rope_dim,
            block_size,
            DType.BF16,
            DType.BF16,
            "plain",
        )


@pytest.mark.parametrize(
    ("kv_lora_rank", "rope_dim", "block_size"),
    [
        (256, 64, 64),
        (512, 128, 64),
        (512, 64, 32),
    ],
)
def test_runner_rejects_unsupported_dimensions_before_cuda(
    kv_lora_rank,
    rope_dim,
    block_size,
):
    from profiling.runners.attention.mla_cache_append import _validate_args

    with pytest.raises(ValueError, match=r"== \(512, 64, 64\)"):
        _validate_args(
            1,
            kv_lora_rank,
            rope_dim,
            block_size,
            DType.BF16,
            DType.BF16,
            "plain",
        )


@pytest.mark.parametrize(
    ("input_dtype", "kv_dtype"),
    [
        (DType.FP16, DType.BF16),
        (DType.BF16, DType.FP16),
        (DType.FP32, DType.FP32),
        (DType.BF16, DType.FP8_E4M3),
    ],
)
def test_runner_rejects_unsupported_dtypes_before_cuda(
    input_dtype,
    kv_dtype,
):
    from profiling.runners.attention.mla_cache_append import _validate_args

    with pytest.raises(ValueError, match="input_dtype=kv_dtype=bf16"):
        _validate_args(
            1,
            512,
            64,
            64,
            input_dtype,
            kv_dtype,
            "plain",
        )


@pytest.mark.parametrize("cache_format", ["fp8_ds_mla", "Plain", ""])
def test_runner_rejects_unsupported_cache_format_before_cuda(cache_format):
    from profiling.runners.attention.mla_cache_append import _validate_args

    with pytest.raises(ValueError, match="cache_format='plain'"):
        _validate_args(
            1,
            512,
            64,
            64,
            DType.BF16,
            DType.BF16,
            cache_format,
        )


def test_runner_rejects_missing_cuda_and_unverified_gpu():
    from profiling.runners.attention.mla_cache_append import (
        _validate_cuda_device,
    )

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


def test_operand_constructor_matches_plain_glm_layout():
    from profiling.runners.attention.mla_cache_append import _build_operands

    operands = _build_operands(
        torch,
        num_tokens=5,
        kv_lora_rank=512,
        rope_dim=64,
        block_size=64,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    assert operands.kv_c.shape == (5, 512)
    assert operands.kv_c.stride() == (512, 1)
    assert operands.k_pe_backing.shape == (5, 1, 64)
    assert operands.k_pe_backing.stride() == (64, 64, 1)
    assert operands.k_pe.shape == (5, 64)
    assert operands.k_pe.stride() == (64, 1)
    assert operands.k_pe.untyped_storage().data_ptr() == (
        operands.k_pe_backing.untyped_storage().data_ptr()
    )
    assert operands.cache.shape == (256, 64, 576)
    assert operands.cache.stride() == (64 * 576, 576, 1)
    assert operands.slot_mapping.shape == (5,)
    assert operands.slot_mapping.dtype is torch.int64
    assert operands.slot_mapping.is_contiguous()
    assert torch.unique(operands.slot_mapping).numel() == 5
    assert int(operands.slot_mapping.min()) >= 0
    assert int(operands.slot_mapping.max()) < 256 * 64
    assert torch.equal(
        operands.block_indices,
        operands.slot_mapping // 64,
    )
    assert torch.equal(operands.block_offsets, operands.slot_mapping % 64)
    assert int(operands.block_indices.max()) < 256
    assert int(operands.block_offsets.max()) < 64


def test_operand_constructor_uses_spare_block_for_long_prefill():
    from profiling.runners.attention.mla_cache_append import _build_operands

    operands = _build_operands(
        torch,
        num_tokens=16384,
        kv_lora_rank=512,
        rope_dim=64,
        block_size=64,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    assert operands.cache.shape == (257, 64, 576)
    assert operands.slot_mapping.numel() == 16384
    assert torch.unique(operands.slot_mapping).numel() == 16384


def test_write_helper_matches_reference_and_mutates_only_cache():
    from profiling.runners.attention.mla_cache_append import (
        _build_operands,
        _write_cache,
    )

    operands = _build_operands(
        torch,
        num_tokens=7,
        kv_lora_rank=512,
        rope_dim=64,
        block_size=64,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    kv_before = operands.kv_c.clone()
    k_pe_before = operands.k_pe.clone()
    mapping_before = operands.slot_mapping.clone()
    expected = operands.cache.clone()
    mla_cache_append_reference(
        operands.kv_c,
        operands.k_pe,
        expected,
        operands.slot_mapping,
    )

    cache_identity = id(operands.cache)
    cache_ptr = operands.cache.untyped_storage().data_ptr()
    _write_cache(operands)

    assert id(operands.cache) == cache_identity
    assert operands.cache.untyped_storage().data_ptr() == cache_ptr
    assert torch.equal(operands.cache, expected)
    assert torch.equal(operands.kv_c, kv_before)
    assert torch.equal(operands.k_pe, k_pe_before)
    assert torch.equal(operands.slot_mapping, mapping_before)
    assert torch.equal(
        operands.cache[
            operands.block_indices,
            operands.block_offsets,
            :512,
        ],
        operands.kv_c,
    )
    assert torch.equal(
        operands.cache[
            operands.block_indices,
            operands.block_offsets,
            512:,
        ],
        operands.k_pe,
    )
    written_slots = set(operands.slot_mapping.tolist())
    untouched_slot = next(slot for slot in range(256 * 64) if slot not in written_slots)
    assert torch.all(operands.cache.view(-1, 576)[untouched_slot] == -1)


def test_logical_traffic_is_2312_bytes_per_glm_token():
    from profiling.runners.attention.mla_cache_append import _logical_bytes

    assert (
        _logical_bytes(
            num_tokens=3,
            kv_lora_rank=512,
            rope_dim=64,
            input_dtype=DType.BF16,
            kv_dtype=DType.BF16,
        )
        == 3 * 2312
    )


def test_generated_facades_are_available():
    assert hasattr(perf_api, "get_mla_cache_append_times")
    assert hasattr(perf_api, "count_missing_mla_cache_append")
