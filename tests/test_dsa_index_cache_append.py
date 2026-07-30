"""Registration and CPU runner-helper tests for ``dsa_index_cache_append``."""

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
from profiling.kernels.dsa_index_cache_append import (
    KIND,
    DsaIndexCacheAppendArgs,
)
from profiling.runners.attention.dsa_index_cache_append_reference import (
    dsa_index_cache_append_reference,
)
from profiling.runners.exceptions import ProfilerNotImplemented

_BACKEND = "torch"
_VLLM_BACKEND = "vllm_cuda"
_CACHE_FORMAT = "page_planar_fp8_fp32_scale"


def test_args_field_order_and_dtype_coercion():
    assert [field.name for field in fields(DsaIndexCacheAppendArgs)] == [
        "num_tokens",
        "index_dim",
        "block_size",
        "quant_block_size",
        "input_dtype",
        "cache_dtype",
        "scale_format",
        "cache_format",
    ]
    args = coerce_args(
        DsaIndexCacheAppendArgs,
        {
            "num_tokens": 128,
            "index_dim": 128,
            "block_size": 64,
            "quant_block_size": 128,
            "input_dtype": "bfloat16",
            "cache_dtype": "fp8_e4m3",
            "scale_format": "ue8m0",
            "cache_format": _CACHE_FORMAT,
        },
    )
    assert args == DsaIndexCacheAppendArgs(
        num_tokens=128,
        index_dim=128,
        block_size=64,
        quant_block_size=128,
        input_dtype=DType.BF16,
        cache_dtype=DType.FP8_E4M3,
        scale_format="ue8m0",
        cache_format=_CACHE_FORMAT,
    )


def test_kind_table_backend_runner_and_support_contract():
    spec = find_kernel_profiler_spec(KIND, _BACKEND)

    assert KIND == "dsa_index_cache_append"
    assert known_backends(KIND) == [_BACKEND, _VLLM_BACKEND]
    assert spec.kernel_kind == KIND
    assert spec.table_name == KIND
    assert spec.args_schema is DsaIndexCacheAppendArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env is None
    assert spec.runner_ref.module_name == ("profiling.runners.attention.dsa_index_cache_append")
    assert spec.runner_ref.function_name == "profile_dsa_index_cache_append_torch"
    assert spec.supports.allows(
        DType.BF16,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H200",
    )
    assert not spec.supports.allows(
        DType.FP16,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H200",
    )
    assert not spec.supports.allows(
        DType.BF16,
        kv_dtype=DType.FP16,
        gpu="NVIDIA H200",
    )
    assert not spec.supports.allows(
        DType.BF16,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H100",
    )


def test_vllm_cuda_registration_reuses_kind_table_args_and_facades():
    torch_spec = find_kernel_profiler_spec(KIND, _BACKEND)
    vllm_spec = find_kernel_profiler_spec(KIND, _VLLM_BACKEND)

    assert vllm_spec.kernel_kind == KIND
    assert vllm_spec.table_name == torch_spec.table_name == KIND
    assert vllm_spec.args_schema is torch_spec.args_schema is DsaIndexCacheAppendArgs
    assert vllm_spec.metric_family is torch_spec.metric_family is MetricFamily.COMPUTE
    assert vllm_spec.batch_outlier_policy == BatchOutlierPolicy()
    assert vllm_spec.subprocess_env == "vllm_env"
    assert vllm_spec.runner_ref.module_name == (
        "profiling.runners.attention.dsa_index_cache_append"
    )
    assert vllm_spec.runner_ref.function_name == "profile_dsa_index_cache_append_vllm_cuda"


def test_vllm_cuda_support_is_bf16_fp8_e4m3_h200_only():
    support = find_kernel_profiler_spec(KIND, _VLLM_BACKEND).supports

    assert support.allows(
        DType.BF16,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H200",
    )
    assert not support.allows(
        DType.FP16,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H200",
    )
    assert not support.allows(
        DType.BF16,
        kv_dtype=DType.FP16,
        gpu="NVIDIA H200",
    )
    assert not support.allows(
        DType.BF16,
        kv_dtype=DType.FP8_E4M3,
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
                "print('vllm' in sys.modules); "
                "print("
                "'profiling.runners.attention.dsa_index_cache_append' "
                "in sys.modules"
                ")"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == ["False", "False", "False"]


def test_runner_refs_resolve_without_importing_torch_or_vllm():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from profiling.db.registry import find_kernel_profiler_spec; "
                "runner = find_kernel_profiler_spec("
                "'dsa_index_cache_append', 'torch').runner_ref.load(); "
                "vllm_runner = find_kernel_profiler_spec("
                "'dsa_index_cache_append', 'vllm_cuda').runner_ref.load(); "
                "print(runner.__module__); "
                "print(runner.__name__); "
                "print(vllm_runner.__module__); "
                "print(vllm_runner.__name__); "
                "print('torch' in sys.modules); "
                "print('vllm' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.dsa_index_cache_append",
        "profile_dsa_index_cache_append_torch",
        "profiling.runners.attention.dsa_index_cache_append",
        "profile_dsa_index_cache_append_vllm_cuda",
        "False",
        "False",
    ]


@pytest.mark.parametrize(
    ("num_tokens", "index_dim", "block_size", "quant_block_size"),
    [
        (0, 128, 64, 128),
        (1, 0, 64, 128),
        (1, 128, 0, 128),
        (1, 128, 64, 0),
    ],
)
def test_runner_rejects_nonpositive_dimensions_before_cuda(
    num_tokens,
    index_dim,
    block_size,
    quant_block_size,
):
    from profiling.runners.attention.dsa_index_cache_append import _validate_args

    with pytest.raises(ValueError, match="must be > 0"):
        _validate_args(
            num_tokens,
            index_dim,
            block_size,
            quant_block_size,
            DType.BF16,
            DType.FP8_E4M3,
            "ue8m0",
            _CACHE_FORMAT,
        )


@pytest.mark.parametrize(
    ("index_dim", "block_size", "quant_block_size"),
    [
        (256, 64, 128),
        (128, 32, 128),
        (128, 64, 64),
    ],
)
def test_runner_rejects_unsupported_dimensions_before_cuda(
    index_dim,
    block_size,
    quant_block_size,
):
    from profiling.runners.attention.dsa_index_cache_append import _validate_args

    with pytest.raises(ValueError, match=r"== \(128, 64, 128\)"):
        _validate_args(
            1,
            index_dim,
            block_size,
            quant_block_size,
            DType.BF16,
            DType.FP8_E4M3,
            "ue8m0",
            _CACHE_FORMAT,
        )


@pytest.mark.parametrize(
    ("input_dtype", "cache_dtype"),
    [
        (DType.FP16, DType.FP8_E4M3),
        (DType.FP32, DType.FP8_E4M3),
        (DType.BF16, DType.FP16),
        (DType.BF16, DType.FP8_E5M2),
    ],
)
def test_runner_rejects_unsupported_dtypes_before_cuda(
    input_dtype,
    cache_dtype,
):
    from profiling.runners.attention.dsa_index_cache_append import _validate_args

    with pytest.raises(
        ValueError,
        match="input_dtype=bf16 and cache_dtype=fp8_e4m3",
    ):
        _validate_args(
            1,
            128,
            64,
            128,
            input_dtype,
            cache_dtype,
            "ue8m0",
            _CACHE_FORMAT,
        )


@pytest.mark.parametrize("scale_format", ["float32", "UE8M0", ""])
def test_runner_rejects_unsupported_scale_format_before_cuda(scale_format):
    from profiling.runners.attention.dsa_index_cache_append import _validate_args

    with pytest.raises(ValueError, match="scale_format='ue8m0'"):
        _validate_args(
            1,
            128,
            64,
            128,
            DType.BF16,
            DType.FP8_E4M3,
            scale_format,
            _CACHE_FORMAT,
        )


@pytest.mark.parametrize("cache_format", ["interleaved", "plain", ""])
def test_runner_rejects_unsupported_cache_format_before_cuda(cache_format):
    from profiling.runners.attention.dsa_index_cache_append import _validate_args

    with pytest.raises(
        ValueError,
        match="cache_format='page_planar_fp8_fp32_scale'",
    ):
        _validate_args(
            1,
            128,
            64,
            128,
            DType.BF16,
            DType.FP8_E4M3,
            "ue8m0",
            cache_format,
        )


def test_runner_rejects_missing_cuda_and_unverified_gpu():
    from profiling.runners.attention.dsa_index_cache_append import (
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


def test_profile_entry_rejects_invalid_args_before_torch_import():
    from profiling.runners.attention.dsa_index_cache_append import (
        profile_dsa_index_cache_append_torch,
    )

    valid = {
        "num_tokens": 1,
        "index_dim": 128,
        "block_size": 64,
        "quant_block_size": 128,
        "input_dtype": DType.BF16,
        "cache_dtype": DType.FP8_E4M3,
        "scale_format": "ue8m0",
        "cache_format": _CACHE_FORMAT,
    }
    invalid_cases = [
        ({"num_tokens": 0}, "must be > 0"),
        ({"index_dim": 256}, r"== \(128, 64, 128\)"),
        ({"block_size": 32}, r"== \(128, 64, 128\)"),
        ({"quant_block_size": 64}, r"== \(128, 64, 128\)"),
        ({"input_dtype": DType.FP16}, "input_dtype=bf16"),
        ({"cache_dtype": DType.FP16}, "cache_dtype=fp8_e4m3"),
        ({"scale_format": "float32"}, "scale_format='ue8m0'"),
        ({"cache_format": "plain"}, "cache_format='page_planar"),
    ]
    for overrides, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            profile_dsa_index_cache_append_torch(**(valid | overrides))


def test_vllm_profile_entry_rejects_invalid_args_before_framework_imports():
    from profiling.runners.attention.dsa_index_cache_append import (
        profile_dsa_index_cache_append_vllm_cuda,
    )

    valid = {
        "num_tokens": 1,
        "index_dim": 128,
        "block_size": 64,
        "quant_block_size": 128,
        "input_dtype": DType.BF16,
        "cache_dtype": DType.FP8_E4M3,
        "scale_format": "ue8m0",
        "cache_format": _CACHE_FORMAT,
    }
    invalid_cases = [
        ({"num_tokens": 0}, "must be > 0"),
        ({"index_dim": 256}, r"== \(128, 64, 128\)"),
        ({"block_size": 32}, r"== \(128, 64, 128\)"),
        ({"quant_block_size": 64}, r"== \(128, 64, 128\)"),
        ({"input_dtype": DType.FP16}, "input_dtype=bf16"),
        ({"cache_dtype": DType.FP16}, "cache_dtype=fp8_e4m3"),
        ({"scale_format": "float32"}, "scale_format='ue8m0'"),
        ({"cache_format": "plain"}, "cache_format='page_planar"),
    ]
    for overrides, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            profile_dsa_index_cache_append_vllm_cuda(**(valid | overrides))


def test_vllm_runner_rejects_missing_cuda_and_unverified_gpu():
    from profiling.runners.attention.dsa_index_cache_append import (
        _validate_vllm_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_vllm_cuda_device(no_cuda)

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
        _validate_vllm_cuda_device(h100)


def test_operand_constructor_matches_glm_page_planar_layout():
    from profiling.runners.attention.dsa_index_cache_append import _build_operands

    operands = _build_operands(
        torch,
        num_tokens=5,
        index_dim=128,
        block_size=64,
        quant_block_size=128,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    assert operands.k.shape == (5, 128)
    assert operands.k.stride() == (128, 1)
    assert operands.k.is_contiguous()
    assert operands.cache.shape == (256, 64, 132)
    assert operands.cache.stride() == (8448, 132, 1)
    assert operands.cache.dtype is torch.uint8
    assert operands.key_plane.shape == (256, 64, 128)
    assert operands.key_plane.stride() == (8448, 128, 1)
    assert operands.scale_plane.shape == (256, 64, 1, 4)
    assert operands.scale_plane.stride() == (8448, 4, 4, 1)
    assert operands.key_plane.untyped_storage().data_ptr() == (
        operands.cache.untyped_storage().data_ptr()
    )
    assert operands.scale_plane.untyped_storage().data_ptr() == (
        operands.cache.untyped_storage().data_ptr()
    )
    assert operands.scale_plane.storage_offset() == 64 * 128
    assert operands.slot_mapping.shape == (5,)
    assert operands.slot_mapping.dtype is torch.int64
    assert operands.slot_mapping.is_contiguous()
    assert torch.unique(operands.slot_mapping).numel() == 5
    assert int(operands.slot_mapping.min()) >= 0
    assert int(operands.slot_mapping.max()) < 256 * 64
    assert torch.equal(operands.block_indices, operands.slot_mapping // 64)
    assert torch.equal(operands.block_offsets, operands.slot_mapping % 64)
    assert int(operands.block_indices.max()) < 256
    assert int(operands.block_offsets.max()) < 64


def test_operand_constructor_uses_spare_block_for_long_prefill():
    from profiling.runners.attention.dsa_index_cache_append import _build_operands

    operands = _build_operands(
        torch,
        num_tokens=16384,
        index_dim=128,
        block_size=64,
        quant_block_size=128,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    assert operands.cache.shape == (257, 64, 132)
    assert operands.slot_mapping.numel() == 16384
    assert torch.unique(operands.slot_mapping).numel() == 16384


def test_vectorized_write_matches_reference_and_mutates_only_cache():
    from profiling.runners.attention.dsa_index_cache_append import (
        _build_operands,
        _write_cache,
    )

    operands = _build_operands(
        torch,
        num_tokens=7,
        index_dim=128,
        block_size=64,
        quant_block_size=128,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    k_before = operands.k.clone()
    mapping_before = operands.slot_mapping.clone()
    expected = operands.cache.clone()
    dsa_index_cache_append_reference(
        operands.k,
        expected,
        operands.slot_mapping,
    )
    cache_ptr = operands.cache.untyped_storage().data_ptr()

    _write_cache(torch, operands, quant_block_size=128)

    assert operands.cache.untyped_storage().data_ptr() == cache_ptr
    assert torch.equal(operands.cache, expected)
    assert torch.equal(operands.k, k_before)
    assert torch.equal(operands.slot_mapping, mapping_before)
    assert torch.equal(
        operands.key_plane[
            operands.block_indices,
            operands.block_offsets,
        ],
        expected.view(256, -1)[:, : 64 * 128].view(256, 64, 128)[
            operands.block_indices,
            operands.block_offsets,
        ],
    )
    assert torch.equal(
        operands.scale_plane[
            operands.block_indices,
            operands.block_offsets,
        ],
        expected.view(256, -1)[:, 64 * 128 :].view(256, 64, 1, 4)[
            operands.block_indices,
            operands.block_offsets,
        ],
    )
    written_slots = set(operands.slot_mapping.tolist())
    untouched_slot = next(slot for slot in range(256 * 64) if slot not in written_slots)
    block, offset = divmod(untouched_slot, 64)
    assert torch.all(operands.key_plane[block, offset] == 0xA5)
    assert torch.all(operands.scale_plane[block, offset] == 0xA5)


def test_logical_traffic_is_396_bytes_per_glm_token():
    from profiling.runners.attention.dsa_index_cache_append import _logical_bytes

    assert (
        _logical_bytes(
            num_tokens=3,
            index_dim=128,
            quant_block_size=128,
            input_dtype=DType.BF16,
            cache_dtype=DType.FP8_E4M3,
        )
        == 3 * 396
    )


def test_generated_facades_are_available():
    assert hasattr(perf_api, "get_dsa_index_cache_append_times")
    assert hasattr(perf_api, "count_missing_dsa_index_cache_append")
