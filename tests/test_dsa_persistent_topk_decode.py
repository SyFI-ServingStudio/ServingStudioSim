"""Registration and Torch-runner tests for persistent DSA decode top-k."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.kernels.dsa_persistent_topk_decode import (
    KIND,
    DsaPersistentTopkDecodeArgs,
)
from profiling.runners.attention.dsa_persistent_topk_decode_reference import (
    dsa_persistent_topk_decode_reference,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

_TORCH_BACKEND = "torch"
_NATIVE_BACKEND = "vllm_cuda"
_BASE_SPEC = {
    "batch_size": 16,
    "context_len": 8192,
    "next_n": 2,
    "max_model_len": 1048576,
    "top_k": 2048,
    "logits_row_stride": 1048576,
    "logits_dtype": "fp32",
    "index_dtype": "int32",
    "context_mode": "uniform",
}


def test_args_field_order_and_dtype_coercion() -> None:
    assert [field.name for field in fields(DsaPersistentTopkDecodeArgs)] == [
        "batch_size",
        "context_len",
        "next_n",
        "max_model_len",
        "top_k",
        "logits_row_stride",
        "logits_dtype",
        "index_dtype",
        "context_mode",
    ]
    args = coerce_args(
        DsaPersistentTopkDecodeArgs,
        _BASE_SPEC | {"batch_size": "16", "next_n": "2"},
    )
    assert args == DsaPersistentTopkDecodeArgs(
        batch_size=16,
        context_len=8192,
        next_n=2,
        max_model_len=1048576,
        top_k=2048,
        logits_row_stride=1048576,
        logits_dtype=DType.FP32,
        index_dtype="int32",
        context_mode="uniform",
    )


def test_registration_support_policy_and_facades() -> None:
    torch_spec = find_kernel_profiler_spec(KIND, _TORCH_BACKEND)
    native_spec = find_kernel_profiler_spec(KIND, _NATIVE_BACKEND)

    assert KIND == "dsa_persistent_topk_decode"
    assert known_backends(KIND) == [_TORCH_BACKEND, _NATIVE_BACKEND]
    for spec in (torch_spec, native_spec):
        assert spec.kernel_kind == spec.table_name == KIND
        assert spec.args_schema is DsaPersistentTopkDecodeArgs
        assert spec.metric_family is MetricFamily.COMPUTE
        assert spec.batch_outlier_policy == BatchOutlierPolicy()
        assert spec.supports.kv is None
        assert spec.supports.allows(DType.FP32, gpu="NVIDIA H200")
        assert not spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
        assert not spec.supports.allows(DType.FP32, gpu="NVIDIA H100")
        assert spec.runner_ref.module_name == (
            "profiling.runners.attention.dsa_persistent_topk_decode"
        )
    assert torch_spec.subprocess_env is None
    assert torch_spec.runner_ref.function_name == ("profile_dsa_persistent_topk_decode_torch")
    assert native_spec.subprocess_env == "vllm_env"
    assert native_spec.runner_ref.function_name == ("profile_dsa_persistent_topk_decode_vllm_cuda")
    assert hasattr(perf_api, "get_dsa_persistent_topk_decode_times")
    assert hasattr(perf_api, "count_missing_dsa_persistent_topk_decode")


def test_registry_barrel_import_is_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.dsa_persistent_topk_decode' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == ["False", "False"]


def test_runner_ref_resolves_without_importing_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from profiling.db.registry import "
                "find_kernel_profiler_spec; runner = find_kernel_profiler_spec("
                "'dsa_persistent_topk_decode', 'torch').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.dsa_persistent_topk_decode",
        "profile_dsa_persistent_topk_decode_torch",
        "False",
    ]


def test_native_runner_ref_resolves_without_importing_heavy_stack() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from profiling.db.registry import "
                "find_kernel_profiler_spec; runner = find_kernel_profiler_spec("
                "'dsa_persistent_topk_decode', 'vllm_cuda').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); print('vllm' in sys.modules); "
                "print('profiling.runners.attention.dsa_persistent_topk_native.loader' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.dsa_persistent_topk_decode",
        "profile_dsa_persistent_topk_decode_vllm_cuda",
        "False",
        "False",
        "False",
    ]


def test_native_source_manifest_and_required_hashes_are_enforced() -> None:
    from profiling.runners.attention.dsa_persistent_topk_native import loader

    asset_root = Path(loader.__file__).resolve().parent
    manifest = json.loads((asset_root / "source_manifest.json").read_text(encoding="utf-8"))
    verified = loader.verify_vendored_sources()
    assert manifest["upstream"] == {
        "project": "vLLM",
        "version": "v0.23.0",
        "commit": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "repository": "https://github.com/vllm-project/vllm",
        "license": "Apache-2.0",
    }
    assert manifest["implementation"] == {
        "kind": "corrected_v0_23_derived",
        "correction_id": "all-cooperative-radix-with-radix-iteration-v1",
        "correction_manifest": "correction_manifest.json",
        "description": (
            "The loader preserves byte-identical upstream assets and compiles "
            "deterministic corrected copies in its worker-local cache."
        ),
    }
    assert verified["upstream/topk.cu"] == (
        "2c90ef9391e1d6bd6ca65c05841597569cf629451f25a1ed4446aa5b34f1d917"
    )
    assert verified["upstream/persistent_topk.cuh"] == (
        "1d92c234493599e4d57d793eda2cf3b8efa246415425dd2ac881935b25b950ee"
    )
    assert verified["correction_manifest.json"] == (
        "a6e00159afa31b110950e1fbfcbe3118df8c549bf56f01052f7eeb99c549d001"
    )
    assert set(verified) == {record["local_path"] for record in manifest["files"]}
    for relative, expected in verified.items():
        assert hashlib.sha256((asset_root / relative).read_bytes()).hexdigest() == expected


def test_native_source_hash_failure_is_clear_and_precedes_build(tmp_path) -> None:
    from profiling.runners.attention.dsa_persistent_topk_native import loader

    copied = tmp_path / "native"
    shutil.copytree(Path(loader.__file__).resolve().parent, copied)
    with (copied / "upstream" / "topk.cu").open("ab") as handle:
        handle.write(b"\n// tampered\n")
    with pytest.raises(loader.NativeExtensionLoadError, match="source hash mismatch.*topk.cu"):
        loader.verify_vendored_sources(copied)


def test_native_correction_derivation_is_deterministic_and_preserves_upstream(
    tmp_path,
) -> None:
    from profiling.runners.attention.dsa_persistent_topk_native import loader

    asset_root = Path(loader.__file__).resolve().parent
    upstream_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (asset_root / "upstream").iterdir()
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    expected = {
        "binding.cpp": "ea1b70c6f70fd2ce683706e5ea0f444a8d57b97747b07609fce74f3497839d6d",
        "persistent_topk.cuh": ("e0aeea8a0bb7d4be12c45b7d643c24489054d8410aa0e07648cb291050411a94"),
        "topk.cu": "937bece0a889b353b1c2e9148a55f00c37a1791cd1d5e6f05bb7fb98da675084",
        "torch_utils.h": "f619e1943039a67a3254f7e0679e1f3e0faa9d8fccae795f3b20df9e91a8a429",
    }

    assert loader._derive_corrected_sources(first) == expected
    assert loader._derive_corrected_sources(second) == expected
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (asset_root / "upstream").iterdir()
    } == upstream_hashes
    assert (first / "topk.cu").read_bytes() == (second / "topk.cu").read_bytes()
    assert (first / "persistent_topk.cuh").read_bytes() == (
        second / "persistent_topk.cuh"
    ).read_bytes()


def test_corrected_source_encodes_all_radix_and_radix_iteration_invariants(
    tmp_path,
) -> None:
    from profiling.runners.attention.dsa_persistent_topk_native import loader

    source_dir = tmp_path / "corrected"
    loader._derive_corrected_sources(source_dir)
    host = (source_dir / "topk.cu").read_text(encoding="utf-8")
    kernel = (source_dir / "persistent_topk.cuh").read_text(encoding="utf-8")

    assert "if (num_rows > 32 && max_smem_per_block >= 128 * 1024)" not in host
    assert "static_cast<uint32_t>(max_seq_len) > static_cast<uint32_t>(TopK)" in host
    assert "FilteredTopK fallback failed" not in host
    assert "if (cta_in_group != 0 && params.max_seq_len <= TopK) return;" in kernel
    assert "uint32_t radix_iter = 0;" in kernel
    assert "barrier_phase, radix_iter, tx);" in kernel
    assert "radix_iter++;" in kernel
    assert "barrier_phase, iter, tx);" not in kernel
    dispatch = kernel[
        kernel.index("for (uint32_t iter = 0;") : kernel.index("}  // namespace persistent")
    ]
    assert "histogram_2048_topk<TopK>" not in dispatch
    assert "histogram_256_topk<TopK>" not in dispatch


def test_native_correction_and_derived_source_tampering_fail(tmp_path) -> None:
    from profiling.runners.attention.dsa_persistent_topk_native import loader

    asset_root = Path(loader.__file__).resolve().parent
    copied = tmp_path / "native"
    shutil.copytree(asset_root, copied)
    correction_path = copied / "correction_manifest.json"
    correction_path.write_text(correction_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(
        loader.NativeExtensionLoadError,
        match="source hash mismatch for correction_manifest.json",
    ):
        loader.verify_vendored_sources(copied)

    source_dir = tmp_path / "derived"
    expected = loader._derive_corrected_sources(source_dir)
    with (source_dir / "persistent_topk.cuh").open("ab") as handle:
        handle.write(b"\n// tampered\n")
    with pytest.raises(loader.NativeExtensionLoadError, match="build-source mismatch"):
        loader._verify_derived_sources(source_dir, expected)


def test_native_fingerprint_and_marker_include_correction_identity(tmp_path) -> None:
    from profiling.runners.attention.dsa_persistent_topk_native import loader

    fake_torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
    )
    source_hashes = loader.verify_vendored_sources()
    correction = loader._read_correction_manifest()
    fingerprint = loader._build_fingerprint(fake_torch, source_hashes, correction)
    assert fingerprint == loader._build_fingerprint(fake_torch, source_hashes, correction)

    source_dir = tmp_path / "corrected_source"
    derived = loader._derive_corrected_sources(source_dir)
    artifact = tmp_path / "extension.so"
    artifact.write_bytes(b"synthetic extension")
    marker = tmp_path / "complete.json"
    loader._write_complete_marker(
        marker,
        artifact=artifact,
        fingerprint=fingerprint,
        source_hashes=source_hashes,
        derived_source_hashes=derived,
        correction=correction,
    )
    loader._validate_complete_marker(
        marker,
        artifact=artifact,
        fingerprint=fingerprint,
        source_hashes=source_hashes,
        derived_source_hashes=derived,
        correction=correction,
        source_dir=source_dir,
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["correction"]["id"] = "tampered"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(loader.NativeExtensionLoadError, match="correction mismatch"):
        loader._validate_complete_marker(
            marker,
            artifact=artifact,
            fingerprint=fingerprint,
            source_hashes=source_hashes,
            derived_source_hashes=derived,
            correction=correction,
            source_dir=source_dir,
        )


def test_native_build_lock_serializes_concurrent_workers(tmp_path) -> None:
    from profiling.runners.attention.dsa_persistent_topk_native.loader import (
        _exclusive_build_lock,
    )

    lock = tmp_path / "extension.lock"
    first_entered = threading.Event()
    allow_first_exit = threading.Event()
    order: list[str] = []

    def first() -> None:
        with _exclusive_build_lock(lock):
            order.append("first-enter")
            first_entered.set()
            assert allow_first_exit.wait(timeout=5)
            order.append("first-exit")

    def second() -> None:
        assert first_entered.wait(timeout=5)
        with _exclusive_build_lock(lock):
            order.append("second-enter")

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout=5)
    time.sleep(0.05)
    assert order == ["first-enter"]
    allow_first_exit.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert order == ["first-enter", "first-exit", "second-enter"]


def test_native_corrected_cache_builds_once_and_reuses_marker(monkeypatch, tmp_path) -> None:
    from profiling.runners.attention.dsa_persistent_topk_native import loader

    fake_torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
    )
    state: dict[str, object | None] = {"op": None}
    build_count = 0
    load_count = 0

    monkeypatch.setattr(loader, "_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(loader, "_validate_runtime", lambda _torch: None)
    monkeypatch.setattr(loader, "_registered_op", lambda _torch: state["op"])

    def fake_build(_torch, *, build_dir, source_dir, extension_name):
        nonlocal build_count
        build_count += 1
        assert source_dir.is_dir()
        artifact = build_dir / f"{extension_name}.so"
        artifact.write_bytes(b"corrected native extension")
        state["op"] = object()
        return artifact

    def fake_load(_torch, artifact):
        nonlocal load_count
        load_count += 1
        assert artifact.read_bytes() == b"corrected native extension"
        state["op"] = object()
        return state["op"]

    monkeypatch.setattr(loader, "_build_library", fake_build)
    monkeypatch.setattr(loader, "_load_library", fake_load)

    first = loader.load_persistent_topk_op(fake_torch)
    assert first is state["op"]
    state["op"] = None  # Model a fresh worker loading the completed cache.
    second = loader.load_persistent_topk_op(fake_torch)
    assert second is state["op"]
    assert first is not second
    assert build_count == 1
    assert load_count == 1
    assert len(list((tmp_path / "cache").glob("*/complete.json"))) == 1


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"batch_size": 0}, "1 <= batch_size <= 256"),
        ({"batch_size": 257}, "1 <= batch_size <= 256"),
        ({"next_n": 0}, "next_n in \\[1, 2\\]"),
        ({"next_n": 3}, "next_n in \\[1, 2\\]"),
        ({"context_len": -1}, "context_len must be >= 0"),
        (
            {"context_len": 0, "next_n": 2},
            "context_len must be >= next_n - 1",
        ),
        ({"max_model_len": 0}, "max_model_len must be > 0"),
        ({"context_len": 1048577}, "context_len must be <= max_model_len"),
        ({"top_k": 1024}, "top_k=2048"),
        ({"logits_row_stride": 0}, "positive and >= max_model_len"),
        ({"logits_row_stride": 1048575}, "positive and >= max_model_len"),
        ({"logits_dtype": DType.BF16}, "logits_dtype=fp32"),
        ({"index_dtype": "int64"}, "index_dtype='int32'"),
        ({"context_mode": "mixed"}, "context_mode='uniform'"),
    ],
)
def test_rejects_unsupported_args_before_allocation(monkeypatch, overrides, match) -> None:
    from profiling.runners.attention import dsa_persistent_topk_decode as runner

    allocated = False

    def fail_if_allocated(*args, **kwargs):
        nonlocal allocated
        allocated = True
        raise AssertionError("operand allocation must not run")

    monkeypatch.setattr(runner, "_build_operands", fail_if_allocated)
    kwargs = dict(_BASE_SPEC)
    kwargs.update(overrides)
    with pytest.raises((TypeError, ValueError), match=match):
        runner.profile_dsa_persistent_topk_decode_torch(**kwargs)
    assert not allocated


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"batch_size": 0}, "1 <= batch_size <= 256"),
        ({"next_n": 3}, "next_n in \\[1, 2\\]"),
        ({"context_len": -1}, "context_len must be >= 0"),
        ({"max_model_len": 0}, "max_model_len must be > 0"),
        ({"top_k": 1024}, "top_k=2048"),
        ({"logits_dtype": DType.BF16}, "logits_dtype=fp32"),
        ({"index_dtype": "int64"}, "index_dtype='int32'"),
        ({"context_mode": "mixed"}, "context_mode='uniform'"),
    ],
)
def test_native_rejects_unsupported_args_before_import_or_allocation(
    monkeypatch, overrides, match
) -> None:
    from profiling.runners.attention import dsa_persistent_topk_decode as runner

    reached_worker_path = False

    def fail_worker_path(*args, **kwargs):
        nonlocal reached_worker_path
        reached_worker_path = True
        raise AssertionError("native worker path must not run")

    monkeypatch.setattr(runner, "_build_native_operands", fail_worker_path)
    monkeypatch.setattr(runner, "_load_native_op", fail_worker_path)
    kwargs = dict(_BASE_SPEC)
    kwargs.update(overrides)
    with pytest.raises((TypeError, ValueError), match=match):
        runner.profile_dsa_persistent_topk_decode_vllm_cuda(**kwargs)
    assert not reached_worker_path


def test_validate_accepts_ordinary_speculative_and_zero_context() -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode import _validate_args

    ordinary = _validate_args(**(_BASE_SPEC | {"batch_size": 1, "context_len": 0, "next_n": 1}))
    speculative = _validate_args(**_BASE_SPEC)
    assert ordinary[:3] == (1, 0, 1)
    assert speculative[:3] == (16, 8192, 2)


def test_rejects_missing_cuda_and_unverified_gpu() -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode import (
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
    with pytest.raises(ProfilerNotImplemented, match="verified only on NVIDIA H200"):
        _validate_cuda_device(h100)


def test_operand_layout_and_rank_two_length_ramp() -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode import _build_operands

    operands = _build_operands(
        torch,
        batch_size=3,
        context_len=5,
        next_n=2,
        max_model_len=8,
        top_k=4,
        logits_row_stride=11,
        device="cpu",
    )

    assert operands.logits_backing.shape == (6, 11)
    assert operands.logits_backing.stride() == (11, 1)
    assert operands.logits.shape == (6, 8)
    assert operands.logits.stride() == (11, 1)
    assert not operands.logits.is_contiguous()
    assert operands.logits.untyped_storage().data_ptr() == (
        operands.logits_backing.untyped_storage().data_ptr()
    )
    assert operands.lengths.shape == (3, 2)
    assert operands.lengths.tolist() == [[4, 5], [4, 5], [4, 5]]
    assert operands.flat_lengths.tolist() == [4, 5, 4, 5, 4, 5]
    assert operands.lengths.dtype is torch.int32
    assert operands.lengths.is_contiguous()
    assert operands.out.shape == (6, 4)
    assert operands.out.stride() == (4, 1)
    assert operands.out.dtype is torch.int32
    assert operands.valid_mask.shape == (6, 8)
    assert operands.long_row_indices.tolist() == [1, 3, 5]
    assert bool(torch.isfinite(operands.logits).all())
    assert bool(torch.any(operands.logits < 0))
    assert bool(torch.any(operands.logits > 0))
    assert operands.logits[0].unique().numel() == 8


def test_operand_layout_for_ordinary_and_fully_padded_rows() -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode import _build_operands

    ordinary = _build_operands(
        torch,
        batch_size=2,
        context_len=7,
        next_n=1,
        max_model_len=8,
        top_k=4,
        logits_row_stride=8,
        device="cpu",
    )
    padded = _build_operands(
        torch,
        batch_size=2,
        context_len=0,
        next_n=1,
        max_model_len=8,
        top_k=4,
        logits_row_stride=8,
        device="cpu",
    )
    assert ordinary.lengths.tolist() == [[7], [7]]
    assert padded.lengths.tolist() == [[0], [0]]
    assert padded.long_row_indices.numel() == 0


def test_native_operand_layout_has_exact_workspace_without_composite_intermediates() -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode import (
        _WORKSPACE_BYTES,
        _build_native_operands,
    )

    operands = _build_native_operands(
        torch,
        batch_size=3,
        context_len=5,
        next_n=2,
        max_model_len=8,
        top_k=4,
        logits_row_stride=11,
        device="cpu",
    )
    assert operands.logits_backing.shape == (6, 11)
    assert operands.logits.shape == (6, 8)
    assert operands.logits.stride() == (11, 1)
    assert operands.lengths.shape == (3, 2)
    assert operands.lengths.tolist() == [[4, 5], [4, 5], [4, 5]]
    assert operands.flat_lengths.tolist() == [4, 5, 4, 5, 4, 5]
    assert operands.out.shape == (6, 4)
    assert operands.workspace.shape == (_WORKSPACE_BYTES,)
    assert operands.workspace.dtype is torch.uint8
    assert operands.workspace.is_contiguous()
    assert not hasattr(operands, "valid_mask")
    assert not hasattr(operands, "natural_output")
    assert not hasattr(operands, "long_row_indices")


def test_vectorized_composite_matches_reference_and_preserves_inputs() -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode import (
        _build_operands,
        _torch_composite,
    )

    operands = _build_operands(
        torch,
        batch_size=2,
        context_len=513,
        next_n=2,
        max_model_len=520,
        top_k=512,
        logits_row_stride=528,
        device="cpu",
    )
    logits_before = operands.logits_backing.clone()
    lengths_before = operands.lengths.clone()
    expected = torch.full_like(operands.out, -99)
    dsa_persistent_topk_decode_reference(
        operands.logits,
        operands.lengths,
        expected,
        top_k=512,
        max_seq_len=513,
    )
    out_ptr = operands.out.untyped_storage().data_ptr()

    actual = _torch_composite(operands)

    assert actual is operands.out
    assert actual.untyped_storage().data_ptr() == out_ptr
    assert torch.equal(actual, expected)
    assert actual[0].tolist() == list(range(512))
    assert all(0 <= index < 513 for index in actual[1].tolist())
    assert torch.equal(actual[0], actual[2])
    assert torch.equal(actual[1], actual[3])
    assert torch.equal(operands.logits_backing, logits_before)
    assert torch.equal(operands.lengths, lengths_before)


def test_semantic_validation_accepts_reference_equivalent_operands() -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode import (
        _build_operands,
        _validate_semantics,
    )

    operands = _build_operands(
        torch,
        batch_size=2,
        context_len=513,
        next_n=2,
        max_model_len=520,
        top_k=512,
        logits_row_stride=528,
        device="cpu",
    )
    _validate_semantics(torch, operands, top_k=512, max_seq_len=513)


def test_composite_short_rows_use_local_indices_and_minus_one() -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode import (
        _build_operands,
        _torch_composite,
    )

    operands = _build_operands(
        torch,
        batch_size=1,
        context_len=3,
        next_n=2,
        max_model_len=8,
        top_k=4,
        logits_row_stride=8,
        device="cpu",
    )
    actual = _torch_composite(operands)
    assert actual.tolist() == [[0, 1, -1, -1], [0, 1, 2, -1]]


def test_composite_long_rows_select_local_top_values() -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode import (
        _build_operands,
        _torch_composite,
    )

    operands = _build_operands(
        torch,
        batch_size=1,
        context_len=7,
        next_n=1,
        max_model_len=8,
        top_k=3,
        logits_row_stride=8,
        device="cpu",
    )
    operands.logits[0, :7].copy_(torch.tensor([-2.0, 7.0, 1.0, 6.0, -3.0, 5.0, 4.0]))
    actual = _torch_composite(operands)
    assert actual[0].tolist() == [1, 3, 5]
    assert operands.logits[0, actual[0].long()].tolist() == [7.0, 6.0, 5.0]


def test_logical_bytes_accounts_only_valid_prefixes_lengths_and_output() -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode import _logical_bytes

    # B=3, next_n=2 yields lengths [4, 5] for each request.
    assert _logical_bytes(batch_size=3, context_len=5, next_n=2, top_k=4) == (
        4 * 3 * (4 + 5) + 4 * 6 + 4 * 6 * 4
    )
    assert _logical_bytes(batch_size=2, context_len=0, next_n=1, top_k=4) == (4 * 2 + 4 * 2 * 4)
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        _logical_bytes(batch_size=0, context_len=5, next_n=2, top_k=4)
    with pytest.raises(ValueError, match="context_len must be >= next_n - 1"):
        _logical_bytes(batch_size=1, context_len=0, next_n=2, top_k=4)


def test_profile_uses_cuda_event_total_and_returns_compute_metrics(monkeypatch) -> None:
    from profiling.runners.attention import dsa_persistent_topk_decode as runner

    operands = runner._build_operands(
        torch,
        batch_size=1,
        context_len=3,
        next_n=1,
        max_model_len=8,
        top_k=4,
        logits_row_stride=8,
        device="cpu",
    )
    monkeypatch.setattr(runner, "_validate_cuda_device", lambda _torch: None)
    monkeypatch.setattr(runner, "_build_operands", lambda *args, **kwargs: operands)
    monkeypatch.setattr(runner, "_validate_semantics", lambda *args, **kwargs: None)

    calls = []

    def fake_cuda_event(kernel, *, warmup):
        calls.append(("timer", warmup, kernel()))
        return 0.5

    def fake_energy(kernel, *, warmup, per_iter_time_ms):
        calls.append(("energy", warmup, per_iter_time_ms, kernel()))
        return 0.25

    monkeypatch.setattr(runner.Timer, "cuda_event", fake_cuda_event)
    monkeypatch.setattr(runner.Energy, "perf", fake_energy)

    metrics = runner.profile_dsa_persistent_topk_decode_torch(
        **(
            _BASE_SPEC
            | {
                "batch_size": 1,
                "context_len": 3,
                "next_n": 1,
            }
        )
    )

    assert calls[0][:2] == ("timer", 5)
    assert calls[0][2] is operands.out
    assert calls[1][:3] == ("energy", 5, 0.5)
    assert calls[1][3] is operands.out
    assert metrics.time_ms == 0.5
    assert metrics.energy_j == 0.25
    assert metrics.tflops == 0.0
    assert metrics.memory_bandwidth_gbps > 0


def test_profile_translates_runtime_failure(monkeypatch) -> None:
    from profiling.runners.attention import dsa_persistent_topk_decode as runner

    monkeypatch.setattr(runner, "_validate_cuda_device", lambda _torch: None)
    monkeypatch.setattr(runner, "_validate_semantics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_build_operands",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        runner.Timer,
        "cuda_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    with pytest.raises(KernelLaunchFailed, match="synthetic failure"):
        runner.profile_dsa_persistent_topk_decode_torch(**_BASE_SPEC)


def test_native_call_forwards_exact_operands_and_metadata() -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode import (
        _build_native_operands,
        _native_call,
    )

    operands = _build_native_operands(
        torch,
        batch_size=2,
        context_len=7,
        next_n=2,
        max_model_len=8,
        top_k=4,
        logits_row_stride=8,
        device="cpu",
    )
    calls = []

    def fake_op(*args):
        calls.append(args)
        return None

    returned = _native_call(fake_op, operands, top_k=4, max_seq_len=7)
    assert returned is None
    assert calls == [
        (
            operands.logits,
            operands.lengths,
            operands.out,
            operands.workspace,
            4,
            7,
        )
    ]


def test_native_semantic_validation_matches_reference_and_preserves_storage() -> None:
    from profiling.runners.attention.dsa_persistent_topk_decode import (
        _build_native_operands,
        _validate_native_semantics,
    )

    operands = _build_native_operands(
        torch,
        batch_size=2,
        context_len=513,
        next_n=2,
        max_model_len=520,
        top_k=512,
        logits_row_stride=528,
        device="cpu",
    )

    def reference_op(logits, lengths, out, workspace, top_k, max_seq_len):
        assert workspace is operands.workspace
        dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=top_k,
            max_seq_len=max_seq_len,
        )
        return None

    _validate_native_semantics(
        torch,
        reference_op,
        operands,
        top_k=512,
        max_seq_len=513,
    )


@pytest.mark.parametrize(
    ("exception_name", "expected_type"),
    [
        ("NativeExtensionUnsupported", ProfilerNotImplemented),
        ("NativeExtensionBuildError", KernelLaunchFailed),
        ("NativeExtensionLoadError", KernelLaunchFailed),
    ],
)
def test_native_loader_failures_translate_to_typed_profiler_errors(
    monkeypatch, exception_name, expected_type
) -> None:
    from profiling.runners.attention import dsa_persistent_topk_decode as runner
    from profiling.runners.attention import dsa_persistent_topk_native as native

    exception_type = getattr(native, exception_name)

    def fail(_torch):
        raise exception_type("synthetic native failure")

    monkeypatch.setattr(native, "load_persistent_topk_op", fail)
    with pytest.raises(expected_type, match="synthetic native failure"):
        runner._load_native_op(torch)


def test_native_profile_times_only_complete_op_and_returns_metrics(monkeypatch) -> None:
    from profiling.runners.attention import dsa_persistent_topk_decode as runner

    operands = runner._build_native_operands(
        torch,
        batch_size=1,
        context_len=3,
        next_n=1,
        max_model_len=8,
        top_k=4,
        logits_row_stride=8,
        device="cpu",
    )
    events = []

    def fake_op(*args):
        events.append(("op", args))
        return None

    monkeypatch.setattr(runner, "_validate_cuda_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_load_native_op", lambda _torch: fake_op)
    monkeypatch.setattr(runner, "_build_native_operands", lambda *args, **kwargs: operands)
    monkeypatch.setattr(runner, "_validate_native_semantics", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append(("sync",)))

    def fake_cuda_event(kernel, *, warmup):
        events.append(("timer", warmup))
        assert kernel() is None
        return 0.25

    def fake_energy(kernel, *, warmup, per_iter_time_ms):
        events.append(("energy", warmup, per_iter_time_ms))
        assert kernel() is None
        return 0.125

    monkeypatch.setattr(runner.Timer, "cuda_event", fake_cuda_event)
    monkeypatch.setattr(runner.Energy, "perf", fake_energy)
    metrics = runner.profile_dsa_persistent_topk_decode_vllm_cuda(
        **(_BASE_SPEC | {"batch_size": 1, "context_len": 3, "next_n": 1})
    )

    assert [event[0] for event in events] == ["sync", "timer", "op", "energy", "op"]
    assert all(event[1][0] is operands.logits for event in events if event[0] == "op")
    assert metrics.time_ms == 0.25
    assert metrics.energy_j == 0.125
    assert metrics.tflops == 0.0
    assert metrics.memory_bandwidth_gbps > 0


def test_native_profile_translates_op_runtime_failure(monkeypatch) -> None:
    from profiling.runners.attention import dsa_persistent_topk_decode as runner

    operands = runner._build_native_operands(
        torch,
        batch_size=1,
        context_len=3,
        next_n=1,
        max_model_len=8,
        top_k=4,
        logits_row_stride=8,
        device="cpu",
    )

    def fail_op(*args):
        raise RuntimeError("synthetic pinned op failure")

    monkeypatch.setattr(runner, "_validate_cuda_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_load_native_op", lambda _torch: fail_op)
    monkeypatch.setattr(runner, "_build_native_operands", lambda *args, **kwargs: operands)
    monkeypatch.setattr(runner, "_validate_native_semantics", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(runner.Timer, "cuda_event", lambda kernel, **kwargs: kernel())
    with pytest.raises(KernelLaunchFailed, match="synthetic pinned op failure"):
        runner.profile_dsa_persistent_topk_decode_vllm_cuda(
            **(_BASE_SPEC | {"batch_size": 1, "context_len": 3, "next_n": 1})
        )
