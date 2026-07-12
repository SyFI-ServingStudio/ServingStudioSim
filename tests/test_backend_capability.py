"""Unit tests for per-(kernel, backend) BackendSupport capability.

CPU tier (unmarked): pure registry introspection, no GPU / no profile.db. This
is the single source of truth the backend-selection validator and the dry-run
`options` column read from, so the declarations are asserted here directly.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.db.registry import (
    BackendSupport,
    backend_supports,
    known_backends,
    supported_backends,
    _validate_registry,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
)
from profiling.db.outlier import BatchOutlierPolicy


def test_gemm_backend_is_locked_to_dtype():
    # Torch layouts ⟺ bf16/fp16, deepgemm ⟺ fp8.
    for kind in ("single_gemm", "grouped_gemm"):
        assert backend_supports(kind, "torch", DType.BF16)
        assert backend_supports(kind, "torch", DType.FP16)
        assert not backend_supports(kind, "torch", DType.FP8_E4M3)
        assert backend_supports(kind, "deepgemm", DType.FP8_E4M3)
        assert not backend_supports(kind, "deepgemm", DType.BF16)
    assert backend_supports("single_gemm", "torch_linear", DType.BF16)
    assert backend_supports("single_gemm", "torch_linear", DType.FP16)
    assert not backend_supports("single_gemm", "torch_linear", DType.FP8_E4M3)


def test_attention_backend_dtype_matrix():
    # Two axes — compute (q) and kv-cache. fp8 *compute* needs fa3/trt; fp8 *KV*
    # alone is fine on fa2 (a current production config), but not on cudnn.
    for kind in (
        "flashinfer_attn_prefill",
        "flashinfer_attn_decode",
        "flashinfer_attn_rect",
    ):
        # full bf16
        assert backend_supports(kind, "fa2", DType.BF16, DType.BF16)
        # bf16 query + fp8 KV cache — the current prod config, MUST validate on fa2.
        assert backend_supports(kind, "fa2", DType.BF16, DType.FP8_E4M3)
        # fp8 compute (query) is NOT supported by fa2.
        assert not backend_supports(kind, "fa2", DType.FP8_E4M3, DType.FP8_E4M3)
        # fp8 compute needs fa3 (or trt).
        assert backend_supports(kind, "fa3", DType.FP8_E4M3, DType.FP8_E4M3)
        assert backend_supports(kind, "trt", DType.FP8_E4M3, DType.FP8_E4M3)
        # cudnn is bf16-only on BOTH axes — fp8 KV is not allowed even with bf16 q.
        assert backend_supports(kind, "cudnn", DType.BF16, DType.BF16)
        assert not backend_supports(kind, "cudnn", DType.BF16, DType.FP8_E4M3)


def test_comm_and_elementwise_are_dtype_agnostic():
    # Size-keyed / byte-keyed: any dtype allowed (dtypes=None).
    for kind in ("all_reduce", "p2p_intra", "p2p_inter"):
        for backend in known_backends(kind):
            assert backend_supports(kind, backend, DType.BF16)
            assert backend_supports(kind, backend, DType.FP8_E4M3)
    assert backend_supports("elementwise", "triton", DType.FP8_E4M3)


def test_supported_backends_filters_options_by_dtype():
    # The dry-run `options` column: filter registered backends to a dtype.
    assert supported_backends("single_gemm", DType.FP8_E4M3) == ["deepgemm"]
    assert supported_backends("single_gemm", DType.BF16) == ["torch", "torch_linear"]
    assert set(supported_backends("flashinfer_attn_prefill", DType.FP8_E4M3)) == {
        "fa3",
        "trt",
    }
    assert set(supported_backends("flashinfer_attn_prefill", DType.BF16)) == {
        "fa2",
        "fa3",
        "cudnn",
    }


def test_trt_attention_is_blackwell_gated():
    # The real trt declaration is GPU-gated: fp8 is legal on B200 but NOT on H200
    # (trtllm-gen kernels are Blackwell-only). An unknown GPU (None) skips the axis.
    for kind in (
        "flashinfer_attn_prefill",
        "flashinfer_attn_decode",
        "flashinfer_attn_rect",
    ):
        assert backend_supports(kind, "trt", DType.FP8_E4M3, DType.FP8_E4M3, gpu="NVIDIA B200")
        assert not backend_supports(kind, "trt", DType.FP8_E4M3, DType.FP8_E4M3, gpu="NVIDIA H200")
        assert backend_supports(kind, "trt", DType.FP8_E4M3, DType.FP8_E4M3)  # gpu=None → skip
        # options at fp8: trt drops on H200 (keeps fa3), returns on B200.
        assert supported_backends(kind, DType.FP8_E4M3, DType.FP8_E4M3, gpu="NVIDIA H200") == ["fa3"]
        assert set(supported_backends(kind, DType.FP8_E4M3, DType.FP8_E4M3, gpu="NVIDIA B200")) == {
            "fa3",
            "trt",
        }


def test_backend_support_allows_gpu_restriction():
    only_b200 = BackendSupport(
        compute=frozenset({DType.FP8_E4M3}), gpus=frozenset({"NVIDIA B200"})
    )
    assert only_b200.allows(DType.FP8_E4M3, gpu="NVIDIA B200")
    assert not only_b200.allows(DType.FP8_E4M3, gpu="NVIDIA H200")
    assert not only_b200.allows(DType.BF16, gpu="NVIDIA B200")
    # gpu=None means "don't check the gpu axis".
    assert only_b200.allows(DType.FP8_E4M3, gpu=None)


def test_validate_registry_rejects_empty_dtype_set():
    bad = KernelProfilerSpec(
        kernel_kind="single_gemm",
        backend="torch",
        supports=BackendSupport(compute=frozenset()),  # empty ≠ None
        runner_ref=RunnerRef(module_name="x", function_name="y"),
        table_name="single_gemm",
        args_schema=type("A", (), {}),
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
    with pytest.raises(ValueError, match="empty compute dtype set"):
        _validate_registry([bad])
