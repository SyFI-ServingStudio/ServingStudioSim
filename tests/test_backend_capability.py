"""Unit tests for per-(kernel, backend) BackendSupport capability.

CPU tier (unmarked): pure registry introspection, no GPU / no profile.db. This
is the single source of truth the backend-selection validator and the dry-run
`options` column read from, so the declarations are asserted here directly.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    _validate_registry,
    backend_supports,
    known_backends,
    supported_backends,
)


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
    assert supported_backends("single_gemm", DType.BF16) == [
        "torch",
        "torch_linear_vllm",
        "torch_linear",
        "sglang_bf16_auto",
        "sglang_fused_a_auto",
    ]
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
    # The real trt declaration is arch-gated: trtllm-gen kernels exist for the
    # SM10x family (sm_100f), so fp8 is legal on B200/B300 but NOT on H200. An
    # unknown GPU (None) skips the axis.
    for kind in (
        "flashinfer_attn_prefill",
        "flashinfer_attn_decode",
        "flashinfer_attn_rect",
    ):
        assert backend_supports(kind, "trt", DType.FP8_E4M3, DType.FP8_E4M3, gpu="NVIDIA B200")
        assert backend_supports(kind, "trt", DType.FP8_E4M3, DType.FP8_E4M3, gpu="NVIDIA B300")
        assert not backend_supports(kind, "trt", DType.FP8_E4M3, DType.FP8_E4M3, gpu="NVIDIA H200")
        assert backend_supports(kind, "trt", DType.FP8_E4M3, DType.FP8_E4M3)  # gpu=None → skip
        # options at fp8: trt drops on H200 (keeps fa3), returns on B200.
        assert supported_backends(kind, DType.FP8_E4M3, DType.FP8_E4M3, gpu="NVIDIA H200") == [
            "fa3"
        ]
        assert set(supported_backends(kind, DType.FP8_E4M3, DType.FP8_E4M3, gpu="NVIDIA B200")) == {
            "fa3",
            "trt",
        }


def test_vllm_mla_rope_is_bf16_on_sm80_and_newer():
    assert not backend_supports("vllm_mla_rope", "vllm_inductor", DType.BF16, gpu="NVIDIA V100")
    assert backend_supports("vllm_mla_rope", "vllm_inductor", DType.BF16, gpu="NVIDIA A100")
    assert backend_supports(
        "vllm_mla_rope",
        "vllm_inductor",
        DType.BF16,
        gpu="NVIDIA B200",
    )
    assert not backend_supports(
        "vllm_mla_rope",
        "vllm_inductor",
        DType.FP16,
        gpu="NVIDIA B200",
    )


def test_backend_support_allows_gpu_restriction():
    only_b200 = BackendSupport(compute=frozenset({DType.FP8_E4M3}), gpus=frozenset({"NVIDIA B200"}))
    assert only_b200.allows(DType.FP8_E4M3, gpu="NVIDIA B200")
    assert not only_b200.allows(DType.FP8_E4M3, gpu="NVIDIA H200")
    assert not only_b200.allows(DType.BF16, gpu="NVIDIA B200")
    # gpu=None means "don't check the gpu axis".
    assert only_b200.allows(DType.FP8_E4M3, gpu=None)


def test_sm_targets_match_arch_specific_and_family_builds():
    hopper_or_sm10x = BackendSupport(compute=None, sm_targets=frozenset({"sm_90a", "sm_100f"}))
    # sm_90a is exactly SM90; sm_100f is every SM10x part.
    for gpu in ("NVIDIA H100", "NVIDIA H200", "NVIDIA B200", "NVIDIA B300", "NVIDIA GB200"):
        assert hopper_or_sm10x.allows(DType.BF16, gpu=gpu), gpu
    for gpu in ("NVIDIA A100", "NVIDIA L40S"):
        assert not hopper_or_sm10x.allows(DType.BF16, gpu=gpu), gpu

    only_sm100 = BackendSupport(compute=None, sm_targets=frozenset({"sm_100a"}))
    assert only_sm100.allows(DType.BF16, gpu="NVIDIA B200")
    assert not only_sm100.allows(DType.BF16, gpu="NVIDIA B300")


def test_min_compute_capability_admits_every_newer_gpu():
    fp8 = BackendSupport(compute=None, min_compute_capability=(8, 9))
    for gpu in ("NVIDIA L40S", "NVIDIA H200", "NVIDIA B300"):
        assert fp8.allows(DType.FP8_E4M3, gpu=gpu), gpu
    assert not fp8.allows(DType.FP8_E4M3, gpu="NVIDIA A100")
    assert not fp8.allows(DType.FP8_E4M3, gpu="NVIDIA A40")


def test_unknown_or_absent_gpu_skips_the_device_check():
    support = BackendSupport(compute=None, sm_targets=frozenset({"sm_100f"}))
    # No GPU, a name outside gpu/spec.json, and a non-CUDA part are not refused:
    # the runner gates on the real device when it profiles.
    assert support.allows(DType.BF16, gpu=None)
    assert support.allows(DType.BF16, gpu="Some Future GPU")
    assert support.allows(DType.BF16, gpu="AMD MI300X")


def test_catalog_aliases_resolve_to_compute_capability():
    from profiling.gpu_catalog import gpu_compute_capability

    assert gpu_compute_capability("NVIDIA H200") == (9, 0)
    assert gpu_compute_capability("H100") == (9, 0)
    assert gpu_compute_capability("NVIDIA B200") == (10, 0)
    assert gpu_compute_capability("NVIDIA B300") == (10, 3)
    assert gpu_compute_capability("NVIDIA L40S") == (8, 9)
    assert gpu_compute_capability("AMD MI300X") is None
    assert gpu_compute_capability("Some Future GPU") is None


def test_no_backend_restricts_by_gpu_name():
    """A missing profile row is a data gap, not a reason to refuse a backend; real
    arch requirements are compute-capability fields, never GPU-name sets."""
    from profiling.db.registry import iter_kernel_profiler_specs

    named = [
        f"{spec.kernel_kind}:{spec.backend}"
        for spec in iter_kernel_profiler_specs()
        if spec.supports.gpus is not None
    ]
    assert named == []


def test_validate_registry_rejects_unknown_sm_target():
    bad = KernelProfilerSpec(
        kernel_kind="single_gemm",
        backend="torch",
        supports=BackendSupport(compute=None, sm_targets=frozenset({"hopper"})),
        runner_ref=RunnerRef(module_name="x", function_name="y"),
        table_name="single_gemm",
        args_schema=type("A", (), {}),
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
    with pytest.raises(ValueError, match="unknown sm target"):
        _validate_registry([bad])


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
