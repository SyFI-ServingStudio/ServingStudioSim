"""The device gate the worker applies from a backend's declared BackendSupport.

Runners no longer repeat capability rules, so this gate is what keeps a backend
off a device its build does not support; it must read the real device.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from fixtures.fake_torch import fake_cuda_torch

from profiling.db.registry import (
    BackendSupport,
    find_kernel_profiler_spec,
    iter_kernel_profiler_specs,
)
from profiling.runners.device import require_cuda_toolkit, unsupported_device
from profiling.runners.exceptions import ProfilerNotImplemented


def _install_torch(monkeypatch, capability, *, name="NVIDIA GPU", available=True):
    torch = fake_cuda_torch(capability, name, available=available)
    monkeypatch.setitem(sys.modules, "torch", torch)


_CAPABILITIES = ((8, 0), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0))
# Expected admission on each capability above, one row per kind of rule. The
# truth is written out, not recomputed from BackendSupport, so a wrong
# declaration or a wrong capability match both fail here.
_PINNED = {
    ("mxfp4_marlin_moe_gemm", "vllm_marlin"): "YYYYYY",  # 8.0+
    ("kv_compress_store", "vllm_triton"): "-YYYYY",  # 8.9+
    ("nvfp4_quant", "vllm_cuda"): "---YYY",  # 10.0+
    ("all_reduce_fusion", "flashinfer_mnnvl"): "---Y--",  # sm_100a
    ("nvfp4_fused_moe", "flashinfer_trtllm_sm100"): "---YY-",  # sm_100f
    ("gdn_chunk_delta_rule", "flashinfer"): "--Y---",  # sm_90a
    ("dsa_sparse_mla_attention", "vllm_flashmla_bf16"): "--YYY-",  # sm_90a, sm_100f
    ("dsa_paged_mqa_logits_decode", "deepgemm_fp8"): "--YYYY",  # + sm_120f
}


@pytest.mark.parametrize(
    ("kind", "backend", "admitted"),
    [pytest.param(*key, row, id=f"{key[0]}-{key[1]}") for key, row in _PINNED.items()],
)
def test_named_backends_are_admitted_exactly_where_pinned(monkeypatch, kind, backend, admitted):
    supports = find_kernel_profiler_spec(kind, backend).supports
    for capability, mark in zip(_CAPABILITIES, admitted, strict=True):
        _install_torch(monkeypatch, capability, name="NVIDIA Test")
        error = unsupported_device(supports, f"{kind} {backend}")
        if mark == "Y":
            assert error is None, capability
        else:
            assert error == (
                f"{kind} {backend} needs {supports.device_rule()}, "
                f"got NVIDIA Test with SM{capability[0]}{capability[1]}"
            ), capability


def test_the_sm10x_family_and_the_fp8_floor_are_capability_rules(monkeypatch):
    family = BackendSupport(compute=None, sm_targets=frozenset({"sm_100f"}))
    fp8 = BackendSupport(compute=None, min_compute_capability=(8, 9))
    for capability, family_ok, fp8_ok in (
        ((10, 0), True, True),
        ((10, 3), True, True),
        ((9, 0), False, True),
        ((8, 9), False, True),
        ((8, 0), False, False),
    ):
        _install_torch(monkeypatch, capability)
        assert (unsupported_device(family, "x") is None) is family_ok
        assert (unsupported_device(fp8, "x") is None) is fp8_ok


def test_every_backend_needs_cuda_and_one_without_a_rule_takes_any_gpu(monkeypatch):
    # Runners no longer check for CUDA themselves, so the gate must refuse a
    # CUDA-less worker even for a backend that declares no capability rule.
    ruleless = BackendSupport(compute=None)
    gated = BackendSupport(compute=None, sm_targets=frozenset({"sm_100f"}))
    _install_torch(monkeypatch, (10, 0), available=False)
    assert unsupported_device(ruleless, "k b") == "CUDA is required for k b"
    assert unsupported_device(gated, "k b") == "CUDA is required for k b"
    for capability in ((7, 0), (8, 0), (12, 0)):
        _install_torch(monkeypatch, capability)
        assert unsupported_device(ruleless, "k b") is None


@pytest.mark.parametrize(
    ("version", "ok"), [("12.8", True), ("13.0.1", True), ("12.6", False), (None, False)]
)
def test_cuda_toolkit_floor(version, ok):
    torch = SimpleNamespace(version=SimpleNamespace(cuda=version))
    if ok:
        require_cuda_toolkit(torch, (12, 8), "x")
    else:
        with pytest.raises(ProfilerNotImplemented, match=r"x requires CUDA >= 12\.8"):
            require_cuda_toolkit(torch, (12, 8), "x")


def test_the_worker_refuses_an_unsupported_device_without_loading_the_runner(monkeypatch, tmp_path):
    # Catches the gate moving after the runner import, or a refused chunk
    # reporting fewer results than specs.
    from profiling.exec import local_worker

    _install_torch(monkeypatch, (9, 0), name="NVIDIA H200")
    spec = next(
        s
        for s in iter_kernel_profiler_specs()
        if s.supports.sm_targets == frozenset({"sm_100f"}) and s.kernel_kind == "bf16_fused_moe"
    )
    monkeypatch.setattr(
        type(spec),
        "load_list_runner",
        lambda _self: pytest.fail("an unsupported device must not load the runner"),
    )
    monkeypatch.setattr(local_worker, "args_to_spec", lambda args: {})
    monkeypatch.setattr(local_worker, "coerce_args", lambda schema, spec: spec)
    monkeypatch.setattr(local_worker, "_current_gpu_name", lambda: "NVIDIA H200")
    monkeypatch.setattr(local_worker, "_runtime_versions", lambda _backend: {})
    request = tmp_path / "in.json"
    output = tmp_path / "out.json"
    request.write_text(
        json.dumps({"kernel_kind": spec.kernel_kind, "specs": [{"backend": spec.backend}] * 2})
    )
    local_worker._worker_main(request, output)
    results = json.loads(output.read_text())["results"]
    assert (
        results
        == [
            {
                "ok": False,
                "error": f"{spec.kernel_kind} {spec.backend} needs sm_100f, "
                "got NVIDIA H200 with SM90",
            }
        ]
        * 2
    )
