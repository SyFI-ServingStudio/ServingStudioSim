"""The device gate the worker applies from a backend's declared BackendSupport.

Runners no longer repeat capability rules, so this gate is what keeps a backend
off a device its build does not support; it must read the real device.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from profiling.db.registry import BackendSupport, iter_kernel_profiler_specs
from profiling.runners.device import require_cuda_toolkit, unsupported_device
from profiling.runners.exceptions import ProfilerNotImplemented


def _install_torch(monkeypatch, capability, *, name="NVIDIA GPU", available=True):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: available,
        get_device_capability=lambda _index: capability,
        get_device_name=lambda _index: name,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


_GATED = sorted(
    (
        pytest.param(
            spec.kernel_kind, spec.backend, spec.supports, id=f"{spec.kernel_kind}-{spec.backend}"
        )
        for spec in iter_kernel_profiler_specs()
        if spec.supports.device_rule() is not None
    ),
    key=lambda param: param.id,
)


@pytest.mark.parametrize(("kind", "backend", "supports"), _GATED)
@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)])
def test_every_gated_backend_is_refused_exactly_where_its_support_says(
    monkeypatch, kind, backend, supports, capability
):
    # Catches a worker gate that disagrees with the declaration the launcher
    # and the public catalog read.
    _install_torch(monkeypatch, capability, name="NVIDIA Test")
    error = unsupported_device(supports, f"{kind} {backend}")
    if supports.allows_compute_capability(capability):
        assert error is None
    else:
        assert error == (
            f"{kind} {backend} needs {supports.device_rule()}, "
            f"got NVIDIA Test with SM{capability[0]}{capability[1]}"
        )


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


def test_a_backend_without_a_rule_never_reads_the_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert unsupported_device(BackendSupport(compute=None), "x") is None


def test_a_gated_backend_without_cuda_is_refused(monkeypatch):
    _install_torch(monkeypatch, (10, 0), available=False)
    supports = BackendSupport(compute=None, sm_targets=frozenset({"sm_100f"}))
    assert unsupported_device(supports, "k b") == "CUDA is required for k b"


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
