from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from profiling.runners.elementwise import nvfp4_quant


def test_nvfp4_quant_profiles_the_unswizzled_vllm_callable(monkeypatch) -> None:
    calls: list[tuple[object, object, bool]] = []
    source = object()
    global_scale = object()

    torch = ModuleType("torch")
    torch.bfloat16 = object()
    torch.float32 = object()
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_capability=lambda _device: (10, 0),
        get_device_name=lambda _device: "NVIDIA B200",
    )
    torch.randn = lambda shape, *, dtype, device: source
    torch.ones = lambda shape, *, dtype, device: global_scale

    ops = SimpleNamespace(
        scaled_fp4_quant=lambda value, scale, *, is_sf_swizzled_layout: calls.append(
            (value, scale, is_sf_swizzled_layout)
        )
    )
    vllm = ModuleType("vllm")
    vllm._custom_ops = ops
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    def fake_cupti(fn, *, kernel_name):
        assert kernel_name == "cvt_fp16_to_fp4_sf_major"
        fn()
        return 2.0

    def fake_energy(fn, *, per_iter_time_ms):
        assert per_iter_time_ms == 2.0
        fn()
        return 3.0

    monkeypatch.setattr(nvfp4_quant.Timer, "cupti", fake_cupti)
    monkeypatch.setattr(nvfp4_quant.Energy, "perf", fake_energy)

    metrics = nvfp4_quant.profile_nvfp4_quant_vllm_cuda(
        num_tokens=2,
        hidden_size=32,
        group_size=16,
        input_dtype="bf16",
        scale_format="linear_e4m3",
    )

    assert calls == [(source, global_scale, False), (source, global_scale, False)]
    assert metrics.time_ms == 2.0
    assert metrics.energy_j == 3.0
    assert metrics.memory_bandwidth_gbps == pytest.approx((128 + 32 + 4) / 0.002 / 1e9)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_tokens": 0}, "num_tokens must be > 0"),
        ({"hidden_size": 31}, "divisible by 16"),
        ({"group_size": 32}, "group_size=16"),
        ({"input_dtype": "fp16"}, "input_dtype=bf16"),
        ({"scale_format": "trtllm_swizzled_e4m3"}, "scale format"),
    ],
)
def test_nvfp4_quant_rejects_nonproduction_cache_identities(kwargs, message) -> None:
    spec = {
        "num_tokens": 2,
        "hidden_size": 32,
        "group_size": 16,
        "input_dtype": "bf16",
        "scale_format": "linear_e4m3",
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        nvfp4_quant._validate_args(**spec)
