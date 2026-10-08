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


def test_nvfp4_quant_swizzled_times_the_linear_kernel_call_and_counts_tile_padding(
    monkeypatch,
) -> None:
    # Catches a swizzled row that times a different call than vLLM's B200 NVFP4
    # linear (a "trtllm" backend switches m <= 32 to the 8x4 layout), filters
    # the unswizzled kernel, or drops the zero-filled 128x4 tile padding.
    calls: list[tuple[object, object, dict[str, object]]] = []
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

    ops = SimpleNamespace(
        scaled_fp4_quant=lambda value, scale, **kwargs: calls.append((value, scale, kwargs))
    )
    vllm = ModuleType("vllm")
    vllm._custom_ops = ops
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr(nvfp4_quant, "_calibrated_global_scale", lambda value: global_scale)

    def fake_cupti(fn, *, kernel_name):
        assert kernel_name == nvfp4_quant._KERNEL_NAMES["swizzled_e4m3"]
        # CUPTI's mangled names for the swizzled and linear kernels.
        assert kernel_name in "_ZN4vllm15cvt_fp16_to_fp4I13__nv_bfloat16Lb0ELb0EEEviiiiPKT_PKfPjS7_"
        assert kernel_name not in (
            "_ZN4vllm24cvt_fp16_to_fp4_sf_majorI13__nv_bfloat16Lb0EEEviiiiiPKT_PKfPjS7_"
        )
        fn()
        return 2.0

    monkeypatch.setattr(nvfp4_quant.Timer, "cupti", fake_cupti)
    monkeypatch.setattr(nvfp4_quant.Energy, "perf", lambda fn, *, per_iter_time_ms: (fn(), 3.0)[1])

    metrics = nvfp4_quant.profile_nvfp4_quant_vllm_cuda(
        num_tokens=2,
        hidden_size=32,
        group_size=16,
        input_dtype="bf16",
        scale_format="swizzled_e4m3",
    )

    expected_call = (
        source,
        global_scale,
        {"is_sf_swizzled_layout": True, "backend": "flashinfer-cutedsl"},
    )
    assert calls == [expected_call, expected_call]
    # BF16 in + FP4 out + a 128-row x 4-group scale tile for 2 rows x 2 groups.
    assert metrics.memory_bandwidth_gbps == pytest.approx((128 + 32 + 128 * 4) / 0.002 / 1e9)


@pytest.mark.parametrize(
    ("num_tokens", "hidden_size", "scale_bytes"),
    [
        (1, 4096, 128 * 256),
        (128, 4096, 128 * 256),
        (129, 4096, 256 * 256),
        (64, 1040, 128 * 68),  # 65 groups pad to 68
    ],
)
def test_nvfp4_quant_swizzled_scale_bytes_pad_rows_and_groups(
    num_tokens, hidden_size, scale_bytes
) -> None:
    # Catches a byte count that follows the unpadded row-major scale size.
    data_bytes = num_tokens * hidden_size * 2 + num_tokens * hidden_size // 2
    assert nvfp4_quant._logical_bytes(num_tokens, hidden_size, "swizzled_e4m3") == (
        data_bytes + scale_bytes
    )
    assert nvfp4_quant._logical_bytes(num_tokens, hidden_size, "linear_e4m3") == (
        data_bytes + num_tokens * hidden_size // 16
    )


def test_nvfp4_quant_swizzled_format_is_vllm_only() -> None:
    # The FlashInfer runner always writes the linear layout; accepting the
    # swizzled identity there would store a mislabeled row.
    spec = {
        "num_tokens": 2,
        "hidden_size": 32,
        "group_size": 16,
        "input_dtype": "bf16",
        "scale_format": "swizzled_e4m3",
    }
    assert nvfp4_quant._validate_args(
        **spec, supported_scale_formats=nvfp4_quant._VLLM_SCALE_FORMATS
    )
    with pytest.raises(ValueError, match="scale format"):
        nvfp4_quant.profile_nvfp4_quant_flashinfer_cutedsl(**spec)


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
