import pytest

from profiling.runners.attention.inv_rope_fp8_quant_vllm_triton import (
    _Launch,
    _logical_bytes,
    _validate_args,
)


@pytest.mark.parametrize("num_tokens", [1, 8192, 8193, 65_536])
def test_any_positive_token_count_is_supported(num_tokens):
    assert _validate_args(num_tokens) == num_tokens


@pytest.mark.parametrize("num_tokens", [0, -1, 1.0])
def test_invalid_token_count_is_rejected(num_tokens):
    with pytest.raises(ValueError):
        _validate_args(num_tokens)


def test_logical_bytes_count_each_semantic_tensor_once():
    assert _logical_bytes(2) == 2 * (2 * 64 * 512 + 8 + 4 * 64 + 64 * 512 + 4 * 8 * 32)


def test_launch_preserves_the_production_public_call_contract():
    calls = []

    def callable(*args, **kwargs):
        calls.append((args, kwargs))
        return "result"

    launch = _Launch(callable, "output", "positions", "cache")
    assert launch.run() == "result"
    assert calls == [
        (
            ("output", "positions", "cache"),
            {
                "n_groups": 8,
                "heads_per_group": 8,
                "nope_dim": 448,
                "rope_dim": 64,
                "quant_group_size": 128,
                "tma_aligned_scales": False,
            },
        )
    ]


@pytest.mark.parametrize(
    ("capability", "accepted"), [((8, 0), False), ((8, 9), True), ((9, 0), True), ((10, 0), True)]
)
def test_gpu_guard_is_the_fp8_capability_floor(capability, accepted):
    from types import SimpleNamespace

    from profiling.runners.attention.inv_rope_fp8_quant_vllm_triton import _require_fp8_gpu
    from profiling.runners.exceptions import ProfilerNotImplemented

    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_capability=lambda _device: capability,
    )
    torch = SimpleNamespace(cuda=cuda)
    if accepted:
        _require_fp8_gpu(torch)
    else:
        with pytest.raises(ProfilerNotImplemented, match="SM89"):
            _require_fp8_gpu(torch)
