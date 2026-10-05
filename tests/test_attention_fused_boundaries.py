import pytest

from profiling.runners.attention.q_kv_rms_norm_vllm_triton import (
    _validate_args as validate_fused_rmsnorm,
)
from profiling.runners.attention.qnorm_rope_kv_insert_vllm_cuda import (
    _validate_args as validate_qnorm_insert,
)
from profiling.runners.exceptions import ProfilerNotImplemented


@pytest.mark.parametrize(
    ("q_dim", "kv_dim", "rms_eps"),
    [(1536, 512, 1.0e-6), (1536, 512, 1.0e-5), (1024, 512, 1.0e-6), (2048, 576, 1.0e-6)],
)
def test_fused_rmsnorm_accepts_any_row_widths(q_dim, kv_dim, rms_eps):
    validate_fused_rmsnorm(128, q_dim, kv_dim, rms_eps, "bf16")


@pytest.mark.parametrize(
    ("args", "error", "match"),
    [
        ((128, 0, 512, 1.0e-6, "bf16"), ValueError, "q_dim"),
        ((128, 1536, -1, 1.0e-6, "bf16"), ValueError, "kv_dim"),
        ((128, 1536, 512, 0.0, "bf16"), ValueError, "rms_eps"),
        ((128, 1536, 512, 1.0e-6, "fp16"), ProfilerNotImplemented, "bf16"),
    ],
)
def test_fused_rmsnorm_rejects_invalid_shapes(args, error, match):
    with pytest.raises(error, match=match):
        validate_fused_rmsnorm(*args)


def test_qnorm_insert_preserves_dp_padding_and_physical_identity():
    identity = (
        64,
        64,
        512,
        64,
        256,
        1.0e-6,
        "bf16",
        "fp8_ds_mla",
        "block_segregated_data_then_scales",
        "ue8m0",
    )
    validate_qnorm_insert(128, 96, identity)
    with pytest.raises(ValueError, match="num_insert_tokens"):
        validate_qnorm_insert(128, 129, identity)


_QNORM_IDENTITY = (
    64,
    64,
    512,
    64,
    256,
    1.0e-6,
    "bf16",
    "fp8_ds_mla",
    "block_segregated_data_then_scales",
    "ue8m0",
)


@pytest.mark.parametrize(
    "overrides",
    [
        {0: 8, 1: 8},
        {0: 16, 1: 16},
        {0: 32, 1: 32},
        {0: 128, 1: 128},
        {0: 24, 1: 32},
        {4: 64},
        {5: 1.0e-5},
    ],
)
def test_qnorm_insert_accepts_any_compiled_head_padding_and_block_size(overrides):
    identity = tuple(overrides.get(index, value) for index, value in enumerate(_QNORM_IDENTITY))
    validate_qnorm_insert(100_000, 96, identity)


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({1: 48}, ProfilerNotImplemented, "padded_heads"),
        ({0: 64, 1: 32}, ProfilerNotImplemented, "padded_heads"),
        ({0: 0}, ValueError, "num_heads"),
        ({2: 576}, ProfilerNotImplemented, "head_dim=512"),
        ({3: 32}, ProfilerNotImplemented, "rope_dim=64"),
        ({4: 0}, ValueError, "block_size"),
        ({5: 0.0}, ValueError, "rms_eps"),
        ({6: "fp16"}, ProfilerNotImplemented, "storage"),
        ({7: "bf16"}, ProfilerNotImplemented, "storage"),
    ],
)
def test_qnorm_insert_rejects_what_the_kernel_cannot_launch(overrides, error, match):
    identity = tuple(overrides.get(index, value) for index, value in enumerate(_QNORM_IDENTITY))
    with pytest.raises(error, match=match):
        validate_qnorm_insert(128, 96, identity)
