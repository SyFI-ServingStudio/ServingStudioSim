import pytest

from profiling.runners.attention.deepseek_v4_fused_q_kv_rmsnorm_vllm_triton import (
    _validate_args as validate_fused_rmsnorm,
)
from profiling.runners.attention.deepseek_v4_qnorm_rope_kv_insert_vllm_cuda import (
    _validate_args as validate_qnorm_insert,
)
from profiling.runners.exceptions import ProfilerNotImplemented


def test_fused_rmsnorm_accepts_only_the_production_identity():
    validate_fused_rmsnorm(128, 1536, 512, 1.0e-6, "bf16")
    with pytest.raises(ProfilerNotImplemented, match="supports"):
        validate_fused_rmsnorm(128, 1024, 512, 1.0e-6, "bf16")


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
