"""Behavioral tests for the fused QK RMSNorm + partial RoPE + gate split."""

import pytest
import torch

from profiling.runners.attention.qk_rms_norm_rope_gate_reference import (
    qk_rms_norm_rope_gate_reference,
)
from profiling.runners.attention.qk_rms_norm_rope_gate_vllm_triton import _validate_args


def test_reference_rotates_only_the_rotary_slice_and_copies_the_gate() -> None:
    # head_dim 4, rotary_dim 2: one rotated pair, two passthrough columns. A
    # full-head rotation or a normalized gate would both change these values.
    head_dim, rotary_dim = 4, 2
    q_gate = torch.tensor([[1.0, 1.0, 1.0, 1.0, 5.0, 6.0, 7.0, 8.0]], dtype=torch.bfloat16)
    k = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.bfloat16)
    weight = torch.ones(head_dim, dtype=torch.bfloat16)
    # Position 0 rotates by 90 degrees: (x, y) -> (-y, x).
    cos_sin_cache = torch.tensor([[0.0, 1.0]])

    q, k_out, gate = qk_rms_norm_rope_gate_reference(
        torch,
        q_gate,
        k,
        weight,
        weight,
        cos_sin_cache,
        torch.tensor([0]),
        0.0,
        head_dim,
        rotary_dim,
    )

    torch.testing.assert_close(q.float(), torch.tensor([[-1.0, 1.0, 1.0, 1.0]]))
    torch.testing.assert_close(k_out.float(), torch.tensor([[-1.0, 1.0, 1.0, 1.0]]))
    torch.testing.assert_close(gate.float(), torch.tensor([[5.0, 6.0, 7.0, 8.0]]))


@pytest.mark.parametrize("rotary_dim", [3, 512])
def test_runner_rejects_rotary_dims_the_kernel_cannot_pair(rotary_dim: int) -> None:
    with pytest.raises(ValueError):
        _validate_args(8, 24, 4, 256, rotary_dim, "bf16")
