"""TensorFold's Triton attention and MLP glue: gated_qk_rms_norm_rope, attn_output_gate,
and the tensorfold_triton rows of silu_and_mul and rms_norm."""

from __future__ import annotations

import math

import pytest
import torch

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.attention.attn_output_gate_reference import attn_output_gate_reference
from profiling.runners.attention.gated_qk_rms_norm_rope_reference import (
    gated_qk_rms_norm_rope_reference,
    rotary_axes,
)
from profiling.runners.exceptions import ProfilerNotImplemented

_ROWS = [
    ("gated_qk_rms_norm_rope", "tensorfold_triton"),
    ("attn_output_gate", "tensorfold_triton"),
    ("silu_and_mul", "tensorfold_triton"),
    ("rms_norm", "tensorfold_triton"),
]


@pytest.mark.parametrize(("kind", "backend"), _ROWS)
def test_rows_run_in_tensorfold_env_and_are_bf16_only(kind: str, backend: str) -> None:
    # Catches a row landing on the project venv (no TensorFold) or advertising
    # fp16, which the kernels' hard tl.bfloat16 casts would silently miscompute.
    spec = find_kernel_profiler_spec(kind, backend)
    assert spec.subprocess_env == "tensorfold_env"
    assert spec.table_name == kind
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA GB10")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA GB10")


def test_gated_reference_reads_queries_not_gates_and_rotates_only_the_rotary_slice() -> None:
    # head_dim 4, rotary_dim 2, position 1 with inv_freq pi/2: a 90-degree turn
    # of the first pair. Reading the gate half, or rotating the whole head,
    # changes these values.
    head_dim = 4
    q_gate = torch.tensor([[1.0, 1.0, 1.0, 1.0, 5.0, 6.0, 7.0, 8.0]], dtype=torch.bfloat16)
    k = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.bfloat16)
    weight = torch.ones(head_dim, dtype=torch.bfloat16)
    inv_freq = torch.tensor([math.pi / 2])

    q, k_out = gated_qk_rms_norm_rope_reference(
        torch, q_gate, k, weight, weight, torch.tensor([1]), inv_freq, 0.0, head_dim
    )

    want = torch.tensor([[[-1.0, 1.0, 1.0, 1.0]]])
    torch.testing.assert_close(q.float(), want, atol=1e-2, rtol=0)
    torch.testing.assert_close(k_out.float(), q.float())


def test_mrope_axes_follow_the_interleaved_sections() -> None:
    # Qwen3.5's (11, 11, 10) over 32 pairs: pair i % 3 == 1 reads height
    # while i < 33, i % 3 == 2 reads width while i < 30, the rest temporal.
    axes = rotary_axes(64, (11, 11, 10))
    assert axes[:6] == [0, 1, 2, 0, 1, 2]
    assert axes[29] == 2 and axes[31] == 1
    assert axes[30] == 0
    assert rotary_axes(64, None) == [0] * 32


def test_output_gate_reference_uses_each_heads_gate_half() -> None:
    # Two heads of width 2: the gate of head h sits after its query. A gate read
    # from the query half, or a head swap, gives a different product.
    out = torch.ones(1, 2, 2)
    q_gate = torch.tensor([[9.0, 9.0, 0.0, 0.0, -9.0, -9.0, 100.0, -100.0]])
    y = attn_output_gate_reference(torch, out, q_gate, 2)
    torch.testing.assert_close(y, torch.tensor([[0.5, 0.5, 1.0, 0.0]]))


def test_gated_runner_refuses_shapes_the_kernel_cannot_run() -> None:
    from profiling.runners.attention.gated_qk_rms_norm_rope_tensorfold_triton import (
        validate_args,
    )

    with pytest.raises(ProfilerNotImplemented):
        validate_args(8, 24, 4, 192, 64, "bf16")  # tl.arange needs a power of two
    with pytest.raises(ProfilerNotImplemented):
        validate_args(8, 24, 4, 256, 64, "fp16")
    with pytest.raises(ValueError):
        validate_args(8, 24, 4, 256, 63, "bf16")


@pytest.mark.parametrize(
    "module",
    [
        "profiling.runners.attention.attn_output_gate_tensorfold_triton",
        "profiling.runners.elementwise.silu_and_mul_tensorfold_triton",
        "profiling.runners.norm.rms_norm_tensorfold_triton",
    ],
)
def test_runners_refuse_widths_off_the_64_group_grid(module: str) -> None:
    # The group-sum store indexes width // 64 groups; a ragged width would drop
    # its tail and record a time for a call TensorFold never makes.
    import importlib

    validate_args = importlib.import_module(module).validate_args
    args = (8, 2, 96, "bf16") if "attn_output_gate" in module else (8, 5000, "bf16")
    with pytest.raises(ProfilerNotImplemented):
        validate_args(*args)


def test_stacked_swiglu_refuses_rows_past_its_int32_row_offset() -> None:
    # Catches a grid point whose row * 2 * hidden_dim offset wraps in int32: the
    # kernel faults (illegal address) and poisons every spec in the batch.
    from profiling.runners.elementwise.silu_and_mul_tensorfold_triton import validate_args

    validate_args(61680, 17408, "bf16")
    with pytest.raises(ProfilerNotImplemented):
        validate_args(65536, 17408, "bf16")
