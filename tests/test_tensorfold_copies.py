"""TensorFold's remaining glue rows: torch copies/casts/gathers and the drafter's Triton
_group_sums, _dconv_kernel and _prep_kernel."""

from __future__ import annotations

import pytest
import torch

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.attention.head_major_qk_norm_rope_reference import (
    head_major_qk_norm_rope_reference,
)
from profiling.runners.elementwise import tensorfold as tf
from profiling.runners.elementwise.dynamic_causal_conv_reference import (
    dynamic_causal_conv_reference,
)
from profiling.runners.gemm.row_group_sums_reference import row_group_sums_reference

_ROWS = [
    ("elementwise", "tensorfold_torch"),
    ("elementwise", "tensorfold_torch_cat"),
    ("elementwise", "tensorfold_torch_strided_copy"),
    ("elementwise", "tensorfold_torch_cast"),
    ("elementwise", "tensorfold_torch_fp32_add"),
    ("elementwise", "tensorfold_torch_foreach_copy"),
    ("elementwise", "tensorfold_torch_cat_strided"),
    ("elementwise", "tensorfold_torch_rotary"),
    ("elementwise", "tensorfold_triton_paths"),
    ("indexed_row_copy", "tensorfold_torch"),
    ("embedding_row_gather", "tensorfold_torch"),
    ("rms_norm", "tensorfold_torch_eager"),
    ("row_group_sums", "tensorfold_triton"),
    ("dynamic_causal_conv", "tensorfold_triton"),
    ("head_major_qk_norm_rope", "tensorfold_triton"),
]


@pytest.mark.parametrize(("kind", "backend"), _ROWS)
def test_rows_run_in_tensorfold_env(kind: str, backend: str) -> None:
    # Catches a row landing on the project venv, whose Torch is not the one
    # TensorFold serves with (and has no tensorfold package for the callables).
    spec = find_kernel_profiler_spec(kind, backend)
    assert spec.subprocess_env == "tensorfold_env"
    assert spec.table_name == kind
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA GB10")


@pytest.mark.parametrize(
    ("call", "args"),
    [
        (tf.cast_direction, (100, 100)),  # a cast changes the width
        (tf.fp32_add_elements, (100, 100)),  # two inputs of the output's size
        (tf.foreach_tensor_bytes, (100, 200)),  # a copy
        (tf.cat_strided_rows, (197184, 197184 + 2)),  # whole 98592-element rows
        (tf.rotary_rows, (8, 512)),  # reads nothing
    ],
)
def test_statements_refuse_ratios_they_cannot_produce(call, args) -> None:
    # Catches a Rust config whose byte rates do not match the statement, which
    # would otherwise be timed as some other statement's bytes.
    with pytest.raises(ValueError):
        call(*args)


def test_cast_direction_follows_the_byte_ratio() -> None:
    assert tf.cast_direction(2048, 4096) == "to_fp32"
    assert tf.cast_direction(4096, 2048) == "to_bf16"


def test_foreach_copy_keeps_every_byte() -> None:
    sizes = tf.foreach_tensor_bytes(100_000, 100_000)
    assert sum(sizes) == 100_000 and max(sizes) == 32 << 10


def test_paths_bytes_carry_the_row_count() -> None:
    assert tf.paths_rows(4 * 88, 4 * 88 * 129, 128) == 88
    with pytest.raises(ValueError):
        tf.paths_rows(4 * 88, 4 * 88 * 128, 128)


def test_group_sums_reference_sums_each_group() -> None:
    x = torch.arange(8.0).reshape(1, 8)
    torch.testing.assert_close(row_group_sums_reference(torch, x, 4), torch.tensor([[6.0, 22.0]]))


def test_dynamic_conv_reference_restarts_every_segment() -> None:
    # Rows 0 and 2 start 2-row segments: they must not read the row before.
    # A reference that mixed across segments gives 1 + 2 at row 2.
    x = torch.ones(4, 2, dtype=torch.bfloat16)
    dyn = torch.zeros(4, 2, 2, 1, dtype=torch.bfloat16)
    base = torch.ones(2, 2, 2, dtype=torch.bfloat16)
    y = dynamic_causal_conv_reference(torch, x, dyn, base, 0, 2, 2)
    torch.testing.assert_close(y.float()[:, 0], torch.tensor([1.0, 2.0, 1.0, 2.0]))
    res = torch.full((4, 2), 10.0, dtype=torch.bfloat16)
    y = dynamic_causal_conv_reference(torch, x, dyn, base, 1, 2, 2, res)
    torch.testing.assert_close(y.float()[:, 0], torch.tensor([11.0, 12.0, 11.0, 12.0]))


def test_head_major_reference_moves_v_and_skips_absent_queries() -> None:
    # One kv head of width 2, no query heads (the context path): k is normed
    # and rotated, v copied head-major unchanged.
    qkv = torch.tensor([[3.0, 4.0, 7.0, 8.0]], dtype=torch.bfloat16)
    ones = torch.ones(2, dtype=torch.bfloat16)
    cos, sin = torch.ones(1, 1), torch.zeros(1, 1)
    q, k, v = head_major_qk_norm_rope_reference(torch, qkv, ones, ones, cos, sin, 0.0, 0, 1, 2)
    assert q is None and k.shape == (1, 1, 2)
    torch.testing.assert_close(v.float(), torch.tensor([[[7.0, 8.0]]]))
    torch.testing.assert_close(k.float(), torch.tensor([[[0.8485, 1.1314]]]), atol=1e-2, rtol=0)
