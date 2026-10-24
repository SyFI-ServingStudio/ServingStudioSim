"""CPU checks for ``gdn_conv_norm_gates``: schema, registration, runner input
validation and the Torch reference's semantics.

GPU timing and the check of TensorFold's ``_gdn_pre`` against the reference run
through the public ``python -m profiling run`` smoke on a GB10.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec, known_backends
from profiling.kernels.gdn_conv_norm_gates import KIND, GdnConvNormGatesArgs
from profiling.runners.attention.gdn_conv_norm_gates_tensorfold_triton import (
    logical_bytes,
    validate_args,
)
from profiling.runners.exceptions import ProfilerNotImplemented

_GEOMETRY = dict(num_qk_heads=16, num_value_heads=48, key_head_dim=128, value_head_dim=128, kernel_size=4)


def test_args_order_matches_the_rust_payload_and_facade_exists():
    # The Rust enumerate emits these names; a reorder or rename splits DB keys.
    assert [f.name for f in fields(GdnConvNormGatesArgs)] == [
        "num_tokens",
        "num_qk_heads",
        "num_value_heads",
        "key_head_dim",
        "value_head_dim",
        "kernel_size",
        "dtype",
    ]
    assert hasattr(perf_api, "get_gdn_conv_norm_gates_times")


def test_registration_runs_tensorfold_triton_in_its_env_bf16_only():
    assert known_backends(KIND) == ["tensorfold_triton"]
    spec = find_kernel_profiler_spec(KIND, "tensorfold_triton")
    assert spec.subprocess_env == "tensorfold_env"
    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.runner_ref.function_name == "profile_gdn_conv_norm_gates_tensorfold_triton"


@pytest.mark.parametrize(
    ("override", "error"),
    [
        # _gdn_pre takes one head width for q, k and v (tl.arange(0, DK)).
        ({"value_head_dim": 64}, ProfilerNotImplemented),
        ({"key_head_dim": 96, "value_head_dim": 96}, ProfilerNotImplemented),
        ({"dtype": "fp16"}, ProfilerNotImplemented),
        # A one-tap filter has no state rows; the kernel indexes kernel_size - 1 of them.
        ({"kernel_size": 1}, ValueError),
        ({"num_tokens": 0}, ValueError),
    ],
)
def test_runner_rejects_shapes_the_kernel_cannot_run(override, error):
    spec = {"num_tokens": 16, **_GEOMETRY, "dtype": "bf16", **override}
    with pytest.raises(error):
        validate_args(**spec)


def test_logical_bytes_counts_kernel_size_window_reads_per_token():
    width = (2 * 16 + 48) * 128
    got = logical_bytes(10, 16, 48, 128, 128, 4)
    assert got == 2 * (10 * (4 * width + width + 2 * 48) + width * 4) + 8 * 10 * 48


def test_reference_reads_state_rows_at_each_rows_stream_offset():
    # Two streams, kernel_size 2 (one state row each). With filter taps (1, 0)
    # each row's output is SiLU of its window's first row: the stream's own
    # state row for a window starting in the state. Reading stream 0's state for
    # a stream-1 row (the bug MULTI mode exists to avoid) would fail this.
    torch = pytest.importorskip("torch")
    from profiling.runners.attention.gdn_conv_norm_gates_reference import gdn_conv_norm_gates_reference

    heads, dim = 1, 2
    width = 3 * dim
    state = torch.tensor([[1.0] * width, [-2.0] * width], dtype=torch.bfloat16)
    qkv = torch.full((2, width), 3.0, dtype=torch.bfloat16)
    weight = torch.tensor([[1.0, 0.0]] * width, dtype=torch.bfloat16)
    windows = torch.tensor([[0, 1], [0, 2]])            # each row: [its state row, its own qkv row]
    sids = torch.tensor([0, 1])
    zeros = torch.zeros(2, heads, dtype=torch.bfloat16)
    params = torch.zeros(heads)
    q, k, v, g, beta = gdn_conv_norm_gates_reference(
        qkv, state, weight, windows, sids, zeros, zeros, params, params,
        num_qk_heads=heads, num_value_heads=heads, key_head_dim=dim, value_head_dim=dim,
    )
    silu = torch.nn.functional.silu
    assert torch.allclose(v[0].float(), silu(torch.tensor(1.0)).expand(1, dim), atol=1e-2)
    assert torch.allclose(v[1].float(), silu(torch.tensor(-2.0)).expand(1, dim), atol=1e-2)
    # q is unit-norm scaled by dim ** -0.5, k unit-norm; g is the decay exp(-softplus(0)), not its log.
    assert torch.allclose(q.float().norm(dim=-1), torch.full((2, 1), dim**-0.5), atol=1e-2)
    assert torch.allclose(k.float().norm(dim=-1), torch.ones(2, 1), atol=1e-2)
    assert torch.allclose(g, torch.full((2, 1), 0.5), atol=1e-6)
    assert torch.allclose(beta, torch.full((2, 1), 0.5))
