"""Focused CPU tests for ``kda_chunk_prefill`` (no GPU, no DB).

Each test names the defect it guards against. The GPU comparison of the
vendored Triton callable against the oracle runs inside the runner, and its
evidence comes from the Slurm smoke.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch

from profiling.db.args import DType
from profiling.kernels.kda_chunk_prefill import KdaChunkPrefillArgs
from profiling.runners.attention import kda_chunk_prefill_vllm_triton as runner
from profiling.runners.attention.kda_chunk_prefill_reference import (
    kda_chunk_prefill_reference,
    kda_safe_gate,
)

# The in-situ capture: mixed iteration 886, one 2019-token prefill + 29 decodes, TP4.
_CAPTURE = dict(
    num_tokens=2048,
    max_sequence_length=2019,
    num_decode_sequences=29,
    num_heads=16,
    head_dim=128,
    dtype="bf16",
)


def test_args_fields_are_the_runner_kwargs() -> None:
    # Defect: schema/DB key order drifting from what the worker passes the runner.
    names = [f.name for f in fields(KdaChunkPrefillArgs)]
    assert names == [
        "num_tokens",
        "max_sequence_length",
        "num_decode_sequences",
        "num_heads",
        "head_dim",
        "dtype",
    ]
    signature = inspect.signature(runner.profile_kda_chunk_prefill_vllm_triton)
    assert list(signature.parameters) == names


def test_canonical_varlen_keeps_decodes_as_their_own_chunks() -> None:
    # Defect: folding co-scheduled decodes into the prefill, which would halve
    # the chunk count (33 instead of 61) that the chunk-parallel kernels launch.
    shape = runner.validate_args(**_CAPTURE)
    assert runner.sequence_lengths(shape) == (1,) * 29 + (2019,)
    assert runner.sequence_boundaries(shape)[-1] == 2048
    assert runner.num_chunks(shape) == 29 + 32
    prefill_only = runner.validate_args(**{**_CAPTURE, "num_decode_sequences": 0})
    assert runner.sequence_lengths(prefill_only) == (2019, 29)
    assert runner.num_chunks(prefill_only) == 33
    chunked = runner.validate_args(
        **{**_CAPTURE, "num_tokens": 8192, "max_sequence_length": 4096, "num_decode_sequences": 0}
    )
    assert runner.sequence_lengths(chunked) == (4096, 4096)


def test_guard_shape_keeps_the_mixed_topology_bounded() -> None:
    # Defect: a correctness witness that drops decodes or the partial tail chunk.
    guard = runner.guard_shape(runner.validate_args(**_CAPTURE))
    lengths = runner.sequence_lengths(guard)
    assert lengths[:8] == (1,) * 8 and lengths[8:] == (1024, 995)
    small = runner.validate_args(
        **{**_CAPTURE, "num_tokens": 300, "max_sequence_length": 200, "num_decode_sequences": 3}
    )
    assert runner.guard_shape(small) == small


@pytest.mark.parametrize(
    "override",
    [
        {"head_dim": 64},
        {"dtype": "fp16"},
        {"num_decode_sequences": 2048},  # pure decode: fused_recurrent_kda, not this path
        {"max_sequence_length": 0},
        {"num_decode_sequences": -1},
        {"num_heads": True},
    ],
)
def test_rejects_shapes_outside_the_verified_path(override: dict) -> None:
    # Defect: silently profiling a path production never takes.
    with pytest.raises(ValueError):
        runner.validate_args(**{**_CAPTURE, **override})


def test_invoke_reproduces_the_production_call() -> None:
    # Defect: timing contiguous q/k/v (skips the callable's three copy launches),
    # or dropping safe_gate / in-kernel l2norm / the final-state write.
    shape = runner.KdaChunkPrefillShape(
        num_tokens=70,
        max_sequence_length=66,
        num_decode_sequences=4,
        num_heads=2,
        head_dim=128,
        dtype=DType.BF16,
    )
    operands = runner.build_operands(torch, shape, device=torch.device("cpu"))
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return "o", "ht"

    assert runner.invoke(fake, operands) == ("o", "ht")
    projection = shape.num_heads * shape.head_dim
    for name in ("q", "k", "v"):
        view = seen[name]
        assert view.shape == (1, 70, 2, 128) and view.dtype is torch.bfloat16
        assert not view.is_contiguous() and view.stride(1) == 3 * projection
        assert view.untyped_storage().data_ptr() == operands.qkv.untyped_storage().data_ptr()
    assert seen["raw_g"].is_contiguous() and seen["raw_g"].dtype is torch.bfloat16
    beta = seen["beta"]
    assert beta.shape == (1, 70, 2) and beta.dtype is torch.float32
    assert bool(((beta > 0) & (beta < 1)).all())
    assert seen["A_log"].shape == (1, 1, 2, 1) and seen["g_bias"].shape == (projection,)
    state = seen["initial_state"]
    assert state.shape == (5, 2, 128, 128) and state.dtype is torch.float32
    assert bool(state[:4].abs().sum(dim=(1, 2, 3)).gt(0).all())
    assert bool(state[4:].eq(0).all())
    assert seen["cu_seqlens"].dtype is torch.int32
    assert seen["cu_seqlens"].tolist() == [0, 1, 2, 3, 4, 70]
    assert seen["safe_gate"] is True and seen["lower_bound"] == -5.0
    assert seen["use_qk_l2norm_in_kernel"] is True and seen["output_final_state"] is True


def test_safe_gate_formula() -> None:
    # Defect: softplus gate or bias/A_log applied on the wrong axis.
    heads, dim = 3, 4
    a_log = torch.tensor([0.0, 1.0, 2.0])
    dt_bias = torch.arange(heads * dim, dtype=torch.float32) / 10
    raw_g = -dt_bias.view(1, heads, dim)  # zero pre-activation -> sigmoid(0)
    torch.testing.assert_close(
        kda_safe_gate(raw_g, a_log, dt_bias, -5.0), torch.full((1, heads, dim), -2.5)
    )
    raw_g = torch.ones(1, heads, dim)
    expected = -5.0 * torch.sigmoid(a_log.exp().view(1, heads, 1) * (1 + dt_bias.view(heads, dim)))
    torch.testing.assert_close(kda_safe_gate(raw_g, a_log, dt_bias, -5.0), expected)


def _oracle_inputs(tokens: int, heads: int = 2, dim: int = 4):
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(tokens, heads, dim, generator=generator)
    k = torch.randn(tokens, heads, dim, generator=generator)
    v = torch.randn(tokens, heads, dim, generator=generator)
    raw_g = torch.randn(tokens, heads, dim, generator=generator)
    return q, k, v, raw_g, torch.zeros(heads), torch.zeros(heads * dim)


def test_oracle_retrieves_a_written_value_in_vk_state_layout() -> None:
    # Defect: [N,H,V,K] vs [K,V] state transposition, or a wrong delta-rule write.
    # With no decay (lower_bound=0), beta=1 and a zero state, one token writes
    # k v^T (unit k), so querying with q=k reads back scale * v.
    q, k, v, raw_g, a_log, dt_bias = _oracle_inputs(1)
    o, state = kda_chunk_prefill_reference(
        k,
        k,
        v,
        raw_g,
        torch.ones(1, 2),
        a_log,
        dt_bias,
        torch.zeros(1, 2, 4, 4),
        [0, 1],
        lower_bound=0.0,
    )
    k_unit = k / k.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(o, v * 4**-0.5, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        state[0], torch.einsum("hv,hk->hvk", v[0], k_unit[0]), rtol=1e-5, atol=1e-5
    )


def test_oracle_decays_state_per_key_channel_and_isolates_sequences() -> None:
    # Defect: per-head (scalar) decay, decay on the value axis, or state leaking
    # across cu_seqlens boundaries. With beta=0 nothing is written, so each
    # sequence's state is its initial state scaled by exp(sum of its gates).
    q, k, v, raw_g, a_log, dt_bias = _oracle_inputs(5)
    initial = torch.randn(2, 2, 4, 4, generator=torch.Generator().manual_seed(1))
    _, state = kda_chunk_prefill_reference(
        q,
        k,
        v,
        raw_g,
        torch.zeros(5, 2),
        a_log,
        dt_bias,
        initial,
        [0, 2, 5],
        lower_bound=-5.0,
    )
    gate = kda_safe_gate(raw_g, a_log, dt_bias, -5.0)
    for sequence, (start, end) in enumerate([(0, 2), (2, 5)]):
        decay = gate[start:end].sum(0).exp()  # [H, K]
        torch.testing.assert_close(state[sequence], initial[sequence] * decay[:, None, :])
