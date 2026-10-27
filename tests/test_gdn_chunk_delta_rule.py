"""Focused CPU tests for ``gdn_chunk_delta_rule`` (no GPU, no DB).

Each test names the defect it guards against. The GPU comparison of FLA's
Triton chain against the oracle runs inside the ``fla_triton`` runner on every
row; its evidence comes from the public-CLI smoke.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch

from profiling.db.args import DType
from profiling.kernels.gdn_chunk_delta_rule import GdnChunkDeltaRuleArgs
from profiling.runners.attention import gdn_chunk_delta_rule_fla_triton as fla
from profiling.runners.attention import gdn_chunk_delta_rule_flashinfer as flashinfer
from profiling.runners.attention.gdn_chunk_delta_rule_reference import (
    gdn_chunk_delta_rule_reference,
)

# Qwen3.8-27B GDN layer at TP1: 16 key heads, 48 value heads, 128/128 dims.
_QWEN = dict(
    num_tokens=8192,
    max_sequence_length=8192,
    num_key_heads=16,
    num_heads=48,
    key_head_dim=128,
    value_head_dim=128,
    dtype="bf16",
)


@pytest.mark.parametrize(
    "runner",
    [
        fla.profile_gdn_chunk_delta_rule_fla_triton,
        flashinfer.profile_gdn_chunk_delta_rule_flashinfer,
    ],
)
def test_both_backends_take_exactly_the_kind_args(runner) -> None:
    # Defect: a backend growing a private field, which would split one kind's
    # DB key between backends and stop the arch switching backends freely.
    names = [f.name for f in fields(GdnChunkDeltaRuleArgs)]
    assert list(inspect.signature(runner).parameters) == names


@pytest.mark.parametrize(
    "override",
    [
        {"key_head_dim": 257},  # chunk_gated_delta_rule_fwd_h asserts K <= 256
        {"dtype": "fp16"},
        {"num_heads": 40},  # not a multiple of 16 key heads
        {"max_sequence_length": 0},
        {"num_tokens": 100, "max_sequence_length": 200},
        {"num_heads": True},
    ],
)
def test_fla_rejects_shapes_outside_the_verified_path(override: dict) -> None:
    # Defect: silently profiling a shape the FLA chain cannot run or vLLM never issues.
    with pytest.raises(ValueError):
        fla.validate_args(**{**_QWEN, **override})


def test_fla_autotune_anchor_is_vllms_warmup_shape_at_the_spec_heads() -> None:
    # Defect: tuning at the row's own shape (grid-order dependence) or at a shape
    # production never tunes at; vLLM tunes once at T = FLA_CHUNK_SIZE.
    row = fla.validate_args(**{**_QWEN, "num_tokens": 300, "max_sequence_length": 200})
    anchor = fla.anchor_shape(row)
    assert (anchor.num_tokens, anchor.max_sequence_length) == (64, 64)
    assert (anchor.num_key_heads, anchor.num_heads, anchor.key_head_dim) == (16, 48, 128)
    assert anchor.dtype is DType.BF16


def test_fla_guard_keeps_two_sequences_and_a_partial_chunk() -> None:
    # Defect: a correctness witness that only covers one full sequence, so the
    # varlen boundary and the partial tail chunk go unchecked.
    guard = fla.guard_shape(fla.validate_args(**{**_QWEN, "num_tokens": 16384}))
    assert fla.sequence_boundaries(guard) == (0, 1024, 2047)
    long_single = fla.guard_shape(fla.validate_args(**_QWEN))
    assert fla.sequence_boundaries(long_single) == (0, 1024, 2047)
    single = fla.validate_args(**{**_QWEN, "num_tokens": 500, "max_sequence_length": 500})
    assert fla.guard_shape(single) == single
    small = fla.validate_args(**{**_QWEN, "num_tokens": 300, "max_sequence_length": 200})
    assert fla.guard_shape(small) == small


def _fake_chunk_metadata():
    calls = []

    def indices(cu_seqlens, chunk_size):
        calls.append(("indices", cu_seqlens.device.type, chunk_size))
        return torch.zeros(1, 2, dtype=cu_seqlens.dtype)

    def offsets(cu_seqlens, chunk_size):
        calls.append(("offsets", cu_seqlens.device.type, chunk_size))
        return torch.zeros(1, dtype=cu_seqlens.dtype)

    return (indices, offsets), calls


def test_fla_invoke_reproduces_the_production_prefill_call() -> None:
    # Defect: timing in-kernel L2 norm, an exponentiated gate, a missing final
    # state, or chunk metadata derived inside the timed call (vLLM's metadata
    # builder precomputes it on the host once per step).
    args = fla.validate_args(
        num_tokens=150, max_sequence_length=100, num_key_heads=2, num_heads=6,
        key_head_dim=32, value_head_dim=16, dtype="bf16",
    )
    metadata, calls = _fake_chunk_metadata()
    operands = fla.build_operands(torch, args, device=torch.device("cpu"), chunk_metadata=metadata)
    assert calls == [("indices", "cpu", 64), ("offsets", "cpu", 64)]
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return "o", "ht"

    assert fla.invoke(fake, operands) == ("o", "ht")
    for name, heads, dim in (("q", 2, 32), ("k", 2, 32), ("v", 6, 16)):
        tensor = seen[name]
        assert tensor.shape == (1, 150, heads, dim) and tensor.dtype is torch.bfloat16
        assert tensor.is_contiguous()
    for name in ("q", "k"):
        norms = seen[name].float().norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-2, rtol=0)
    g, beta = seen["g"], seen["beta"]
    assert g.shape == beta.shape == (1, 150, 6)
    assert g.dtype is beta.dtype is torch.float32
    assert bool((g < 0).all()), "FLA takes the log-space decay, not exp(g)"
    assert bool(((beta > 0) & (beta < 1)).all())
    state = seen["initial_state"]
    assert state.shape == (2, 6, 16, 32) and state.dtype is torch.float32
    assert bool(state.eq(0).all())
    assert seen["cu_seqlens"].dtype is torch.int32
    assert seen["cu_seqlens"].tolist() == [0, 100, 150]
    assert seen["chunk_indices"] is operands.chunk_indices
    assert seen["chunk_offsets"] is operands.chunk_offsets
    assert seen["use_qk_l2norm_in_kernel"] is False
    assert seen["output_final_state"] is True
    assert "core_attn_out" not in seen and "scale" not in seen


def _oracle_inputs(tokens: int, key_heads: int = 1, heads: int = 2, dim: int = 4):
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(tokens, key_heads, dim, generator=generator)
    k = torch.randn(tokens, key_heads, dim, generator=generator)
    k = torch.nn.functional.normalize(k, dim=-1)
    v = torch.randn(tokens, heads, dim, generator=generator)
    g = torch.nn.functional.logsigmoid(torch.randn(tokens, heads, generator=generator))
    return q, k, v, g


def test_oracle_reads_back_a_written_value_through_grouped_heads_in_vk_layout() -> None:
    # Defect: value head h reading the wrong key head, or the [N,H,V,K] state
    # transposed. No decay, beta=1 and a zero state: one token writes k v^T, so
    # querying with q=k (unit) reads back scale * v on every value head.
    q, k, v, _ = _oracle_inputs(1, key_heads=1, heads=2)
    o, state = gdn_chunk_delta_rule_reference(
        k, k, v, torch.zeros(1, 2), torch.ones(1, 2), torch.zeros(1, 2, 4, 4), [0, 1],
        g_is_log=True,
    )
    torch.testing.assert_close(o, v * 4**-0.5, rtol=1e-5, atol=1e-5)
    expected = torch.einsum("hv,k->hvk", v[0], k[0, 0])
    torch.testing.assert_close(state[0], expected, rtol=1e-5, atol=1e-5)


def test_oracle_gate_conventions_agree_and_decay_is_per_head_per_sequence() -> None:
    # Defect: the two backends' gate forms (log for FLA, exp for FlashInfer)
    # diverging in the oracle, a decay applied per channel instead of per head,
    # or state leaking across cu_seqlens boundaries.
    q, k, v, g = _oracle_inputs(5)
    initial = torch.randn(2, 2, 4, 4, generator=torch.Generator().manual_seed(1))
    beta = torch.rand(5, 2, generator=torch.Generator().manual_seed(2))
    log_form = gdn_chunk_delta_rule_reference(q, k, v, g, beta, initial, [0, 2, 5], g_is_log=True)
    exp_form = gdn_chunk_delta_rule_reference(
        q, k, v, g.exp(), beta, initial, [0, 2, 5], g_is_log=False
    )
    for log_part, exp_part in zip(log_form, exp_form, strict=True):
        torch.testing.assert_close(log_part, exp_part)
    _, state = gdn_chunk_delta_rule_reference(
        q, k, v, g, torch.zeros(5, 2), initial, [0, 2, 5], g_is_log=True
    )
    for sequence, (start, end) in enumerate([(0, 2), (2, 5)]):
        decay = g[start:end].sum(0).exp()  # [H]
        torch.testing.assert_close(state[sequence], initial[sequence] * decay[:, None, None])
