"""CPU tests for the Lamport RS + short-conv + AG + residual + RMSNorm kind."""

from __future__ import annotations

import subprocess
import sys

import torch

from profiling.kernels.rs_sconv_ag_residual_rms_norm import RsSconvAgResidualRmsNormArgs
from profiling.runners.comm.rs_sconv_ag_residual_rms_norm_reference import (
    rs_sconv_ag_residual_rms_norm_reference,
)
from profiling.runners.comm.vllm_lamport_triton import conv_cache_layout, run_lamport_batch


def test_args_field_order_matches_rust_enumerate_and_runner_kwargs():
    # Rust's enumerate emits these keys in this order; a reorder breaks DB keys.
    assert list(RsSconvAgResidualRmsNormArgs.__dataclass_fields__) == [
        "num_gpus",
        "num_tokens",
        "hidden_dim",
        "conv_kernel_size",
        "has_shared",
        "dtype",
        "fabric",
    ]


def _manual(partials, shared, residual, weight, gamma, eps, history, positions, seq_idx):
    """Element loop over the documented math, with a dense per-request history."""
    world, tokens, hidden = partials.shape
    kernel = weight.shape[1]
    x = torch.zeros(tokens, hidden, dtype=torch.bfloat16)
    for t in range(tokens):
        acc = torch.zeros(hidden)
        for r in range(world):
            v = partials[r, t].float()
            if shared is not None:
                v = v + shared[r, t].float()
            acc = acc + v.to(torch.bfloat16).float()
        x[t] = acc.to(torch.bfloat16)
    # The production call inserts every token's x before any conv reads.
    history = history.clone()
    for t in range(tokens):
        history[int(seq_idx[t]), int(positions[t])] = x[t]
    normed = torch.zeros_like(x)
    residual_out = torch.zeros_like(x)
    for t in range(tokens):
        conv = torch.zeros(hidden)
        for i in range(kernel):
            p = int(positions[t]) - (kernel - 1) + i
            if p >= 0:
                conv = conv + history[int(seq_idx[t]), p].float() * weight[:, i].float()
        y = (conv + x[t].float()).to(torch.bfloat16)
        h = (residual[t].float() + y.float()).to(torch.bfloat16)
        residual_out[t] = h
        inv = torch.rsqrt(h.float().square().mean() + eps)
        normed[t] = (h.float() * inv * gamma.float()).to(torch.bfloat16)
    return normed, residual_out, x


def test_reference_matches_manual_math_with_same_batch_and_cached_taps():
    # Catches wrong rounding points, a lost sink partial, and taps that read the
    # stale cache instead of this call's inserts (or vice versa).
    gen = torch.Generator().manual_seed(5)
    world, hidden, kernel = 2, 8, 4

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)

    # Request 0: a prefill chunk at positions 1..3 (position 0 cached, -1/-2 empty).
    # Request 1: one decode token at position 9 (taps 6..8 cached).
    positions = torch.tensor([1, 2, 3, 9])
    seq_idx = torch.tensor([0, 0, 0, 1], dtype=torch.int32)
    tokens = positions.numel()
    partials = rnd(world, tokens, hidden, scale=0.5)
    shared = rnd(world, tokens, hidden, scale=0.5)
    residual = rnd(tokens, hidden)
    weight = rnd(hidden, kernel, scale=0.3)
    gamma = rnd(hidden, scale=0.1) + 1
    history = rnd(2, 12, hidden)
    prior = torch.stack(
        [
            torch.stack([history[int(s), max(int(p) - 3 + i, 0)] for i in range(kernel - 1)])
            for p, s in zip(positions, seq_idx, strict=True)
        ]
    )

    for shared_partial in (None, shared):
        actual = rs_sconv_ag_residual_rms_norm_reference(
            partials, shared_partial, residual, weight, gamma, 1e-6, prior, positions, seq_idx
        )
        expected = _manual(
            partials, shared_partial, residual, weight, gamma, 1e-6, history, positions, seq_idx
        )
        for got, want in zip(actual, expected, strict=True):
            assert got.dtype is torch.bfloat16
            torch.testing.assert_close(got, want, rtol=8e-3, atol=8e-3)


def test_inkling_small_tp4_cache_layout_matches_production_assertions():
    # sconv_swa_attn.py: ws = 4096/8, 2 heads per rank, D = next_pow2(2*128 + 2*512).
    attn = conv_cache_layout(4096, 4, has_shared=False)
    mlp = conv_cache_layout(4096, 4, has_shared=True)
    assert attn == {"ws": 512, "heads_per_rank": 2, "head_size": 2048, "off_s": 256}
    assert mlp["off_s"] == 768 and mlp["off_s"] + mlp["ws"] <= mlp["head_size"]


def test_unsupported_specs_fail_before_spawning_ranks():
    base = dict(
        num_gpus=4,
        num_tokens=8,
        hidden_dim=4096,
        conv_kernel_size=4,
        has_shared=False,
        dtype="bf16",
        fabric="nvlink",
    )
    for bad in ({"conv_kernel_size": 3}, {"dtype": "fp16"}, {"num_tokens": 16385}):
        (result,) = run_lamport_batch([{**base, **bad}], layout="decode")
        assert result.metrics is None and result.error
    (result,) = run_lamport_batch([{**base, "num_gpus": 3}], layout="decode")
    assert result.error and "TP" in result.error


def test_kernel_import_is_lazy():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels.rs_sconv_ag_residual_rms_norm; "
                "print('profiling.runners.comm.vllm_lamport_triton' in sys.modules, "
                "'torch' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False False"
