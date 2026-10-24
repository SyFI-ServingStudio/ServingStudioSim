"""CPU tests for ``sliding_block_attention`` (TensorFold ``tensorfold_triton`` backend).

The GPU path is covered by the public profiling smoke, whose correctness check
runs this reference; these protect the reference mask and the argument gate.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.runners.attention import sliding_block_attention_tensorfold as runner
from profiling.runners.attention.sliding_block_attention_reference import (
    sliding_block_attention_reference,
)
from profiling.runners.exceptions import ProfilerNotImplemented

torch = pytest.importorskip("torch")


def _naive(q, kb, vb, kc, vc, window, scale, causal):
    """One head and one row at a time, float64, keys listed explicitly."""
    heads, length, dim = q.shape
    group = heads // kb.shape[0]
    n = kc.shape[1]
    out = torch.zeros(length, heads * dim, dtype=torch.float64)
    for h in range(heads):
        g = h // group
        for r in range(length):
            ks = [kc[g, j] for j in range(n) if n + r - j < window + 1]
            vs = [vc[g, j] for j in range(n) if n + r - j < window + 1]
            last = r + 1 if causal else length
            ks += [kb[g, j] for j in range(last)]
            vs += [vb[g, j] for j in range(last)]
            k, v = torch.stack(ks).double(), torch.stack(vs).double()
            p = torch.softmax(k @ q[h, r].double() * scale, dim=0)
            out[r, h * dim : (h + 1) * dim] = p @ v
    return out


@pytest.mark.parametrize("causal", [False, True])
def test_reference_masks_the_window_and_the_block(causal):
    # Catches a reference that lets the first rows see context keys the window
    # drops, mixes up the GQA grouping, or applies causality to the context.
    torch.manual_seed(0)
    q = torch.randn(4, 3, 8)
    kb, vb = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    kc, vc = torch.randn(2, 6, 8), torch.randn(2, 6, 8)
    got = sliding_block_attention_reference(
        torch, q, kb, vb, kc, vc, window=5, scale=0.3, causal=causal
    )
    want = _naive(q, kb, vb, kc, vc, 5, 0.3, causal)
    torch.testing.assert_close(got.double(), want, rtol=1e-5, atol=1e-6)


_ARGS = dict(
    num_qo_heads=32,
    num_kv_heads=8,
    head_dim=128,
    dtype=DType.BF16,
    block_rows=16,
    window=2047,
    causal=False,
    batch_size=8,
    context_len=2047,
)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("dtype", DType.FP16, ProfilerNotImplemented),
        ("head_dim", 96, ProfilerNotImplemented),
        ("context_len", 0, ValueError),
        ("context_len", 2048, ValueError),
        ("num_qo_heads", 30, ValueError),
        ("causal", 0, ValueError),
    ],
)
def test_calls_the_drafter_never_makes_are_refused(field, value, error):
    # Catches rows for an empty context (launch_blocks skips such streams), a
    # context past the window, a broken GQA group, or a non-bool causal flag.
    runner.validate_args(**_ARGS)
    with pytest.raises(error):
        runner.validate_args(**{**_ARGS, field: value})
