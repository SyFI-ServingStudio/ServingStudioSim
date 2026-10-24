"""CPU tests for the ``tensorfold_chain`` backend of ``gdn_chunk_delta_rule``.

The GPU path (TensorFold's JIT CUDA chain) is covered by the public profiling
smoke, whose correctness check runs this oracle; these protect the oracle and
the argument gate, which can go wrong silently without a GPU.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.runners.attention import gdn_chunk_delta_rule_tensorfold as runner
from profiling.runners.exceptions import ProfilerNotImplemented

torch = pytest.importorskip("torch")


def _naive(q, k, v, g, beta, state):
    """Gated delta rule one head and one row at a time, in float64."""
    rows, heads = v.shape[:2]
    group = heads // q.shape[1]
    s = state.double().clone()
    out = torch.empty(rows, heads, v.shape[2], dtype=torch.float64)
    for t in range(rows):
        for h in range(heads):
            kt, qt = k[t, h // group].double(), q[t, h // group].double()
            decayed = s[h] * g[t, h].double()
            error = v[t, h].double() - decayed @ kt
            s[h] = decayed + beta[t, h].double() * torch.outer(error, kt)
            out[t, h] = s[h] @ qt
    return out, s


def test_oracle_matches_a_per_head_delta_rule():
    # Catches an oracle with the wrong key-head grouping (h // group), the
    # update order (decay before the error term, output after the update) or
    # the (value, key) state layout: it would then accept a wrong kernel.
    torch.manual_seed(0)
    rows, hk, hv, d = 5, 2, 6, 128
    q = torch.randn(rows, hk, d) / d
    k = torch.nn.functional.normalize(torch.randn(rows, hk, d), dim=-1)
    v = torch.randn(rows, hv, d)
    g = torch.rand(rows, hv) * 0.5 + 0.5
    beta = torch.rand(rows, hv)
    state = 0.1 * torch.randn(hv, d, d)
    out, final = runner.reference(torch, q, k, v, g, beta, state)
    want_out, want_final = _naive(q, k, v, g, beta, state)
    torch.testing.assert_close(out.double(), want_out, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(final.double(), want_final, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("field", "value"),
    [("dtype", DType.FP16), ("key_head_dim", 64), ("value_head_dim", 256)],
)
def test_shapes_the_chain_does_not_build_are_refused(field, value):
    # Catches a runner that would record a time for a dtype or head dim that
    # gdn.cpp `prefill` rejects (it hard-codes bf16 values and 128-wide heads).
    args = dict(
        num_tokens=64,
        max_sequence_length=64,
        num_key_heads=16,
        num_heads=48,
        key_head_dim=128,
        value_head_dim=128,
        dtype=DType.BF16,
    )
    runner.validate_args(**args)
    with pytest.raises(ProfilerNotImplemented):
        runner.validate_args(**{**args, field: value})
