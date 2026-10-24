"""CPU tests for ``gdn_state_replay`` (TensorFold ``tensorfold_cuda`` backend).

The GPU path (TensorFold's JIT CUDA replay) is covered by the public profiling
smoke, whose correctness check runs this oracle; these protect the oracle, the
row split and the argument gate, which can go wrong silently without a GPU.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.runners.attention import gdn_state_replay_tensorfold as runner
from profiling.runners.attention.gdn_state_replay_reference import gdn_state_replay_reference
from profiling.runners.exceptions import ProfilerNotImplemented

torch = pytest.importorskip("torch")


def _naive(k, v, g, beta, state, path):
    """Gated delta rule one head and one row at a time, in float64."""
    heads = v.shape[1]
    group = heads // k.shape[1]
    s = state.double().clone()
    for t in path:
        for h in range(heads):
            kt = k[t, h // group].double()
            decayed = s[h] * g[t, h].double()
            error = v[t, h].double() - decayed @ kt
            s[h] = decayed + beta[t, h].double() * torch.outer(error, kt)
    return s


def test_oracle_replays_only_the_path_rows_in_order():
    # Catches an oracle that walks the whole window instead of the accepted
    # rows, uses the wrong key-head grouping, or applies the decay after the
    # error term: it would then accept a wrong kernel.
    torch.manual_seed(0)
    rows, hk, hv, d = 7, 2, 6, 128
    k = torch.nn.functional.normalize(torch.randn(rows, hk, d), dim=-1)
    v = torch.randn(rows, hv, d)
    g = torch.rand(rows, hv) * 0.5 + 0.5
    beta = torch.rand(rows, hv)
    state = 0.1 * torch.randn(hv, d, d)
    path = [0, 2, 3, 6]
    got = gdn_state_replay_reference(torch, k, v, g, beta, state, path)
    torch.testing.assert_close(
        got.double(), _naive(k, v, g, beta, state, path), rtol=1e-4, atol=1e-5
    )
    assert not torch.equal(
        got, gdn_state_replay_reference(torch, k, v, g, beta, state, range(rows))
    )


def test_rows_split_evenly_and_keep_the_total():
    # Catches a split that drops rows or leaves a stream empty (the kernel
    # would then time fewer rows than the key says).
    assert runner.even_split(17, 4) == [5, 4, 4, 4]
    assert runner.even_split(16, 16) == [1] * 16
    assert sum(runner.even_split(250, 16)) == 250


_ARGS = dict(
    num_layers=48,
    batch_size=4,
    num_qk_heads=16,
    num_value_heads=48,
    key_head_dim=128,
    value_head_dim=128,
    dtype=DType.BF16,
    state_dtype=DType.FP32,
    num_tokens=40,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dtype", DType.FP16),
        ("state_dtype", DType.BF16),
        ("key_head_dim", 64),
        ("value_head_dim", 256),
    ],
)
def test_shapes_the_replay_does_not_build_are_refused(field, value):
    # Catches a runner that would record a time for a dtype or head dim that
    # gdn.cpp replay / gdn.py replay_table reject (bf16 rows, fp32 128x128 states).
    runner.validate_args(**_ARGS)
    with pytest.raises(ProfilerNotImplemented):
        runner.validate_args(**{**_ARGS, field: value})


@pytest.mark.parametrize("num_tokens", [3, 4 * 128 + 1])
def test_streams_need_one_to_128_rows(num_tokens):
    # Catches a key with an empty stream (gdn._commit refuses empty paths) or a
    # window past attention.MAX_NODES.
    with pytest.raises(ValueError):
        runner.validate_args(**{**_ARGS, "num_tokens": num_tokens})
