"""CPU tests for the ``tensorfold_pattn`` backend of ``flashinfer_attn_prefill``.

The GPU path (TensorFold's JIT CUDA kernel) is covered by the public profiling
smoke; these protect the parts that can silently go wrong without a GPU.
"""

from __future__ import annotations

import math

import pytest

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.attention import flashinfer_attn_prefill_tensorfold as runner
from profiling.runners.exceptions import ProfilerNotImplemented

torch = pytest.importorskip("torch")


def _naive(q, k_cache, v_cache, p0, scale):
    """Row-by-row softmax over keys [0, p0 + i], GQA head h -> h // group."""
    w, h, d = q.shape
    group = h // k_cache.shape[1]
    out = torch.empty(w, h, d, dtype=torch.float64)
    for i in range(w):
        for head in range(h):
            keys = k_cache[: p0 + i + 1, head // group].double()
            vals = v_cache[: p0 + i + 1, head // group].double()
            p = torch.softmax(keys @ q[i, head].double() * scale, dim=0)
            out[i, head] = p @ vals
    return out


def test_oracle_is_causal_over_the_prefix():
    # Catches an oracle whose mask ignores p0 or is off by one: it would then
    # accept a kernel that attends to the wrong keys.
    torch.manual_seed(0)
    p0, w, h, hk, d = 5, 7, 6, 2, 16
    q = torch.randn(w, h, d)
    k = torch.randn(p0 + w + 3, hk, d)  # spare rows past p0 + w must be ignored
    v = torch.randn(p0 + w + 3, hk, d)
    got = runner.reference(torch, q, k, v, p0, 1 / math.sqrt(d))
    torch.testing.assert_close(
        got.double(), _naive(q, k, v, p0, 1 / math.sqrt(d)), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("q_dtype", DType.FP8_E4M3), ("kv_dtype", DType.FP8_E4M3), ("head_dim", 96)],
)
def test_unsupported_shapes_are_refused_before_any_launch(field, value):
    # Catches a runner that would hand TensorFold an fp8 cache or a head_dim its
    # kernel does not build, and record a time for a call that cannot run.
    args = dict(
        num_qo_heads=24,
        num_kv_heads=4,
        head_dim=256,
        q_dtype=DType.BF16,
        kv_dtype=DType.BF16,
        o_dtype=DType.BF16,
        prefix_len=0,
        append_len=16,
    )
    args[field] = value
    with pytest.raises(ProfilerNotImplemented):
        runner.validate_args(**args)


def test_backend_runs_in_tensorfold_env():
    # Catches the row landing on the project venv, which has no TensorFold.
    spec = find_kernel_profiler_spec("flashinfer_attn_prefill", "tensorfold_pattn")
    assert spec.subprocess_env == "tensorfold_env"
    assert spec.supports.min_compute_capability == (8, 9)
