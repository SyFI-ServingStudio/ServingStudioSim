"""CPU tests for the ``rms_norm`` ``tensorfold_torch`` backend (TensorFold's DFlash2 drafter norm)."""

from __future__ import annotations

import pytest

from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.rms_norm import KIND, RmsNormArgs
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.norm import rms_norm_tensorfold_torch as runner

torch = pytest.importorskip("torch")


def test_backend_runs_in_tensorfold_env_on_the_kind_schema():
    # Catches a registration that would run in the default image (another
    # Torch than the one TensorFold serves with) or on another schema.
    spec = find_kernel_profiler_spec(KIND, "tensorfold_torch")
    assert spec.subprocess_env == "tensorfold_env"
    assert spec.args_schema is RmsNormArgs


def test_oracle_matches_torch_rms_norm():
    # Catches an oracle that normalizes by the mean instead of the mean square
    # or skips the weight: it would then pass a wrong kernel.
    x = torch.randn(5, 64).to(torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn(64)).to(torch.bfloat16)
    want = torch.nn.functional.rms_norm(x.float(), (64,), w.float(), 1e-6)
    got = runner.reference(torch, x, w, 1e-6)
    torch.testing.assert_close(got.float(), want, rtol=1e-2, atol=1e-2)


def test_only_bf16_rows_are_measured():
    # The drafter's activations are bf16; an fp16 row is a call it never makes.
    with pytest.raises(ProfilerNotImplemented):
        runner.validate_args(16, 5120, "fp16")
    with pytest.raises(ValueError):
        runner.validate_args(0, 5120, "bf16")
