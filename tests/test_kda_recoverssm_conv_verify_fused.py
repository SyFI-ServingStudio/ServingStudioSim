"""CPU checks for the fused conv + verify runner (backend cuda_persistent_wy_mma).

The kernel needs an SM10x GPU and the vLLM fork; these tests pin the argument
contract, the shapes it refuses, and that the vendored source ships with it.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest

from profiling.kernels.kda_recoverssm_conv_verify import KdaRecoverSSMConvVerifyArgs
from profiling.runners.attention import kda_recoverssm_conv_verify_fused as runner

_ARGS = dict(
    batch_size=4,
    num_heads=2,
    head_dim=128,
    spec_query_len=8,
    dtype="bf16",
    conv_weight_dtype="fp32",
)
_PROFILE = runner.profile_kda_recoverssm_conv_verify_cuda_persistent_wy_mma


def test_args_fields_are_the_runner_kwargs() -> None:
    # Defect: schema/DB key order drifting from what the worker passes the runner.
    signature = inspect.signature(_PROFILE)
    assert list(signature.parameters) == [f.name for f in fields(KdaRecoverSSMConvVerifyArgs)]


@pytest.mark.parametrize("window", [8, 4])
def test_sources_are_present(window: int) -> None:
    # Defect: the runner shipped without the kernel it builds, or a window
    # mapped to a source that cannot build it.
    source, _, flags = runner._BUILDS[window]
    assert source.is_file()
    text = source.read_text()
    assert "k_fused" in text and f"L_({runner._RUNG}, {runner._MINB}, {runner._ABL}, 0)" in text
    if window != 8:
        assert f"-DSQ={window}" in flags and "#ifndef SQ" in text


@pytest.mark.parametrize(
    "override",
    [{"spec_query_len": 6}, {"conv_weight_dtype": "bf16"}, {"head_dim": 64}, {"dtype": "fp16"}],
)
def test_rejects_shapes_the_kernel_cannot_run(override: dict) -> None:
    # Defect: timing an 8-token, fp32-tap, K=128 kernel on a shape it gets wrong.
    with pytest.raises(ValueError):
        _PROFILE(**{**_ARGS, **override})
