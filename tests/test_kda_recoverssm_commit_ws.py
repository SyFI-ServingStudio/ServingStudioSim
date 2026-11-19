"""CPU checks for the persistent warp-specialized commit runner (backend cuda_persistent_ws).

The kernel needs an SM10x GPU and the vLLM fork; these tests pin the argument
contract, the shapes it refuses, and that the vendored source ships with it.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest

from profiling.kernels.kda_recoverssm_commit import KdaRecoverSSMCommitArgs
from profiling.runners.attention import kda_recoverssm_commit_ws as runner

_ARGS = dict(
    batch_size=4,
    commit_len=4,
    num_heads=2,
    head_dim=128,
    spec_query_len=8,
    num_layers=2,
    dtype="bf16",
    state_dtype="bf16",
)


_PROFILES = [
    runner.profile_kda_recoverssm_commit_cuda_persistent_ws,
    runner.profile_kda_recoverssm_commit_cuda_persistent_ws_tf32_mma,
]


@pytest.mark.parametrize("profile", _PROFILES)
def test_args_fields_are_the_runner_kwargs(profile) -> None:
    # Defect: schema/DB key order drifting from what the worker passes the runner.
    signature = inspect.signature(profile)
    assert list(signature.parameters) == [f.name for f in fields(KdaRecoverSSMCommitArgs)]


@pytest.mark.parametrize(
    ("variant", "kernel"), [(runner.FFMA, "k_commit4"), (runner.TF32_MMA, "k_commit7")]
)
def test_vendored_source_is_present(variant, kernel: str) -> None:
    # Defect: the runner shipped without the kernel it builds, or a variant
    # pointing at the other variant's source.
    assert variant.source.is_file()
    assert kernel in variant.source.read_text()
    assert variant.byte_exact is (variant is runner.FFMA)


@pytest.mark.parametrize("profile", _PROFILES)
@pytest.mark.parametrize(
    "override",
    [{"state_dtype": "fp32"}, {"head_dim": 64}, {"spec_query_len": 9, "commit_len": 9}],
)
def test_rejects_shapes_the_kernel_cannot_fold(profile, override: dict) -> None:
    # Defect: timing a bf16-only, K=128, Q<=8 kernel on a shape it gets wrong.
    with pytest.raises(ValueError):
        profile(**{**_ARGS, **override})
