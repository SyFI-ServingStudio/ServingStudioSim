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


def test_args_fields_are_the_runner_kwargs() -> None:
    # Defect: schema/DB key order drifting from what the worker passes the runner.
    signature = inspect.signature(runner.profile_kda_recoverssm_commit_cuda_persistent_ws)
    assert list(signature.parameters) == [f.name for f in fields(KdaRecoverSSMCommitArgs)]


def test_vendored_source_is_present() -> None:
    # Defect: the runner shipped without the kernel it builds.
    assert runner._SOURCE.is_file()
    assert "k_commit4" in runner._SOURCE.read_text()


@pytest.mark.parametrize(
    "override",
    [{"state_dtype": "fp32"}, {"head_dim": 64}, {"spec_query_len": 9, "commit_len": 9}],
)
def test_rejects_shapes_the_kernel_cannot_fold(override: dict) -> None:
    # Defect: timing a bf16-only, K=128, Q<=8 kernel on a shape it gets wrong.
    with pytest.raises(ValueError):
        runner.profile_kda_recoverssm_commit_cuda_persistent_ws(**{**_ARGS, **override})
