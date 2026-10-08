"""Focused CPU tests for the ``kda_chunk_prefill`` FlashInfer backends (no GPU, no DB).

Each test names the defect it guards against. The GPU comparison of each
FlashInfer backend against the oracle runs inside the runner; its evidence
comes from the Slurm smoke.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.exec.env import ENV_REGISTRY, ProfileEnv
from profiling.kernels.kda_chunk_prefill import KIND, KdaChunkPrefillArgs
from profiling.runners.attention import kda_chunk_prefill_flashinfer as runner

_BACKENDS = (runner.TIRX, runner.CUTE_PERSISTENT)
_CAPTURE = dict(
    num_tokens=2048,
    max_sequence_length=2019,
    num_decode_sequences=29,
    num_heads=16,
    head_dim=128,
    dtype="bf16",
)


@pytest.mark.parametrize(
    "entry",
    [
        runner.profile_kda_chunk_prefill_flashinfer_tirx,
        runner.profile_kda_chunk_prefill_flashinfer_cute_persistent,
    ],
)
def test_entry_points_take_the_schema_fields_in_order(entry) -> None:
    # Defect: the worker passing spec kwargs that the entry point names differently.
    assert list(inspect.signature(entry).parameters) == [
        f.name for f in fields(KdaChunkPrefillArgs)
    ]


@pytest.mark.parametrize("backend", _BACKENDS, ids=lambda b: b.name)
def test_invoke_reproduces_the_vllm_flashinfer_call(backend) -> None:
    # Defect: timing pre-packed q/k/v (drops the three copy launches), passing a
    # pre-sigmoided fp32 beta or a precomputed gate (moves fused work out of the
    # call), int32 offsets (adds a conversion launch), or leaving FlashInfer to
    # choose its own backend.
    shape = runner.validate_args(
        backend,
        num_tokens=70,
        max_sequence_length=66,
        num_decode_sequences=4,
        num_heads=8,
        head_dim=128,
        dtype="bf16",
    )
    operands = runner.build_operands(torch, shape, device=torch.device("cpu"))
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return kwargs["output"], None

    runner.invoke(fake, backend, operands)
    for name in ("q", "k", "v"):
        packed = seen[name]
        strided = getattr(operands, name)
        assert not strided.is_contiguous() and packed.is_contiguous()
        assert packed.untyped_storage().data_ptr() != operands.qkv.untyped_storage().data_ptr()
        assert torch.equal(packed, strided)
    assert seen["g"].shape == (1, 70, 8, 128) and seen["g"].dtype is torch.bfloat16
    beta = seen["beta"]
    assert beta.shape == (1, 70, 8) and beta.dtype is torch.bfloat16
    assert bool((beta < 0).any())  # a logit, not a sigmoid output
    assert seen["beta_is_logit"] is True and seen["use_gate_in_kernel"] is True
    assert seen["use_qk_l2norm_in_kernel"] is True and seen["lower_bound"] == -5.0
    assert seen["A_log"].shape == (8,) and seen["dt_bias"].shape == (8 * 128,)
    assert seen["initial_state"] is operands.initial_state
    assert seen["initial_state"].dtype is torch.float32
    assert seen["output"] is operands.output and seen["output_final_state"] is False
    assert seen["cu_seqlens"].dtype is torch.int64
    assert seen["cu_seqlens"].tolist() == [0, 1, 2, 3, 4, 70]
    assert seen["scale"] == pytest.approx(128**-0.5)
    assert seen["backend"] == backend.flashinfer_backend


@pytest.mark.parametrize("backend", _BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize(
    "override",
    [
        {"head_dim": 64},  # both backends are D128-only
        {"num_heads": 12},  # heads are tiled in groups of eight
        {"num_heads": 4},
        {"num_decode_sequences": 2048},  # pure decode: not this path
        {"max_sequence_length": 1},  # query length 1 is always a decode
        {"dtype": "fp16"},
    ],
)
def test_rejects_shapes_the_backend_cannot_take(backend, override: dict) -> None:
    # Defect: launching a shape FlashInfer refuses, or profiling an unreachable path.
    with pytest.raises(ValueError):
        runner.validate_args(backend, **{**_CAPTURE, **override})


@pytest.mark.parametrize("num_heads", [8, 16, 64])
def test_accepts_the_grid_extremes(num_heads: int) -> None:
    # Defect: a limit check that refuses cells of the profiled grid: the most
    # sequences (64 decodes + 512 x 64-token prefills) and the longest prefill.
    for backend in _BACKENDS:
        for tokens, length, decodes in ((32832, 64, 64), (32832, 16384, 64)):
            shape = runner.validate_args(backend, tokens, length, decodes, num_heads, 128, "bf16")
            assert shape.num_tokens == tokens


@pytest.mark.parametrize("backend", _BACKENDS, ids=lambda b: b.name)
def test_rejects_32_bit_index_overflow(backend) -> None:
    # Defect: launching a call whose token or state index overflows int32,
    # which both backends refuse (a wide-H long prefill, or many short ones).
    with pytest.raises(ValueError, match="2\\*\\*31"):
        runner.validate_args(backend, 2**21, 2**21, 0, 8, 128, "bf16")
    with pytest.raises(ValueError, match="2\\*\\*31"):
        runner.validate_args(backend, 2**16, 2, 0, 64, 128, "bf16")


@pytest.mark.parametrize("backend", ["flashinfer_tirx", "flashinfer_cute_persistent"])
def test_registered_for_sm100_and_sm103_only_in_their_own_env(backend: str) -> None:
    # Defect: claiming the SM10x family (CC 10.7, 11.0) or Hopper, which both
    # FlashInfer backends refuse, or running in the project venv, whose
    # FlashInfer predates these backends.
    spec = find_kernel_profiler_spec(KIND, backend)
    supports = spec.supports
    assert supports.allows(DType.BF16) and not supports.allows(DType.FP16)
    assert supports.allows_compute_capability((10, 0))
    assert supports.allows_compute_capability((10, 3))
    for capability in ((9, 0), (10, 7), (11, 0), (12, 0)):
        assert not supports.allows_compute_capability(capability)
    assert spec.subprocess_env == "flashinfer_kda_env"
    env = ENV_REGISTRY["flashinfer_kda_env"]
    assert isinstance(env, ProfileEnv) and env.isolated_site_packages
    assert spec.row_provenance_ref is not None


def test_row_provenance_names_the_flashinfer_commit() -> None:
    # Defect: rows from a nightly carrying no source commit, so a later
    # FlashInfer build cannot be told apart in backend_version.
    note = runner.row_provenance_tirx(**_CAPTURE)
    assert "flashinfer-python" in note and "tirx-kernels" in note
    assert "flashinfer commit" in note
