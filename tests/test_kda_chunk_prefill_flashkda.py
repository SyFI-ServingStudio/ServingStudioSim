"""Focused CPU tests for the ``kda_chunk_prefill`` ``flashkda`` backend.

Each test names the defect it guards against. FlashKDA itself only runs on
SM90+ GPUs, so its comparison against the per-token oracle runs inside the
runner, and its evidence comes from the Slurm smoke.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch

from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.kda_chunk_prefill import KdaChunkPrefillArgs
from profiling.runners.attention import kda_chunk_prefill_flashkda as runner

_CAPTURE = dict(
    num_tokens=2048,
    max_sequence_length=2019,
    num_decode_sequences=29,
    num_heads=16,
    head_dim=128,
    dtype="bf16",
)


def test_runner_kwargs_are_the_args_fields() -> None:
    # Defect: the worker passing schema fields the flashkda entry does not take.
    names = [f.name for f in fields(KdaChunkPrefillArgs)]
    assert list(inspect.signature(runner.profile_kda_chunk_prefill_flashkda).parameters) == names


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"head_dim": 64}, "head_dim=128"),  # flash_kda.cpp: D == 128 only
        ({"head_dim": 256}, "head_dim=128"),
        ({"dtype": "fp16"}, "flashkda requires dtype=bf16"),
        ({"max_sequence_length": 1, "num_tokens": 30}, "unreachable"),
        ({"num_decode_sequences": 2048}, "at least one full-length prefill"),
    ],
)
def test_rejects_shapes_flashkda_or_the_prefill_path_cannot_take(
    override: dict, match: str
) -> None:
    # Defect: profiling a head dim FlashKDA refuses at launch, or a topology the
    # prefill branch never sees, instead of failing before the GPU work.
    with pytest.raises(ValueError, match=match):
        runner.validate_args(**{**_CAPTURE, **override})


def test_capability_gate_matches_vllms_flashkda_dispatch() -> None:
    # Defect: offering FlashKDA on a GPU vLLM would run the Triton path on
    # (SM80, SM89), or refusing an SM10x/SM12x part vLLM sends to FlashKDA.
    support = find_kernel_profiler_spec("kda_chunk_prefill", "flashkda").supports
    for capability in [(9, 0), (10, 0), (10, 3), (12, 0), (12, 1)]:
        assert support.allows_compute_capability(capability), capability
    for capability in [(8, 0), (8, 9), (11, 0)]:
        assert not support.allows_compute_capability(capability), capability


def test_invoke_reproduces_vllms_flashkda_prefill_call() -> None:
    # Defect: timing pre-contiguous q/k/v (skips the three copy launches vLLM
    # pays), passing sigmoided or fp32 beta (FlashKDA sigmoids raw bf16 logits
    # in-kernel), or reordering the positional fwd arguments.
    shape = runner.validate_args(
        num_tokens=70,
        max_sequence_length=66,
        num_decode_sequences=4,
        num_heads=2,
        head_dim=128,
        dtype="bf16",
    )
    sizes: list = []

    def workspace_size(tokens: int, heads: int, sequences: int) -> int:
        sizes.append((tokens, heads, sequences))
        return 64

    operands = runner.build_operands(torch, shape, workspace_size, device=torch.device("cpu"))
    assert sizes == [(70, 2, 5)]
    seen: list = []
    runner.invoke(lambda *args: seen.extend(args), operands, shape.head_dim)
    q, k, v, g, beta, scale, out, workspace, a_log, dt_bias, lower_bound = seen[:11]
    h0, ht, cu, cs, co = seen[11:]
    assert len(seen) == 16

    for copy, view in ((q, operands.q), (k, operands.k), (v, operands.v)):
        assert not view.is_contiguous() and view.stride(1) == 3 * 2 * 128
        assert copy.is_contiguous() and copy.data_ptr() != view.data_ptr()
        assert torch.equal(copy, view)
    assert g is operands.g and g.is_contiguous()
    assert beta.shape == (1, 70, 2) and beta.dtype is torch.bfloat16
    assert not beta.is_contiguous() and beta.data_ptr() != 0
    assert beta.stride(1) == 3 * 2 * 128 + 2 + 2 * 128  # merged in_proj row stride
    assert bool((beta < 0).any())  # raw logits, not post-sigmoid
    assert scale == pytest.approx(128**-0.5) and lower_bound == -5.0
    assert out is operands.out and out.shape == (1, 70, 2, 128) and out.dtype is torch.bfloat16
    assert workspace is operands.workspace and workspace.dtype is torch.uint8
    assert a_log.shape == (2,) and a_log.dtype is torch.float32
    assert dt_bias.shape == (2, 128) and dt_bias.dtype is torch.float32
    assert h0 is operands.initial_state and h0.shape == (5, 2, 128, 128)
    assert h0.dtype is torch.float32 and ht is operands.final_state
    assert bool(h0[:4].abs().sum(dim=(1, 2, 3)).gt(0).all()) and bool(h0[4:].eq(0).all())
    assert cu.dtype is torch.int32 and cu.tolist() == [0, 1, 2, 3, 4, 70]
    assert cs is None and co is None
