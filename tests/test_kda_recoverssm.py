"""Focused CPU tests for ``kda_recoverssm_verify`` / ``kda_recoverssm_commit``.

No GPU and no DB. Each test names the defect it guards against. The GPU
comparison of the vLLM-fork Triton ops against these oracles runs inside the
runners, and its evidence comes from the Slurm smoke.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch

from profiling.kernels.kda_recoverssm_commit import KdaRecoverSSMCommitArgs
from profiling.kernels.kda_recoverssm_verify import KdaRecoverSSMVerifyArgs
from profiling.runners.attention import kda_recoverssm_commit_tilelang as tilelang_runner
from profiling.runners.attention import kda_recoverssm_flashinfer as flashinfer_runner
from profiling.runners.attention import kda_recoverssm_vllm_triton as runner
from profiling.runners.attention.kda_recoverssm_reference import (
    compact_conv_state_reference,
    kda_recoverssm_commit_reference,
    kda_recoverssm_verify_reference,
)
from profiling.runners.attention.kda_recurrent_decode_reference import (
    kda_recurrent_decode_reference,
)

_VERIFY = dict(
    batch_size=4, num_heads=8, head_dim=128, spec_query_len=8, dtype="bf16", state_dtype="fp32"
)
_COMMIT = dict(
    batch_size=4,
    commit_len=8,
    num_heads=8,
    head_dim=128,
    spec_query_len=8,
    num_layers=2,
    dtype="bf16",
    state_dtype="fp32",
)


def test_args_fields_are_the_runner_kwargs() -> None:
    # Defect: schema/DB key order drifting from what the worker passes the runner.
    verify = [f.name for f in fields(KdaRecoverSSMVerifyArgs)]
    assert verify == [
        "batch_size",
        "num_heads",
        "head_dim",
        "spec_query_len",
        "dtype",
        "state_dtype",
    ]
    signature = inspect.signature(runner.profile_kda_recoverssm_verify_vllm_triton)
    assert list(signature.parameters) == verify
    signature = inspect.signature(flashinfer_runner.profile_kda_recoverssm_verify_flashinfer_cute)
    assert list(signature.parameters) == verify
    commit = [f.name for f in fields(KdaRecoverSSMCommitArgs)]
    assert commit == [
        "batch_size",
        "commit_len",
        "num_heads",
        "head_dim",
        "spec_query_len",
        "num_layers",
        "dtype",
        "state_dtype",
    ]
    signature = inspect.signature(runner.profile_kda_recoverssm_commit_vllm_triton)
    assert list(signature.parameters) == commit
    signature = inspect.signature(tilelang_runner.profile_kda_recoverssm_commit_tilelang)
    assert list(signature.parameters) == commit


@pytest.mark.parametrize(
    "override",
    [
        {"dtype": "fp16"},
        {"spec_query_len": 1},
        {"batch_size": 0},
        {"num_heads": True},
        {"state_dtype": "fp16"},
    ],
)
def test_verify_rejects_shapes_outside_the_production_path(override: dict) -> None:
    # Defect: profiling a window with no drafts or a dtype the kg record cannot hold.
    with pytest.raises(ValueError):
        runner.validate_verify_args(**{**_VERIFY, **override})


@pytest.mark.parametrize(
    "override", [{"state_dtype": "fp32"}, {"head_dim": 64}, {"spec_query_len": 17}]
)
def test_flashinfer_verify_rejects_what_the_wy_kernel_cannot_take(override: dict) -> None:
    # Defect: a flashinfer_cute row for a pool dtype, head width or window the
    # WY kernel rejects (or silently routes elsewhere) at launch.
    flashinfer_runner.validate_args(**{**_VERIFY, "state_dtype": "bf16"})
    with pytest.raises(ValueError):
        flashinfer_runner.validate_args(**{**_VERIFY, "state_dtype": "bf16", **override})


@pytest.mark.parametrize(
    "override", [{"commit_len": 0}, {"commit_len": 9}, {"num_layers": 0}, {"dtype": "fp16"}]
)
def test_commit_rejects_unreachable_lengths(override: dict) -> None:
    # Defect: timing a commit_len the plan kernel would clamp (0 skips every
    # program; > window reads past the records).
    with pytest.raises(ValueError):
        runner.validate_commit_args(**{**_COMMIT, **override})


@pytest.mark.parametrize("head_dim", [64, 192])
def test_tilelang_commit_rejects_heads_its_fold_tile_cannot_cover(head_dim: int) -> None:
    # Defect: a tilelang row for a head width the whole-row fold tile would
    # under-cover or read past, timed before any launch fails.
    with pytest.raises(ValueError):
        tilelang_runner.profile_kda_recoverssm_commit_tilelang(**{**_COMMIT, "head_dim": head_dim})


def _window(batch: int, spec: int, heads: int = 2, dim: int = 8, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)

    def randn(*size: int) -> torch.Tensor:
        return torch.randn(*size, generator=generator)

    tokens = batch * spec
    q, k, v, raw_g = (randn(tokens, heads, dim) for _ in range(4))
    raw_beta = randn(tokens, heads)
    a_log = torch.tensor([0.5, 3.0]).log()
    dt_bias = randn(heads * dim) * 0.1
    pool = randn(batch + 2, heads, dim, dim)
    slots = torch.tensor([batch + 1 - i for i in range(batch)], dtype=torch.int32)
    return q, k, v, raw_g, raw_beta, a_log, dt_bias, pool, slots


def _decode_steps(q, k, v, raw_g, raw_beta, a_log, dt_bias, pool, slots, spec, steps):
    """Run the independent plain-decode oracle token by token."""
    batch = slots.shape[0]
    view = lambda x: x.view(batch, spec, *x.shape[1:])  # noqa: E731
    outputs = []
    for t in range(steps):
        o, pool = kda_recurrent_decode_reference(
            view(q)[:, t],
            view(k)[:, t],
            view(v)[:, t],
            view(raw_g)[:, t],
            view(raw_beta)[:, t],
            a_log,
            dt_bias,
            pool,
            slots,
            lower_bound=-5.0,
        )
        outputs.append(o)
    return torch.stack(outputs, dim=1).reshape(batch * steps, *q.shape[1:]), pool


def test_verify_oracle_is_the_decode_recurrence_without_writing_state() -> None:
    # Defect: a verify oracle that drifts from the KDA step (gate axis, beta
    # sigmoid, q scale) or that advances the checkpoint it must only read.
    spec = 4
    inputs = _window(batch=3, spec=spec)
    q, k, v, raw_g, raw_beta, a_log, dt_bias, pool, slots = inputs
    before = pool.clone()
    out, correction, kg = kda_recoverssm_verify_reference(
        q, k, v, raw_g, raw_beta, a_log, dt_bias, pool, slots, spec, lower_bound=-5.0
    )
    expected_out, _ = _decode_steps(*inputs, spec, spec)
    torch.testing.assert_close(out, expected_out.float(), rtol=1e-5, atol=1e-5)
    assert torch.equal(pool, before)
    assert correction.shape == (3, 2, spec, 8) and kg.shape == (3, 2, spec, 16)
    # The kg record is the raw key then the raw gate logits, per position.
    assert torch.equal(kg[1, :, 2, :8], k.view(3, spec, 2, 8)[1, 2])
    assert torch.equal(kg[1, :, 2, 8:], raw_g.view(3, spec, 2, 8)[1, 2])


@pytest.mark.parametrize("commit_len", [1, 3, 4])
def test_commit_oracle_reaches_the_state_after_the_accepted_prefix(commit_len: int) -> None:
    # Defect: folding records in the wrong order, decaying the value axis, or
    # folding the whole window instead of the accepted prefix.
    spec = 4
    inputs = _window(batch=3, spec=spec, seed=1)
    q, k, v, raw_g, raw_beta, a_log, dt_bias, pool, slots = inputs
    _, correction, kg = kda_recoverssm_verify_reference(
        q, k, v, raw_g, raw_beta, a_log, dt_bias, pool, slots, spec, lower_bound=-5.0
    )
    committed = kda_recoverssm_commit_reference(
        pool[slots.long()],
        correction,
        kg,
        a_log,
        dt_bias,
        torch.full((3,), commit_len, dtype=torch.int32),
        lower_bound=-5.0,
    )
    _, expected_pool = _decode_steps(*inputs, spec, commit_len)
    torch.testing.assert_close(committed, expected_pool[slots.long()], rtol=1e-5, atol=1e-5)


def test_conv_compaction_keeps_the_history_ending_at_the_accepted_token() -> None:
    # Defect: an off-by-one that keeps the window ending one token early or late.
    conv = torch.arange(2 * 3 * 10, dtype=torch.float32).view(2, 3, 10)
    out = compact_conv_state_reference(conv, torch.tensor([1, 8]), history=3)
    assert torch.equal(out[0, :, :3], conv[0, :, 0:3])
    assert torch.equal(out[1, :, :3], conv[1, :, 7:10])
    assert torch.equal(out[:, :, 3:], conv[:, :, 3:])


def test_verify_operands_follow_the_kimi_k3_layout() -> None:
    # Defect: timing contiguous q/k/v copies the production call never makes,
    # a contiguous beta, or records outside the layer's cache page.
    shape = runner.validate_verify_args(**{**_VERIFY, "num_heads": 2})
    operands = runner.build_verify_operands(torch, shape, device=torch.device("cpu"))
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return kwargs["out"]

    runner.invoke_verify(fake, operands, shape.spec_query_len)
    tokens, width = 32, 3 * 256
    for name in ("q", "k", "v"):
        view = seen[name]
        assert view.shape == (1, tokens, 2, 128) and view.stride(1) == width
        assert view.data_ptr() >= operands.conv_out.data_ptr()
    assert seen["raw_beta"].shape == (1, tokens, 2) and not seen["raw_beta"].is_contiguous()
    assert seen["raw_g"].is_contiguous() and seen["raw_g"].dtype is torch.bfloat16
    assert seen["A_log"].shape == (2,) and seen["dt_bias"].shape == (256,)
    assert seen["lower_bound"] == -5.0 and seen["spec_query_len"] == 8
    assert seen["query_start_loc"].tolist() == [0, 8, 16, 24, 32]
    assert sorted(seen["state_indices"].tolist()) == [1, 2, 3, 4]
    page = operands.page
    assert seen["checkpoint_state"] is page.checkpoint
    assert page.checkpoint.shape == (5, 2, 128, 128) and page.checkpoint.dtype is torch.float32
    assert page.correction.shape == (5, 2, 8, 128) and page.correction.dtype is torch.float32
    assert page.kg.shape == (5, 2, 8, 256) and page.kg.dtype is torch.bfloat16
    assert page.conv.shape == (5, 10, width)
    page_bytes = page.raw.numel() // 5
    for view in (page.conv, page.checkpoint, page.correction, page.kg):
        assert view.untyped_storage().data_ptr() == page.raw.untyped_storage().data_ptr()
        assert view.stride(0) * view.element_size() == page_bytes


def test_commit_context_gets_every_layer_and_no_align_metadata() -> None:
    # Defect: committing one layer when the call spans a cache group, or timing
    # the prefix-caching align path production does not take here.
    shape = runner.validate_commit_args(**{**_COMMIT, "num_heads": 2, "commit_len": 3})
    created: dict = {}
    calls: list = []

    class FakeContext:
        @classmethod
        def create(cls, layers, *, spec_query_len, max_num_reqs):
            created.update(layers=layers, spec=spec_query_len, max_reqs=max_num_reqs)
            return cls()

        def commit(self, *args, **kwargs):
            calls.append((args, kwargs))

    operands = runner.build_commit_operands(torch, FakeContext, shape, device=torch.device("cpu"))
    runner.invoke_commit(operands)
    assert len(created["layers"]) == 2 and created["spec"] == 8 and created["max_reqs"] == 4
    layer = created["layers"][0]
    assert [t.shape for t in layer.kv_cache] == [
        (5, 10, 768),
        (5, 2, 128, 128),
        (5, 2, 8, 128),
        (5, 2, 8, 256),
    ]
    assert layer.gate_lower_bound == -5.0 and layer.local_num_heads == 2
    (accepted, slots, qsl), kwargs = calls[0]
    assert accepted.tolist() == [3, 3, 3, 3]
    assert qsl.tolist() == [0, 8, 16, 24, 32] and 0 not in slots.tolist()
    assert kwargs["block_table"] is None and kwargs["request_indices"] is None
