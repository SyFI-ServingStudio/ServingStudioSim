"""Focused CPU tests for ``sigmoid_gated_rms_norm``. No GPU and no DB."""

from __future__ import annotations

import inspect
from dataclasses import fields

import torch

from profiling.kernels.sigmoid_gated_rms_norm import SigmoidGatedRmsNormArgs
from profiling.runners.norm import sigmoid_gated_rms_norm_vllm_fla_triton as runner
from profiling.runners.norm.sigmoid_gated_rms_norm_reference import (
    sigmoid_gated_rms_norm_reference,
)


def test_args_fields_are_the_runner_kwargs() -> None:
    # Defect: schema/DB key order drifting from what the worker passes the runner.
    names = [f.name for f in fields(SigmoidGatedRmsNormArgs)]
    assert names == ["m", "hidden", "dtype"]
    signature = inspect.signature(runner.profile_sigmoid_gated_rms_norm_vllm_fla_triton)
    assert list(signature.parameters) == names


def test_oracle_gates_with_sigmoid_not_silu() -> None:
    # Defect: an oracle that applies gdn_gated_rms_norm's SiLU gate (g * sigmoid(g))
    # or its 1e-6 epsilon, which would pass a kernel computing the wrong norm.
    x = torch.tensor([[3.0, -4.0], [1e-3, 0.0]])
    g = torch.tensor([[0.0, 2.0], [-1.0, 1.0]])
    weight = torch.tensor([2.0, 0.5])
    rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
    expected = x * rstd * weight * torch.sigmoid(g)
    torch.testing.assert_close(sigmoid_gated_rms_norm_reference(x, g, weight), expected)
    silu_gated = x * rstd * weight * g * torch.sigmoid(g)
    assert not torch.allclose(expected, silu_gated)
