"""gated_activation: the NVFP4 fused-MoE GEMM1 activation arg (SwiGLU default, Kimi SiTU)."""

from __future__ import annotations

import pytest

from profiling.db.args import canonical_args
from profiling.kernels.nvfp4_fused_moe import Nvfp4FusedMoeArgs
from profiling.runners import tactic_labels
from profiling.runners.moe import gated_activation as ga
from profiling.runners.moe.nvfp4_fused_moe import _validate_args

KIMI = "situ:beta=4,linear_beta=25"


def _spec(**extra):
    base = dict(
        num_tokens=4,
        hidden_size=3584,
        intermediate_size=3072,
        num_experts=8,
        num_local_experts=4,
        top_k=2,
        input_dtype="bf16",
        weight_format="nvfp4_e2m1",
        group_size=16,
        routing_method="deepseek_v3",
        n_group=1,
        topk_group=1,
        routed_scaling_numerator=1,
        routed_scaling_denominator=1,
        per_expert_batches=(2, 2, 1, 1, 1, 1, 0, 0),
    )
    return {**base, **extra}


def test_default_is_the_old_swiglu_call():
    act = ga.parse(None)
    assert act.is_default and act.activation_type == 3 and act.label_suffix() == ""
    assert ga.parse("swiglu") == act


def test_kimi_situ_parses_to_vllm_constants():
    act = ga.parse(KIMI)
    assert (act.activation_type, act.beta, act.linear_beta) == (10, 4.0, 25.0)
    assert ga.situ(4.0, 25.0) == KIMI
    assert act.label_suffix() == ".situ-b4-lb25"


@pytest.mark.parametrize(
    "bad", ["situ:beta=4.0,linear_beta=25", "situ", "situ:beta=0,linear_beta=25", "gelu"]
)
def test_rejects_non_canonical_or_unsupported(bad):
    with pytest.raises(ValueError):
        ga.parse(bad)


def test_absent_and_explicit_swiglu_share_one_key():
    explicit = canonical_args(Nvfp4FusedMoeArgs(**_spec(gated_activation="swiglu")))
    assert explicit == Nvfp4FusedMoeArgs(**_spec())
    assert Nvfp4FusedMoeArgs(**_spec(gated_activation=KIMI)) != explicit


def test_validate_and_label():
    assert _validate_args(**_spec())["gated_activation"] is None
    assert _validate_args(**_spec(gated_activation="swiglu"))["gated_activation"] is None
    args = _validate_args(**_spec(gated_activation=KIMI))
    assert args["gated_activation"] == KIMI
    base = tactic_labels.nvfp4_fused_moe_vllm_routed(**_spec())
    assert tactic_labels.nvfp4_fused_moe_vllm_routed(**args) == base.replace(
        ".i3072", ".i3072.situ-b4-lb25"
    )
    with pytest.raises(ValueError):
        _validate_args(stack="sglang", **_spec(routing_method="minimax2", gated_activation=KIMI))
