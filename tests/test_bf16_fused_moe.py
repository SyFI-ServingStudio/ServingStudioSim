from types import SimpleNamespace

import pytest

from profiling.runners.moe import bf16_fused_moe as runner


def _args(**overrides):
    return {
        "num_tokens": 4,
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_experts": 8,
        "num_local_experts": 4,
        "top_k": 2,
        "dtype": "bf16",
        "routing_method": "minimax2",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_numerator": 3,
        "routed_scaling_denominator": 2,
        "per_expert_batches": (2, 1, 1, 0, 1, 1, 1, 1),
        **overrides,
    }


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"dtype": "fp16"}, "dtype=bf16"),
        ({"routing_method": "deepseek_v3"}, "unsupported routing method"),
        ({"hidden_size": 192}, "divisible by 128"),
        ({"per_expert_batches": (2,) * 8}, "must sum"),
        ({"per_expert_batches": (5, 1, 1, 1, 0, 0, 0, 0)}, "one row per token"),
        ({"num_local_experts": 3}, "must divide"),
        ({"top_k": 33}, "top_k"),
        ({"n_group": 2}, "n_group=topk_group=1"),
        ({"routed_scaling_denominator": 0}, "must be positive"),
    ],
)
def test_invalid_shape_fails_before_loading_cuda(monkeypatch, overrides, message):
    def unexpected_runtime():
        pytest.fail("invalid shapes must not load a GPU runtime")

    monkeypatch.setattr(runner, "_load_runtime", unexpected_runtime)
    with pytest.raises(ValueError, match=message):
        runner.profile_bf16_fused_moe_sm100(**_args(**overrides))


def test_public_runner_times_only_the_production_callable(monkeypatch):
    events = []
    calls = []
    operands = runner._Operands("logits", "bias", "hidden", "w1", "w2")

    def production(**kwargs):
        events.append("launch")
        calls.append(kwargs)
        return "output"

    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: events.append("sync")))
    monkeypatch.setattr(runner, "_load_runtime", lambda: (torch, production, None))
    monkeypatch.setattr(runner, "_require_b200", lambda _: events.append("device"))
    monkeypatch.setattr(
        runner, "_check_small_correctness", lambda *args: events.append("correctness")
    )

    def prepare(torch, callable_, prepare_weights, args):
        events.append("prepare")
        return runner._Launch(callable_, operands, args)

    monkeypatch.setattr(runner, "_prepare_launch", prepare)
    monkeypatch.setattr(runner, "_validate_output", lambda *args: events.append("validate"))

    def time_call(fn, **kwargs):
        assert kwargs == {"kernel_name": None, "interval_union": True}
        events.append("cupti")
        fn()
        return 2.0

    def energy_call(fn, **kwargs):
        assert kwargs == {"per_iter_time_ms": 2.0}
        events.append("energy")
        fn()
        return 3.0

    monkeypatch.setattr(runner.Timer, "cupti", time_call)
    monkeypatch.setattr(runner.Energy, "perf", energy_call)
    metrics = runner.profile_bf16_fused_moe_sm100(**_args())
    assert events == [
        "device",
        "correctness",
        "prepare",
        "launch",
        "sync",
        "validate",
        "cupti",
        "launch",
        "energy",
        "launch",
    ]
    assert (
        calls
        == [
            {
                "routing_logits": "logits",
                "routing_bias": "bias",
                "hidden_states": "hidden",
                "gemm1_weights": "w1",
                "gemm2_weights": "w2",
                "num_experts": 8,
                "top_k": 2,
                "n_group": 1,
                "topk_group": 1,
                "intermediate_size": 128,
                "local_expert_offset": 0,
                "local_num_experts": 4,
                "routed_scaling_factor": 1.5,
                "routing_method_type": 7,
            }
        ]
        * 3
    )
    assert metrics.time_ms == 2.0
    assert metrics.energy_j == 3.0
    # Four local rows, three active experts; global routing and final output.
    assert metrics.memory_bandwidth_gbps == pytest.approx(
        (80 + 1024 + 3 * 98304 + 1024) / 0.002 / 1e9
    )
    assert metrics.tflops == pytest.approx(4 * 6 * 128 * 128 / 0.002 / 1e12)


def test_no_local_assignments_charge_only_router_and_output():
    args = runner._validate_args(**_args(per_expert_batches=(0, 0, 0, 0, 2, 2, 2, 2)))
    assert runner._logical_bytes(args) == 80 + 1024
