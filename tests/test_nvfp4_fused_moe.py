from __future__ import annotations

import pytest

from launcher.backends import dedup_roles, validate_backend_map
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import BackendSupport, known_backends, supported_backends
from profiling.kernels.nvfp4_fused_moe import KIND, Nvfp4FusedMoeArgs
from profiling.runners.moe.exact_topk import exact_topk_ids
from profiling.runners.moe.nvfp4_fused_moe import (
    ROUTING_METHODS,
    _forced_routing,
    _logical_bytes,
    _validate_args,
    tuning_label,
    vllm_tune_max_num_tokens,
)


def test_exact_topk_ids_realize_distinct_expert_counts() -> None:
    batches = (4, 3, 3, 2)
    ids = exact_topk_ids(num_tokens=6, top_k=2, per_expert_batches=batches)

    flattened = [expert for row in ids for expert in row]
    assert [flattened.count(expert) for expert in range(4)] == list(batches)
    assert all(len(row) == len(set(row)) == 2 for row in ids)


def test_exact_topk_ids_reject_impossible_expert_degree() -> None:
    with pytest.raises(ValueError, match="more than one row per token"):
        exact_topk_ids(num_tokens=2, top_k=2, per_expert_batches=(3, 1))


def _args() -> dict:
    # One EP rank of four: 32 local experts of 128, half of them idle.
    per_expert_batches = tuple(8 if expert % 2 == 0 else 0 for expert in range(128))
    return {
        "num_tokens": 64,
        "hidden_size": 6144,
        "intermediate_size": 1536,
        "num_experts": 128,
        "num_local_experts": 32,
        "top_k": 8,
        "routing_method": "minimax2",
        "per_expert_batches": per_expert_batches,
    }


def test_logical_bytes_follow_the_finalize_mode_of_the_dispatch() -> None:
    """The two dispatches do not produce the same output.

    vLLM finalizes to one row per input token; SGLang defers and writes one
    unfinalized row per local expert assignment. Billing both at the finalized
    size would misreport bandwidth by ``2 * hidden * (tokens - local_rows)`` on
    every deferred row, in whichever direction the rank's share happens to fall.
    """
    args = _args()
    local_rows = sum(args["per_expert_batches"][: args["num_local_experts"]])
    assert local_rows == 128  # 16 active local experts x 8 rows

    finalized = _logical_bytes(args, do_finalize=True)
    deferred = _logical_bytes(args, do_finalize=False)

    hidden = args["hidden_size"]
    assert deferred - finalized == 2 * hidden * (local_rows - args["num_tokens"])
    assert finalized != deferred


def test_precomputed_routing_bills_selected_ids_and_weights_not_logits() -> None:
    args = _args()
    routed = _logical_bytes(args, do_finalize=True, precomputed_routing=True)
    logits = _logical_bytes(args, do_finalize=True)
    tokens, experts, top_k = args["num_tokens"], args["num_experts"], args["top_k"]
    assert logits - routed == 2 * tokens * experts + 2 * experts - 8 * tokens * top_k


def test_routed_tuning_bound_covers_the_dp_gathered_batch() -> None:
    assert vllm_tune_max_num_tokens(1) == 8192
    assert vllm_tune_max_num_tokens(8192) == 8192
    assert vllm_tune_max_num_tokens(8193) == 16384
    assert vllm_tune_max_num_tokens(65536) == 65536


def test_logical_bytes_charge_weights_only_for_active_local_experts() -> None:
    """Weight traffic dominates at small batch, so idle experts must not count.

    Every expert this rank owns holds the same weights; charging all 32 when only
    16 receive a row would roughly double the reported bytes at decode shapes.
    """
    args = _args()
    dense = dict(args, per_expert_batches=tuple(4 for _ in range(128)))

    hidden, intermediate = args["hidden_size"], args["intermediate_size"]
    per_expert_weights = (
        intermediate * hidden
        + intermediate * hidden // 8
        + hidden * intermediate // 2
        + hidden * intermediate // 16
    )
    # Same 128 local rows either way, so the whole difference is weight traffic
    # plus the three FP32 per-expert scale vectors.
    assert sum(dense["per_expert_batches"][: dense["num_local_experts"]]) == 128
    extra = _logical_bytes(dense, do_finalize=True) - _logical_bytes(args, do_finalize=True)
    assert extra == 16 * (per_expert_weights + 3 * 4)


def test_nvfp4_dtype_round_trips_and_has_no_torch_dtype() -> None:
    # The wire literal is the profile.db `weight_format` value, unchanged.
    assert DType("nvfp4_e2m1") is DType.NVFP4_E2M1
    assert DType.from_value("nvfp4_e2m1") is DType.NVFP4_E2M1
    assert DType.from_value(DType.NVFP4_E2M1).value == "nvfp4_e2m1"
    assert DType.NVFP4_E2M1.size_bytes() == 0.5
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="nvfp4_e2m1 is not supported by the torch runner"):
        DType.NVFP4_E2M1.torch()


def test_weight_format_coerces_to_the_nvfp4_dtype() -> None:
    args = coerce_args(
        Nvfp4FusedMoeArgs,
        {
            **_args(),
            "input_dtype": "bf16",
            "weight_format": "nvfp4_e2m1",
            "group_size": 16,
            "routing_method": "minimax2",
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_numerator": 5,
            "routed_scaling_denominator": 2,
        },
    )
    assert args.weight_format is DType.NVFP4_E2M1
    assert args.input_dtype is DType.BF16


_FP8_BLOCK_BACKEND = "flashinfer_trtllm_fp8_block_sm100"


def test_backends_gate_on_the_weight_format_not_the_bf16_input() -> None:
    """The Rust config tags weight_format as the compute dtype, so the launcher
    asks for backends at nvfp4_e2m1 (or fp8_e4m3 for the FP8 block-scale
    backend). input_dtype (bf16) is only the activation before quantization."""
    nvfp4 = sorted(set(known_backends(KIND)) - {_FP8_BLOCK_BACKEND})
    assert sorted(supported_backends(KIND, DType.NVFP4_E2M1, None, "NVIDIA B200")) == nvfp4
    assert supported_backends(KIND, DType.FP8_E4M3, None, "NVIDIA B200") == [_FP8_BLOCK_BACKEND]
    assert supported_backends(KIND, DType.BF16, None, "NVIDIA B200") == []
    assert supported_backends(KIND, DType.NVFP4_E2M1, None, "NVIDIA H200") == []
    # A backend declared at bf16, as these were before, fails the gate.
    bf16_only = BackendSupport(compute=frozenset({DType.BF16}), sm_targets=frozenset({"sm_100f"}))
    assert not bf16_only.allows(DType.NVFP4_E2M1, None, "NVIDIA B200")


def test_launcher_validation_accepts_nvfp4_roles_and_rejects_bf16_ones() -> None:
    record = {
        "pool": "main",
        "name": "unified.moe.experts",
        "kind": KIND,
        "gpu": "NVIDIA B200",
        "compute_dtype": "nvfp4_e2m1",
        "kv_dtype": None,
        "config": {},
        "backends": ["flashinfer_trtllm_sm100"],
    }
    (role,) = dedup_roles([record])
    assert role.compute is DType.NVFP4_E2M1
    assert set(role.options) == set(known_backends(KIND)) - {_FP8_BLOCK_BACKEND}
    backend_map = {"main": {"unified.moe.experts": ["flashinfer_trtllm_sm100"]}}
    assert validate_backend_map(backend_map, [role]) == []
    (bf16_role,) = dedup_roles([{**record, "compute_dtype": "bf16"}])
    errors = validate_backend_map(backend_map, [bf16_role])
    assert any("unsupported for nvfp4_fused_moe at dtype=bf16" in e for e in errors)


def test_runner_accepts_the_nvfp4_weight_format_as_dtype_or_wire_string() -> None:
    spec = {
        **_args(),
        "input_dtype": DType.BF16,
        "group_size": 16,
        "routing_method": "minimax2",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_numerator": 5,
        "routed_scaling_denominator": 2,
    }
    for weight_format in (DType.NVFP4_E2M1, "nvfp4_e2m1"):
        assert _validate_args(**spec, weight_format=weight_format)["weight_format"] == "nvfp4_e2m1"
    with pytest.raises(ValueError, match="nvfp4_e2m1 weights"):
        _validate_args(**spec, weight_format="int4")


def _glm_ep4_spec(**overrides: object) -> dict:
    # A 288-expert, top-8 layer on one EP4 rank (72 local experts), 16 tokens,
    # half of this rank's experts idle.
    batches = [0] * 288
    for expert in range(0, 64, 2):
        batches[expert] = 4
    spec = {
        "num_tokens": 16,
        "hidden_size": 4096,
        "intermediate_size": 2048,
        "num_experts": 288,
        "num_local_experts": 72,
        "top_k": 8,
        "input_dtype": "bf16",
        "weight_format": "nvfp4_e2m1",
        "group_size": 16,
        "routing_method": "deepseek_v3",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_numerator": 5,
        "routed_scaling_denominator": 2,
        "per_expert_batches": tuple(batches),
    }
    spec.update(overrides)
    return spec


def test_routing_methods_map_to_flashinfer_routing_method_types() -> None:
    # flashinfer.fused_moe.RoutingMethodType: DeepSeekV3 = 2, MiniMax2 = 7.
    assert ROUTING_METHODS == {"minimax2": 7, "deepseek_v3": 2}


def test_vllm_backends_accept_ungrouped_deepseek_v3_routing() -> None:
    args = _validate_args(**_glm_ep4_spec())
    assert args["routing_method"] == "deepseek_v3"


@pytest.mark.parametrize(
    ("override", "stack", "message"),
    [
        ({"n_group": 8, "topk_group": 4}, "vllm", "ungrouped"),
        ({}, "sglang", "SGLang"),
        ({"routing_method": "renormalize"}, "vllm", "routing method"),
    ],
)
def test_validation_rejects_routing_the_measured_call_would_not_honor(
    override: dict, stack: str, message: str
) -> None:
    """Grouped routing would ignore the forced histogram, and the SGLang
    backend is only measured for MiniMax2, so these keys would mislabel rows."""
    with pytest.raises(ValueError, match=message):
        _validate_args(stack=stack, **_glm_ep4_spec(**override))


def test_forced_deepseek_v3_routing_realizes_the_histogram_with_fp32_inputs() -> None:
    """The monolithic call routes from logits, so DeepSeekV3 selection over the
    forced FP32 logits and zero FP32 bias must pick exactly the row's experts,
    at the dtype vLLM passes for an FP32 router."""
    torch = pytest.importorskip("torch")
    from profiling.runners.moe.fp8_block_fused_moe_reference import deepseek_v3_routing

    args = _validate_args(**_glm_ep4_spec())
    ids = exact_topk_ids(
        num_tokens=args["num_tokens"], top_k=8, per_expert_batches=args["per_expert_batches"]
    )
    logits, bias = _forced_routing(torch, ids, args, device="cpu")
    assert logits.dtype is torch.float32 and bias.dtype is torch.float32

    chosen, weights = deepseek_v3_routing(torch, logits, bias, top_k=8, routed_scaling_factor=2.5)
    counts = torch.bincount(chosen.flatten(), minlength=288).tolist()
    assert counts == list(args["per_expert_batches"])
    torch.testing.assert_close(weights.sum(dim=-1), torch.full((16,), 2.5))


def test_forced_minimax2_routing_keeps_bf16_logits_and_bias() -> None:
    torch = pytest.importorskip("torch")
    args = _validate_args(**_glm_ep4_spec(routing_method="minimax2"))
    ids = exact_topk_ids(
        num_tokens=args["num_tokens"], top_k=8, per_expert_batches=args["per_expert_batches"]
    )
    logits, bias = _forced_routing(torch, ids, args, device="cpu")
    assert logits.dtype is torch.bfloat16 and bias.dtype is torch.bfloat16
    chosen = torch.topk(logits.float(), 8).indices.tolist()
    assert [set(row) for row in chosen] == [set(row) for row in ids]


def test_logical_bytes_charge_routing_at_the_router_element_size() -> None:
    """DeepSeekV3 rows read FP32 logits and bias, MiniMax2 rows BF16; the routed
    backend reads the same pre-selected IDs and weights for both."""
    deepseek = _validate_args(**_glm_ep4_spec())
    minimax = _validate_args(**_glm_ep4_spec(routing_method="minimax2"))
    tokens, experts = 16, 288
    assert _logical_bytes(deepseek, do_finalize=True) - _logical_bytes(
        minimax, do_finalize=True
    ) == 2 * (tokens * experts + experts)
    assert _logical_bytes(deepseek, do_finalize=True, precomputed_routing=True) == _logical_bytes(
        minimax, do_finalize=True, precomputed_routing=True
    )


def test_tuning_label_separates_configurations_flashinfer_keys_alike() -> None:
    """FlashInfer's persisted key is the bucketed input shapes alone, which are
    identical for EP4, EP8, MoE-TP4 and MoE-TP8 ranks of one model; each must
    tune into its own file or later rows replay the first one's tactic."""
    configs = [(72, 2048), (36, 2048), (288, 512), (288, 256)]
    labels = {
        tuning_label(
            _glm_ep4_spec(num_local_experts=local, intermediate_size=inter), "vllm", routed
        )
        for local, inter in configs
        for routed in (False, True)
    }
    assert len(labels) == 2 * len(configs)
