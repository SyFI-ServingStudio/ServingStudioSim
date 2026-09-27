"""CPU-side registry, schema, and semantic checks for Kimi-K3 profiling."""

from dataclasses import fields

import pytest
import torch

from profiling.db import DType, find_kernel_profiler_spec, known_backends
from profiling.db.batch import coerce_args


def test_k3_kinds_and_backends_are_registered_without_runner_imports():
    assert set(known_backends("kda_recurrent_decode")) == {"torch", "sglang_triton"}
    assert set(known_backends("kda_fused_decode")) == {"sglang_fused"}
    assert set(known_backends("mla_decode_attention")) == {
        "sglang_cutedsl_mla",
        "sglang_trtllm_mla",
        "sglang_triton",
    }
    assert set(known_backends("mxfp4_fused_moe")) == {
        "sglang_trtllm_mxfp4",
        "sglang_trtllm_mxfp4_prefill",
    }
    assert set(known_backends("causal_conv1d_prefill")) == {"sglang_triton"}
    assert set(known_backends("kda_chunk_prefill")) == {"sglang_triton"}
    assert set(known_backends("mla_prefill_attention")) == {"sglang_trtllm_mla"}
    assert set(known_backends("mla_prefix_gather")) == {"sglang_triton"}
    assert set(known_backends("mla_merge_state")) == {"sglang_triton"}
    assert set(known_backends("k3_situ_and_mul_prefill")) == {"sglang_k3"}
    assert set(known_backends("k3_add3_prefill")) == {"sglang_k3"}
    assert set(known_backends("k3_attn_res_prefill")) == {"sglang_k3"}
    for kind, backend in (
        ("single_gemm", "sglang_k3_raw_bf16"),
        ("gemm_fp32_output", "sglang_k3_fp32_auto"),
        ("mxfp4_fused_moe", "sglang_trtllm_mxfp4_prefill"),
    ):
        assert find_kernel_profiler_spec(kind, backend).subprocess_env == "sglang_k3_env"
    for kind in (
        "kda_recurrent_decode",
        "kda_fused_decode",
        "mla_decode_attention",
        "mxfp4_fused_moe",
    ):
        for backend in known_backends(kind):
            spec = find_kernel_profiler_spec(kind, backend)
            assert spec.table_name == kind
            expected_env = (
                None if kind == "kda_recurrent_decode" and backend == "torch" else "sglang_k3_env"
            )
            assert spec.subprocess_env == expected_env


def test_k3_argument_field_order_is_the_db_contract():
    from profiling.kernels.kda_recurrent_decode import KdaRecurrentDecodeArgs
    from profiling.kernels.mla_decode_attention import MlaDecodeAttentionArgs
    from profiling.kernels.mla_prefill_attention import MlaPrefillAttentionArgs
    from profiling.kernels.mxfp4_fused_moe import Mxfp4FusedMoeArgs

    assert [field.name for field in fields(KdaRecurrentDecodeArgs)] == [
        "batch_size",
        "num_heads",
        "head_k_dim",
        "head_v_dim",
        "dtype",
        "state_dtype",
        "lower_bound",
    ]
    assert [field.name for field in fields(MlaDecodeAttentionArgs)] == [
        "num_heads",
        "kv_lora_rank",
        "rope_dim",
        "q_dtype",
        "kv_dtype",
        "page_size",
        "batch_size",
        "kv_len",
    ]
    assert [field.name for field in fields(Mxfp4FusedMoeArgs)][-4:] == [
        "routed_scaling_factor",
        "gemm1_alpha",
        "gemm1_clamp_limit",
        "per_expert_batches",
    ]
    assert [field.name for field in fields(MlaPrefillAttentionArgs)][-5:] == [
        "batch_size",
        "q_len",
        "kv_len",
        "prefix_len",
        "num_prefix_chunks",
    ]


def test_k3_specs_coerce_representative_json_shapes():
    kda = coerce_args(
        find_kernel_profiler_spec("kda_recurrent_decode", "sglang_triton").args_schema,
        {
            "batch_size": 32,
            "num_heads": 12,
            "head_k_dim": 128,
            "head_v_dim": 128,
            "dtype": "bf16",
            "state_dtype": "fp32",
            "lower_bound": -5.0,
        },
    )
    assert kda.dtype is DType.BF16 and kda.lower_bound == -5.0
    mla = coerce_args(
        find_kernel_profiler_spec("mla_decode_attention", "sglang_cutedsl_mla").args_schema,
        {
            "num_heads": 12,
            "kv_lora_rank": 512,
            "rope_dim": 64,
            "q_dtype": "bf16",
            "kv_dtype": "bf16",
            "page_size": 64,
            "batch_size": 128,
            "kv_len": 8192,
        },
    )
    assert mla.page_size == 64
    mixed_mla = coerce_args(
        find_kernel_profiler_spec("mla_decode_attention", "sglang_trtllm_mla").args_schema,
        {
            "num_heads": 12,
            "kv_lora_rank": 512,
            "rope_dim": 64,
            "q_dtype": "bf16",
            "kv_dtype": "fp8_e4m3",
            "page_size": 64,
            "batch_size": 128,
            "kv_len": 8192,
        },
    )
    assert mixed_mla.q_dtype is DType.BF16
    assert mixed_mla.kv_dtype is DType.FP8_E4M3
    moe = coerce_args(
        find_kernel_profiler_spec("mxfp4_fused_moe", "sglang_trtllm_mxfp4").args_schema,
        {
            "num_tokens": 16,
            "hidden_size": 3584,
            "intermediate_size": 3072,
            "num_experts": 896,
            "num_local_experts": 112,
            "top_k": 16,
            "input_dtype": "bf16",
            "weight_format": "mxfp4_e2m1_ue8m0",
            "group_size": 32,
            "routing_method": "deepseek_v3_sigmoid",
            "activation": "situ",
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "gemm1_alpha": 4.0,
            "gemm1_clamp_limit": 25.0,
            "per_expert_batches": [16] * 16 + [0] * 880,
        },
    )
    assert moe.per_expert_batches[:16] == (16,) * 16


def test_mla_prefill_chunk_contract_covers_a_short_remainder():
    from profiling.runners.attention.kimi_k3_prefill import _prefix_chunk_lengths

    assert _prefix_chunk_lengths(131_072, 131_072, 1) == (131_072,)
    assert _prefix_chunk_lengths(245_760, 131_072, 2) == (131_072, 114_688)
    assert _prefix_chunk_lengths(0, 16_384, 0) == (16_384,)
    with pytest.raises(ValueError, match="num_prefix_chunks"):
        _prefix_chunk_lengths(245_760, 131_072, 1)


@pytest.mark.parametrize(
    ("num_experts", "top_k"),
    [(112, 2), (112, 16), (896, 16)],
)
def test_mxfp4_runner_accepts_rank_local_and_global_routing_widths(num_experts, top_k):
    from profiling.runners.moe.mxfp4_fused_moe import _validate_args

    counts = [top_k] * 16 + [0] * (num_experts - 16)
    args = _validate_args(
        num_tokens=16,
        hidden_size=3584,
        intermediate_size=3072,
        num_experts=num_experts,
        num_local_experts=112,
        top_k=top_k,
        input_dtype="bf16",
        weight_format="mxfp4_e2m1_ue8m0",
        group_size=32,
        routing_method="deepseek_v3_sigmoid",
        activation="situ",
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        gemm1_alpha=4.0,
        gemm1_clamp_limit=25.0,
        per_expert_batches=counts,
    )
    assert len(args["per_expert_batches"]) == num_experts


def test_kda_reference_is_cpu_testable_and_updates_state():
    from profiling.runners.attention.kda_recurrent_decode import (
        kda_recurrent_decode_reference,
    )

    batch, heads, key_dim, value_dim = 2, 3, 4, 5
    generator = torch.Generator().manual_seed(7)
    q = torch.randn((1, batch, heads, key_dim), generator=generator, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn((1, batch, heads, value_dim), generator=generator, dtype=torch.bfloat16)
    a = torch.randn((batch, heads, key_dim), generator=generator, dtype=torch.bfloat16)
    b = torch.randn((batch, heads), generator=generator, dtype=torch.bfloat16)
    A_log = torch.zeros(heads, dtype=torch.float32)
    dt_bias = torch.zeros(heads * key_dim, dtype=torch.float32)
    state = torch.randn((batch + 1, heads, value_dim, key_dim), generator=generator)
    before = state.clone()
    output = kda_recurrent_decode_reference(q, k, v, a, b, A_log, dt_bias, state)
    assert output.shape == (1, batch, heads, value_dim)
    assert torch.isfinite(output).all()
    assert not torch.equal(state[1:], before[1:])
    assert torch.equal(state[0], before[0])


def test_kda_production_operands_match_sglang_decode_layout():
    from profiling.runners.attention.kda_recurrent_decode import (
        _build_operands,
        _validate_args,
    )

    args = _validate_args(
        32,
        12,
        128,
        128,
        "bf16",
        "fp32",
        -5.0,
        production=True,
    )
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    assert operands.q.shape == (1, 32, 12, 128)
    assert operands.a.shape == (32, 12 * 128)
    assert operands.b.shape == (1, 32, 12)
    assert operands.A_log.dtype is torch.float32
    assert operands.dt_bias.dtype is torch.float32


def test_kda_fused_operands_match_sglang_argument_contract():
    from profiling.runners.attention.kda_fused_decode import (
        _build_operands,
        _invoke,
        _validate_args,
    )

    args = _validate_args(2, 12, 128, 128, "bf16", "fp32", -5.0)
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    expected_shapes = {
        "mixed_qkv": (2, 4608),
        "a": (2, 1536),
        "b": (2, 12),
        "conv_states": (3, 3, 4608),
        "w_q_t": (4, 1536),
        "w_k_t": (4, 1536),
        "w_v_t": (4, 1536),
        "conv_bias": (4608,),
        "A_log": (12,),
        "dt_bias": (1536,),
        "onorm_g": (2, 1536),
        "onorm_weight": (128,),
        "ssm_states": (3, 12, 128, 128),
        "cache_indices": (2,),
    }
    assert {name: tuple(value.shape) for name, value in operands.items()} == expected_shapes
    for name in ("w_q_t", "w_k_t", "w_v_t", "conv_bias", "A_log", "dt_bias", "onorm_weight"):
        assert operands[name].dtype is torch.float32
    for name in ("mixed_qkv", "a", "b", "conv_states", "onorm_g"):
        assert operands[name].dtype is torch.bfloat16
    assert operands["ssm_states"].dtype is torch.float32
    assert operands["cache_indices"].dtype is torch.int32

    captured: dict[str, object] = {}

    def fake_kernel(*positional: object, **keyword: object) -> None:
        captured["positional"] = positional
        captured["keyword"] = keyword

    _invoke(fake_kernel, operands, args)
    positional = captured["positional"]
    assert isinstance(positional, tuple)
    assert all(
        actual is operands[name]
        for actual, name in zip(
            positional,
            (
                "mixed_qkv",
                "a",
                "b",
                "conv_states",
                "w_q_t",
                "w_k_t",
                "w_v_t",
                "conv_bias",
                "A_log",
                "dt_bias",
                "onorm_g",
                "onorm_weight",
                "ssm_states",
                "cache_indices",
            ),
            strict=True,
        )
    )
    assert captured["keyword"] == {
        "scale": 128**-0.5,
        "onorm_eps": 1e-6,
        "lower_bound": -5.0,
    }


def test_mla_reference_handles_paged_latent_cache_on_cpu():
    from profiling.runners.attention.mla_decode_attention import mla_decode_attention_reference

    query = torch.randn((2, 1, 3, 8), dtype=torch.bfloat16)
    cache = torch.randn((4, 1, 4, 8), dtype=torch.bfloat16)
    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_lens = torch.tensor([5, 3], dtype=torch.int32)
    output = mla_decode_attention_reference(
        query,
        cache,
        block_tables,
        seq_lens,
        kv_lora_rank=6,
        rope_dim=2,
    )
    assert output.shape == (2, 3, 6)
    assert output.dtype is torch.bfloat16
    assert torch.isfinite(output).all()


def test_k3_shape_gates_are_runner_validation_not_registry_fallbacks():
    from profiling.runners.attention.gdn_causal_conv_decode_sglang_triton import (
        _validate_args as validate_conv,
    )
    from profiling.runners.attention.gdn_gated_rms_norm_sglang_triton import (
        _validate_args as validate_norm,
    )
    from profiling.runners.attention.mla_decode_attention import _validate_args as validate_mla

    with pytest.raises(ValueError, match="page_size=64"):
        validate_mla(12, 512, 64, "bf16", "bf16", 16, 1, 8192, backend="sglang_cutedsl_mla")
    with pytest.raises(ValueError, match="64 < num_heads < 128"):
        validate_mla(96, 512, 64, "bf16", "bf16", 64, 1, 8192, backend="sglang_trtllm_mla")
    assert (
        validate_mla(96, 512, 64, "bf16", "bf16", 64, 1, 8192, backend="sglang_triton").num_heads
        == 96
    )
    assert (
        validate_mla(12, 512, 64, "fp8_e4m3", "bf16", 64, 1, 8192, backend="sglang_triton").q_dtype
        is DType.FP8_E4M3
    )
    with pytest.raises(ValueError, match="4608 or 36864"):
        validate_conv(1, 8192, 4, "bf16", "bf16")
    with pytest.raises(ValueError, match="hidden=128"):
        validate_norm(1, 256, "bf16")


@pytest.mark.parametrize(
    "backend",
    ["sglang_cutedsl_mla", "sglang_trtllm_mla", "sglang_triton"],
)
def test_mla_runner_pads_short_page_tables_for_every_backend(backend):
    from profiling.runners.attention.mla_decode_attention import (
        _padded_kv_len,
        _pages_per_request,
        _validate_args,
    )

    args = _validate_args(12, 512, 64, "bf16", "bf16", 64, 1, 64, backend=backend)
    assert _padded_kv_len(args.kv_len) == 128
    assert _pages_per_request(args) == 2


def test_mla_runner_rejects_kv_allocations_beyond_the_profile_pool():
    from profiling.runners.attention.mla_decode_attention import (
        _build_operands,
        _validate_args,
    )
    from profiling.runners.exceptions import ProfilerNotImplemented

    args = _validate_args(
        12,
        512,
        64,
        "bf16",
        "fp8_e4m3",
        64,
        128,
        1_048_576,
        backend="sglang_cutedsl_mla",
    )
    with pytest.raises(ProfilerNotImplemented, match="48 GiB"):
        _build_operands(None, args)
