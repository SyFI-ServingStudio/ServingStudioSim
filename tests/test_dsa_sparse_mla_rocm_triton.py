"""Focused CPU tests for the MI300X rope-free BF16 sparse-MLA Triton backends.

No GPU, no DB. The GPU correctness check (call vs rope-free Torch reference) and
the Triton-path assertion run inside the runner on the ROCm host; their evidence
comes from the Slurm step. Each test names the defect it guards against.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
import torch

import profiling.runners.attention._rocm_triton_mla_sparse_common as common
from profiling.db.args import DType
from profiling.kernels import dsa_sparse_mla_attention as decode_kind
from profiling.kernels import dsa_sparse_mla_prefill as prefill_kind
from profiling.runners.attention import dsa_sparse_mla_attention_rocm_triton as decode
from profiling.runners.attention import dsa_sparse_mla_prefill_rocm_triton as prefill
from profiling.runners.exceptions import ProfilerNotImplemented


def _decode_spec(**overrides: object) -> dict:
    spec = {
        "num_queries": 32,
        "num_cache_tokens": 4096,
        "num_heads": 16,
        "num_kv_heads": 1,
        "selected_k": 2048,
        "latent_dim": 512,
        "rope_dim": 0,
        "value_dim": 512,
        "softmax_scale": 0.0625,
        "q_dtype": "bf16",
        "cache_dtype": "bf16",
        "index_dtype": "int32",
        "output_dtype": "bf16",
        "valid_counts": "u:2048x32",
        "index_distribution": "unique_scattered_pages",
        "cache_layout": "token_major_mqa_bf16_latent",
    }
    spec.update(overrides)
    return spec


def _prefill_spec(**overrides: object) -> dict:
    spec = {
        "query_context_pairs": ((8, 2048),),
        "num_heads": 16,
        "num_kv_heads": 1,
        "selected_k": 2048,
        "latent_dim": 512,
        "rope_dim": 0,
        "value_dim": 512,
        "softmax_scale": 0.0625,
        "q_dtype": "bf16",
        "cache_dtype": "bf16",
        "index_dtype": "int32",
        "output_dtype": "bf16",
        "index_distribution": "recent_contiguous",
        "cache_layout": "token_major_mqa_bf16_latent",
    }
    spec.update(overrides)
    return spec


def test_both_backends_registered_for_mi300x_only() -> None:
    # Defect: the MI300X backend leaking onto a NVIDIA GPU, or the B200/H200 rows
    # changing. Each is a separate backend of the same kind, gated to MI300X,
    # BF16 (not FP8), and routed to the vllm_rocm_env image.
    from profiling.db.registry import find_kernel_profiler_spec

    for kind, fn in (
        ("dsa_sparse_mla_attention", "profile_dsa_sparse_mla_attention_rocm_triton"),
        ("dsa_sparse_mla_prefill", "profile_dsa_sparse_mla_prefill_rocm_triton"),
    ):
        spec = find_kernel_profiler_spec(kind, "rocm_triton_mla_sparse")
        assert spec.supports.gpus == frozenset({"MI300X"})
        assert spec.supports.compute == frozenset({DType.BF16})
        assert spec.supports.kv == frozenset({DType.BF16})
        assert spec.subprocess_env == "vllm_rocm_env"
        assert spec.runner_ref.function_name == fn

    # B200 fp8 rows untouched.
    b200 = find_kernel_profiler_spec("dsa_sparse_mla_attention", "flashinfer_trtllm_fp8")
    assert b200.supports.gpus == frozenset({"NVIDIA B200"})
    assert b200.supports.compute == frozenset({DType.FP8_E4M3})
    assert b200.subprocess_env == "vllm_env"


def test_profile_signatures_match_kind_arg_order() -> None:
    # Defect: the ROCm runner drifting from the kind's args schema, so the worker
    # passes a coordinate the runner reads in a different order.
    decode_params = list(inspect.signature(decode.profile_dsa_sparse_mla_attention_rocm_triton).parameters)
    decode_fields = [f.name for f in dataclasses.fields(decode_kind.DsaSparseMlaAttentionArgs)]
    assert decode_params == decode_fields

    prefill_params = list(inspect.signature(prefill.profile_dsa_sparse_mla_prefill_rocm_triton).parameters)
    prefill_fields = [f.name for f in dataclasses.fields(prefill_kind.DsaSparseMlaPrefillArgs)]
    assert prefill_params == prefill_fields


def test_rocprof_builders_registered() -> None:
    # Defect: the rocprof launch driver cannot rebuild a (kind, backend), so the
    # under-tracer replay fails.
    from profiling.profilers.rocprof_run import _BUILDERS

    assert ("dsa_sparse_mla_attention", "rocm_triton_mla_sparse") in _BUILDERS
    assert ("dsa_sparse_mla_prefill", "rocm_triton_mla_sparse") in _BUILDERS


def test_validation_rejects_fp8_and_wrong_rope() -> None:
    # Defect: silently accepting the B200 fp8 / rope=64 coordinate, which would
    # mislabel a row. The MI300X backend is BF16 and rope-free only.
    with pytest.raises(ProfilerNotImplemented):
        decode._validate(**_decode_spec(rope_dim=64))
    with pytest.raises(ProfilerNotImplemented):
        decode._validate(**_decode_spec(q_dtype="fp8_e4m3"))
    with pytest.raises(ProfilerNotImplemented):
        decode._validate(**_decode_spec(selected_k=1024))
    with pytest.raises(ProfilerNotImplemented):
        prefill._validate(**_prefill_spec(rope_dim=64))
    with pytest.raises(ProfilerNotImplemented):
        prefill._validate(**_prefill_spec(cache_dtype="fp8_e4m3"))


def test_prefill_valid_counts_match_b200_derivation() -> None:
    # Defect: the prefill ramp drifting from the B200 runner's causal derivation,
    # so the MI300X and B200 rows describe different workloads.
    from profiling.runners.attention.dsa_sparse_mla_prefill import _validate_args as b200_validate

    pairs = ((4, 10), (2, 2048))
    _p, counts = prefill._validate(**_prefill_spec(query_context_pairs=pairs))
    # The B200 validator derives the same ramp (it rejects our bf16 storage only
    # after computing valid_counts, so compute it the same hand way here).
    expected: list[int] = []
    for qc, cl in pairs:
        expected.extend(min(p + 1, 2048) for p in range(cl - qc, cl))
    assert list(counts) == expected


def test_reference_is_rope_free_selected_softmax() -> None:
    # Defect: a wrong reference oracle (e.g. a 576-wide rope split), which would
    # make the GPU correctness check pass against the wrong math. Verify the
    # rope-free reference equals an independent einsum over the selected rows.
    torch.manual_seed(0)
    num_queries, num_heads, k, cache = 3, 2, 4, 16
    dense = torch.full((num_queries, common.SELECTED_K), -1, dtype=torch.int32)
    lengths = torch.zeros(num_queries, dtype=torch.int32)
    for i in range(num_queries):
        dense[i, :k].copy_(torch.randperm(cache)[:k].to(torch.int32))
        lengths[i] = k
    batch = common.RaggedBatch(
        dense_indices=dense,
        lengths=lengths,
        num_cache_tokens=cache,
        num_queries=num_queries,
        valid_counts=(k,) * num_queries,
    )
    q = torch.randn(num_queries, num_heads, common.HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.randn(cache, common.NUM_KV_HEADS, common.HEAD_DIM, dtype=torch.bfloat16)
    out = common.reference_output(torch, q, kv, batch, softmax_scale=common.SOFTMAX_SCALE)
    assert out.shape == (num_queries, num_heads, common.VALUE_DIM)
    assert torch.isfinite(out).all()

    kv_rows = kv[:, 0, :].float()
    for i in range(num_queries):
        idx = dense[i, :k].to(torch.int64)
        sel = kv_rows.index_select(0, idx)
        scores = (q[i].float() @ sel.t()) * common.SOFTMAX_SCALE
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        ref = (probs @ sel).to(torch.bfloat16).float()
        assert torch.allclose(out[i].float(), ref, atol=1e-2, rtol=1e-2)


def test_logical_flops_and_bytes_rope_free() -> None:
    # Defect: counting a 576-wide score path or a nonzero rope read, which would
    # overstate TFLOPS/GB/s vs the rope-free kernel's real traffic.
    flops = common.logical_flops(num_queries=32, num_heads=16, selected_k=2048)
    assert flops == 2 * 32 * 16 * 2048 * (512 + 512)
    byts = common.logical_bytes(
        num_queries=2, num_heads=16, selected_k=2048, valid_counts=(2048, 2048)
    )
    expected = (
        2 * 2 * 16 * 512  # q
        + 4 * 2 * 2048  # index
        + 2 * (2048 + 2048) * 512  # cache
        + 2 * 2 * 16 * 512  # output
        + 8 * 2 * 16  # max/lse
    )
    assert byts == expected
