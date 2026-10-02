"""Focused CPU tests for the MI300X ``nvfp4_fused_moe`` AITER FP8-block backend.

No GPU, no DB. The GPU correctness check of the AITER call against the Torch
reference and the AITER-vs-Triton path assertion run inside the runner on the
ROCm host; their evidence comes from the Slurm step. Each test names the defect
it guards against.
"""

from __future__ import annotations

import inspect
import subprocess
import sys

import torch

from profiling.runners.moe import fp8_block_fused_moe as b200
from profiling.runners.moe import rocm_aiter_fp8_block_fused_moe as rocm
from profiling.runners.moe.exact_topk import exact_topk_ids


def _spec(**overrides: object) -> dict:
    # GLM-5.3-Flash on one EP4 rank, 16 tokens, half the local experts idle --
    # identical coordinates to the B200 fp8-block row.
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
        "weight_format": "fp8_e4m3",
        "group_size": 128,
        "routing_method": "deepseek_v3",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_numerator": 5,
        "routed_scaling_denominator": 2,
        "per_expert_batches": tuple(batches),
    }
    spec.update(overrides)
    return spec


def test_backend_registered_for_mi300x_only() -> None:
    # Defect: the MI300X backend leaking onto a NVIDIA GPU, or the B200 rows
    # changing. The AITER row is a separate backend of the same kind, gated to
    # MI300X and routed to the vllm_rocm_env image.
    from profiling.db.registry import find_kernel_profiler_spec

    spec = find_kernel_profiler_spec("nvfp4_fused_moe", "rocm_aiter_fp8_block")
    assert spec.supports.gpus == frozenset({"MI300X"})
    assert spec.supports.compute == frozenset({__import__("profiling.db.args", fromlist=["DType"]).DType.FP8_E4M3})
    assert spec.subprocess_env == "vllm_rocm_env"
    assert spec.runner_ref.function_name == "profile_rocm_aiter_fp8_block_fused_moe"
    assert spec.args_schema.__name__ == "Nvfp4FusedMoeArgs"

    b200_spec = find_kernel_profiler_spec("nvfp4_fused_moe", "flashinfer_trtllm_fp8_block_sm100")
    assert b200_spec.supports.gpus == frozenset({"NVIDIA B200"})
    assert b200_spec.subprocess_env == "vllm_env"


def test_profile_signature_matches_kind_arg_order() -> None:
    # Defect: the ROCm runner drifting from the kind's arg order, so the worker
    # passes positional args the runner reads in a different order.
    rocm_params = list(inspect.signature(rocm.profile_rocm_aiter_fp8_block_fused_moe).parameters)
    b200_params = list(inspect.signature(b200.profile_fp8_block_fused_moe_sm100).parameters)
    assert rocm_params == b200_params


def test_rocprof_builder_registered() -> None:
    # Defect: the rocprof launch driver cannot rebuild this (kind, backend), so
    # the under-tracer replay fails. The builder must be wired into _BUILDERS.
    from profiling.profilers.rocprof_run import _BUILDERS

    assert ("nvfp4_fused_moe", "rocm_aiter_fp8_block") in _BUILDERS


def test_torch_operands_are_amd_native_and_match_histogram() -> None:
    # Defect: wrong weight/scale shapes (would fail AITER's shuffle K%128/N%16
    # constraints or the block-scale layout), or topk ids that do not realize
    # per_expert_batches. Built on CPU; the only GPU-only step is the shuffle.
    args = b200._validate_args(**_spec())
    operands = rocm.build_torch_operands(args, device=torch.device("cpu"))

    experts, hidden, inter = 72, 4096, 2048
    assert operands["w13"].shape == (experts, 2 * inter, hidden)
    assert operands["w2"].shape == (experts, hidden, inter)
    assert operands["w13"].dtype is torch.float8_e4m3fn and operands["w2"].dtype is torch.float8_e4m3fn
    # shuffle_weight constraints on each per-expert matrix.
    assert hidden % 128 == 0 and (2 * inter) % 16 == 0  # w13: K=hidden, N=2*inter
    assert inter % 128 == 0 and hidden % 16 == 0  # w2: K=inter, N=hidden

    assert operands["w13_scale"].shape == (experts, 2 * inter // 128, hidden // 128)
    assert operands["w2_scale"].shape == (experts, hidden // 128, inter // 128)
    assert operands["w13_scale"].dtype is torch.float32 and operands["w2_scale"].dtype is torch.float32

    assert operands["hidden_states"].shape == (16, hidden)
    assert operands["hidden_states"].dtype is torch.bfloat16

    assert operands["topk_ids"].shape == (16, 8) and operands["topk_ids"].dtype is torch.int32
    assert operands["topk_weights"].shape == (16, 8) and operands["topk_weights"].dtype is torch.float32
    # The ids must realize per_expert_batches exactly (the histogram the Rust
    # sweep handed down).
    counts = [0] * 288
    for row in operands["topk_ids"].tolist():
        assert len(set(row)) == 8  # distinct experts per token
        for expert in row:
            counts[expert] += 1
    assert tuple(counts) == args["per_expert_batches"]
    # topk weights are a per-token simplex scaled by the routed factor (5/2).
    rowsum = operands["topk_weights"].sum(dim=-1)
    assert torch.allclose(rowsum, torch.full_like(rowsum, 2.5), atol=1e-5)


def test_first_run_pins_are_set_from_the_mi300x_capture() -> None:
    # Pinned from the first real MI300X rocpd capture (Q6/Q7): the whole fused
    # call issues D=5 dispatches; the AITER block-scale GEMM and sorting HIP
    # strings are the path-assertion anchors; the Triton kernel must be forbidden.
    assert rocm._DISPATCHES_PER_LAUNCH == 5
    assert rocm._AITER_KERNEL_SUBSTR == "GridwiseMoeGemmBlockScale"
    assert rocm._SORTING_KERNEL_SUBSTR == "moe_sorting"
    assert rocm._TRITON_KERNEL_SUBSTR == "fused_moe_kernel"


def test_runner_is_import_light() -> None:
    # Importing the runner must not eager-import torch or vllm; both are built
    # lazily only when a measurement runs on the ROCm host.
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import profiling.runners.moe.rocm_aiter_fp8_block_fused_moe as r; "
            "print('torch' in sys.modules, 'vllm' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False False"
