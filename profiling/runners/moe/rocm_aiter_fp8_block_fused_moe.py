"""AITER FP8 block-scale fused MoE on MI300X (the GLM-5.3-Flash routed MoE).

This is the ROCm/MI300X backend of the ``nvfp4_fused_moe`` kind, the single
largest share of GLM-5.3-Flash iteration time. It is the AMD analog of the B200
``flashinfer_trtllm_fp8_block_sm100`` backend (``fp8_block_fused_moe.py``):
instead of FlashInfer's ``trtllm_fp8_block_scale_moe`` it times the call vLLM's
ROCm path makes with AITER enabled,

    ``vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe
      .rocm_aiter_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids,
          moe_config, activation=MoEActivation.SILU, quant_config=..., ...)``

which, for Silu + FP8 block-scale (``block_shape=[128, 128]``, per-token-group-128
activations) and these GLM dims (288 experts / 72 local / top-8 / hidden 4096 /
inter 2048 / group 128), dispatches the AITER kernel
``aiter.fmoe_fp8_blockscale_g1u1`` plus ``aiter.moe_sorting_fwd`` -- confirmed
production (not the Triton fallback) by static introspection of the pinned image;
see ``doc/pr/glm53-flash-mi300x/tier-a-amd-backends.md`` Q4.

Operands are AMD-native, built from scratch (no cross-layout conversion at
measurement time, exactly as the B200 runner builds its own FlashInfer layout):

- FP8 E4M3 expert weights in checkpoint order ``w13 = [gate; up]`` stacked and
  ``w2``, each pre-shuffled with ``aiter.ops.shuffle.shuffle_weight(w,
  layout=(16, 16))`` (the shuffle vLLM's AITER FP8-MoE prep applies; requires
  ``K % 128 == 0`` and ``N % 16 == 0``, which the GLM dims satisfy).
- Plain FP32 128x128 weight block scales (``w1_scale`` / ``w2_scale``) -- no
  FlashInfer BlockMajorK / UE8M0 packing.
- BF16 hidden states; the call quantizes them to FP8 with per-token-group-128
  FP32 activation scales internally (``fc_scale_blkn = 128``), so ``a1q_scale``
  is left ``None``.

Routing is fed directly: we hand the call ``topk_ids`` / ``topk_weights`` built
from ``per_expert_batches`` via the shared ``exact_topk_ids``, so the router is
bypassed (the AITER expert call takes ids/weights, unlike the FlashInfer B200
call which runs routing in-kernel from forced logits). The correctness check
compares against a Torch reference driven by the SAME ids/weights.

Arg coordinates are identical to the B200 FP8-block row: the kind's
``Nvfp4FusedMoeArgs`` is reused unchanged and validated by the same
``_validate_args`` as ``fp8_block_fused_moe.py``, so the MI300X and B200 rows sit
at the same point in ``(num_tokens, hidden, inter, experts, ..., per_expert_batches)``
space and differ only by ``(gpu_name, backend)``.

Timing is kernel-only via rocprofv3 (``measure_registered_via_rocprofv3``), the
ROCm counterpart of the CUPTI path the B200 backend uses. The fused call is a
compound of several dispatches per launch (sorting, FC1+SwiGLU g1u1, FC2,
finalize); AITER/CK MoE kernels ``@autotune`` on the first launch (a variable
dispatch burst), then cache their winner in-process, so every later launch
issues a constant ``D`` dispatches. As with KDA chunked prefill this is timed
with the autotune-robust trailing fold ``dispatches_per_launch=D`` (sum the last
``rep*D`` dispatches, skipping the leading build-shuffle + autotune prefix), not
``fold_per_launch`` (which the burst would corrupt).

FIRST-REAL-RUN PINS (open questions Q6/Q7, resolvable only on an MI300X):
``_DISPATCHES_PER_LAUNCH`` (the steady-state ``D``) and ``_AITER_KERNEL_SUBSTR``
(the emitted CK/ASM HIP kernel string for the g1u1 block-scale GEMM). Until they
are pinned the profile raises ``ProfilerNotImplemented`` rather than guessing a
time. The path-selection guardrail (AITER ran, Triton did not) is asserted from
the capture's kernel-name set before any number is reported.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from profiling.db.args import DType
from profiling.runners.attention.kda_recurrent_decode_torch_rocm import (
    _device_arch,
    _device_name,
    _is_mi300_series,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.moe.exact_topk import exact_topk_ids

# Reuse the B200 FP8-block validation verbatim so the MI300X row lands at exactly
# the same arg coordinates (same constraints, same per-expert histogram checks).
from profiling.runners.moe.fp8_block_fused_moe import _validate_args
from profiling.runners.moe.fp8_block_fused_moe_reference import (
    dequantize_blocks,
    dequantize_token_groups,
    quantize_token_groups,
)

_KIND = "nvfp4_fused_moe"
_BACKEND = "rocm_aiter_fp8_block"
_GPU_SKU_TOKEN = "MI300X"
_GROUP_SIZE = 128
# GLM-5.3-Flash `swiglu_limit`, same as the B200 runner's SWIGLU clamp.
_SWIGLU_LIMIT = 10.0
_SEED = 0
# Enough warm-up to cover the one-time AITER/CK autotune so every timed rep is
# steady-state, mirroring the B200 runner's "3 warm-up calls after autotune".
_WARMUP = 6
_REP = 20
_W13_SCALE = (0.05, 0.15)
_W2_SCALE = (0.005, 0.015)

_MODULE = "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe"
_CALLABLE = "rocm_aiter_fused_experts"

# --- Pinned from the first real MI300X capture (gfx942, aiter v0.1.21.x,
# vLLM 0.3.1.dev190; inspect_aiter_fp8_moe.py). ------------------------------
# Steady-state GPU-dispatch count of one rocm_aiter_fused_experts call once the
# AITER/CK kernels are built/cached (every launch from warm-up #2 onward): the
# whole fused call issues D=5 dispatches (sorting + the two block-scale grouped
# GEMMs + the scatter/finalize helpers), constant across the trailing reps
# (period-5 tail, 134 total = 4-dispatch build/shuffle prefix + 26 launches*5).
# The trailing fold sums the last rep*D dispatches, skipping the prefix.
_DISPATCHES_PER_LAUNCH: int | None = 5
# The emitted AITER FP8 block-scale MoE grouped-GEMM HIP kernel string is
# `void ck::kernel_moe_gemm<ck::GridwiseMoeGemmBlockScale<... ck::f8_fnuz_t,
# ck::f8_fnuz_t, ... MulABScaleExpertWeightA8W8blkscale ...>>` (the CK realization
# of aiter.fmoe_fp8_blockscale_g1u1 / QuantType per_1x128); the sorting/scatter
# kernel is `aiter::opus_moe_sorting_entry<aiter::MoeSortingKernel<...>>`.
_AITER_KERNEL_SUBSTR: str | None = "GridwiseMoeGemmBlockScale"
# At small per-expert batch sizes AITER's fused_moe keeps control (its own
# moe_sorting + per-group quant) but dispatches the 1-stage ASM block-scale
# g1u1 GEMM (`aiter::fmoe_..._blockscaleFp8_g1u1_...`) rather than the CK
# 2-stage `GridwiseMoeGemmBlockScale` realization. Both are the real AITER
# fp8 block-scale MoE GEMM, so either one counts as "AITER ran".
_AITER_ASM_GEMM_SUBSTR: str = "blockscaleFp8_g1u1"
_SORTING_KERNEL_SUBSTR: str | None = "moe_sorting"
# The Triton FP8-MoE kernel that MUST be absent (silent-fallback guard): the
# `@triton.jit def fused_moe_kernel` in vLLM's fused_moe.py.
_TRITON_KERNEL_SUBSTR = "fused_moe_kernel"


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    """Import torch + the AITER MoE callable and weight-shuffle lazily.

    Nothing here is imported at module load, so the CPU registration tests can
    import this runner on a non-ROCm host. On the real image this resolves the
    exact production entry point vLLM-ROCm dispatches with AITER enabled.
    """

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc
    if getattr(torch.version, "hip", None) is None:
        raise ProfilerNotImplemented(f"{_BACKEND} requires a ROCm (HIP) torch build")
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"a ROCm device is required for {_BACKEND}")
    if not _is_mi300_series(torch):
        raise ProfilerNotImplemented(
            f"{_BACKEND} is verified only on {_GPU_SKU_TOKEN} (CDNA3 gfx942), got "
            f"name={_device_name(torch)!r} arch={_device_arch(torch)!r}"
        )
    try:
        from aiter.ops.shuffle import shuffle_weight
    except Exception as exc:  # pragma: no cover - requires the vLLM-ROCm image
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires AITER (aiter.ops.shuffle.shuffle_weight)"
        ) from exc
    try:
        module = importlib.import_module(_MODULE)
        fused_experts = getattr(module, _CALLABLE)
    except Exception as exc:  # pragma: no cover - requires the vLLM-ROCm image
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires {_MODULE}.{_CALLABLE} (vLLM-ROCm AITER MoE path)"
        ) from exc
    return torch, fused_experts, shuffle_weight, module


def _fp8_dtype(torch: Any, device: Any) -> Any:
    """The FP8 E4M3 variant for this device.

    CDNA3 (gfx942) uses ``float8_e4m3fnuz`` (what ``aiter`` and vLLM's quant
    config expect via ``current_platform.fp8_dtype()``); the OCP ``float8_e4m3fn``
    NVIDIA uses is unsupported by aiter's quant bindings there. On CPU (the
    registration test) there is no platform, so fall back to the OCP dtype.
    """

    if getattr(device, "type", None) == "cuda":
        try:
            from vllm.platforms import current_platform

            return current_platform.fp8_dtype()
        except Exception:
            pass
    return torch.float8_e4m3fn


def _random_fp8(
    torch: Any, shape: tuple[int, ...], generator: Any, *, device: Any, fp8_dtype: Any
) -> Any:
    values = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    return values.to(fp8_dtype).to(device)


def _random_scales(
    torch: Any,
    shape: tuple[int, ...],
    bounds: tuple[float, float],
    generator: Any,
    *,
    device: Any,
) -> Any:
    low, high = bounds
    values = torch.rand(shape, generator=generator, dtype=torch.float32)
    return (low + (high - low) * values).to(device)


def build_torch_operands(args: dict[str, Any], *, device: Any) -> dict[str, Any]:
    """Build the AMD-native synthetic operands with pure torch (no AITER).

    Returned on ``device`` (``"cpu"`` in the registration test, the ROCm device
    in the runner). Weights are checkpoint-order FP8 ``w13 = [gate; up]`` stacked
    and ``w2``; scales are plain FP32 128x128 blocks; hidden states are BF16;
    ``topk_ids`` / ``topk_weights`` realize ``per_expert_batches`` exactly. The
    AITER weight shuffle is applied separately (GPU only) in ``_build_case``.
    """

    import torch

    experts = args["num_local_experts"]
    hidden = args["hidden_size"]
    inter = args["intermediate_size"]
    tokens = args["num_tokens"]
    top_k = args["top_k"]

    gen = torch.Generator(device="cpu").manual_seed(_SEED)
    fp8_dtype = _fp8_dtype(torch, device)
    # Checkpoint order: w13 is [2*inter, hidden] per expert ([gate; up]); w2 is
    # [hidden, inter]. shuffle_weight needs K % 128 == 0 and N % 16 == 0.
    w13 = _random_fp8(torch, (experts, 2 * inter, hidden), gen, device=device, fp8_dtype=fp8_dtype)
    w2 = _random_fp8(torch, (experts, hidden, inter), gen, device=device, fp8_dtype=fp8_dtype)
    w13_scale = _random_scales(
        torch, (experts, 2 * inter // _GROUP_SIZE, hidden // _GROUP_SIZE), _W13_SCALE, gen,
        device=device,
    )
    w2_scale = _random_scales(
        torch, (experts, hidden // _GROUP_SIZE, inter // _GROUP_SIZE), _W2_SCALE, gen,
        device=device,
    )

    hidden_states = (
        torch.randn((tokens, hidden), generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
    )

    ids = exact_topk_ids(
        num_tokens=tokens, top_k=top_k, per_expert_batches=args["per_expert_batches"]
    )
    topk_ids = torch.tensor(ids, dtype=torch.int32, device=device)
    # Positive per-token weights, renormalized then scaled by the routed factor,
    # as the router would produce. Fed directly to the call and to the reference.
    raw = torch.rand((tokens, top_k), generator=gen, dtype=torch.float32)
    scale = args["routed_scaling_numerator"] / args["routed_scaling_denominator"]
    topk_weights = (raw / raw.sum(dim=-1, keepdim=True) * scale).to(device)

    return {
        "w13": w13,
        "w2": w2,
        "w13_scale": w13_scale,
        "w2_scale": w2_scale,
        "hidden_states": hidden_states,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
    }


def _build_case(args: dict[str, Any]) -> dict[str, Any]:
    """GPU-only: build operands, AITER-shuffle the weights, bind the call."""

    torch, fused_experts, shuffle_weight, _module = _load_runtime()
    device = torch.device("cuda", torch.cuda.current_device())
    operands = build_torch_operands(args, device=device)

    # AITER weight pre-shuffle (vLLM's AITER FP8-MoE prep: layout=(16, 16) on both
    # the stacked gate-up and the down weights). Done once, before the timed
    # launches, so its dispatches land in the prefix the trailing fold skips.
    w13_shuffled = shuffle_weight(operands["w13"], layout=(16, 16))
    w2_shuffled = shuffle_weight(operands["w2"], layout=(16, 16))

    moe_config = _build_moe_config(torch, args)
    quant_config = _build_quant_config(args, operands["w13_scale"], operands["w2_scale"])
    activation = _silu_activation()

    def kernel() -> Any:
        return fused_experts(
            operands["hidden_states"],
            w13_shuffled,
            w2_shuffled,
            operands["topk_weights"],
            operands["topk_ids"],
            moe_config,
            activation=activation,
            quant_config=quant_config,
            a1q_scale=None,
        )

    return {
        "torch": torch,
        "kernel": kernel,
        "operands": operands,
        "w13_shuffled": w13_shuffled,
        "w2_shuffled": w2_shuffled,
        "warmup": _WARMUP,
        "rep": _REP,
    }


def _silu_activation() -> Any:
    # MoEActivation lives in the activation module (pinned on the real image).
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return MoEActivation.SILU


def _build_moe_config(torch: Any, args: dict[str, Any]) -> Any:
    """Construct the FusedMoEConfig the AITER expert call consumes.

    Pinned against the installed vLLM-ROCm (``v0.3.1.dev190``) by mirroring
    vLLM's own construction in ``fused_moe/layer.py`` and the CI/testing
    ``FusedMoEParallelConfig`` (single rank, ``use_ep=False``). For this
    standalone single-rank measurement the config's expert count is the rank's
    local experts (the synthetic ``w1``/``w2`` carry ``num_local_experts`` and
    ``topk_ids`` index into them), which is exactly the per-rank grouped GEMM.
    """

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    local = args["num_local_experts"]
    parallel = FusedMoEParallelConfig(
        tp_size=1, pcp_size=1, dp_size=1, ep_size=1,
        tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0,
        sp_size=1, use_ep=False,
        all2all_backend="allgather_reducescatter", enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=local,
        experts_per_token=args["top_k"],
        hidden_dim=args["hidden_size"],
        intermediate_size=args["intermediate_size"],
        num_local_experts=local,
        num_logical_experts=local,
        activation=MoEActivation.SILU,
        device=torch.device("cuda", torch.cuda.current_device()),
        routing_method=RoutingMethodType.DeepSeekV3,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        intermediate_size_per_partition=args["intermediate_size"],
        swiglu_limit=_SWIGLU_LIMIT,
    )


def _build_quant_config(args: dict[str, Any], w13_scale: Any, w2_scale: Any) -> Any:
    """FP8 w8a8 block-scale quant config: plain FP32 128x128 weight scales.

    Mirrors vLLM's fp8 "normal" block-scale branch (``fp8.py``): w1/w2 scales +
    ``block_shape=[128,128]``, activations quantized per-token-group-128 inside
    the call, and the GLM SwiGLU clamp forwarded as ``gemm1_clamp_limit``.
    """

    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config

    return fp8_w8a8_moe_quant_config(
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
        block_shape=[_GROUP_SIZE, _GROUP_SIZE],
        gemm1_clamp_limit=_SWIGLU_LIMIT,
    )


def build_rocm_aiter_fp8_block_fused_moe_kernel(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    input_dtype: DType | str,
    weight_format: DType | str,
    group_size: int,
    routing_method: str,
    n_group: int,
    topk_group: int,
    routed_scaling_numerator: int,
    routed_scaling_denominator: int,
    per_expert_batches: tuple[int, ...],
) -> dict[str, Any]:
    """Build the timed callable + fixed operands (shared with the rocprof driver)."""

    args = _validate_args(**dict(locals()))
    return _build_case(args)


def _reference_output(
    torch: Any, operands: dict[str, Any], args: dict[str, Any], *, clamp_limit: float | None
) -> Any:
    """FP32 ``[T, H]`` reference for the GIVEN topk ids/weights (router bypassed).

    The AITER expert call is handed ids/weights directly, so the reference loops
    experts over those same ids/weights rather than re-deriving routing. Reuses
    the shared block/token-group (de)quant so the numerics match the B200 ref.
    """

    x = dequantize_token_groups(
        torch,
        *_quantize_hidden(torch, operands["hidden_states"],
                          fp8_dtype=_fp8_dtype(torch, operands["hidden_states"].device)),
    )
    ids = operands["topk_ids"]
    weights = operands["topk_weights"]
    inter = args["intermediate_size"]
    output = torch.zeros_like(x)
    for local in range(args["num_local_experts"]):
        rows, slots = torch.nonzero(ids == local, as_tuple=True)
        if rows.numel() == 0:
            continue
        gate_up = x[rows] @ dequantize_blocks(
            torch, operands["w13"][local], operands["w13_scale"][local]
        ).t()
        gate, up = gate_up[:, :inter], gate_up[:, inter:]
        if clamp_limit is not None:
            gate = torch.clamp(gate, max=clamp_limit)
            up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
        act = quantize_token_groups(torch, torch.nn.functional.silu(gate) * up)
        down = act @ dequantize_blocks(torch, operands["w2"][local], operands["w2_scale"][local]).t()
        output.index_add_(0, rows, down * weights[rows, slots, None])
    return output


def _quantize_hidden(torch: Any, hidden: Any, *, fp8_dtype: Any) -> tuple[Any, Any]:
    """Round-trip BF16 hidden to FP8 with per-token-group-128 FP32 scales."""

    rows, cols = hidden.shape
    grouped = hidden.float().reshape(rows, cols // _GROUP_SIZE, _GROUP_SIZE)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / 448.0
    q = (grouped / scale).to(fp8_dtype).reshape(rows, cols)
    return q, scale.squeeze(-1)


def _check_correctness(torch: Any, built: dict[str, Any], args: dict[str, Any]) -> dict[str, float]:
    """Run the production call once and compare with the Torch reference."""

    actual = built["kernel"]().float()
    torch.cuda.synchronize()
    expected = _reference_output(torch, built["operands"], args, clamp_limit=_SWIGLU_LIMIT)

    def rel(a: Any, b: Any) -> float:
        return float((a - b).norm() / b.norm().clamp(min=1e-30))

    rel_l2 = rel(actual, expected)
    stats = {
        "rel_l2": rel_l2,
        "max_abs": float((actual - expected).abs().max()),
        "finite": float(bool(torch.isfinite(actual).all())),
    }
    # Loose tolerance: FP8 block-scale GEMMs accumulate differently than the
    # dequant-then-BF16 reference. A gross mismatch (>0.1 rel-L2) means wrong
    # weight shuffle / scale layout, not rounding.
    if not stats["finite"] or rel_l2 > 0.1:
        raise KernelLaunchFailed(
            f"{_BACKEND} correctness check failed: rel_l2={rel_l2:.4f} finite={stats['finite']}"
        )
    return stats


def _spec_from_args(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_tokens": args["num_tokens"],
        "hidden_size": args["hidden_size"],
        "intermediate_size": args["intermediate_size"],
        "num_experts": args["num_experts"],
        "num_local_experts": args["num_local_experts"],
        "top_k": args["top_k"],
        "input_dtype": DType.from_value(args["input_dtype"]).value,
        "weight_format": DType.from_value(args["weight_format"]).value,
        "group_size": args["group_size"],
        "routing_method": args["routing_method"],
        "n_group": args["n_group"],
        "topk_group": args["topk_group"],
        "routed_scaling_numerator": args["routed_scaling_numerator"],
        "routed_scaling_denominator": args["routed_scaling_denominator"],
        "per_expert_batches": list(args["per_expert_batches"]),
    }


def _assert_aiter_path(spec: dict[str, Any]) -> int:
    """Prove AITER ran (and Triton did not) from the captured kernel-name set.

    A standalone rocprofv3 capture of the same launch driver, read for its kernel
    names only (durations discarded). Reuses the profiler's rocprofv3 locator and
    rocpd name reader; this is the mandatory silent-fallback guard, run once
    before any timing. Requires the HIP symbol pins to be set.
    """

    from profiling.profilers.rocprof_kernel_profiler import _captured_names, _find_rocprofv3

    if _AITER_KERNEL_SUBSTR is None or _SORTING_KERNEL_SUBSTR is None:
        raise ProfilerNotImplemented(
            f"{_BACKEND}: AITER HIP kernel substrings not yet pinned (Q6); "
            "read them off the first MI300X rocpd capture"
        )
    rocprofv3 = _find_rocprofv3()
    with tempfile.TemporaryDirectory(prefix="vibesim-aiter-moe-") as tmp:
        rocpd_dir = Path(tmp) / "rocpd"
        driver = [
            "python3", "-m", "profiling.profilers.rocprof_run",
            "--kind", _KIND, "--backend", _BACKEND,
            "--spec", json.dumps(spec), "--warmup", str(_WARMUP), "--rep", str(_REP),
        ]
        cmd = [
            rocprofv3, "--kernel-trace", "--output-format", "rocpd",
            "-d", str(rocpd_dir), "-o", "run", "--", *driver,
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "")[-1500:]
            raise KernelLaunchFailed(f"{_BACKEND} path-assertion capture failed: {tail}")
        db_files = sorted(rocpd_dir.rglob("*.db"))
        if not db_files:
            raise KernelLaunchFailed(f"{_BACKEND} path-assertion produced no rocpd database")
        names = [n for n in _captured_names(str(db_files[0])) if n]
    has_ck_gemm = any(_AITER_KERNEL_SUBSTR in n for n in names)
    has_asm_gemm = any(_AITER_ASM_GEMM_SUBSTR in n for n in names)
    has_aiter_gemm = has_ck_gemm or has_asm_gemm
    has_sorting = any(_SORTING_KERNEL_SUBSTR in n for n in names)
    has_triton = any(_TRITON_KERNEL_SUBSTR in n for n in names)
    # AITER must own the call: its fused-MoE GEMM ran (CK 2-stage or ASM
    # 1-stage) AND its sorting ran, and no Triton fused-MoE fallback leaked in.
    if not (has_aiter_gemm and has_sorting) or has_triton:
        raise KernelLaunchFailed(
            f"{_BACKEND} is NOT on the AITER path: aiter_gemm={has_aiter_gemm} "
            f"(ck={has_ck_gemm} asm={has_asm_gemm}) aiter_sorting={has_sorting} "
            f"triton_fallback={has_triton}. Refusing to mislabel a non-AITER run as AITER."
        )
    # Steady-state dispatches-per-launch D for the trailing-mean timing fold.
    # CK 2-stage keeps the pinned D (path unchanged). The ASM 1-stage tiny-M
    # path issues a different, smaller D (sorting + per-group quant + the single
    # g1u1 GEMM); recover it from this capture, which is
    #   [small one-time prefix] + (warmup+rep)*D,
    # exact while the prefix is smaller than one sweep of launches.
    if has_ck_gemm:
        if _DISPATCHES_PER_LAUNCH is None:
            raise ProfilerNotImplemented(
                f"{_BACKEND}: CK per-launch dispatch count D not yet pinned (Q7)"
            )
        return int(_DISPATCHES_PER_LAUNCH)
    launches = _WARMUP + _REP
    d = len(names) // launches
    prefix = len(names) - d * launches
    if d < 1 or prefix >= launches:
        raise KernelLaunchFailed(
            f"{_BACKEND} ASM path: cannot recover dispatches-per-launch from "
            f"{len(names)} dispatches over {launches} launches (d={d} prefix={prefix})."
        )
    return d


def _logical_bytes(args: dict[str, Any]) -> int:
    """Useful algorithmic traffic of this rank's call (same counts as B200)."""

    local_batches = args["per_expert_batches"][: args["num_local_experts"]]
    local_rows = sum(local_batches)
    active_experts = sum(batch > 0 for batch in local_batches)
    tokens = args["num_tokens"]
    hidden = args["hidden_size"]
    inter = args["intermediate_size"]
    experts = args["num_experts"]
    blocks = _GROUP_SIZE * _GROUP_SIZE

    routing = 4 * tokens * experts + 4 * experts
    activations = local_rows * (hidden + 4 * hidden // _GROUP_SIZE)
    weights = active_experts * (3 * inter * hidden + 4 * 3 * inter * hidden // blocks)
    output = 2 * hidden * tokens
    return routing + activations + weights + output


def profile_rocm_aiter_fp8_block_fused_moe(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    input_dtype: DType | str,
    weight_format: DType | str,
    group_size: int,
    routing_method: str,
    n_group: int,
    topk_group: int,
    routed_scaling_numerator: int,
    routed_scaling_denominator: int,
    per_expert_batches: tuple[int, ...],
) -> ComputeMetrics:
    """Profile one GLM-5.3-Flash AITER FP8 block-scale fused MoE call on MI300X."""

    args = _validate_args(**dict(locals()))
    try:
        torch, _fused, _shuffle, _module = _load_runtime()
        from profiling.profilers.rocprof_kernel_profiler import (
            measure_registered_via_rocprofv3,
        )

        built = _build_case(args)
        _check_correctness(torch, built, args)

        spec = _spec_from_args(args)
        # Mandatory silent-fallback guard: prove AITER (not Triton) ran. Returns
        # the steady-state dispatches-per-launch D for the timing fold (CK keeps
        # the pinned D; the ASM 1-stage tiny-M path recovers its smaller D).
        dispatches_per_launch = _assert_aiter_path(spec)

        time_ms = measure_registered_via_rocprofv3(
            kind=_KIND,
            backend=_BACKEND,
            spec=spec,
            kernel_name_contains=None,
            warmup=_WARMUP,
            rep=_REP,
            dispatches_per_launch=dispatches_per_launch,
        )
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{_BACKEND} ran out of device memory") from exc
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    local_rows = sum(args["per_expert_batches"][: args["num_local_experts"]])
    h, i = args["hidden_size"], args["intermediate_size"]
    flops = 2 * local_rows * (h * 2 * i + i * h)
    elapsed = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed / 1e12 if elapsed > 0 else 0.0,
        memory_bandwidth_gbps=_logical_bytes(args) / elapsed / 1e9 if elapsed > 0 else 0.0,
        energy_j=0.0,
    )


__all__ = [
    "build_rocm_aiter_fp8_block_fused_moe_kernel",
    "build_torch_operands",
    "profile_rocm_aiter_fp8_block_fused_moe",
]
