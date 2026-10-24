"""TensorFold fused gate|up GEMM + SwiGLU + NVFP4 quant: the ``tensorfold_ck`` backend of ``gemm_swiglu_quant``.

L1a-only: build inputs, check the output against the PyTorch reference, time
one launch, return metrics.

Production path (TensorFold perf-study 6469730a): a checkpoint-math NVFP4 dense
MLP's prompt rows go through ``tensorfold.cuda.nvfp4.checkpoint.mlp_prompt``
(``families/qwen3_5/cuda/prefill.py::_mlp``):

1. ``quant4(x, gate.act)``: bf16 rows -> NVFP4 rows (its own launch, the input
   quantization; built here once, outside timing);
2. ``_ext().gemm_gu_ck(...)``: the timed launch, ``gemm_kernel<A4, 128, 256, 2,
   4, 3, 2, false, 2>`` (``SWIGLU_FP32`` picks EPI 2): gate and up of the same
   rows, SiLU(gate)·up in fp32, written as NVFP4 rows under ``1 / down.act``;
3. ``_gemm(A4, ..., down)``: the down projection, a ``single_gemm`` row.

``mlp_prompt`` takes this path only when gate and up share their static input
scale (``act``; their ``weight_scale_2`` stay separate, as two alphas), both are
``n`` wide with ``n`` whole 64-column tiles, and ``k`` and ``n`` are multiples
of 128; otherwise it returns None and the layer runs unfused. The runner refuses
those shapes rather than time a launch production never issues there. The
kernel's tile and launch shape are fixed (no chip or row-count switch), so its
time depends on (m, n, k) only.

Runs in ``tensorfold_env``: TensorFold JIT-compiles the extension against its
own Torch on the first call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BLOCK = 16
# Scale bytes and codes the fused epilogue may round differently from the
# reference: its fp32 accumulation order differs from torch's matmul, so a value
# sitting on an e4m3 or e2m1 rounding boundary can land on either side.
MAX_SCALE_MISMATCH = 1e-3
MAX_CODE_MISMATCH = 1e-3
MAX_REL_ERR = 5e-3
KERNEL_NAME = "gemm_kernel"
_FP4_MAX = 6.0
_E4M3_MAX = 448.0


@dataclass
class FusedMlpOperands:
    """The timed call's arguments and what the check needs."""

    rows: Any  # checkpoint.Rows4 of the activation
    gate: Any  # Fp4Linear (checkpoint math)
    up: Any
    codes: Any  # output codes [m, n/2]
    scales: Any  # output scales [n/64, mpad, 4]
    qg: float  # 1 / down.act
    x_deq: Any  # fp32 [m, k], the activation as the kernel reads it
    wg_deq: Any  # fp32 [n, k]
    wu_deq: Any


def _validate(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int]:
    dtype = DType.from_value(dtype)
    if dtype is not DType.NVFP4_E2M1:
        raise ProfilerNotImplemented(
            f"tensorfold_ck gemm_swiglu_quant is NVFP4-only but got dtype={dtype.value}"
        )
    m, n, k = int(m), int(n), int(k)
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"m, n, k must be positive, got m={m}, n={n}, k={k}")
    # checkpoint.mlp_prompt's fused-path conditions (gate.npad == gate.n,
    # gate.k % 128, down.k % 128 with down.k == n).
    if n % 128 or k % 128:
        raise ValueError(
            f"TensorFold fuses gate|up only when n and k are multiples of 128, got n={n}, k={k}"
        )
    return m, n, k


def _random_checkpoint_weight(torch: Any, n: int, k: int, generator: Any) -> tuple[Any, Any, float]:
    """A checkpoint NVFP4 weight: random e2m1 codes, e4m3 block scales, an fp32 global scale."""
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=generator)
    block = torch.empty((n, k // BLOCK), dtype=torch.float32, device="cuda").uniform_(
        16.0, 448.0, generator=generator
    )
    weight_scale = block.to(torch.float8_e4m3fn)
    # A unit-variance weight: e2m1 codes average |v| ~ 2.3, scales ~ 230.
    return codes, weight_scale, 1.0 / (2.3 * 230.0)


def prepare_operands(m: int, n: int, k: int, *, seed: int = 0) -> FusedMlpOperands:
    import torch
    from tensorfold.cuda.nvfp4 import checkpoint
    from tensorfold.cuda.nvfp4.linear import Fp4Linear

    from profiling.runners.gemm import gemm_swiglu_quant_reference as ref

    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda", generator=generator)
    act = float(x.float().abs().amax()) / (_FP4_MAX * _E4M3_MAX)

    linears, dequantized = [], []
    for _ in range(2):
        w, ws, gs = _random_checkpoint_weight(torch, n, k, generator)
        linears.append(Fp4Linear.from_checkpoint(w, ws, gs, act=act))
        dequantized.append(ref.dequantize_weight(torch, w, ws, gs))
        del w, ws
    gate, up = linears

    rows = checkpoint.quant4(x, act)  # the input quantization: another kind, untimed
    del x
    x_deq = ref.dequantize_rows(torch, rows.codes, rows.scales, act)

    # The down projection's static input scale, calibrated on this activation.
    h = ref.swiglu_fp32(torch, x_deq, dequantized[0], dequantized[1])
    down_act = float(h.abs().amax()) / (_FP4_MAX * _E4M3_MAX) or 1.0
    del h
    qg = checkpoint._inv(down_act)

    mpad = rows.scales.shape[1]
    codes = torch.empty((m, n // 2), dtype=torch.uint8, device="cuda")
    scales = torch.empty((n // 64, mpad, 4), dtype=torch.uint8, device="cuda")
    return FusedMlpOperands(rows, gate, up, codes, scales, qg, x_deq, *dequantized)


def make_launch(ops: FusedMlpOperands) -> Any:
    """``mlp_prompt``'s fused launch, exactly as it calls it."""
    from tensorfold.cuda.nvfp4 import checkpoint

    ext = checkpoint._ext()
    gate, up = ops.gate, ops.up
    args = (
        ops.rows.codes, ops.rows.scales, gate.words, gate.bs, up.words, up.bs,
        checkpoint.alpha(gate.act, gate.scale), checkpoint.alpha(up.act, up.scale),
        gate.npad, gate.k, ops.qg, ops.codes, ops.scales, checkpoint.SWIGLU_FP32,
    )

    def launch() -> None:
        ext.gemm_gu_ck(*args)

    return launch


def check_against_reference(ops: FusedMlpOperands) -> dict[str, float]:
    """Kernel output vs the PyTorch reference; raises when it is outside tolerance."""
    import torch

    from profiling.runners.gemm import gemm_swiglu_quant_reference as ref

    m = ops.codes.shape[0]
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        want_codes, want_scales, h = ref.gemm_swiglu_quant_reference(
            torch, ops.x_deq, ops.wg_deq, ops.wu_deq, ops.qg
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    got_scales = ops.scales[:, :m, :]
    scale_mismatch = float((got_scales != want_scales).float().mean())
    scale_ulps = int((got_scales.int() - want_scales.int()).abs().max())
    code_mismatch = float((ops.codes != want_codes).float().mean())
    got = ref.dequantize_rows(torch, ops.codes, ops.scales, 1.0 / ops.qg)
    want = ref.dequantize_rows(torch, want_codes, want_scales.contiguous(), 1.0 / ops.qg)
    rel_err = float((got - want).norm() / want.norm().clamp_min(1e-30))
    quant_err = float((want - h).norm() / h.norm().clamp_min(1e-30))
    stats = {
        "scale_mismatch": scale_mismatch,
        "scale_max_ulps": scale_ulps,
        "code_mismatch": code_mismatch,
        "rel_err_vs_reference": rel_err,
        "reference_quant_err": quant_err,
    }
    if (
        scale_mismatch > MAX_SCALE_MISMATCH
        or scale_ulps > 1
        or code_mismatch > MAX_CODE_MISMATCH
        or rel_err > MAX_REL_ERR
    ):
        raise KernelLaunchFailed(f"gemm_gu_ck output disagrees with the PyTorch reference: {stats}")
    return stats


def logical_bytes(m: int, n: int, k: int) -> float:
    """Packed e2m1 codes plus one e4m3 scale per 16 elements of x, both weights and the output."""
    return (m * k + 2 * n * k + m * n) * (0.5 + 1.0 / BLOCK)


def profile_gemm_swiglu_quant_tensorfold_ck(
    m: int, n: int, k: int, dtype: DType | str
) -> ComputeMetrics:
    m, n, k = _validate(m, n, k, dtype)
    try:
        import torch
        from tensorfold.cuda.nvfp4 import checkpoint
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "TensorFold (tensorfold_env) is required for the tensorfold_ck fused MLP"
        ) from exc
    if not checkpoint.available():
        raise ProfilerNotImplemented("TensorFold's NVFP4 checkpoint math needs compute capability 12.x")

    try:
        ops = prepare_operands(m, n, k)
        launch = make_launch(ops)
        launch()  # the first call JIT-compiles the extension
        torch.cuda.synchronize()
        check_against_reference(ops)
        ops.x_deq = ops.wg_deq = ops.wu_deq = None
        torch.cuda.empty_cache()

        time_ms = Timer.cupti(launch, kernel_name=KERNEL_NAME)
        energy_j = Energy.perf(launch, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError("tensorfold_ck gemm_swiglu_quant ran out of CUDA memory") from exc
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc

    seconds = time_ms / 1000.0
    flops = 2 * m * (2 * n) * k
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / seconds / 1e12) if time_ms > 0 else 0.0,
        memory_bandwidth_gbps=float(logical_bytes(m, n, k) / seconds / 1e9) if time_ms > 0 else 0.0,
        energy_j=float(energy_j),
    )
