"""FLA Triton runner for the GDN prefill chunked delta rule.

This runner only allocates tensors, times one public call, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

It times ``chunk_gated_delta_rule`` from
``vllm.third_party.flash_linear_attention.ops``, the callable
``ChunkGatedDeltaRule.forward_native`` (``vllm/model_executor/layers/mamba/gdn/
qwen_gdn_linear_attn.py``) returns whenever vLLM resolves the GDN prefill
backend to Triton/FLA: any CUDA device other than SM90, or SM10x/SM12x when
``head_k_dim != 128`` or the CUDA runtime is older than 13. The operands are
the ones ``QwenGatedDeltaNetAttention._forward_core`` passes on the prefill
branch, after ``fused_post_conv_prep(..., apply_l2norm=True,
output_g_exp=False)``:

- q/k ``[1,T,Hk,K]`` and v ``[1,T,H,V]``, contiguous bf16, q/k already
  L2-normalized, so ``use_qk_l2norm_in_kernel=False``;
- ``g`` the FP32 log-space decay ``[1,T,H]``. FLA's ``chunk_local_cumsum``
  consumes it directly; unlike the FlashInfer path there is no ``exp(g)``
  launch anywhere on this path;
- ``beta`` FP32 ``[1,T,H]``, already sigmoided;
- ``initial_state`` FP32 ``[N,H,V,K]``, the gathered SSM states;
- ``cu_seqlens`` int32 plus the ``chunk_indices`` / ``chunk_offsets`` the
  GDN metadata builder precomputes on the host for every step, so the call
  never derives them itself; ``output_final_state=True``; no ``core_attn_out``.

One call is a fixed seven-launch chain: ``chunk_local_cumsum_scalar_kernel``,
``chunk_scaled_dot_kkt_fwd_kernel``, the BF16 ``torch.zeros_like`` fill of
``solve_tril``'s output (``FillFunctor<BFloat16>``),
``merge_16x16_to_64x64_inverse_kernel``, ``recompute_w_u_fwd_kernel``,
``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` and ``chunk_fwd_kernel_o``.
The fill is inside the call (``solve_tril`` allocates it), so timing is the
CUPTI sum over every launch (``kernel_name=None``). The state gather, the
fresh-state zeroing and the final-state scatter are outside the call.

Autotune state. Six of the FLA kernels are ``@autotune``d on keys that hold
H/K/V/BT but not the token count, so the first shape a process tunes fixes
the configs for every later row. vLLM tunes them once at start-up in
``_warmup_prefill_kernels``: one 64-token sequence (``T = FLA_CHUNK_SIZE``)
from a zero state. The registry row disables autotune persistence
(``TRITON_CACHE_AUTOTUNING=0``) and the runner tunes at that same anchor,
at the spec's heads, dims and dtype, before a worker's first row of that
state key; ``row_provenance`` records the anchor and a digest of the configs.

FLOPs and bytes are the kind's semantic logical counts (shared with the
FlashInfer runner), not physical Triton work.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import exact_int as _exact_int
from profiling.runners.attention._gdn_common import load_required_callable
from profiling.runners.attention.gdn_chunk_delta_rule_flashinfer import (
    _Args,
    _canonical_sequence_boundaries,
    _logical_bytes,
    _semantic_flops,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.triton_autotune_pin import AutotunePin

_BACKEND = "gdn_chunk_delta_rule:fla_triton"
_PACKAGE = "vllm.third_party.flash_linear_attention.ops"
_CALLABLE = "chunk_gated_delta_rule"
# FLA_CHUNK_SIZE: the chunk kernels hard-code BT=64.
CHUNK_SIZE = 64
# chunk_gated_delta_rule_fwd_h asserts K <= 256 (four 64-wide state blocks).
_MAX_KEY_HEAD_DIM = 256
# FLA's own chunk-vs-naive tolerance for the gated delta rule (get_err_ratio).
RMSE_RATIO_TOL = 5e-3
# vLLM's _warmup_prefill_kernels: one FLA_CHUNK_SIZE-token sequence.
AUTOTUNE_ANCHOR = (CHUNK_SIZE, CHUNK_SIZE)
_PIN = AutotunePin(("vllm.third_party.flash_linear_attention",))
# Correctness witness bound: the per-token oracle is O(T) small launches.
_GUARD_MAX_LENGTH = 1024


@dataclass(frozen=True)
class _Operands:
    q: Any
    k: Any
    v: Any
    g: Any
    beta: Any
    initial_state: Any
    cu_seqlens: Any
    chunk_indices: Any
    chunk_offsets: Any
    boundaries: tuple[int, ...]


def validate_args(
    num_tokens: int,
    max_sequence_length: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> _Args:
    args = _Args(
        num_tokens=_exact_int("num_tokens", num_tokens),
        max_sequence_length=_exact_int("max_sequence_length", max_sequence_length),
        num_key_heads=_exact_int("num_key_heads", num_key_heads),
        num_heads=_exact_int("num_heads", num_heads),
        key_head_dim=_exact_int("key_head_dim", key_head_dim),
        value_head_dim=_exact_int("value_head_dim", value_head_dim),
        dtype=DType.from_value(dtype),
    )
    if args.max_sequence_length < 1 or args.num_tokens < args.max_sequence_length:
        raise ValueError("num_tokens must be >= max_sequence_length >= 1")
    if args.num_key_heads < 1 or args.num_heads % args.num_key_heads != 0:
        raise ValueError("num_heads must be a positive multiple of num_key_heads")
    if not 1 <= args.key_head_dim <= _MAX_KEY_HEAD_DIM:
        raise ValueError(f"{_BACKEND} requires 1 <= key_head_dim <= {_MAX_KEY_HEAD_DIM}")
    if args.value_head_dim < 1:
        raise ValueError("value_head_dim must be >= 1")
    if args.dtype is not DType.BF16:
        raise ValueError(f"{_BACKEND} requires dtype=bf16, got {args.dtype}")
    return args


def sequence_boundaries(args: _Args) -> tuple[int, ...]:
    """The kind's canonical partition: full-length sequences, then the remainder."""
    return tuple(_canonical_sequence_boundaries(args.num_tokens, args.max_sequence_length))


def guard_shape(args: _Args) -> _Args:
    """A bounded witness keeping several chunks, a second sequence and a tail."""
    length = min(args.max_sequence_length, _GUARD_MAX_LENGTH)
    tokens = min(args.num_tokens, 2 * length - 1) if args.num_tokens > length else length
    guard = replace(args, num_tokens=tokens, max_sequence_length=length)
    return args if guard == args else guard


def _vllm_chunk_metadata() -> tuple[Any, Any]:
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )

    return prepare_chunk_indices, prepare_chunk_offsets


def build_operands(
    torch: Any,
    args: _Args,
    *,
    device: Any,
    random_state: bool = False,
    chunk_metadata: tuple[Any, Any] | None = None,
) -> _Operands:
    """Operands as fused_post_conv_prep leaves them; chunk metadata host-built.

    ``chunk_metadata`` is ``(prepare_chunk_indices, prepare_chunk_offsets)``;
    the default is vLLM's own pair, which the GDN metadata builder runs on the
    host copy of ``cu_seqlens`` once per step.
    """
    prepare_chunk_indices, prepare_chunk_offsets = chunk_metadata or _vllm_chunk_metadata()
    tokens = args.num_tokens
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(*size: int) -> Any:
        return torch.randn(*size, generator=generator, dtype=torch.float32, device=device)

    def unit(x: Any) -> Any:
        # l2norm_fwd: x / sqrt(sum(x^2) + 1e-6), returned in the input dtype.
        return (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)

    q = unit(normal(1, tokens, args.num_key_heads, args.key_head_dim))
    k = unit(normal(1, tokens, args.num_key_heads, args.key_head_dim))
    v = normal(1, tokens, args.num_heads, args.value_head_dim).to(torch.bfloat16)
    # Log-space decay in (-inf, 0), FLA's own test distribution.
    g = torch.nn.functional.logsigmoid(normal(1, tokens, args.num_heads))
    beta = torch.sigmoid(normal(1, tokens, args.num_heads))
    boundaries = sequence_boundaries(args)
    state_shape = (len(boundaries) - 1, args.num_heads, args.value_head_dim, args.key_head_dim)
    initial_state = (
        normal(*state_shape)
        if random_state
        else torch.zeros(state_shape, dtype=torch.float32, device=device)
    )
    cu_seqlens_cpu = torch.tensor(boundaries, dtype=torch.int32)
    return _Operands(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens_cpu.to(device),
        chunk_indices=prepare_chunk_indices(cu_seqlens_cpu, CHUNK_SIZE).to(device),
        chunk_offsets=prepare_chunk_offsets(cu_seqlens_cpu, CHUNK_SIZE).to(device),
        boundaries=boundaries,
    )


def invoke(callable_: Any, operands: _Operands) -> Any:
    """The exact keyword set the qwen GDN prefill branch passes."""
    return callable_(
        q=operands.q,
        k=operands.k,
        v=operands.v,
        g=operands.g,
        beta=operands.beta,
        initial_state=operands.initial_state,
        output_final_state=True,
        cu_seqlens=operands.cu_seqlens,
        chunk_indices=operands.chunk_indices,
        chunk_offsets=operands.chunk_offsets,
        use_qk_l2norm_in_kernel=False,
    )


def check_correctness(
    torch: Any, callable_: Any, operands: _Operands, args: _Args
) -> tuple[float, float]:
    """Compare one call against the naive oracle; return (o, state) RMSE ratios."""
    from profiling.runners.attention.gdn_chunk_delta_rule_reference import (
        gdn_chunk_delta_rule_reference,
        rmse_ratio,
    )

    names = ("q", "k", "v", "g", "beta", "initial_state", "cu_seqlens")
    snapshots = {name: getattr(operands, name).clone() for name in names}
    output, final_state = invoke(callable_, operands)
    torch.cuda.synchronize()
    sequences = len(operands.boundaries) - 1
    expected_o_shape = (1, args.num_tokens, args.num_heads, args.value_head_dim)
    if tuple(output.shape) != expected_o_shape or output.dtype is not torch.bfloat16:
        raise AssertionError(f"unexpected output {tuple(output.shape)} {output.dtype}")
    if (
        tuple(final_state.shape)
        != (sequences, args.num_heads, args.value_head_dim, args.key_head_dim)
        or final_state.dtype is not torch.float32
    ):
        raise AssertionError(f"unexpected final state {tuple(final_state.shape)}")
    if not (torch.isfinite(output).all() and torch.isfinite(final_state).all()):
        raise AssertionError("non-finite GDN output or state")
    for name, snapshot in snapshots.items():
        if not torch.equal(getattr(operands, name), snapshot):
            raise AssertionError(f"chunk_gated_delta_rule mutated {name}")

    expected_o, expected_state = gdn_chunk_delta_rule_reference(
        operands.q.squeeze(0),
        operands.k.squeeze(0),
        operands.v.squeeze(0),
        operands.g.squeeze(0),
        operands.beta.squeeze(0),
        operands.initial_state,
        operands.boundaries,
        g_is_log=True,
    )
    o_err = rmse_ratio(expected_o, output.squeeze(0))
    state_err = rmse_ratio(expected_state, final_state)
    if not (o_err < RMSE_RATIO_TOL and state_err < RMSE_RATIO_TOL):
        raise AssertionError(
            f"GDN mismatch vs naive oracle: o rmse ratio {o_err:.2e}, "
            f"state rmse ratio {state_err:.2e} (tol {RMSE_RATIO_TOL:.0e})"
        )
    return o_err, state_err


def load_callable() -> Any:
    return load_required_callable(
        importlib.import_module,
        backend=_BACKEND,
        module_name=_PACKAGE,
        callable_name=_CALLABLE,
        environment_label="vllm_env",
    )


def profile_gdn_chunk_delta_rule_fla_triton(
    num_tokens: int,
    max_sequence_length: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile vLLM's FLA chunk_gated_delta_rule exactly as the GDN prefill calls it."""
    args = validate_args(
        num_tokens,
        max_sequence_length,
        num_key_heads,
        num_heads,
        key_head_dim,
        value_head_dim,
        dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc
    try:
        callable_ = load_callable()
        device = torch.device("cuda", torch.cuda.current_device())
        pin_autotune(torch, callable_, args, device=device)
        guard = guard_shape(args)
        check_correctness(
            torch,
            callable_,
            build_operands(torch, guard, device=device, random_state=True),
            guard,
        )
        operands = build_operands(torch, args, device=device)

        def kernel() -> Any:
            return invoke(callable_, operands)

        # Every call allocates its intermediates, output and final state and
        # reads initial_state without writing it, so no reset is needed.
        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=None)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    elapsed = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=_semantic_flops(args) / elapsed / 1e12 if elapsed > 0 else 0.0,
        memory_bandwidth_gbps=_logical_bytes(args) / elapsed / 1e9 if elapsed > 0 else 0.0,
        energy_j=float(energy_j),
    )


def anchor_shape(args: _Args) -> _Args:
    tokens, length = AUTOTUNE_ANCHOR
    return replace(args, num_tokens=tokens, max_sequence_length=length)


def _autotune_state(args: _Args) -> tuple[int, int, int, int, str]:
    return (
        args.num_key_heads,
        args.num_heads,
        args.key_head_dim,
        args.value_head_dim,
        args.dtype.value,
    )


def pin_autotune(torch: Any, callable_: Any, args: _Args, *, device: Any) -> str:
    """Tune once per worker and state key at vLLM's warm-up shape."""
    anchor = anchor_shape(args)
    label = f"T{anchor.num_tokens}/L{anchor.max_sequence_length}/H{anchor.num_heads}"

    def tune() -> None:
        invoke(callable_, build_operands(torch, anchor, device=device))
        torch.cuda.synchronize()

    return _PIN.ensure(_autotune_state(args), label, tune)


def row_provenance(
    num_tokens: int,
    max_sequence_length: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> str | None:
    """The autotune note for a row this worker just measured (registry hook)."""
    del num_tokens, max_sequence_length
    return _PIN.note(
        (num_key_heads, num_heads, key_head_dim, value_head_dim, DType.from_value(dtype).value)
    )


__all__ = [
    "AUTOTUNE_ANCHOR",
    "CHUNK_SIZE",
    "anchor_shape",
    "build_operands",
    "check_correctness",
    "guard_shape",
    "invoke",
    "load_callable",
    "pin_autotune",
    "profile_gdn_chunk_delta_rule_fla_triton",
    "row_provenance",
    "sequence_boundaries",
    "validate_args",
]
