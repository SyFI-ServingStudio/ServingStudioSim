"""vLLM-fork Triton runners for the KDA RecoverSSM verify and commit kinds.

These runners only allocate tensors, time one public call, and return
metrics. DB writes, subprocess selection and registry routing live in L1b.

Both time the vLLM fork's Kimi-K3 RecoverSSM ops
(``vllm.models.kimi_k3.nvidia.ops.recoverssm``), the only KDA RecoverSSM
implementation; Kimi-K3's KDA layer is the same op GLM-5.3-Flash uses
(per-key-channel gate with ``lower_bound``, sigmoid beta, q/k l2norm).

Cache pages. vLLM stores one KDA layer's states in one page per block, in
``get_state_shape`` order: conv ``[state_len, conv_dim]`` (SD layout, bf16),
the checkpoint ``[H, V, K]`` (``state_dtype``: fp32 by default, bf16 with
``--mamba-ssm-cache-dtype bfloat16``), the fp32 correction record ``[H, S, V]``
and the key/gate record ``[H, S, 2K]`` (activation dtype). Each is a strided
view of the layer's raw page buffer whose block stride is the whole page.
``_layer_page`` reproduces that layout; ``state_len = conv_size - 1 + S - 1``
with GLM-5.3-Flash's ``short_conv_kernel_size = 4``. Slot 0 is the NULL block
and the B requests own a fixed scattered permutation of slots ``1..B``.

``kda_recoverssm_verify``. One ``kda_recoverssm_verify`` call made exactly as
``KimiK3DeltaAttention._forward`` makes it on a spec-decode step, every
request a full ``S = spec_query_len`` window (``T = B * S`` tokens):

- q, k and v are ``1 n h d`` views of the contiguous ``spec_conv_out``
  ``[T, 3*H*K]`` that ``causal_conv1d_update`` writes (row stride ``3*H*K``);
- ``raw_g`` is the contiguous ``[1, T, H, K]`` ``f_b_proj`` output;
- ``raw_beta`` is the ``[1, T, H]`` column slice of the merged input
  projection (GLM-5.3-Flash's ``q|k|v|beta|f_a|g_a`` width), so it is
  row-strided;
- ``A_log`` is fp32 ``[H]``, ``dt_bias`` fp32 ``[H*K]``;
- ``out`` is the contiguous ``[1, T, H, V]`` layer output slice.

The op takes strided operands directly, so there are no ``.contiguous()``
copies on this path and the call is one ``_kda_recoverssm_verify_kernel``
launch (grid ``(V/32, B, H)``, 4 warps, 2 stages, no autotune). It reads the
checkpoint and never writes it; it writes ``out`` and the two records.

``kda_recoverssm_commit``. One ``KDARecoverSSMCommitContext.commit`` call,
the accepted-state commit ``RecoverSSMState.commit_step`` makes once per KDA
KV-cache group after sampling. The context spans ``num_layers`` layers and the
call launches three kernels, all counted:

- ``_prepare_commit_plan_kernel`` (grid ``(B,)``) clamps each request's
  accepted count to its window and plans the destination slots;
- ``_compact_conv_state_kernel`` (grid ``(conv_dim/256, B, L)``) shifts each
  layer's conv window so it ends at the last accepted token;
- ``_commit_kda_state_kernel`` (grid ``(V/32, B, L*H)``) reads each
  checkpoint once, folds ``commit_len`` records into it, and writes it once.

Every request accepts ``commit_len`` tokens. The non-align cache mode is
measured (no prefix caching: ``block_table`` is None), so the committed state
goes back to the checkpoint's own slot and no boundary state is written.
``request_indices`` is None (request i is spec decode i).

Timing is the CUPTI sum over the call's launches (``kernel_name=None``).
Every timed verify call rewrites the same records; every timed commit call
advances the checkpoints and conv windows in place, as production does. The
fold is contractive (decay < 1, bounded records), so the states stay finite.
Correctness is checked once against the Torch oracles, outside timing, on
the first ``CHECK_REQUESTS`` requests.

FLOPs and bytes are semantic logical counts, not physical Triton work.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import exact_int as _exact_int
from profiling.runners.attention._gdn_common import load_required_callable
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_MODULE = "vllm.models.kimi_k3.nvidia.ops.recoverssm"
_ENV = "vllm_upstream_fork_env"
_VERIFY_BACKEND = "kda_recoverssm_verify:vllm_triton"
_COMMIT_BACKEND = "kda_recoverssm_commit:vllm_triton"
# GLM-5.3-Flash `linear_attn_config.gate_lower_bound` and short conv width.
LOWER_BOUND = -5.0
CONV_KERNEL_SIZE = 4
# RMSE-ratio tolerances against the fp32 Torch oracles. `out` is bf16, so its
# bound covers one bf16 rounding; the fp32 records and states only differ by
# summation order.
OUTPUT_RMSE_RATIO_TOL = 4e-3
CORRECTION_RMSE_RATIO_TOL = 1e-3
STATE_RMSE_RATIO_TOL = 1e-3
# A bf16 checkpoint rounds the committed state once on the store.
BF16_STATE_RMSE_RATIO_TOL = 4e-3
STATE_DTYPES = (DType.FP32, DType.BF16)


# --------------------------------------------------------------------------
# Shapes and validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KdaRecoverSSMShape:
    batch_size: int
    num_heads: int
    head_dim: int
    spec_query_len: int
    dtype: DType
    state_dtype: DType = DType.FP32
    commit_len: int = 0
    num_layers: int = 1

    @property
    def tokens(self) -> int:
        return self.batch_size * self.spec_query_len

    @property
    def projection(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def projected_width(self) -> int:
        """Merged in_proj width: q, k, v, beta (H), f_a (K), g_a (K)."""
        return 3 * self.projection + self.num_heads + 2 * self.head_dim

    @property
    def conv_dim(self) -> int:
        return 3 * self.projection

    @property
    def conv_state_len(self) -> int:
        """``conv_kernel_size - 1 + num_spec`` (``kda_state_shape``)."""
        return CONV_KERNEL_SIZE - 1 + self.spec_query_len - 1

    @property
    def conv_history_len(self) -> int:
        return self.conv_state_len - self.spec_query_len + 1


def _validate_common(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> KdaRecoverSSMShape:
    shape = KdaRecoverSSMShape(
        batch_size=_exact_int("batch_size", batch_size),
        num_heads=_exact_int("num_heads", num_heads),
        head_dim=_exact_int("head_dim", head_dim),
        spec_query_len=_exact_int("spec_query_len", spec_query_len),
        dtype=DType.from_value(dtype),
        state_dtype=DType.from_value(state_dtype),
    )
    if shape.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if shape.num_heads < 1:
        raise ValueError("num_heads must be >= 1")
    # BK = next_power_of_2(K) and BV = min(next_power_of_2(V), 32) mask any
    # tail, so any positive head dim launches.
    if shape.head_dim < 1:
        raise ValueError("head_dim must be >= 1")
    # A RecoverSSM window is the target token plus at least one draft.
    if shape.spec_query_len < 2:
        raise ValueError("spec_query_len must be >= 2 (one target token plus drafts)")
    # The kg record must match the activation dtype; the fork serves bf16.
    if shape.dtype is not DType.BF16:
        raise ValueError(f"KDA RecoverSSM requires dtype=bf16, got {shape.dtype}")
    if shape.state_dtype not in STATE_DTYPES:
        raise ValueError(
            f"KDA RecoverSSM state_dtype must be fp32 or bf16, got {shape.state_dtype}"
        )
    return shape


def validate_verify_args(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> KdaRecoverSSMShape:
    return _validate_common(batch_size, num_heads, head_dim, spec_query_len, dtype, state_dtype)


def validate_commit_args(
    batch_size: int,
    commit_len: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    num_layers: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> KdaRecoverSSMShape:
    base = _validate_common(batch_size, num_heads, head_dim, spec_query_len, dtype, state_dtype)
    commit_len = _exact_int("commit_len", commit_len)
    num_layers = _exact_int("num_layers", num_layers)
    # Spec decode always accepts the target token, and the plan kernel clamps
    # to the window: 1..spec_query_len are the reachable lengths.
    if not 1 <= commit_len <= base.spec_query_len:
        raise ValueError(f"commit_len must be in 1..spec_query_len, got {commit_len}")
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    return KdaRecoverSSMShape(
        batch_size=base.batch_size,
        num_heads=base.num_heads,
        head_dim=base.head_dim,
        spec_query_len=base.spec_query_len,
        dtype=base.dtype,
        state_dtype=base.state_dtype,
        commit_len=commit_len,
        num_layers=num_layers,
    )


def state_slots(batch_size: int) -> tuple[int, ...]:
    """Distinct non-NULL slots in a fixed scattered order (slot 0 is NULL)."""
    stride = 7 if batch_size % 7 else 1
    return tuple(1 + (i * stride) % batch_size for i in range(batch_size))


# --------------------------------------------------------------------------
# Operands
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerPage:
    """One KDA layer's cache page views, in ``get_state_shape`` order."""

    raw: Any
    conv: Any  # [blocks, state_len, conv_dim] bf16 (SD layout)
    checkpoint: Any  # [blocks, H, V, K] state_dtype
    correction: Any  # [blocks, H, S, V] fp32
    kg: Any  # [blocks, H, S, 2K] bf16
    a_log: Any  # [H] fp32
    dt_bias: Any  # [H*K] fp32


def _layer_page(torch: Any, shape: KdaRecoverSSMShape, *, device: Any, generator: Any) -> LayerPage:
    heads, dim, spec = shape.num_heads, shape.head_dim, shape.spec_query_len
    blocks = shape.batch_size + 1
    parts = (
        ((shape.conv_state_len, shape.conv_dim), torch.bfloat16),
        ((heads, dim, dim), state_torch_dtype(torch, shape.state_dtype)),
        ((heads, spec, dim), torch.float32),
        ((heads, spec, 2 * dim), torch.bfloat16),
    )
    sizes = [_numel(dims) * _itemsize(torch, dtype) for dims, dtype in parts]
    page_bytes = sum(sizes)
    raw = torch.empty(blocks * page_bytes, dtype=torch.uint8, device=device)
    views, offset = [], 0
    for (dims, dtype), size in zip(parts, sizes, strict=True):
        item = _itemsize(torch, dtype)
        inner = []
        stride = 1
        for extent in reversed(dims):
            inner.insert(0, stride)
            stride *= extent
        views.append(
            torch.as_strided(
                raw.view(dtype),
                (blocks, *dims),
                (page_bytes // item, *inner),
                offset // item,
            )
        )
        offset += size
    conv, checkpoint, correction, kg = views
    for view in views:
        view.copy_(torch.randn(view.shape, generator=generator, device=device, dtype=torch.float32))
    a_log = (
        torch.empty(heads, dtype=torch.float32, device=device)
        .uniform_(1.0, 16.0, generator=generator)
        .log()
    )
    dt_bias = (
        torch.randn(heads * dim, generator=generator, device=device, dtype=torch.float32) * 0.1
    )
    return LayerPage(raw, conv, checkpoint, correction, kg, a_log, dt_bias)


def state_torch_dtype(torch: Any, state_dtype: DType) -> Any:
    return torch.bfloat16 if state_dtype is DType.BF16 else torch.float32


def _numel(dims: tuple[int, ...]) -> int:
    out = 1
    for extent in dims:
        out *= extent
    return out


def _itemsize(torch: Any, dtype: Any) -> int:
    return torch.empty((), dtype=dtype).element_size()


@dataclass(frozen=True)
class VerifyOperands:
    conv_out: Any
    projected: Any
    q: Any
    k: Any
    v: Any
    raw_g: Any
    raw_beta: Any
    page: LayerPage
    query_start_loc: Any
    state_indices: Any
    out: Any


def build_verify_operands(torch: Any, shape: KdaRecoverSSMShape, *, device: Any) -> VerifyOperands:
    tokens, heads, dim, projection = shape.tokens, shape.num_heads, shape.head_dim, shape.projection
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(*size: int) -> Any:
        return torch.randn(*size, generator=generator, dtype=torch.float32, device=device).to(
            torch.bfloat16
        )

    conv_out = normal(tokens, shape.conv_dim)
    q, k, v = (part.view(1, tokens, heads, dim) for part in conv_out.split(projection, dim=-1))
    projected = normal(tokens, shape.projected_width)
    raw_beta = projected[:, 3 * projection : 3 * projection + heads].unsqueeze(0)
    raw_g = normal(tokens, projection).view(1, tokens, heads, dim)
    page = _layer_page(torch, shape, device=device, generator=generator)
    query_start_loc = (
        torch.arange(shape.batch_size + 1, dtype=torch.int32, device=device) * shape.spec_query_len
    )
    state_indices = torch.tensor(state_slots(shape.batch_size), dtype=torch.int32, device=device)
    out = torch.empty(1, tokens, heads, dim, dtype=torch.bfloat16, device=device)
    return VerifyOperands(
        conv_out=conv_out,
        projected=projected,
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        page=page,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        out=out,
    )


def invoke_verify(callable_: Any, operands: VerifyOperands, spec_query_len: int) -> Any:
    """The exact keyword set Kimi-K3 ``_forward`` passes on the RecoverSSM branch."""
    page = operands.page
    return callable_(
        q=operands.q,
        k=operands.k,
        v=operands.v,
        raw_g=operands.raw_g,
        raw_beta=operands.raw_beta,
        A_log=page.a_log,
        dt_bias=page.dt_bias,
        lower_bound=LOWER_BOUND,
        checkpoint_state=page.checkpoint,
        correction_cache=page.correction,
        kg_cache=page.kg,
        query_start_loc=operands.query_start_loc,
        state_indices=operands.state_indices,
        spec_query_len=spec_query_len,
        out=operands.out,
    )


@dataclass(frozen=True)
class CommitOperands:
    pages: tuple[LayerPage, ...]
    num_accepted_tokens: Any
    state_indices: Any
    query_start_loc: Any
    # The fork's commit context over ``pages``; None for a backend that brings
    # its own tables.
    context: Any = None


def commit_layers(pages: tuple[LayerPage, ...], shape: KdaRecoverSSMShape) -> list[Any]:
    """The per-layer attributes ``KDARecoverSSMCommitContext.create`` reads."""
    return [
        SimpleNamespace(
            kv_cache=(page.conv, page.checkpoint, page.correction, page.kg),
            A_log=page.a_log,
            dt_bias=page.dt_bias,
            local_num_heads=shape.num_heads,
            head_dim=shape.head_dim,
            gate_lower_bound=LOWER_BOUND,
        )
        for page in pages
    ]


def build_commit_operands(
    torch: Any, context_cls: Any, shape: KdaRecoverSSMShape, *, device: Any
) -> CommitOperands:
    operands = build_commit_pages(torch, shape, device=device)
    context = context_cls.create(
        commit_layers(operands.pages, shape),
        spec_query_len=shape.spec_query_len,
        max_num_reqs=shape.batch_size,
    )
    return replace(operands, context=context)


def build_commit_pages(torch: Any, shape: KdaRecoverSSMShape, *, device: Any) -> CommitOperands:
    """Every layer's page and the commit metadata, without the fork's context."""
    generator = torch.Generator(device=device).manual_seed(42)
    pages = tuple(
        _layer_page(torch, shape, device=device, generator=generator)
        for _ in range(shape.num_layers)
    )
    batch = shape.batch_size
    return CommitOperands(
        pages=pages,
        num_accepted_tokens=torch.full(
            (batch,), shape.commit_len, dtype=torch.int32, device=device
        ),
        state_indices=torch.tensor(state_slots(batch), dtype=torch.int32, device=device),
        query_start_loc=torch.arange(batch + 1, dtype=torch.int32, device=device)
        * shape.spec_query_len,
    )


def invoke_commit(operands: CommitOperands) -> None:
    """``KimiK3KDAMetadata.commit_recoverssm_state`` without align metadata."""
    operands.context.commit(
        operands.num_accepted_tokens,
        operands.state_indices,
        operands.query_start_loc,
        request_indices=None,
        block_table=None,
        num_computed_tokens=None,
        mamba_block_size=None,
    )


# --------------------------------------------------------------------------
# Correctness (outside timing)
# --------------------------------------------------------------------------


CHECK_REQUESTS = 16
"""Requests checked against the oracle. Requests are independent in both
kernels, so a prefix of the batch is a complete check and keeps the Python
oracle cheap at batch 512 and many layers."""


def check_verify(
    torch: Any,
    callable_: Any,
    operands: VerifyOperands,
    shape: KdaRecoverSSMShape,
    *,
    invoke: Any = None,
    output_tol: float = OUTPUT_RMSE_RATIO_TOL,
    correction_tol: float = CORRECTION_RMSE_RATIO_TOL,
) -> None:
    """Compare one call against the oracle on the first requests. The
    records it writes are rewritten by every timed call, so nothing is restored.
    ``invoke`` and the tolerances let another backend of the same kind reuse
    the check with its own call and its own numerics."""
    from profiling.runners.attention.kda_chunk_prefill_reference import rmse_ratio
    from profiling.runners.attention.kda_recoverssm_reference import (
        kda_recoverssm_verify_reference,
    )

    page = operands.page
    n = min(shape.batch_size, CHECK_REQUESTS)
    tokens = slice(0, n * shape.spec_query_len)
    slots = operands.state_indices[:n].long()
    expected_o, expected_corr, expected_kg = kda_recoverssm_verify_reference(
        operands.q[0, tokens],
        operands.k[0, tokens],
        operands.v[0, tokens],
        operands.raw_g[0, tokens],
        operands.raw_beta[0, tokens],
        page.a_log,
        page.dt_bias,
        page.checkpoint,
        operands.state_indices[:n],
        shape.spec_query_len,
        lower_bound=LOWER_BOUND,
    )
    checkpoint_before = page.checkpoint[slots].clone()
    output = (invoke or invoke_verify)(callable_, operands, shape.spec_query_len)
    torch.cuda.synchronize()
    if output is not operands.out:
        raise AssertionError("kda_recoverssm_verify must write the given out")
    if not torch.equal(page.checkpoint[slots], checkpoint_before):
        raise AssertionError("kda_recoverssm_verify wrote the checkpoint")
    if not torch.isfinite(output).all():
        raise AssertionError("non-finite verify output")
    o_err = rmse_ratio(expected_o, output[0, tokens])
    corr_err = rmse_ratio(expected_corr, page.correction[slots])
    if not torch.equal(page.kg[slots], expected_kg):
        raise AssertionError("key/gate record mismatch")
    if not (o_err < output_tol and corr_err < correction_tol):
        raise AssertionError(
            f"verify mismatch vs oracle: o rmse ratio {o_err:.2e} "
            f"(tol {output_tol:.0e}), correction {corr_err:.2e} "
            f"(tol {correction_tol:.0e})"
        )


def check_commit(
    torch: Any, operands: CommitOperands, shape: KdaRecoverSSMShape, *, invoke: Any = None
) -> None:
    """Compare one call against the oracles on the first requests of every
    layer. Timed calls keep advancing the same states, as production does, so
    the checked call is not undone. ``invoke`` lets another backend of the
    kind run its own call over the same pages."""
    from profiling.runners.attention.kda_chunk_prefill_reference import rmse_ratio
    from profiling.runners.attention.kda_recoverssm_reference import (
        compact_conv_state_reference,
        kda_recoverssm_commit_reference,
    )

    n = min(shape.batch_size, CHECK_REQUESTS)
    slots = operands.state_indices[:n].long()
    commit_lens = operands.num_accepted_tokens[:n].cpu()
    null_states = [page.checkpoint[0].clone() for page in operands.pages]
    expected = []
    for page in operands.pages:
        state = kda_recoverssm_commit_reference(
            page.checkpoint[slots],
            page.correction[slots],
            page.kg[slots],
            page.a_log,
            page.dt_bias,
            commit_lens,
            lower_bound=LOWER_BOUND,
        )
        conv = compact_conv_state_reference(
            page.conv[slots].transpose(-1, -2), commit_lens, shape.conv_history_len
        )
        expected.append((state, conv))
    if invoke is None:
        invoke_commit(operands)
    else:
        invoke()
    torch.cuda.synchronize()
    for layer, (page, (state, conv)) in enumerate(zip(operands.pages, expected, strict=True)):
        actual = page.checkpoint[slots]
        if not torch.isfinite(actual).all():
            raise AssertionError(f"non-finite committed state in layer {layer}")
        err = rmse_ratio(state, actual)
        tol = BF16_STATE_RMSE_RATIO_TOL if shape.state_dtype is DType.BF16 else STATE_RMSE_RATIO_TOL
        if err >= tol:
            raise AssertionError(
                f"commit state mismatch in layer {layer}: rmse ratio {err:.2e} (tol {tol:.0e})"
            )
        if not torch.equal(page.conv[slots].transpose(-1, -2), conv):
            raise AssertionError(f"conv compaction mismatch in layer {layer}")
        if not torch.equal(page.checkpoint[0], null_states[layer]):
            raise AssertionError(f"NULL slot 0 was written in layer {layer}")


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def _import_torch(backend: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {backend}") from exc
    return torch


def _metrics(time_ms: float, energy_j: float, flops: float, nbytes: float) -> ComputeMetrics:
    elapsed = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed / 1e12 if elapsed > 0 else 0.0,
        memory_bandwidth_gbps=nbytes / elapsed / 1e9 if elapsed > 0 else 0.0,
        energy_j=float(energy_j),
    )


def profile_kda_recoverssm_verify_vllm_triton(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    """Profile one GLM-5.3-Flash KDA RecoverSSM verify call."""
    shape = validate_verify_args(
        batch_size, num_heads, head_dim, spec_query_len, dtype, state_dtype
    )
    torch = _import_torch(_VERIFY_BACKEND)
    try:
        callable_ = load_required_callable(
            importlib.import_module,
            backend=_VERIFY_BACKEND,
            module_name=_MODULE,
            callable_name="kda_recoverssm_verify",
            environment_label=_ENV,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        operands = build_verify_operands(torch, shape, device=device)
        check_verify(torch, callable_, operands, shape)

        def kernel() -> Any:
            return invoke_verify(callable_, operands, shape.spec_query_len)

        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=None)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_VERIFY_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_VERIFY_BACKEND} failed: {exc}") from exc
    return _metrics(time_ms, energy_j, verify_flops(shape), verify_bytes(shape))


def profile_kda_recoverssm_commit_vllm_triton(
    batch_size: int,
    commit_len: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    num_layers: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    """Profile one KDA RecoverSSM accepted-state commit over ``num_layers`` layers."""
    shape = validate_commit_args(
        batch_size,
        commit_len,
        num_heads,
        head_dim,
        spec_query_len,
        num_layers,
        dtype,
        state_dtype,
    )
    torch = _import_torch(_COMMIT_BACKEND)
    try:
        context_cls = load_required_callable(
            importlib.import_module,
            backend=_COMMIT_BACKEND,
            module_name=_MODULE,
            callable_name="KDARecoverSSMCommitContext",
            environment_label=_ENV,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        operands = build_commit_operands(torch, context_cls, shape, device=device)
        check_commit(torch, operands, shape)

        def kernel() -> None:
            invoke_commit(operands)

        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=None)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_COMMIT_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_COMMIT_BACKEND} failed: {exc}") from exc
    return _metrics(time_ms, energy_j, commit_flops(shape), commit_bytes(shape))


# --------------------------------------------------------------------------
# Logical work
# --------------------------------------------------------------------------


def verify_flops(shape: KdaRecoverSSMShape) -> float:
    """Per token and head: the K x V decay, the k^T.S read, the rank-1 update
    and the q.S read-out (1 + 3 * 2 = 7 K*V), as the plain decode counts it."""
    return float(shape.tokens * shape.num_heads * 7 * shape.head_dim * shape.head_dim)


def verify_bytes(shape: KdaRecoverSSMShape) -> float:
    """q/k/v/raw_g and raw_beta in, out and the two records out, the
    checkpoint read once per request."""
    elements = shape.tokens * shape.projection
    checkpoint = (
        shape.batch_size * shape.num_heads * shape.head_dim * shape.head_dim
    ) * state_bytes(shape)
    records = shape.tokens * shape.num_heads * shape.head_dim * (4 + 2 * 2)
    return float(
        4 * elements * 2 + shape.tokens * shape.num_heads * 2 + elements * 2 + records + checkpoint
    )


def state_bytes(shape: KdaRecoverSSMShape) -> int:
    return 2 if shape.state_dtype is DType.BF16 else 4


def commit_flops(shape: KdaRecoverSSMShape) -> float:
    """Per accepted token and head: the rank-1 update scaled by the running
    decay and its accumulation (3 K*V), plus the final K x V decay and add."""
    kv = shape.head_dim * shape.head_dim
    per_head = (3 * shape.commit_len + 2) * kv
    return float(shape.num_layers * shape.batch_size * shape.num_heads * per_head)


def commit_bytes(shape: KdaRecoverSSMShape) -> float:
    """Per layer: the checkpoint read and written once, the accepted records
    read, and the conv history read and written."""
    heads, dim = shape.num_heads, shape.head_dim
    state = heads * dim * dim * state_bytes(shape) * 2
    records = shape.commit_len * heads * dim * (4 + 2 * 2)
    conv = shape.conv_dim * shape.conv_history_len * 2 * 2
    return float(shape.num_layers * shape.batch_size * (state + records + conv))


__all__ = [
    "KdaRecoverSSMShape",
    "CHECK_REQUESTS",
    "build_commit_operands",
    "build_commit_pages",
    "build_verify_operands",
    "check_commit",
    "check_verify",
    "commit_bytes",
    "commit_flops",
    "invoke_commit",
    "invoke_verify",
    "profile_kda_recoverssm_commit_vllm_triton",
    "profile_kda_recoverssm_verify_vllm_triton",
    "state_bytes",
    "state_slots",
    "state_torch_dtype",
    "validate_commit_args",
    "validate_verify_args",
    "verify_bytes",
    "verify_flops",
]
