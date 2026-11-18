"""vLLM-fork CuTe runners for the KDA RecoverSSM conv + verify and commit kinds.

These runners only allocate tensors, time one public call, and return
metrics. DB writes, subprocess selection and registry routing live in L1b.

Both time what the fork's RecoverSSM KDA layer (``vllm.models.kimi_k3.nvidia``)
selects on SM10x when ``can_use_cute_recoverssm`` holds (``kda.py``
``recoverssm_backend = "cute"``), as the fork lays out its operands at
5629b26: a layer's page holds the conv window and the checkpoint (``kv_cache[:2]``,
bf16), and the verify records live outside the KV cache, one row per request of
the speculative batch (``KDARecoverSSMRecords``).

``kda_recoverssm_conv_verify``. One ``CuteRecoverSSMVerify.__call__`` made as
``_recoverssm_verify`` makes it: ``mixed_qkv`` is the ``[T, 3HK]`` column slice
of the merged ``q|k|v|beta|f_a|g_a`` projection (row-strided), ``raw_beta`` its
``[T, H]`` slice, ``raw_gate`` the contiguous ``f_b_proj`` output, ``out`` the
``[T, HV]`` layer output. Every request verifies a full window; request ``b``
accepted ``b % S + 1`` tokens last step, which places its conv history. The conv
page is ``[state_len, dim]`` (backend ``vllm_cute``, the fork's default layout,
``cutedsl/verify.py``) or ``[dim, state_len]`` (``vllm_cute_dim_first``,
``VLLM_SSM_CONV_STATE_LAYOUT=DS``, the warp-specialized ``verify_ws.py``). Each
call rewrites the records and the conv windows from the same inputs, so timed
calls repeat the checked one.

``kda_recoverssm_commit`` (backend ``vllm_cute``). One
``KDARecoverSSMCommitContext.commit`` over ``num_layers`` layers with the context
built as ``backend="cute"``: the Triton plan and conv compaction, then
``cute_recoverssm_commit``. The pages come from the Triton runner; each layer's
records are its page records gathered to request rows, so the Triton runner's
check applies unchanged.

Timing is the CUPTI sum over the call's launches (``kernel_name=None``).
Correctness is checked once against the Torch oracles, outside timing, on the
first ``CHECK_REQUESTS`` requests.
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
from profiling.runners.attention.kda_recoverssm_vllm_triton import (
    CHECK_REQUESTS,
    CONV_KERNEL_SIZE,
    LOWER_BOUND,
    OUTPUT_RMSE_RATIO_TOL,
    KdaRecoverSSMShape,
    _import_torch,
    _metrics,
    build_commit_pages,
    check_commit,
    commit_bytes,
    commit_flops,
    invoke_commit,
    state_slots,
    validate_commit_args,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_KERNELS = "vllm.models.kimi_k3.nvidia.recoverssm.kernels"
_OPS = "vllm.models.kimi_k3.nvidia.ops.recoverssm"
_ENV = "vllm_env"
_COMMIT_BACKEND = "kda_recoverssm_commit:vllm_cute"
_HEAD_DIM = 128  # the CuTe kernels' K = V
_MAX_WINDOW = 8
# Tensor-core MMAs round their operands to bf16; the records keep fp32.
CORRECTION_RMSE_RATIO_TOL = 4e-3
KG_RMSE_RATIO_TOL = 1e-4
CONV_WEIGHT_DTYPES = (DType.BF16, DType.FP32)


@dataclass(frozen=True)
class ConvVerifyShape:
    batch_size: int
    num_heads: int
    head_dim: int
    spec_query_len: int
    conv_weight_dtype: DType
    dim_first: bool

    @property
    def tokens(self) -> int:
        return self.batch_size * self.spec_query_len

    @property
    def projection(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def conv_dim(self) -> int:
        return 3 * self.projection

    @property
    def projected_width(self) -> int:
        """Merged in_proj width: q, k, v, beta (H), f_a (K), g_a (K)."""
        return self.conv_dim + self.num_heads + 2 * self.head_dim

    @property
    def conv_state_len(self) -> int:
        """``conv_kernel_size - 1 + num_spec`` (``kda_state_shape``)."""
        return CONV_KERNEL_SIZE - 1 + self.spec_query_len - 1


def validate_conv_verify_args(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    dtype: DType | str,
    conv_weight_dtype: DType | str,
    *,
    dim_first: bool,
) -> ConvVerifyShape:
    shape = ConvVerifyShape(
        batch_size=_exact_int("batch_size", batch_size),
        num_heads=_exact_int("num_heads", num_heads),
        head_dim=_exact_int("head_dim", head_dim),
        spec_query_len=_exact_int("spec_query_len", spec_query_len),
        conv_weight_dtype=DType.from_value(conv_weight_dtype),
        dim_first=dim_first,
    )
    if shape.batch_size < 1 or shape.num_heads < 1:
        raise ValueError("batch_size and num_heads must be >= 1")
    # can_use_cute_recoverssm: head_dim 128, a 1..8-token window, bf16 activations.
    if shape.head_dim != _HEAD_DIM:
        raise ValueError(f"the CuTe verify needs head_dim {_HEAD_DIM}, got {shape.head_dim}")
    if not 2 <= shape.spec_query_len <= _MAX_WINDOW:
        raise ValueError(f"spec_query_len must be in 2..{_MAX_WINDOW}, got {shape.spec_query_len}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ValueError(f"the CuTe verify needs dtype=bf16, got {dtype}")
    if shape.conv_weight_dtype not in CONV_WEIGHT_DTYPES:
        raise ValueError(f"conv_weight_dtype must be bf16 or fp32, got {shape.conv_weight_dtype}")
    return shape


@dataclass(frozen=True)
class ConvVerifyOperands:
    projected: Any  # [T, PW] bf16 merged projection
    mixed_qkv: Any  # [T, 3HK] view
    raw_beta: Any  # [T, H] view
    raw_gate: Any  # [T, HK]
    conv_page: Any  # stored page view: [slots, L, C] or [slots, C, L]
    conv: Any  # [slots, L, C] logical view of conv_page
    state: Any  # [slots, H, V, K] bf16
    conv_weight: Any  # [C, 4]
    a_log: Any
    dt_bias: Any
    records: Any  # KDARecoverSSMRecords [B, H, S, *]
    query_start_loc: Any
    state_indices: Any
    accepted: Any
    out: Any  # [T, HV]


def build_conv_verify_operands(
    torch: Any, records_cls: Any, shape: ConvVerifyShape, *, device: Any
) -> ConvVerifyOperands:
    tokens, heads, dim = shape.tokens, shape.num_heads, shape.head_dim
    channels, rows = shape.conv_dim, shape.conv_state_len
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(*size: int, scale: float = 1.0) -> Any:
        return torch.randn(*size, generator=generator, dtype=torch.float32, device=device) * scale

    blocks = shape.batch_size + 1  # slot 0 is the NULL block
    conv_elems, state_elems = rows * channels, heads * dim * dim
    page = conv_elems + state_elems  # bf16 elements; both parts are 16-byte multiples
    raw = torch.empty(blocks * page, dtype=torch.bfloat16, device=device)
    if shape.dim_first:
        conv_page = torch.as_strided(raw, (blocks, channels, rows), (page, rows, 1), 0)
        conv = conv_page.transpose(1, 2)
    else:
        conv_page = torch.as_strided(raw, (blocks, rows, channels), (page, channels, 1), 0)
        conv = conv_page
    state = torch.as_strided(raw, (blocks, heads, dim, dim), (page, dim * dim, dim, 1), conv_elems)
    conv.copy_(normal(blocks, rows, channels, scale=0.5))
    state.copy_(normal(blocks, heads, dim, dim, scale=0.05))
    projected = normal(tokens, shape.projected_width, scale=0.5).to(torch.bfloat16)
    weight_dtype = torch.float32 if shape.conv_weight_dtype is DType.FP32 else torch.bfloat16
    a_log = (
        torch.empty(heads, dtype=torch.float32, device=device)
        .uniform_(1.0, 16.0, generator=generator)
        .log()
    )
    return ConvVerifyOperands(
        projected=projected,
        mixed_qkv=projected[:, :channels],
        raw_beta=projected[:, channels : channels + heads],
        raw_gate=normal(tokens, shape.projection).to(torch.bfloat16),
        conv_page=conv_page,
        conv=conv,
        state=state,
        conv_weight=normal(channels, CONV_KERNEL_SIZE, scale=0.3).to(weight_dtype).contiguous(),
        a_log=a_log,
        dt_bias=normal(heads * dim, scale=0.1).contiguous(),
        records=records_cls.allocate(
            shape.batch_size, heads, shape.spec_query_len, dim, torch.bfloat16, device
        ),
        query_start_loc=torch.arange(shape.batch_size + 1, dtype=torch.int32, device=device)
        * shape.spec_query_len,
        state_indices=torch.tensor(state_slots(shape.batch_size), dtype=torch.int32, device=device),
        accepted=torch.tensor(
            [b % shape.spec_query_len + 1 for b in range(shape.batch_size)],
            dtype=torch.int32,
            device=device,
        ),
        out=torch.empty(tokens, shape.projection, dtype=torch.bfloat16, device=device),
    )


def invoke_conv_verify(verify: Any, operands: ConvVerifyOperands) -> None:
    """The exact keyword set ``_recoverssm_verify`` passes on the CuTe branch."""
    ok = verify(
        mixed_qkv=operands.mixed_qkv,
        raw_gate=operands.raw_gate,
        raw_beta=operands.raw_beta,
        A_log=operands.a_log,
        dt_bias=operands.dt_bias,
        lower_bound=LOWER_BOUND,
        state=operands.state,
        records=operands.records,
        query_start_loc=operands.query_start_loc,
        state_indices=operands.state_indices,
        num_accepted_tokens=operands.accepted,
        conv_state=operands.conv_page,
        conv_weight=operands.conv_weight,
        out=operands.out,
    )
    if not ok:
        raise KernelLaunchFailed("CuteRecoverSSMVerify declined the operands (misaligned)")


def conv_window_reference(torch: Any, operands: ConvVerifyOperands, shape: ConvVerifyShape, n: int):
    """fp32 conv + SiLU of the first ``n`` requests, and each one's rolled window:
    history = the 3 rows ending at its last accepted token, then its window."""
    spec, taps = shape.spec_query_len, CONV_KERNEL_SIZE
    weight = operands.conv_weight.float()
    outputs, windows = [], []
    for b in range(n):
        slot, accepted = int(operands.state_indices[b]), int(operands.accepted[b])
        history = operands.conv[slot, accepted - 1 : accepted - 1 + taps - 1].float()
        x = operands.mixed_qkv[b * spec : (b + 1) * spec].float()
        full = torch.cat([history, x], 0)  # [3 + S, C]
        y = sum(full[j : j + spec] * weight[:, j][None] for j in range(taps))
        outputs.append(torch.nn.functional.silu(y))
        windows.append(
            torch.cat(
                [history.to(torch.bfloat16), operands.mixed_qkv[b * spec : (b + 1) * spec]], 0
            )[-shape.conv_state_len :]
        )
    return torch.cat(outputs, 0), torch.stack(windows)


def check_conv_verify(
    torch: Any,
    verify: Any,
    operands: ConvVerifyOperands,
    shape: ConvVerifyShape,
    *,
    invoke: Any = None,
) -> dict[str, float]:
    """Compare one call against the oracles on the first requests; return the
    RMSE ratios. ``invoke`` lets another backend of the kind run its own call
    over the same operands."""
    from profiling.runners.attention.kda_chunk_prefill_reference import rmse_ratio
    from profiling.runners.attention.kda_recoverssm_reference import (
        kda_recoverssm_verify_reference,
    )

    n = min(shape.batch_size, CHECK_REQUESTS)
    heads, dim, spec = shape.num_heads, shape.head_dim, shape.spec_query_len
    tokens = n * spec
    conv_out, windows = conv_window_reference(torch, operands, shape, n)
    conv_out = conv_out.to(torch.bfloat16)
    hk = shape.projection
    q, k, v = (conv_out[:, i * hk : (i + 1) * hk].reshape(tokens, heads, dim) for i in range(3))
    slots = operands.state_indices[:n].long()
    expected_o, expected_corr, expected_kg = kda_recoverssm_verify_reference(
        q,
        k,
        v,
        operands.raw_gate[:tokens].view(tokens, heads, dim),
        operands.raw_beta[:tokens],
        operands.a_log,
        operands.dt_bias,
        operands.state,
        operands.state_indices[:n],
        spec,
        lower_bound=LOWER_BOUND,
    )
    state_before = operands.state[slots].clone()
    if invoke is None:
        invoke_conv_verify(verify, operands)
    else:
        invoke()
    torch.cuda.synchronize()
    if not torch.equal(operands.state[slots], state_before):
        raise AssertionError("the conv + verify wrote the checkpoint")
    out = operands.out[:tokens].view(tokens, heads, dim)
    if not torch.isfinite(out).all():
        raise AssertionError("non-finite verify output")
    errors = {
        "out": (rmse_ratio(expected_o, out), OUTPUT_RMSE_RATIO_TOL),
        "correction": (
            rmse_ratio(expected_corr, operands.records.correction[:n]),
            CORRECTION_RMSE_RATIO_TOL,
        ),
        "key/gate": (rmse_ratio(expected_kg, operands.records.key_gate[:n]), KG_RMSE_RATIO_TOL),
    }
    bad = {
        name: f"{err:.2e} (tol {tol:.0e})" for name, (err, tol) in errors.items() if not err < tol
    }
    if bad:
        raise AssertionError(f"conv + verify mismatch vs oracle: {bad}")
    if not torch.equal(operands.conv[slots], windows):
        raise AssertionError("rolled conv window mismatch")
    return {name: err for name, (err, _) in errors.items()}


def _conv_verify(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    dtype: DType | str,
    conv_weight_dtype: DType | str,
    *,
    dim_first: bool,
) -> ComputeMetrics:
    backend = "kda_recoverssm_conv_verify:" + ("vllm_cute_dim_first" if dim_first else "vllm_cute")
    shape = validate_conv_verify_args(
        batch_size,
        num_heads,
        head_dim,
        spec_query_len,
        dtype,
        conv_weight_dtype,
        dim_first=dim_first,
    )
    torch = _import_torch(backend)
    try:
        verify_cls = load_required_callable(
            importlib.import_module,
            backend=backend,
            module_name=_KERNELS,
            callable_name="CuteRecoverSSMVerify",
            environment_label=_ENV,
        )
        records_cls = load_required_callable(
            importlib.import_module,
            backend=backend,
            module_name=_OPS,
            callable_name="KDARecoverSSMRecords",
            environment_label=_ENV,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        operands = build_conv_verify_operands(torch, records_cls, shape, device=device)
        verify = verify_cls(
            num_heads=shape.num_heads,
            window=shape.spec_query_len,
            conv_dim_first=dim_first,
            conv_weight_dtype=operands.conv_weight.dtype,
            device=device,
        )
        verify.compile()
        check_conv_verify(torch, verify, operands, shape)

        def kernel() -> None:
            invoke_conv_verify(verify, operands)

        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=None)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{backend} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{backend} failed: {exc}") from exc
    return _metrics(time_ms, energy_j, conv_verify_flops(shape), conv_verify_bytes(shape))


def profile_kda_recoverssm_conv_verify_vllm_cute(**kwargs: Any) -> ComputeMetrics:
    """The fork's CuTe conv + verify on its default ``[state_len, dim]`` conv pages."""
    return _conv_verify(**kwargs, dim_first=False)


def profile_kda_recoverssm_conv_verify_vllm_cute_dim_first(**kwargs: Any) -> ComputeMetrics:
    """The fork's CuTe conv + verify on ``[dim, state_len]`` conv pages (verify_ws)."""
    return _conv_verify(**kwargs, dim_first=True)


# --------------------------------------------------------------------------
# Commit (backend vllm_cute)
# --------------------------------------------------------------------------


def build_cute_commit_operands(
    torch: Any, context_cls: Any, shape: KdaRecoverSSMShape, *, device: Any
):
    """The Triton runner's pages, with each layer's records gathered to request
    rows (the fork keeps them outside the KV cache, by request)."""
    operands = build_commit_pages(torch, shape, device=device)
    slots = operands.state_indices.long()
    layers = [
        SimpleNamespace(
            kv_cache=(page.conv, page.checkpoint),
            recoverssm_records=SimpleNamespace(
                correction=page.correction[slots].contiguous(),
                key_gate=page.kg[slots].contiguous(),
            ),
            A_log=page.a_log,
            dt_bias=page.dt_bias,
            local_num_heads=shape.num_heads,
            head_dim=shape.head_dim,
            gate_lower_bound=LOWER_BOUND,
        )
        for page in operands.pages
    ]
    context = context_cls.create(
        layers,
        spec_query_len=shape.spec_query_len,
        max_num_reqs=shape.batch_size,
        backend="cute",
    )
    return replace(operands, context=context)


def profile_kda_recoverssm_commit_vllm_cute(
    batch_size: int,
    commit_len: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    num_layers: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    """Profile one CuTe-fold RecoverSSM commit over ``num_layers`` layers."""
    shape = validate_commit_args(
        batch_size, commit_len, num_heads, head_dim, spec_query_len, num_layers, dtype, state_dtype
    )
    if shape.state_dtype is not DType.BF16 or shape.head_dim != _HEAD_DIM:
        raise ValueError("the CuTe commit folds bf16 checkpoints with head_dim 128 only")
    torch = _import_torch(_COMMIT_BACKEND)
    try:
        context_cls = load_required_callable(
            importlib.import_module,
            backend=_COMMIT_BACKEND,
            module_name=_OPS,
            callable_name="KDARecoverSSMCommitContext",
            environment_label=_ENV,
        )
        compile_state_commit = load_required_callable(
            importlib.import_module,
            backend=_COMMIT_BACKEND,
            module_name=_KERNELS,
            callable_name="compile_state_commit",
            environment_label=_ENV,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        compile_state_commit(shape.num_heads, shape.spec_query_len, device, align=False)
        operands = build_cute_commit_operands(torch, context_cls, shape, device=device)
        # The fold reads the request-row records: the page records the check
        # reads hold the same values.
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


def conv_verify_flops(shape: ConvVerifyShape) -> float:
    """The verify's 7 K*V per token and head, plus 4 taps per conv channel."""
    kv = shape.head_dim * shape.head_dim
    return float(
        shape.tokens * shape.num_heads * 7 * kv
        + 2 * CONV_KERNEL_SIZE * shape.tokens * shape.conv_dim
    )


def conv_verify_bytes(shape: ConvVerifyShape) -> float:
    """Raw q/k/v, gate and beta in; out and the records out; the bf16
    checkpoint read once; the conv history in and the rolled window out."""
    tokens, hk = shape.tokens, shape.projection
    activations = 2 * tokens * (shape.conv_dim + hk + shape.num_heads + hk)
    records = tokens * hk * (4 + 2 * 2)
    checkpoint = 2 * shape.batch_size * shape.num_heads * shape.head_dim * shape.head_dim
    conv = 2 * shape.batch_size * shape.conv_dim * (CONV_KERNEL_SIZE - 1 + shape.conv_state_len)
    return float(activations + records + checkpoint + conv)


__all__ = [
    "ConvVerifyShape",
    "build_conv_verify_operands",
    "build_cute_commit_operands",
    "check_conv_verify",
    "conv_verify_bytes",
    "conv_verify_flops",
    "conv_window_reference",
    "invoke_conv_verify",
    "profile_kda_recoverssm_commit_vllm_cute",
    "profile_kda_recoverssm_conv_verify_vllm_cute",
    "profile_kda_recoverssm_conv_verify_vllm_cute_dim_first",
    "validate_conv_verify_args",
]
