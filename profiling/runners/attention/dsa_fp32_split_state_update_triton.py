"""Profile the vendored Triton DSA state update from an fp32 split row (backend ``triton``).

The kernel (``state_writer_vendor/triton_post_plus.py``, ``post_plus``) reads
the fp32 output of the DSA layer's hidden-state GEMM ([q latent | kv latent |
indexer k | gate score | fp32 head weights]) and in one launch replaces five
fork launches: the row-split epilogue (``post_a``), ``fused_q_kv_rmsnorm``,
``concat_and_cache_mla`` (fp8), the positions int64 -> int32 cast and
``kpool_decode_update_and_maybe_write_cache_batched``.

Before timing, the launch is compared byte for byte with that incumbent chain
run on the same operands (``check_against_incumbent``). The fork ops are the
image's ``vllm.models.common.ops.fused_q_kv_rmsnorm`` and
``vllm.models.glm5next.nvidia.ops.kpool_compress`` (md5-identical to the fork
sources).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.dsa_pooled_index_cache_update import (
    _Shape as _PoolShape,
)
from profiling.runners.attention.dsa_pooled_index_cache_update import (
    build_operands as build_pool_operands,
)
from profiling.runners.attention.dsa_pooled_index_cache_update import (
    completed_pools,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "dsa_fp32_split_state_update:triton"
# (q_lora_cols, kv_lora_cols, index_dim) compiled into _post_plus_kernel.
_COMPILED_GEOMETRY = (1536, 512, 128)
# Bit-identical with the fork's norm reduction tree (4 warps differs by 1 ulp in
# a few q values past ~384 rows).
NUM_WARPS = 8
_RMS_EPS = 1e-6
_LN_EPS = 1e-6
# A non-power-of-two static scale exercises the kv path's division.
_KV_SCALE = 0.37
# Spare MLA cache slots no row maps, so a stray write lands where the check sees it.
_MLA_PAGE = 64


@dataclass(frozen=True)
class _Shape:
    num_requests: int
    next_n: int
    q_lora_cols: int
    kv_lora_cols: int
    index_dim: int
    fp32_tail_cols: int
    pool_size: int
    tail_ring: int
    block_size: int

    @property
    def rows(self) -> int:
        return self.num_requests * self.next_n

    @property
    def width(self) -> int:
        return self.q_lora_cols + self.kv_lora_cols + 2 * self.index_dim + self.fp32_tail_cols

    def pool(self) -> _PoolShape:
        return _PoolShape(
            self.num_requests,
            self.next_n,
            self.pool_size,
            self.tail_ring,
            self.index_dim,
            self.block_size,
        )


def validate_args(
    num_requests: int,
    next_n: int,
    q_lora_cols: int,
    kv_lora_cols: int,
    index_dim: int,
    fp32_tail_cols: int,
    pool_size: int,
    tail_ring: int,
    block_size: int,
) -> _Shape:
    shape = _Shape(
        num_requests,
        next_n,
        q_lora_cols,
        kv_lora_cols,
        index_dim,
        fp32_tail_cols,
        pool_size,
        tail_ring,
        block_size,
    )
    for name, value in vars(shape).items():
        minimum = 0 if name == "fp32_tail_cols" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if shape.tail_ring < shape.pool_size or shape.tail_ring % shape.pool_size:
        raise ValueError(
            "tail_ring must be a multiple of pool_size, "
            f"got tail_ring={shape.tail_ring}, pool_size={shape.pool_size}"
        )
    geometry = (shape.q_lora_cols, shape.kv_lora_cols, shape.index_dim)
    if geometry != _COMPILED_GEOMETRY:
        raise ProfilerNotImplemented(
            f"{_BACKEND} has the row geometry {_COMPILED_GEOMETRY} compiled in, got {geometry}"
        )
    if shape.pool_size > 4:
        raise ProfilerNotImplemented(f"{_BACKEND} pools at most 4 tokens, got {shape.pool_size}")
    return shape


def logical_bytes(shape: _Shape) -> int:
    """Row reads and writes plus the pooled-cache traffic (as dsa_pooled_index_cache_update)."""
    q, kv, d = shape.q_lora_cols, shape.kv_lora_cols, shape.index_dim
    per_row = (
        4 * (q + kv + 2 * d)  # fp32 row read (the tail is not read)
        + 2 * q  # normed q (bf16)
        + kv  # fp8 MLA cache row
        + 2 * 2 * d  # k and gate score (bf16) out
        + 2 * 2 * d  # k and gate score into the tail ring
        + 20  # MLA slot (int64), positions (int64), pool slot and tail slot (int32)
    )
    pools = completed_pools(shape.pool())
    return shape.rows * per_row + pools * ((shape.pool_size - 1) * 4 * d + d + 4)


@dataclass(frozen=True)
class _Operands:
    y: Any
    q_weight: Any
    kv_weight: Any
    k_weight: Any
    k_bias: Any
    kv_scale: Any
    mla_slots: Any
    mla_cache: Any
    pool: Any  # dsa_pooled_index_cache_update._Operands (positions int32)
    positions64: Any
    q_out: Any
    k_out: Any
    gate_out: Any


def build_operands(torch: Any, shape: _Shape, device: Any) -> _Operands:
    pool = build_pool_operands(torch, shape.pool(), device)
    g = torch.Generator(device=device).manual_seed(29)
    rows = shape.rows
    y = torch.randn((rows, shape.width), dtype=torch.float32, device=device, generator=g)

    def weight(n: int, dtype: Any) -> Any:
        return (1.0 + 0.1 * torch.randn(n, device=device, generator=g)).to(dtype)

    mla_slots_total = (rows // _MLA_PAGE + 2) * _MLA_PAGE
    mla_slots = torch.randperm(mla_slots_total, device=device, generator=g)[:rows]
    return _Operands(
        y=y,
        q_weight=weight(shape.q_lora_cols, torch.bfloat16),
        kv_weight=weight(shape.kv_lora_cols, torch.bfloat16),
        k_weight=weight(shape.index_dim, torch.float32),
        k_bias=0.1 * torch.randn(shape.index_dim, device=device, generator=g),
        kv_scale=torch.tensor([_KV_SCALE], dtype=torch.float32, device=device),
        mla_slots=mla_slots.to(torch.int64).contiguous(),
        mla_cache=torch.zeros(
            mla_slots_total // _MLA_PAGE,
            _MLA_PAGE,
            shape.kv_lora_cols,
            dtype=torch.uint8,
            device=device,
        ),
        pool=pool,
        positions64=pool.positions.reshape(-1).to(torch.int64).contiguous(),
        q_out=torch.empty((rows, shape.q_lora_cols), dtype=torch.bfloat16, device=device),
        k_out=torch.empty((rows, shape.index_dim), dtype=torch.bfloat16, device=device),
        gate_out=torch.empty((rows, shape.index_dim), dtype=torch.bfloat16, device=device),
    )


def launch(post_plus: Any, o: _Operands, shape: _Shape, num_warps: int = NUM_WARPS) -> None:
    p = o.pool
    post_plus(
        o.y,
        o.q_out,
        o.q_weight,
        o.kv_weight,
        _RMS_EPS,
        o.mla_cache,
        o.mla_slots,
        o.kv_scale,
        o.k_out,
        o.gate_out,
        o.k_weight,
        o.k_bias,
        _LN_EPS,
        p.kv_cache,
        p.tail,
        p.tail_slot_mapping,
        p.ape,
        p.slot_mapping.reshape(-1),
        o.positions64,
        shape.next_n,
        pool_size=shape.pool_size,
        mx=_mx_scales(),
        num_warps=num_warps,
    )


def _mx_scales() -> bool:
    from vllm.models.glm5next.nvidia.ops.kpool_compress import mqa_logits_use_mx_scales

    return bool(mqa_logits_use_mx_scales())


def _bytes_differing(torch: Any, a: Any, b: Any) -> int:
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum().item())


def check_against_incumbent(
    torch: Any, post_plus: Any, o: _Operands, shape: _Shape, num_warps: int = NUM_WARPS
) -> dict[str, int]:
    """Run the incumbent chain and this launch from the same state; return bytes differing.

    The chain: post_a (row split + k LayerNorm) -> fused_q_kv_rmsnorm ->
    concat_and_cache_mla(fp8, kv_scale) -> positions.to(int32) -> fork kpool
    update. The fork kpool is fed this launch's own k, so a 1-ulp k LayerNorm
    difference (reduction order) is charged to ``k_ulp_gt1`` only and the cache
    state is compared exactly.
    """
    from vllm import _custom_ops as vops
    from vllm.models.common.ops import fused_q_kv_rmsnorm
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        kpool_decode_update_and_maybe_write_cache_batched as kpool_update,
    )

    from profiling.runners.norm.fp32_row_split_layer_norm_reference import (
        NEAR_ZERO_ATOL,
        bf16_ulp_distance,
    )
    from profiling.runners.norm.row_split_vendor.triton_post_a import post_a

    p = o.pool
    rows, q, kv, d = shape.rows, shape.q_lora_cols, shape.kv_lora_cols, shape.index_dim
    mla0, idx0, tail0 = o.mla_cache.clone(), p.kv_cache.clone(), p.tail.clone()
    y0 = o.y.clone()

    launch(post_plus, o, shape, num_warps)
    torch.cuda.synchronize()
    got = dict(q=o.q_out.clone(), k=o.k_out.clone(), gate=o.gate_out.clone())
    got_mla, got_idx, got_tail = o.mla_cache.clone(), p.kv_cache.clone(), p.tail.clone()

    qkv = torch.empty((rows, q + kv), dtype=torch.bfloat16, device=o.y.device)
    k_ref = torch.empty((rows, d), dtype=torch.bfloat16, device=o.y.device)
    gate_ref = torch.empty_like(k_ref)
    post_a(o.y, qkv, k_ref, gate_ref, o.k_weight, o.k_bias, _LN_EPS)
    q_c, kv_c = qkv.split([q, kv], dim=-1)
    q_ref, kv_normed = fused_q_kv_rmsnorm(q_c, kv_c, o.q_weight, o.kv_weight, _RMS_EPS)
    mla_ref = mla0.clone()
    k_pe = torch.empty((rows, 0), dtype=torch.bfloat16, device=o.y.device)
    vops.concat_and_cache_mla(kv_normed, k_pe, mla_ref, o.mla_slots, "fp8", o.kv_scale)
    idx_ref, tail_ref = idx0.clone(), tail0.clone()
    view = (shape.num_requests, shape.next_n, d)
    kpool_update(
        idx_ref,
        tail_ref,
        p.tail_slot_mapping,
        got["k"].view(view),
        gate_ref.view(view),
        p.ape,
        p.slot_mapping,
        o.positions64.to(torch.int32).view(shape.num_requests, shape.next_n),
        shape.pool_size,
        d,
        round_scale=True,
    )
    torch.cuda.synchronize()
    ulp = bf16_ulp_distance(torch, got["k"], k_ref)
    k_off = (ulp > 1) & ((got["k"].float() - k_ref.float()).abs() > NEAR_ZERO_ATOL)
    result = dict(
        q=_bytes_differing(torch, got["q"], q_ref),
        gate=_bytes_differing(torch, got["gate"], gate_ref),
        mla_cache=_bytes_differing(torch, got_mla, mla_ref),
        index_cache=_bytes_differing(torch, got_idx, idx_ref),
        tail=_bytes_differing(torch, got_tail, tail_ref),
        k_bytes=_bytes_differing(torch, got["k"], k_ref),
        k_ulp_gt1=int(k_off.sum().item()),
        y_modified=_bytes_differing(torch, o.y, y0),
        # Page-planar cache: a changed 132-byte row is not one pool, only a write.
        index_rows_written=int((got_idx != idx0).any(dim=-1).sum().item()),
    )
    # Restore the starting state so the timed launches see the same operands.
    o.mla_cache.copy_(mla0)
    p.kv_cache.copy_(idx0)
    p.tail.copy_(tail0)
    return result


def assert_accepted(diff: dict[str, int]) -> None:
    exact = ("q", "gate", "mla_cache", "index_cache", "tail", "k_ulp_gt1", "y_modified")
    bad = {name: diff[name] for name in exact if diff[name]}
    if bad:
        raise KernelLaunchFailed(f"{_BACKEND} differs from the incumbent chain: {bad}")


def profile_dsa_fp32_split_state_update_triton(
    num_requests: int,
    next_n: int,
    q_lora_cols: int,
    kv_lora_cols: int,
    index_dim: int,
    fp32_tail_cols: int,
    pool_size: int,
    tail_ring: int,
    block_size: int,
) -> ComputeMetrics:
    shape = validate_args(
        num_requests,
        next_n,
        q_lora_cols,
        kv_lora_cols,
        index_dim,
        fp32_tail_cols,
        pool_size,
        tail_ring,
        block_size,
    )
    try:
        import torch

        from profiling.runners.attention.state_writer_vendor.triton_post_plus import post_plus
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires Torch and Triton") from exc
    try:
        import vllm.models.glm5next.nvidia.ops.kpool_compress  # noqa: F401
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} checks against the vLLM fork's ops") from exc
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        operands = build_operands(torch, shape, device)
        diff = check_against_incumbent(torch, post_plus, operands, shape)
        assert_accepted(diff)
        if diff["index_rows_written"] == 0:
            raise KernelLaunchFailed(f"{_BACKEND} wrote no pooled index-cache entry")

        def run() -> None:
            launch(post_plus, operands, shape)

        time_ms = Timer.cupti(run, warmup=3, kernel_name=None)
        energy_j = Energy.perf(run, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        float(time_ms), 0.0, logical_bytes(shape) / seconds / 1e9, float(energy_j)
    )


__all__ = [
    "assert_accepted",
    "check_against_incumbent",
    "logical_bytes",
    "profile_dsa_fp32_split_state_update_triton",
    "validate_args",
]
