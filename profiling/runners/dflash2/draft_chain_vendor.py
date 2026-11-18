# ruff: noqa -- vendored kernel source, kept byte-identical below the header
"""Triton kernels of the DFlash2 draft chain that are not in the vLLM fork.

Needed by ``profiling.runners.dflash2.draft_chain``. In-house kernels with no
upstream source; the kernel bodies are unchanged from their development copy.

- ``qk_norm_rope_fp8``: copy of the fork's Beam ``_qk_norm_rope_kernel`` extended
  with an fp8 query output (static q scale) and an fp8 K/V paged-cache write
  (static k/v scales); replaces q_norm, k_norm, RoPE, the query QuantFP8 and
  reshape_and_cache_flash. Not bit-exact: about 95% of fp8 codes equal the
  incumbent's, the rest differ by rounding near zero.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _qk_norm_rope_fp8_kernel(
    QKV, QW, KW, POS, ROPE, OQ, KV_CACHE, SLOTS, QS, KS, VS, M,
    QKV_STRIDE: tl.constexpr, CACHE_BLOCK_STRIDE: tl.constexpr, CACHE_HEAD_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr, PAGE_SIZE: tl.constexpr, TOKENS: tl.constexpr,
    Q_HEADS: tl.constexpr, K_HEADS: tl.constexpr, DIM: tl.constexpr, EPS: tl.constexpr,
    FP8_MAX: tl.constexpr, RND: tl.constexpr = 3,
):
    token = tl.program_id(0).to(tl.int64) * TOKENS + tl.arange(0, TOKENS)
    head = tl.program_id(1)
    c = tl.arange(0, DIM)
    mask = (token[:, None] < M) & (c[None, :] < DIM)
    is_q = head < Q_HEADS
    col0 = tl.where(is_q, head * DIM, Q_HEADS * DIM + (head - Q_HEADS) * DIM)
    x = tl.load(QKV + token[:, None] * QKV_STRIDE + col0 + c[None, :], mask, 0).to(tl.float32)
    squares = (x * x).reshape((TOKENS, 4, 32))
    total = tl.sum(tl.sum(squares, 2), 1)
    y = x * tl.rsqrt(total / DIM + EPS)[:, None]
    w = tl.where(is_q, tl.load(QW + c).to(tl.float32), tl.load(KW + c).to(tl.float32))
    if RND & 1:
        y = y.to(QKV.dtype.element_ty).to(tl.float32)
    y = y * w[None, :]
    if RND & 2:
        y = y.to(QKV.dtype.element_ty).to(tl.float32)
    half: tl.constexpr = DIM // 2
    partner = tl.where(c < half, c + half, c - half)
    second = c >= half
    offset = c % half
    other = tl.gather(y, tl.broadcast_to(partner[None, :], (TOKENS, DIM)), axis=1)
    pos = tl.load(POS + token, token < M, 0).to(tl.int64)
    cos = tl.load(ROPE + pos[:, None] * DIM + offset[None, :], mask, 0).to(tl.float32)
    sin = tl.load(ROPE + pos[:, None] * DIM + half + offset[None, :], mask, 0).to(tl.float32)
    y = tl.where(second[None, :], other * sin + y * cos, y * cos - other * sin)
    if RND & 4:
        y = y.to(QKV.dtype.element_ty).to(tl.float32)
    if is_q:
        qs = tl.load(QS)
        qv = tl.minimum(tl.maximum(y / qs, -FP8_MAX), FP8_MAX)
        tl.store(OQ + (token[:, None] * Q_HEADS + head) * DIM + c[None, :], qv.to(OQ.dtype.element_ty), mask)
    else:
        h = head - Q_HEADS
        slot = tl.load(SLOTS + token, token < M, -1).to(tl.int64)
        cmask = mask & (slot[:, None] >= 0)
        address = (slot[:, None] // PAGE_SIZE * CACHE_BLOCK_STRIDE + slot[:, None] % PAGE_SIZE * CACHE_TOKEN_STRIDE
                   + h * CACHE_HEAD_STRIDE + c[None, :])
        ks = tl.load(KS)
        kv = tl.minimum(tl.maximum(y / ks, -FP8_MAX), FP8_MAX)
        tl.store(KV_CACHE + address, kv.to(KV_CACHE.dtype.element_ty), cmask)
        v = tl.load(QKV + token[:, None] * QKV_STRIDE + (Q_HEADS + K_HEADS) * DIM + h * DIM + c[None, :], cmask, 0).to(tl.float32)
        vs = tl.load(VS)
        vv = tl.minimum(tl.maximum(v / vs, -FP8_MAX), FP8_MAX)
        tl.store(KV_CACHE + address + DIM, vv.to(KV_CACHE.dtype.element_ty), cmask)


def qk_norm_rope_fp8(qkv, q_weight, k_weight, eps, positions, cos_sin, kv_cache_fp8, slots, q_scale, k_scale,
                     v_scale, q_heads, k_heads, head_dim, out_q=None, tokens=4, warps=4, rnd=7):
    """q -> fp8 [M, q_heads, hd]; K (normed, roped) and V -> fp8 cache [blocks, kv_heads, page, 2*hd]."""
    M = qkv.shape[0]
    if out_q is None:
        out_q = torch.empty(M, q_heads, head_dim, dtype=torch.float8_e4m3fn, device=qkv.device)
    _qk_norm_rope_fp8_kernel[(triton.cdiv(M, tokens), q_heads + k_heads)](
        qkv, q_weight, k_weight, positions, cos_sin, out_q, kv_cache_fp8, slots, q_scale, k_scale, v_scale, M,
        qkv.stride(0), kv_cache_fp8.stride(0), kv_cache_fp8.stride(1), kv_cache_fp8.stride(2), kv_cache_fp8.shape[2],
        tokens, q_heads, k_heads, head_dim, eps, 448.0, rnd, num_warps=warps, enable_fp_fusion=False)
    return out_q
