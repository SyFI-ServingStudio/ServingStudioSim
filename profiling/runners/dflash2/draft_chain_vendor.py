# ruff: noqa -- vendored kernel source, kept byte-identical below the header
"""Triton kernels of the DFlash2 draft chain that are not in the vLLM fork.

Needed by ``profiling.runners.dflash2.draft_chain``. In-house kernels with no
upstream source; the kernel bodies are unchanged from their development copy.

- ``qk_norm_rope_fp8``: copy of the fork's Beam ``_qk_norm_rope_kernel`` extended
  with an fp8 query output (static q scale) and an fp8 K/V paged-cache write
  (static k/v scales); replaces q_norm, k_norm, RoPE, the query QuantFP8 and
  reshape_and_cache_flash. Not bit-exact: about 95% of fp8 codes equal the
  incumbent's, the rest differ by rounding near zero.
- ``ThresholdTopK``: exact top-K with FlashInfer's deterministic tie order
  (top_k(sorted=True, deterministic=True)), values fp32 and ids int64 (the
  fork's casts folded in); ids and values identical to FlashInfer, including
  ties and an in-kernel overflow fallback.
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


# FlashInfer's deterministic tie order (top_k(sorted=True, deterministic=True), multi-CTA radix, topk.cuh): a row
# is cut into FI_CHUNK-wide CTA chunks; inside a chunk, elements equal to the pivot are taken in thread-strided
# order (thread tx owns j = tx, tx+1024, ...), and the final stable sort keeps that order among equal values. So
# the order is (value desc, rank asc) with rank(v) = chunk(v)*FI_CHUNK + (j % 1024) * NJ + j // 1024,
# j = v % FI_CHUNK.
@triton.jit
def _fi_rank(v, FI_CHUNK: tl.constexpr, NJ: tl.constexpr):
    c = v // FI_CHUNK
    j = v - c * FI_CHUNK
    return c * (1024 * NJ) + (j % 1024) * NJ + j // 1024


# ------------------------------------------------------------------ threshold top-k (exact, FlashInfer tie order)
# A: per (row, slice) max.  B: per (row, slice): T = 16th largest slice max of the row (a lower bound of the row's
# 16th largest value: 16 slices each hold an element >= T); a slice whose max < T exits without reading; otherwise
# its elements >= T are appended to the row's candidate list (atomic counter).  C: per row, exact top-16 of the
# candidates under (value desc, FlashInfer rank asc); writes ids int64 / values fp32 and re-zeroes the counter.
@triton.jit
def _thr_a(X, SM, N, X_STRIDE, NS: tl.constexpr, SLICE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    offs = s * SLICE + tl.arange(0, SLICE)
    x = tl.load(X + row * X_STRIDE + offs, mask=offs < N, other=float("-inf")).to(tl.float32)
    tl.store(SM + row * NS + s, tl.max(x, 0))


@triton.jit
def _thr_t(SM, T, NS: tl.constexpr, NSB: tl.constexpr, K: tl.constexpr):
    # T = K-th largest slice max (with multiplicity), once per row
    row = tl.program_id(0).to(tl.int64)
    so = tl.arange(0, NSB)
    sm = tl.load(SM + row * NS + so, mask=so < NS, other=float("-inf"))
    srt = tl.sort(sm, descending=True)
    tl.store(T + row, tl.max(tl.where(so == K - 1, srt, float("-inf")), 0))


@triton.jit
def _thr_b(X, SM, T, CNT, CV, CI, N, X_STRIDE, NS: tl.constexpr, SLICE: tl.constexpr, CAP: tl.constexpr,
           ATOMIC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    mine = tl.load(SM + row * NS + s)
    t = tl.load(T + row)
    if mine >= t:
        offs = s * SLICE + tl.arange(0, SLICE)
        x = tl.load(X + row * X_STRIDE + offs, mask=offs < N, other=float("-inf")).to(tl.float32)
        keep = x >= t
        if ATOMIC:
            # few elements pass: one masked atomic per kept element gives its slot, no block-wide scan
            pos = tl.atomic_add(CNT + row + offs * 0, 1, mask=keep, sem="relaxed")
        else:
            n = tl.sum(keep.to(tl.int32), 0)
            base = tl.atomic_add(CNT + row, n)
            pos = base + tl.cumsum(keep.to(tl.int32), 0) - 1
        ok = keep & (pos < CAP)
        tl.store(CV + row * CAP + pos, x, mask=ok)
        tl.store(CI + row * CAP + pos, offs, mask=ok)


@triton.jit
def _thr_c(CNT, CV, CI, OV, OI, OVF, X, N, X_STRIDE, CAP: tl.constexpr, K: tl.constexpr, FI_CHUNK: tl.constexpr,
           NJ: tl.constexpr, FB: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    n = tl.load(CNT + row)
    offs = tl.arange(0, CAP)
    m = offs < n
    x = tl.load(CV + row * CAP + offs, mask=m, other=float("-inf"))
    ids = tl.load(CI + row * CAP + offs, mask=m, other=0)
    big = 2147483647
    rank = tl.where(m, _fi_rank(ids, FI_CHUNK, NJ), big)
    kk = tl.arange(0, K)
    outv = tl.full([K], float("-inf"), tl.float32)
    outi = tl.full([K], 0, tl.int32)
    for j in tl.static_range(K):
        mx = tl.max(x, 0)
        rk = tl.min(tl.where(x == mx, rank, big), 0)
        sel = rank == rk
        idx = tl.max(tl.where(sel, ids, -1), 0)
        outv = tl.where(kk == j, mx, outv)
        outi = tl.where(kk == j, idx, outi)
        x = tl.where(sel, float("-inf"), x)
        rank = tl.where(sel, big, rank)
    if n > CAP:
        # Overflow (more than CAP elements tie at or above T): exact fallback, K full-row passes, each picking the
        # largest (value desc, rank asc) element strictly after the previous pick. Graph-safe; slow only here.
        prev_v = tl.full([], float("inf"), tl.float32)
        prev_r = tl.full([], -1, tl.int32)
        bo = tl.arange(0, FB)
        for jf in range(K):
            best_v = tl.full([], float("-inf"), tl.float32)
            best_r = tl.full([], 2147483647, tl.int32)
            best_i = tl.full([], 0, tl.int32)
            for c in range(0, N, FB):
                o = c + bo
                valid = o < N
                xv = tl.load(X + row * X_STRIDE + o, mask=valid, other=float("-inf")).to(tl.float32)
                rv = _fi_rank(o, FI_CHUNK, NJ)
                elig = valid & ((xv < prev_v) | ((xv == prev_v) & (rv > prev_r)))
                xv = tl.where(elig, xv, float("-inf"))
                cm = tl.max(xv, 0)
                cr = tl.min(tl.where(elig & (xv == cm), rv, big), 0)
                ci = tl.max(tl.where(elig & (rv == cr), o, -1), 0)
                better = (cr < big) & ((cm > best_v) | ((cm == best_v) & (cr < best_r)))
                best_v = tl.where(better, cm, best_v)
                best_r = tl.where(better, cr, best_r)
                best_i = tl.where(better, ci, best_i)
            outv = tl.where(kk == jf, best_v, outv)
            outi = tl.where(kk == jf, best_i, outi)
            prev_v = best_v
            prev_r = best_r
    tl.store(OV + row * K + kk, outv)
    tl.store(OI + row * K + kk, outi.to(tl.int64))
    tl.store(OVF + row, n)          # candidates seen (overflow if > CAP)
    tl.store(CNT + row, 0)


class ThresholdTopK:
    def __init__(self, rows, n, k, device, slice_=4096, cap=1024, wa=4, wb=4, wc=4, fi_chunk=77440, atomic=True):
        self.atomic = atomic
        self.n, self.k, self.slice, self.cap = n, k, slice_, cap
        self.ns = triton.cdiv(n, slice_)
        assert self.ns >= k
        self.fi_chunk, self.nj = fi_chunk, triton.cdiv(fi_chunk, 1024)
        self.sm = torch.empty(rows, self.ns, dtype=torch.float32, device=device)
        self.cnt = torch.zeros(rows, dtype=torch.int32, device=device)
        self.cv = torch.empty(rows, cap, dtype=torch.float32, device=device)
        self.ci = torch.empty(rows, cap, dtype=torch.int32, device=device)
        self.ov = torch.empty(rows, k, dtype=torch.float32, device=device)
        self.oi = torch.empty(rows, k, dtype=torch.int64, device=device)
        self.seen = torch.zeros(rows, dtype=torch.int32, device=device)
        self.t = torch.empty(rows, dtype=torch.float32, device=device)
        self.w = (wa, wb, wc)

    def __call__(self, logits):
        r = logits.shape[0]
        _thr_a[(r, self.ns)](logits, self.sm, self.n, logits.stride(0), self.ns, self.slice, num_warps=self.w[0])
        _thr_t[(r,)](self.sm, self.t, self.ns, triton.next_power_of_2(self.ns), self.k, num_warps=1)
        _thr_b[(r, self.ns)](logits, self.sm, self.t, self.cnt, self.cv, self.ci, self.n, logits.stride(0), self.ns,
                             self.slice, self.cap, self.atomic, num_warps=self.w[1])
        _thr_c[(r,)](self.cnt, self.cv, self.ci, self.ov, self.oi, self.seen, logits, self.n, logits.stride(0),
                     self.cap, self.k, self.fi_chunk, self.nj, 4096, num_warps=self.w[2])
        return self.ov[:r], self.oi[:r]
