# ruff: noqa  -- vendored verbatim (see the module docstring)
"""Vendored lane J draft-chain kernels (not in the vLLM fork).

Source: the MoE session's lane J v3 delivery for the DFlash2 draft step,
`wt-glm53-decode-investigate/tmp/draft/fused_ops.py` (2026-11-14, final job
22234). Copied as is; only the import of triton changed (the fork re-exports the
same `triton` / `triton.language` through `vllm.triton_utils`).

- `qk_norm_rope_fp8`: copy of the fork's Beam `_qk_norm_rope_kernel` extended
  with an fp8 query output (static q scale) and an fp8 K/V paged-cache write
  (static k/v scales); replaces q_norm, k_norm, RoPE, the query QuantFP8 and
  reshape_and_cache_flash. Not bit-exact: about 95% of fp8 codes equal the
  incumbent's, the rest differ by rounding near zero (lane J notes §4).
- `ThresholdTopK`: exact top-K with FlashInfer's deterministic tie order
  (top_k(sorted=True, deterministic=True)), values fp32 and ids int64 (the
  fork's casts folded in); ids and values 100% identical to FlashInfer,
  including ties and an in-kernel overflow fallback.
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


@triton.jit
def _silu_mul_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cb = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(X + row * 2 * N + cb).to(tl.float32)
    u = tl.load(X + row * 2 * N + N + cb).to(tl.float32)
    tl.store(Y + row * N + cb, (g / (1.0 + tl.exp(-g)) * u).to(Y.dtype.element_ty))


# ------------------------------------------------------------------ two-stage top-k over the vocabulary
# Exact top-K with FlashInfer's deterministic tie order (top_k(sorted=True, deterministic=True), multi-CTA radix,
# topk.cuh): a row is cut into FI_CHUNK-wide CTA chunks; inside a chunk, elements equal to the pivot are taken in
# thread-strided order (thread tx owns j = tx, tx+1024, ...), and the final stable sort keeps that order among equal
# values. So the order is (value desc, rank asc) with rank(v) = chunk(v)*FI_CHUNK + (j % 1024) * NJ + j // 1024,
# j = v % FI_CHUNK.  Stage 1: grid (rows, chunks), top-K of a CHUNK-wide slice by K rounds of (max, min rank among
# the max, mask).  Stage 2: one program per row merges chunks*K candidates the same way; writes ids int64 and values
# fp32 (the fork's .to(int64) and .float() casts folded in).
@triton.jit
def _fi_rank(v, FI_CHUNK: tl.constexpr, NJ: tl.constexpr):
    c = v // FI_CHUNK
    j = v - c * FI_CHUNK
    return c * (1024 * NJ) + (j % 1024) * NJ + j // 1024


@triton.jit
def _topk_stage1(X, CV, CI, N, X_STRIDE, NCHUNK: tl.constexpr, CHUNK: tl.constexpr, K: tl.constexpr,
                 FI_CHUNK: tl.constexpr, NJ: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    offs = ch * CHUNK + tl.arange(0, CHUNK)
    m = offs < N
    x = tl.load(X + row * X_STRIDE + offs, mask=m, other=float("-inf")).to(tl.float32)
    rank = _fi_rank(offs, FI_CHUNK, NJ)
    big = 2147483647
    kk = tl.arange(0, K)
    outv = tl.full([K], float("-inf"), tl.float32)
    outi = tl.full([K], 0, tl.int32)
    for j in tl.static_range(K):
        mx = tl.max(x, 0)
        rk = tl.min(tl.where(x == mx, rank, big), 0)
        sel = rank == rk
        idx = tl.max(tl.where(sel, offs, -1), 0)
        outv = tl.where(kk == j, mx, outv)
        outi = tl.where(kk == j, idx, outi)
        x = tl.where(sel, float("-inf"), x)
    base = (row * NCHUNK + ch) * K
    tl.store(CV + base + kk, outv)
    tl.store(CI + base + kk, outi)


@triton.jit
def _topk_stage2(CV, CI, OV, OI, NCAND: tl.constexpr, BLOCK: tl.constexpr, K: tl.constexpr,
                 FI_CHUNK: tl.constexpr, NJ: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < NCAND
    x = tl.load(CV + row * NCAND + offs, mask=m, other=float("-inf"))
    ids = tl.load(CI + row * NCAND + offs, mask=m, other=0)
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
    tl.store(OV + row * K + kk, outv)
    tl.store(OI + row * K + kk, outi.to(tl.int64))


class TwoStageTopK:
    def __init__(self, rows, n, k, chunk, device, warps1=4, warps2=8, fi_chunk=77440):
        self.rows, self.n, self.k, self.chunk = rows, n, k, chunk
        self.nchunk = triton.cdiv(n, chunk)
        self.fi_chunk, self.nj = fi_chunk, triton.cdiv(fi_chunk, 1024)
        self.cv = torch.empty(rows, self.nchunk, k, dtype=torch.float32, device=device)
        self.ci = torch.empty(rows, self.nchunk, k, dtype=torch.int32, device=device)
        self.ov = torch.empty(rows, k, dtype=torch.float32, device=device)
        self.oi = torch.empty(rows, k, dtype=torch.int64, device=device)
        self.w1, self.w2 = warps1, warps2

    def __call__(self, logits):
        r = logits.shape[0]
        _topk_stage1[(r, self.nchunk)](logits, self.cv, self.ci, self.n, logits.stride(0), self.nchunk, self.chunk,
                                       self.k, self.fi_chunk, self.nj, num_warps=self.w1)
        nc = self.nchunk * self.k
        _topk_stage2[(r,)](self.cv, self.ci, self.ov, self.oi, nc, triton.next_power_of_2(nc), self.k,
                           self.fi_chunk, self.nj, num_warps=self.w2)
        return self.ov[:r], self.oi[:r]



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


# ---------------------------------------------------------------- conv_proj: split-K GEMM partials + conv that reduces them
@triton.jit
def _splitk_gemm_kernel(X, Wt, P, M, N: tl.constexpr, K: tl.constexpr, X_STRIDE, KS: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """P[s, m, n] = sum_{k in split s} X[m, k] * W[n, k] (fp32 partials, no reduce kernel)."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    s = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    k0 = s * KS
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, KS, BK):
        rk = k0 + kk + tl.arange(0, BK)
        a = tl.load(X + rm[:, None] * X_STRIDE + rk[None, :], mask=rm[:, None] < M, other=0.0)
        b = tl.load(Wt + rn[None, :] * K + rk[:, None])
        acc = tl.dot(a, b, acc)
    tl.store(P + (s * M + rm[:, None]) * N + rn[None, :], acc, mask=rm[:, None] < M)


@triton.jit
def _conv_partials_kernel(X, P, BASE, OUT, CO1, M, S: tl.constexpr, NCOL: tl.constexpr, NG: tl.constexpr,
                          C: tl.constexpr, BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr, EB: tl.constexpr):
    """Grouped conv (2 taps) whose delta = bf16(sum_s P[s]) for columns [0, 2*NG) (co[:,0]); also writes the reduced
    co[:,1] columns [2*NG, 4*NG) as bf16 into CO1 [M, 2*NG] for conv_residual_norm. Same math as
    _dflash2_grouped_conv_kernel with delta rounded to bf16 like the cuBLAS output."""
    nb: tl.constexpr = C // EB
    row = tl.program_id(0) // nb
    cb = tl.program_id(0) % nb
    ch = cb * EB + tl.arange(0, EB)
    grp = ch // GROUP_SIZE
    pos = row % BLOCK_SIZE
    d0 = tl.zeros((EB,), tl.float32)
    d1 = tl.zeros((EB,), tl.float32)
    for s in tl.static_range(S):
        base_p = P + (s * M + row) * NCOL
        d0 += tl.load(base_p + grp)
        d1 += tl.load(base_p + NG + grp)
    d0 = d0.to(tl.bfloat16).to(tl.float32)
    d1 = d1.to(tl.bfloat16).to(tl.float32)
    x0 = tl.load(X + row * C + ch).to(tl.float32)
    acc = (tl.load(BASE + ch).to(tl.float32) + d0) * x0
    tm = pos >= 1
    x1 = tl.load(X + (row - 1) * C + ch, mask=tm & (ch < C), other=0.0).to(tl.float32)
    c1 = tl.where(tm, tl.load(BASE + C + ch).to(tl.float32) + d1, 0.0)
    acc += c1 * x1
    tl.store(OUT + row * C + ch, acc.to(tl.bfloat16))
    # reduce the co[:,1] half: 2*NG columns spread over the nb programs of this row
    PER: tl.constexpr = 2 * NG // nb
    cc = cb * PER + tl.arange(0, PER)
    r1 = tl.zeros((PER,), tl.float32)
    for s in tl.static_range(S):
        r1 += tl.load(P + (s * M + row) * NCOL + 2 * NG + cc)
    tl.store(CO1 + row * (2 * NG) + cc, r1.to(tl.bfloat16))


class ConvProj:
    """conv_proj GEMM (N=1024) as split-K partials, consumed by a conv kernel that reduces them (no splitKreduce)."""

    def __init__(self, M, device, S=8, BM=64, BN=64, BK=128, warps=4, stages=3, eb=1024, cwarps=4):
        self.M, self.S, self.BM, self.BN, self.BK, self.w, self.st = M, S, BM, BN, BK, warps, stages
        self.eb, self.cw = eb, cwarps
        self.P = torch.empty(S, M, 1024, dtype=torch.float32, device=device)
        self.out = torch.empty(M, 4096, dtype=torch.bfloat16, device=device)
        self.co1 = torch.empty(M, 512, dtype=torch.bfloat16, device=device)

    def gemm(self, x, w):
        M = x.shape[0]
        KS = 4096 // self.S
        grid = (triton.cdiv(M, self.BM), 1024 // self.BN, self.S)
        _splitk_gemm_kernel[grid](x, w, self.P, M, 1024, 4096, x.stride(0), KS, self.BM, self.BN, self.BK,
                                  num_warps=self.w, num_stages=self.st)

    def conv(self, x, base):
        M = x.shape[0]
        _conv_partials_kernel[(M * (4096 // self.eb),)](x, self.P, base, self.out, self.co1, M, self.S, 1024, 256,
                                                        4096, 8, 16, self.eb, num_warps=self.cw)
        return self.out, self.co1

    def __call__(self, x, w, base):
        self.gemm(x, w)
        return self.conv(x, base)
