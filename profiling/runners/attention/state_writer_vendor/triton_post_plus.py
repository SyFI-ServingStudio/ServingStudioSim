"""Lane K: small DSA state-writer kernels (GLM-5.3-Flash decode, B200).

K1 kpool_par      replaces kpool_decode_update_and_maybe_write_cache_batched (+ the positions int64->int32 cast)
                  One program per request; all NEXT_N tokens processed in parallel as a [BT, 128] tile. The fork's
                  serial order is reproduced exactly: every pool slot's source is the newest earlier token of the same
                  request that stashed into the same (block, ring slot), else the ring as it was at kernel start.
                  All ring reads happen before a CTA barrier, all stashes after it; a stash that a later token of
                  the same request overwrites is dropped (the fork's final ring state).
K2 norm_cache     fused_q_kv_rmsnorm whose kv half writes the fp8 MLA cache row directly (replaces norm + cache_append).
K3 post_plus      lane F's A2 post kernel + q/kv RMSNorm + fp8 cache append + kpool (+ pos cast), one launch.
                  Programs [0, 2M): (row, part) part 0 = q rmsnorm (or raw bf16), part 1 = kv rmsnorm -> fp8 cache.
                  Programs [2M, 2M+NR): one request: k LayerNorm + gate (bf16, also stored) + kpool update.
Q  quant_fp8      static per-tensor fp8 quant (scaled_fp8_quant replacement), bf16 [n] -> fp8 [n].
"""
import torch
import triton
import triton.language as tl

try:
    from vllm.v1.attention.ops.common import pack_ue8m0_x4
except Exception:  # pragma: no cover
    pack_ue8m0_x4 = None


@triton.jit
def _fwht_stage(x, N: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (N,))


@triton.jit
def _div_by_scalar_rn(y, s, r):
    """IEEE round-to-nearest y / s for a scalar s with r = rn(1 / s): q = rn(y r), one fma residual
    correction (Markstein). Replaces a per-element div.rn (slow software sequence) on normal-range data."""
    q = y * r
    e = tl.math.fma(-q, s + tl.zeros_like(y), y)
    return tl.math.fma(e, r + tl.zeros_like(y), q)


@triton.jit
def _hadamard_rows(x, BT: tl.constexpr):
    """Hadamard-128 on every row of a [BT, 128] tile, same butterfly order and scale as the fork's _hadamard128."""
    N: tl.constexpr = BT * 128
    x = tl.reshape(x, (N,))
    x = _fwht_stage(x, N, BT * 64, 1)
    x = _fwht_stage(x, N, BT * 32, 2)
    x = _fwht_stage(x, N, BT * 16, 4)
    x = _fwht_stage(x, N, BT * 8, 8)
    x = _fwht_stage(x, N, BT * 4, 16)
    x = _fwht_stage(x, N, BT * 2, 32)
    x = _fwht_stage(x, N, BT, 64)
    x = x * 0.08838834764831845
    return tl.reshape(x, (BT, 128))


@triton.jit
def _kpool_tile(key, score_cur, t, tm, req, NEXT_N,
                buf_fp8_ptr, buf_fp32_ptr, tail_kv_ptr, tail_slot_mapping_ptr,
                key_ptr, key_stride_b, key_stride_t, slot_score_ptr, ss_stride_b, ss_stride_t,
                ape_ptr, ape_stride_0, slot_mapping_ptr, positions_ptr,
                PAGE_SIZE: tl.constexpr, BUF_NUMEL_PER_PAGE: tl.constexpr, POOL_SIZE: tl.constexpr, RING: tl.constexpr,
                TAIL_BLOCK_ELEMS: tl.constexpr, KPOOL_HEAD: tl.constexpr, HEAD_DIM: tl.constexpr,
                S_OFFSET_NBYTES_IN_PAGE: tl.constexpr, ROUND_SCALE: tl.constexpr, MX_SCALE: tl.constexpr,
                BT: tl.constexpr, SRC_FROM_INPUT: tl.constexpr):
    """key / score_cur: [BT, 128] fp32 (bf16 values) of this request's tokens. Does the completion writes and
    the ring stashes. SRC_FROM_INPUT: in-batch pool sources are read back from key_ptr / slot_score_ptr (they
    hold exactly the stashed bf16 values); else they would have to come from registers (not used)."""
    offs = tl.arange(0, 128)
    idx = req * NEXT_N + t
    cache_loc = tl.load(slot_mapping_ptr + idx, mask=tm, other=-1)
    pos = tl.load(positions_ptr + idx, mask=tm, other=-1).to(tl.int32)
    tail_slot = tl.load(tail_slot_mapping_ptr + idx, mask=tm, other=-1)
    safe_pos = tl.maximum(pos, 0)
    pos_valid = (cache_loc >= 0) & (pos >= 0)
    slot = safe_pos % POOL_SIZE
    phys_slot = safe_pos % RING
    block = tl.maximum(tail_slot, 0).to(tl.int64) // RING
    block_base = block * TAIL_BLOCK_ELEMS
    stash_valid = (pos >= 0) & (tail_slot >= 0) & tm
    comp = pos_valid & (slot == POOL_SIZE - 1) & tm

    # pairwise [t, t'] relations
    tt = t[:, None]
    tp = t[None, :]
    dup_later = tl.max(((tp > tt) & comp[None, :] & (cache_loc[None, :] == cache_loc[:, None])).to(tl.int32), axis=1) > 0
    comp = comp & (~dup_later)
    same_blk = block[None, :] == block[:, None]
    earlier = (tp < tt) & stash_valid[None, :] & same_blk

    pool_start = safe_pos - slot
    max_score = tl.full((BT, 128), -float("inf"), tl.float32)
    # pass 1 (scores kept in registers; pass 2 recomputes nothing)
    s0 = tl.zeros((BT, 128), tl.float32)
    s1 = tl.zeros((BT, 128), tl.float32)
    s2 = tl.zeros((BT, 128), tl.float32)
    s3 = tl.zeros((BT, 128), tl.float32)
    k0 = tl.zeros((BT, 128), tl.float32)
    k1 = tl.zeros((BT, 128), tl.float32)
    k2 = tl.zeros((BT, 128), tl.float32)
    k3 = tl.zeros((BT, 128), tl.float32)
    for j in tl.static_range(0, POOL_SIZE):
        phys_j = (pool_start + j) % RING
        m = earlier & (phys_slot[None, :] == phys_j[:, None])
        src = tl.max(tl.where(m, tp, -1), axis=1)                       # newest earlier stash, else -1
        is_cur = slot == j
        need = comp & (~is_cur)
        from_in = need & (src >= 0)
        from_ring = need & (src < 0)
        ring_off = block_base + phys_j * HEAD_DIM
        srcc = tl.maximum(src, 0)
        kb = tl.load(tail_kv_ptr + ring_off[:, None] + offs[None, :], mask=from_ring[:, None], other=0.0).to(tl.float32)
        sb = tl.load(tail_kv_ptr + ring_off[:, None] + KPOOL_HEAD + offs[None, :], mask=from_ring[:, None], other=0.0).to(tl.float32)
        ki = tl.load(key_ptr + req * key_stride_b + srcc[:, None] * key_stride_t + offs[None, :], mask=from_in[:, None], other=0.0).to(tl.float32)
        si = tl.load(slot_score_ptr + req * ss_stride_b + srcc[:, None] * ss_stride_t + offs[None, :], mask=from_in[:, None], other=0.0).to(tl.float32)
        kj = tl.where(is_cur[:, None], key, tl.where(from_in[:, None], ki, kb))
        sj = tl.where(is_cur[:, None], score_cur, tl.where(from_in[:, None], si, sb))
        sj += tl.load(ape_ptr + j * ape_stride_0 + offs).to(tl.float32)[None, :]
        max_score = tl.maximum(max_score, sj)
        if j == 0:
            s0 = sj
            k0 = kj
        elif j == 1:
            s1 = sj
            k1 = kj
        elif j == 2:
            s2 = sj
            k2 = kj
        else:
            s3 = sj
            k3 = kj
    acc = tl.zeros((BT, 128), tl.float32)
    denom = tl.zeros((BT, 128), tl.float32)
    p = tl.exp(s0 - max_score)
    denom += p
    acc += k0 * p
    if POOL_SIZE > 1:
        p = tl.exp(s1 - max_score)
        denom += p
        acc += k1 * p
    if POOL_SIZE > 2:
        p = tl.exp(s2 - max_score)
        denom += p
        acc += k2 * p
    if POOL_SIZE > 3:
        p = tl.exp(s3 - max_score)
        denom += p
        acc += k3 * p
    x = (acc / denom).to(tl.bfloat16).to(tl.float32)
    x = _hadamard_rows(x, BT).to(tl.bfloat16).to(tl.float32)
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
    else:
        scale = absmax * fp8_max_inv
    quantized = tl.minimum(tl.maximum((x * (1.0 / scale)[:, None] if ROUND_SCALE else x / scale[:, None]), -fp8_max), fp8_max)
    loc = tl.maximum(cache_loc, 0).to(tl.int64)
    pg = loc // PAGE_SIZE
    tk = loc % PAGE_SIZE
    tl.store(buf_fp8_ptr + (pg * BUF_NUMEL_PER_PAGE + tk * HEAD_DIM)[:, None] + offs[None, :], quantized, mask=comp[:, None])
    s_off = pg * BUF_NUMEL_PER_PAGE // 4 + S_OFFSET_NBYTES_IN_PAGE // 4 + tk
    if MX_SCALE:
        tl.store(buf_fp32_ptr.to(tl.pointer_type(tl.int32)) + s_off, pack_ue8m0_x4(scale), mask=comp)
    else:
        tl.store(buf_fp32_ptr + s_off, scale, mask=comp)

    # all ring reads above happen before any stash below (other threads of the CTA may stash a slot we read)
    tl.debug_barrier()
    later_same = (tp > tt) & stash_valid[None, :] & same_blk & (phys_slot[None, :] == phys_slot[:, None])
    overwritten = tl.max(later_same.to(tl.int32), axis=1) > 0
    do_stash = stash_valid & (~overwritten)
    st_off = block_base + phys_slot * HEAD_DIM
    tl.store(tail_kv_ptr + st_off[:, None] + offs[None, :], key.to(tl.bfloat16), mask=do_stash[:, None])
    tl.store(tail_kv_ptr + st_off[:, None] + KPOOL_HEAD + offs[None, :], score_cur.to(tl.bfloat16), mask=do_stash[:, None])


@triton.jit
def _kpool_tile2(key, score_cur, t, tm, req, NEXT_N,
                 buf_fp8_ptr, buf_fp32_ptr, tail_kv_ptr, tail_slot_mapping_ptr,
                 key_ptr, key_stride_b, key_stride_t, slot_score_ptr, ss_stride_b, ss_stride_t,
                 ape_ptr, ape_stride_0, slot_mapping_ptr, positions_ptr,
                 PAGE_SIZE: tl.constexpr, BUF_NUMEL_PER_PAGE: tl.constexpr, POOL_SIZE: tl.constexpr, RING: tl.constexpr,
                 TAIL_BLOCK_ELEMS: tl.constexpr, KPOOL_HEAD: tl.constexpr, HEAD_DIM: tl.constexpr,
                 S_OFFSET_NBYTES_IN_PAGE: tl.constexpr, ROUND_SCALE: tl.constexpr, MX_SCALE: tl.constexpr,
                 BT: tl.constexpr, CT: tl.constexpr, ABL: tl.constexpr):
    """v2: same semantics as _kpool_tile, but the completion math runs only on the completing rows, compacted
    into [CT, 128] chunks (a runtime loop over ceil(ncomp / CT) chunks; 1 chunk for 8 contiguous tokens, pool 4).
    The current token's own key / score are read back from key_ptr / slot_score_ptr (identical bf16 values).
    ABL (timing ablations only, output invalid): 1 no pool gathers, 2 no Hadamard, 4 no barrier, 8 no math,
    16 no completion work at all."""
    offs = tl.arange(0, 128)
    idx = req * NEXT_N + t
    cache_loc = tl.load(slot_mapping_ptr + idx, mask=tm, other=-1)
    pos = tl.load(positions_ptr + idx, mask=tm, other=-1).to(tl.int32)
    tail_slot = tl.load(tail_slot_mapping_ptr + idx, mask=tm, other=-1)
    safe_pos = tl.maximum(pos, 0)
    pos_valid = (cache_loc >= 0) & (pos >= 0)
    slot = safe_pos % POOL_SIZE
    phys_slot = safe_pos % RING
    block = tl.maximum(tail_slot, 0).to(tl.int64) // RING
    stash_valid = (pos >= 0) & (tail_slot >= 0) & tm
    comp = pos_valid & (slot == POOL_SIZE - 1) & tm
    tt = t[:, None]
    tp = t[None, :]
    # a completion whose cache row a later token of this request rewrites is dead (the fork's last write wins)
    dup_later = tl.max(((tp > tt) & comp[None, :] & (cache_loc[None, :] == cache_loc[:, None])).to(tl.int32), axis=1) > 0
    comp = comp & (~dup_later)
    ci = comp.to(tl.int32)
    ncomp = tl.sum(ci, axis=0)
    crank = tl.cumsum(ci, axis=0) - 1
    if ABL & 16:
        ncomp = 0
    tq = tl.arange(0, BT)[None, :]
    for c0 in range(0, ncomp, CT):
        c = c0 + tl.arange(0, CT)
        cm = c < ncomp
        sel = comp[None, :] & (crank[None, :] == c[:, None])                 # [CT, BT] one-hot
        row = tl.sum(tl.where(sel, tq, 0), axis=1)
        r_pos = tl.sum(tl.where(sel, safe_pos[None, :], 0), axis=1)
        r_blk = tl.sum(tl.where(sel, block[None, :], 0), axis=1)
        r_loc = tl.sum(tl.where(sel, cache_loc[None, :], 0), axis=1)
        r_slot = r_pos % POOL_SIZE
        r_base = r_blk * TAIL_BLOCK_ELEMS
        pool_start = r_pos - r_slot
        earlier = (tq < row[:, None]) & stash_valid[None, :] & (block[None, :] == r_blk[:, None])
        max_score = tl.full((CT, 128), -float("inf"), tl.float32)
        s0 = tl.zeros((CT, 128), tl.float32)
        s1 = tl.zeros((CT, 128), tl.float32)
        s2 = tl.zeros((CT, 128), tl.float32)
        s3 = tl.zeros((CT, 128), tl.float32)
        k0 = tl.zeros((CT, 128), tl.float32)
        k1 = tl.zeros((CT, 128), tl.float32)
        k2 = tl.zeros((CT, 128), tl.float32)
        k3 = tl.zeros((CT, 128), tl.float32)
        for j in tl.static_range(0, POOL_SIZE):
            phys_j = (pool_start + j) % RING
            m = earlier & (phys_slot[None, :] == phys_j[:, None])
            src = tl.max(tl.where(m, tq, -1), axis=1)
            is_cur = r_slot == j
            if ABL & 1:
                is_cur = r_slot >= 0
            in_row = tl.where(is_cur, row, src)
            from_in = cm & (in_row >= 0)
            from_ring = cm & (in_row < 0)
            in_rowc = tl.maximum(in_row, 0)
            ring_off = r_base + phys_j * HEAD_DIM
            ki = tl.load(key_ptr + req * key_stride_b + in_rowc[:, None] * key_stride_t + offs[None, :], mask=from_in[:, None], other=0.0).to(tl.float32)
            si = tl.load(slot_score_ptr + req * ss_stride_b + in_rowc[:, None] * ss_stride_t + offs[None, :], mask=from_in[:, None], other=0.0).to(tl.float32)
            kb = tl.load(tail_kv_ptr + ring_off[:, None] + offs[None, :], mask=from_ring[:, None], other=0.0).to(tl.float32)
            sb = tl.load(tail_kv_ptr + ring_off[:, None] + KPOOL_HEAD + offs[None, :], mask=from_ring[:, None], other=0.0).to(tl.float32)
            kj = tl.where(from_in[:, None], ki, kb)
            sj = tl.where(from_in[:, None], si, sb)
            sj += tl.load(ape_ptr + j * ape_stride_0 + offs).to(tl.float32)[None, :]
            max_score = tl.maximum(max_score, sj)
            if j == 0:
                s0 = sj
                k0 = kj
            elif j == 1:
                s1 = sj
                k1 = kj
            elif j == 2:
                s2 = sj
                k2 = kj
            else:
                s3 = sj
                k3 = kj
        if ABL & 8:
            x = k0
            scale = tl.max(s0, axis=1)
            quantized = x
        else:
            acc = tl.zeros((CT, 128), tl.float32)
            denom = tl.zeros((CT, 128), tl.float32)
            p = tl.exp(s0 - max_score)
            denom += p
            acc += k0 * p
            if POOL_SIZE > 1:
                p = tl.exp(s1 - max_score)
                denom += p
                acc += k1 * p
            if POOL_SIZE > 2:
                p = tl.exp(s2 - max_score)
                denom += p
                acc += k2 * p
            if POOL_SIZE > 3:
                p = tl.exp(s3 - max_score)
                denom += p
                acc += k3 * p
            x = (acc / denom).to(tl.bfloat16).to(tl.float32)
            if not (ABL & 2):
                x = _hadamard_rows(x, CT)
            x = x.to(tl.bfloat16).to(tl.float32)
            absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
            if ROUND_SCALE:
                scale = tl.exp2(tl.ceil(tl.log2(absmax * (1.0 / 448.0))))
            else:
                scale = absmax * (1.0 / 448.0)
            quantized = tl.minimum(tl.maximum((x * (1.0 / scale)[:, None] if ROUND_SCALE else x / scale[:, None]), -448.0), 448.0)
        loc = tl.maximum(r_loc, 0).to(tl.int64)
        pg = loc // PAGE_SIZE
        tk = loc % PAGE_SIZE
        tl.store(buf_fp8_ptr + (pg * BUF_NUMEL_PER_PAGE + tk * HEAD_DIM)[:, None] + offs[None, :], quantized, mask=cm[:, None])
        s_off = pg * BUF_NUMEL_PER_PAGE // 4 + S_OFFSET_NBYTES_IN_PAGE // 4 + tk
        if MX_SCALE:
            tl.store(buf_fp32_ptr.to(tl.pointer_type(tl.int32)) + s_off, pack_ue8m0_x4(scale), mask=cm)
        else:
            tl.store(buf_fp32_ptr + s_off, scale, mask=cm)
    if not (ABL & 4):
        tl.debug_barrier()
    same_blk = block[None, :] == block[:, None]
    later_same = (tp > tt) & stash_valid[None, :] & same_blk & (phys_slot[None, :] == phys_slot[:, None])
    overwritten = tl.max(later_same.to(tl.int32), axis=1) > 0
    do_stash = stash_valid & (~overwritten)
    st_off = block * TAIL_BLOCK_ELEMS + phys_slot * HEAD_DIM
    tl.store(tail_kv_ptr + st_off[:, None] + offs[None, :], key.to(tl.bfloat16), mask=do_stash[:, None])
    tl.store(tail_kv_ptr + st_off[:, None] + KPOOL_HEAD + offs[None, :], score_cur.to(tl.bfloat16), mask=do_stash[:, None])


@triton.jit
def _kpool_par_kernel(buf_fp8_ptr, buf_fp32_ptr, tail_kv_ptr, tail_slot_mapping_ptr,
                      key_ptr, key_stride_b, key_stride_t, slot_score_ptr, ss_stride_b, ss_stride_t,
                      ape_ptr, ape_stride_0, slot_mapping_ptr, positions_ptr, NEXT_N,
                      PAGE_SIZE: tl.constexpr, BUF_NUMEL_PER_PAGE: tl.constexpr, POOL_SIZE: tl.constexpr,
                      RING: tl.constexpr, TAIL_BLOCK_ELEMS: tl.constexpr, KPOOL_HEAD: tl.constexpr,
                      HEAD_DIM: tl.constexpr, S_OFFSET_NBYTES_IN_PAGE: tl.constexpr, ROUND_SCALE: tl.constexpr,
                      MX_SCALE: tl.constexpr, BT: tl.constexpr, VER: tl.constexpr, CT: tl.constexpr, ABL: tl.constexpr):
    req = tl.program_id(0)
    t = tl.arange(0, BT)
    tm = t < NEXT_N
    offs = tl.arange(0, 128)
    key = tl.load(key_ptr + req * key_stride_b + t[:, None] * key_stride_t + offs[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
    sc = tl.load(slot_score_ptr + req * ss_stride_b + t[:, None] * ss_stride_t + offs[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
    if VER == 1:
        _kpool_tile(key, sc, t, tm, req, NEXT_N, buf_fp8_ptr, buf_fp32_ptr, tail_kv_ptr, tail_slot_mapping_ptr,
                    key_ptr, key_stride_b, key_stride_t, slot_score_ptr, ss_stride_b, ss_stride_t,
                    ape_ptr, ape_stride_0, slot_mapping_ptr, positions_ptr,
                    PAGE_SIZE, BUF_NUMEL_PER_PAGE, POOL_SIZE, RING, TAIL_BLOCK_ELEMS, KPOOL_HEAD, HEAD_DIM,
                    S_OFFSET_NBYTES_IN_PAGE, ROUND_SCALE, MX_SCALE, BT, True)
    else:
        _kpool_tile2(key, sc, t, tm, req, NEXT_N, buf_fp8_ptr, buf_fp32_ptr, tail_kv_ptr, tail_slot_mapping_ptr,
                     key_ptr, key_stride_b, key_stride_t, slot_score_ptr, ss_stride_b, ss_stride_t,
                     ape_ptr, ape_stride_0, slot_mapping_ptr, positions_ptr,
                     PAGE_SIZE, BUF_NUMEL_PER_PAGE, POOL_SIZE, RING, TAIL_BLOCK_ELEMS, KPOOL_HEAD, HEAD_DIM,
                     S_OFFSET_NBYTES_IN_PAGE, ROUND_SCALE, MX_SCALE, BT, CT, ABL)


def _common_meta(kv_cache, tail_kv_cache, head_dim, pool_size, round_scale, mx):
    page_size = kv_cache.shape[1]
    return dict(PAGE_SIZE=page_size, BUF_NUMEL_PER_PAGE=kv_cache.stride(0), POOL_SIZE=pool_size,
                RING=tail_kv_cache.shape[2], TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
                KPOOL_HEAD=tail_kv_cache.stride(1), HEAD_DIM=head_dim,
                S_OFFSET_NBYTES_IN_PAGE=page_size * head_dim, ROUND_SCALE=round_scale or mx, MX_SCALE=mx)


def kpool_par(kv_cache, tail_kv_cache, tail_slot_mapping, key, slot_score, ape, slot_mapping, positions,
              pool_size, head_dim=128, round_scale=True, mx=True, num_warps=4, ver=2, ct=2, abl=0):
    """Drop-in for kpool_decode_update_and_maybe_write_cache_batched. positions may be int32 or int64
    (the fork's .to(torch.int32) launch is not needed)."""
    nr, nn = key.shape[0], key.shape[1]
    if nr == 0 or nn == 0:
        return
    assert head_dim == 128 and pool_size <= 4
    assert tail_kv_cache.shape[2] >= pool_size
    BT = max(2, triton.next_power_of_2(nn))
    _kpool_par_kernel[(nr,)](
        kv_cache.view(torch.float8_e4m3fn), kv_cache.view(torch.float32), tail_kv_cache, tail_slot_mapping,
        key, key.stride(0), key.stride(1), slot_score, slot_score.stride(0), slot_score.stride(1),
        ape, ape.stride(0), slot_mapping, positions, nn,
        BT=BT, VER=ver, CT=ct, ABL=abl, num_warps=num_warps,
        **_common_meta(kv_cache, tail_kv_cache, head_dim, pool_size, round_scale, mx))


# ------------------------------------------------------------------ K2: q/kv rmsnorm with the fp8 cache write fused
@triton.jit
def _norm_cache_kernel(q_ptr, q_out_ptr, q_w_ptr, q_in_stride, q_out_stride,
                       kv_ptr, kv_w_ptr, kv_in_stride, cache_ptr, slot_ptr, kv_scale_ptr, CACHE_ENTRY_STRIDE,
                       eps, Q_SIZE: tl.constexpr, KV_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                       ABL: tl.constexpr, KV_BLOCK: tl.constexpr):
    tok = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    block = tl.arange(0, BLOCK_SIZE)
    if part == 0:
        mask = block < Q_SIZE
        w = tl.load(q_w_ptr + block, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(q_ptr + tok * q_in_stride + block, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / Q_SIZE
        rr = tl.rsqrt(var + eps)
        y = x * rr * w
        tl.store(q_out_ptr + tok * q_out_stride + block, y.to(tl.bfloat16), mask=mask)
    else:
        kb = tl.arange(0, KV_BLOCK)
        kmask = kb < KV_SIZE
        if ABL & 4:
            slot = tok
        else:
            slot = tl.load(slot_ptr + tok)
        sc = tl.load(kv_scale_ptr)
        wk_ = tl.load(kv_w_ptr + kb, mask=kmask, other=0.0).to(tl.float32)
        xk = tl.load(kv_ptr + tok * kv_in_stride + kb, mask=kmask, other=0.0).to(tl.float32)
        vk = tl.sum(xk * xk, axis=0) / KV_SIZE
        rk = tl.rsqrt(vk + eps)
        yk = (xk * rk * wk_).to(tl.bfloat16).to(tl.float32)
        if ABL & 1:   # timing ablation: fork-like contiguous bf16 kv out (cache_ptr reused as scratch)
            tl.store(cache_ptr.to(tl.pointer_type(tl.bfloat16)) + tok * KV_SIZE + kb, yk.to(tl.bfloat16), mask=kmask)
        else:
            if ABL & 2:
                y8 = yk.to(tl.float8e4nv)
            else:
                if ABL & 8:
                    y8 = tl.math.div_rn(yk, sc + tl.zeros_like(yk)).to(tl.float8e4nv)   # CUDA x / scale (IEEE), slow
                else:
                    y8 = _div_by_scalar_rn(yk, sc, tl.math.div_rn(sc * 0.0 + 1.0, sc)).to(tl.float8e4nv)
            tl.store(cache_ptr + slot * CACHE_ENTRY_STRIDE + kb, y8, mask=kmask & (slot >= 0))


def norm_cache(qc, kv, q_w, kv_w, eps, cache, slot64, kv_scale, q_out=None, abl=0, kv_block=0, num_warps=0):
    """fused_q_kv_rmsnorm (fork) + concat_and_cache_mla(kv_normed, k_pe(width 0), cache, slot, 'fp8', kv_scale).
    cache: [blocks, page, 512] uint8 (entry stride = 512 B = kv_lora, no rope part). Returns q_out."""
    M = qc.shape[0]
    if q_out is None:
        q_out = torch.empty(qc.shape, dtype=qc.dtype, device=qc.device)
    assert cache.shape[-1] == kv.shape[1]
    bs = triton.next_power_of_2(max(qc.shape[1], kv.shape[1]))
    _norm_cache_kernel[(M, 2)](qc, q_out, q_w, qc.stride(0), q_out.stride(0), kv, kv_w, kv.stride(0),
                               cache.view(torch.float8_e4m3fn), slot64, kv_scale, cache.shape[-1], eps,
                               Q_SIZE=qc.shape[1], KV_SIZE=kv.shape[1], BLOCK_SIZE=bs, ABL=abl,
                               KV_BLOCK=kv_block or bs, num_warps=num_warps or (8 if bs >= 2048 else 4))
    return q_out


# ------------------------------------------------------------------ K3: A2 post + q/kv norm + fp8 cache + kpool
@triton.jit
def _post_plus_kernel(y, qout_ptr, q_w_ptr, kv_w_ptr, cache_ptr, slot_ptr, kv_scale_ptr, CACHE_ENTRY_STRIDE, eps,
                      k_out_ptr, gs_out_ptr, kn_w_ptr, kn_b_ptr, ln_eps, M,
                      buf_fp8_ptr, buf_fp32_ptr, tail_kv_ptr, tail_slot_mapping_ptr,
                      ape_ptr, ape_stride_0, slot_mapping_ptr, positions_ptr, NEXT_N,
                      PAGE_SIZE: tl.constexpr, BUF_NUMEL_PER_PAGE: tl.constexpr, POOL_SIZE: tl.constexpr,
                      RING: tl.constexpr, TAIL_BLOCK_ELEMS: tl.constexpr, KPOOL_HEAD: tl.constexpr,
                      HEAD_DIM: tl.constexpr, S_OFFSET_NBYTES_IN_PAGE: tl.constexpr, ROUND_SCALE: tl.constexpr,
                      MX_SCALE: tl.constexpr, BT: tl.constexpr, QNORM: tl.constexpr, DO_KPOOL: tl.constexpr,
                      LDY: tl.constexpr, VER: tl.constexpr, CT: tl.constexpr, REQ_FIRST: tl.constexpr):
    pid0 = tl.program_id(0)
    NRQ = M // NEXT_N
    if REQ_FIRST:
        # request programs (long serial kpool chain) get the lowest ids so they start in the first wave
        pid = tl.where(pid0 < NRQ, pid0 + 2 * M, pid0 - NRQ)
    else:
        pid = pid0
    if pid < 2 * M:
        r = (pid // 2).to(tl.int64)
        part = pid % 2
        block = tl.arange(0, 2048)
        base = y + r * LDY
        if part == 0:
            mask = block < 1536
            x = tl.load(base + block, mask=mask, other=0.0).to(tl.bfloat16).to(tl.float32)
            if QNORM:
                w = tl.load(q_w_ptr + block, mask=mask, other=0.0).to(tl.float32)
                var = tl.sum(x * x, axis=0) / 1536
                rr = tl.rsqrt(var + eps)
                x = x * rr * w
            tl.store(qout_ptr + r * 1536 + block, x.to(tl.bfloat16), mask=mask)
        else:
            mask = block < 512
            slot = tl.load(slot_ptr + r)
            sc = tl.load(kv_scale_ptr)
            w = tl.load(kv_w_ptr + block, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(base + 1536 + block, mask=mask, other=0.0).to(tl.bfloat16).to(tl.float32)
            var = tl.sum(x * x, axis=0) / 512
            rr = tl.rsqrt(var + eps)
            yv = (x * rr * w).to(tl.bfloat16).to(tl.float32)
            tl.store(cache_ptr + slot * CACHE_ENTRY_STRIDE + block, _div_by_scalar_rn(yv, sc, tl.math.div_rn(sc * 0.0 + 1.0, sc)).to(tl.float8e4nv), mask=mask & (slot >= 0))
    else:
        req = pid - 2 * M
        t = tl.arange(0, BT)
        tm = t < NEXT_N
        offs = tl.arange(0, 128)
        rows = (req * NEXT_N + t).to(tl.int64)
        kw = tl.load(y + rows[:, None] * LDY + 2048 + offs[None, :], mask=tm[:, None], other=0.0).to(tl.bfloat16).to(tl.float32)
        g = tl.load(y + rows[:, None] * LDY + 2176 + offs[None, :], mask=tm[:, None], other=0.0).to(tl.bfloat16)
        mean = tl.sum(kw, 1) / 128.0
        d = kw - mean[:, None]
        kvar = tl.sum(d * d, 1) / 128.0
        rs = 1.0 / tl.sqrt(kvar + ln_eps)
        kk = (d * rs[:, None] * tl.load(kn_w_ptr + offs)[None, :] + tl.load(kn_b_ptr + offs)[None, :]).to(tl.bfloat16)
        tl.store(k_out_ptr + rows[:, None] * 128 + offs[None, :], kk, mask=tm[:, None])
        tl.store(gs_out_ptr + rows[:, None] * 128 + offs[None, :], g, mask=tm[:, None])
        if DO_KPOOL:
            # in-batch pool sources are read back from k_out / gs_out (written above by this same program);
            # the barrier inside _kpool_tile comes after those reads, so make the writes visible first.
            tl.debug_barrier()
            if VER == 1:
                _kpool_tile(kk.to(tl.float32), g.to(tl.float32), t, tm, req, NEXT_N, buf_fp8_ptr, buf_fp32_ptr,
                            tail_kv_ptr, tail_slot_mapping_ptr,
                            k_out_ptr, NEXT_N * 128, 128, gs_out_ptr, NEXT_N * 128, 128,
                            ape_ptr, ape_stride_0, slot_mapping_ptr, positions_ptr,
                            PAGE_SIZE, BUF_NUMEL_PER_PAGE, POOL_SIZE, RING, TAIL_BLOCK_ELEMS, KPOOL_HEAD, HEAD_DIM,
                            S_OFFSET_NBYTES_IN_PAGE, ROUND_SCALE, MX_SCALE, BT, True)
            else:
                _kpool_tile2(kk.to(tl.float32), g.to(tl.float32), t, tm, req, NEXT_N, buf_fp8_ptr, buf_fp32_ptr,
                             tail_kv_ptr, tail_slot_mapping_ptr,
                             k_out_ptr, NEXT_N * 128, 128, gs_out_ptr, NEXT_N * 128, 128,
                             ape_ptr, ape_stride_0, slot_mapping_ptr, positions_ptr,
                             PAGE_SIZE, BUF_NUMEL_PER_PAGE, POOL_SIZE, RING, TAIL_BLOCK_ELEMS, KPOOL_HEAD, HEAD_DIM,
                             S_OFFSET_NBYTES_IN_PAGE, ROUND_SCALE, MX_SCALE, BT, CT, 0)


def post_plus(y32, q_out, q_w, kv_w, eps, cache, slot64, kv_scale, k_out, gs_out, kn_w, kn_b, ln_eps,
              idx_cache, tail, tail_slot, ape, dec_slot, positions, next_n, pool_size=4, qnorm=True,
              do_kpool=True, mx=True, num_warps=8, ver=2, ct=2, req_first=True):
    """y32: [M, 2336] fp32 = A2 GEMM output, rows [q_a 1536 | kv_a 512 | wk 128 | gate 128 | head_w 32].
    Uniform decode: M = NR * next_n, request r owns rows [r*next_n, (r+1)*next_n)."""
    M = y32.shape[0]
    assert M % next_n == 0
    nr = M // next_n
    BT = max(2, triton.next_power_of_2(next_n))
    _post_plus_kernel[(2 * M + nr,)](
        y32, q_out, q_w, kv_w, cache.view(torch.float8_e4m3fn), slot64, kv_scale, cache.shape[-1], eps,
        k_out, gs_out, kn_w, kn_b, ln_eps, M,
        idx_cache.view(torch.float8_e4m3fn), idx_cache.view(torch.float32), tail, tail_slot,
        ape, ape.stride(0), dec_slot, positions, next_n,
        BT=BT, QNORM=qnorm, DO_KPOOL=do_kpool, LDY=y32.stride(0), num_warps=num_warps, VER=ver, CT=ct, REQ_FIRST=req_first,
        **_common_meta(idx_cache, tail, 128, pool_size, True, mx))


# ------------------------------------------------------------------ Q: static fp8 quant
@triton.jit
def _quant_kernel(x_ptr, o_ptr, s_ptr, n, BLOCK: tl.constexpr, DIV: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s = tl.load(s_ptr)
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    if DIV:
        y = tl.math.div_rn(x, s + tl.zeros_like(x))
    else:
        y = x * tl.math.div_rn(s * 0.0 + 1.0, s)   # vLLM static_scaled_fp8_quant: x * (1 / scale)
    y = tl.minimum(tl.maximum(y, -448.0), 448.0)
    tl.store(o_ptr + offs, y.to(tl.float8e4nv), mask=m)


def quant_fp8(x, scale, out=None, block=4096, num_warps=4, div=False):
    n = x.numel()
    if out is None:
        out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    _quant_kernel[(triton.cdiv(n, block),)](x, out, scale, n, BLOCK=block, DIV=div, num_warps=num_warps)
    return out


# ------------------------------------------------------------------ V: v_up as a Triton batched GEMM
@triton.jit
def _vup_kernel(x_ptr, w_ptr, o_ptr, M, H: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, W_KMAJOR: tl.constexpr):
    """out[m, h, n] = sum_k x[m, h, k] * W[h, k, n]; x [M, H, K] bf16, W [H, K, N] (or [H, N, K] if W_KMAJOR)."""
    h = tl.program_id(0)
    pm = tl.program_id(1)
    pn = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm = rm < M
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.range(0, K, BK):
        a = tl.load(x_ptr + rm[:, None].to(tl.int64) * (H * K) + h * K + k0 + rk[None, :], mask=mm[:, None], other=0.0)
        if W_KMAJOR:
            b = tl.load(w_ptr + h * (N * K) + rn[None, :] * K + k0 + rk[:, None])
        else:
            b = tl.load(w_ptr + h * (K * N) + (k0 + rk[:, None]) * N + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(o_ptr + rm[:, None].to(tl.int64) * (H * N) + h * N + rn[None, :], acc.to(tl.bfloat16), mask=mm[:, None])


def vup_triton(x, w, out, BM=128, BN=128, BK=64, stages=3, warps=4, w_kmajor=False):
    M, H, K = x.shape
    N = out.shape[-1]
    grid = (H, triton.cdiv(M, BM), N // BN)
    _vup_kernel[grid](x, w, out, M, H=H, K=K, N=N, BM=BM, BN=BN, BK=BK, W_KMAJOR=w_kmajor,
                      num_stages=stages, num_warps=warps)
    return out
