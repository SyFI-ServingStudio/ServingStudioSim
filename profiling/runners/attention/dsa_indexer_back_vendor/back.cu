// Vendored DSA indexer back kernels: the source the dsa_pooled_topk_page_table (entry e2a, 512 threads) and
// dsa_pooled_logits_page_table (entry topk6, stage 4, pf 1) rows were measured on. Kept byte-identical to the
// measured source below this three-line header; tests/test_dsa_pooled_page_table.py pins it.
// Lane E: DSA indexer back for GLM-5.3-Flash decode (B200).
//  - e2a: one kernel replacing fill / int64 cast / seqlen / expand_pools_and_append_tail / copy / remap
//         (bit-identical to the fork chain, including COMPACT_TO_FRONT order and valid counts)
//  - pf0 / pf1: post-top-k floor kernels (flat bytes; + the dependent block-table gather)
//  - topk<NT, STAGE>: own top-512 over pool logits, one CTA per row, sample threshold + filter + exact
//         radix select in smem (exact fallback over the global row); STAGE ladder:
//           0 stream the row (sum)            -> design floor of the row-per-CTA read
//           1 + sample threshold
//           2 + filter + candidate append
//           3 + exact select, writes pool_topk [M, 512]
//           4 + fused e2a epilogue, writes phys [M, 2176] + valid [M]
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>

#define FULL 0xffffffffu
constexpr int KPOOL = 4;
constexpr int SELK = 512;
constexpr int TOPK = SELK * KPOOL;  // 2048
constexpr int PT_W = 2176;
constexpr int CAP = 4096;           // smem candidate capacity
constexpr int NSAMP_F4 = 512;       // 512 float4 = 2048 samples (256 chunks x 8 floats)

__device__ __forceinline__ int lane_id() { return threadIdx.x & 31; }
__device__ __forceinline__ unsigned lanemask_lt() { unsigned m; asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m)); return m; }
__device__ __forceinline__ uint32_t fkey(float f) {
  uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// exclusive scan over NT threads; returns exclusive prefix, writes block total to *total. s_w: >= NT/32+1 ints
template <int NT>
__device__ __forceinline__ int block_excl_scan(int v, int* total, int* s_w) {
  constexpr int NW = NT / 32;
  int lane = lane_id(), w = threadIdx.x >> 5;
  int incl = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(FULL, incl, o); if (lane >= o) incl += y; }
  if (lane == 31) s_w[w] = incl;
  __syncthreads();
  if (w == 0) {
    int x = lane < NW ? s_w[lane] : 0;
    int xi = x;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(FULL, xi, o); if (lane >= o) xi += y; }
    if (lane < NW) s_w[lane] = xi - x;
    if (lane == 31) s_w[NW] = xi;
  }
  __syncthreads();
  int r = s_w[w] + incl - v;
  *total = s_w[NW];
  __syncthreads();
  return r;
}

// ------------------------------------------------------------------------------------------------
// post-top-k row (fork semantics). Thread t owns pools [PPT*t, PPT*t+PPT) (columns 4*PPT*t ...).
// pid: pool ids (int32, -1 = none). Columns [TOPK, TOPK+3) are the tail, [TOPK+3, PT_W) are -1.
template <int NT, int PPT>
__device__ __forceinline__ void post_row(const int (&pid)[PPT], int req, int seq, const int* __restrict__ bt,
                                         long bt_s0, long bt_s1, int max_blocks, int bsr,
                                         int* __restrict__ out_row, int* __restrict__ valid_out, int* s_w) {
  const int t = threadIdx.x, lane = lane_id();
  // history: one block-table load per pool (the 4 tokens of a pool share sign and page)
  bool ok[PPT];
  int base[PPT];
  int tok0[PPT];
#pragma unroll
  for (int j = 0; j < PPT; ++j) {
    tok0[j] = (int)((unsigned)pid[j] * 4u);  // == int32(int64(pid) * 4)
    int blk = tok0[j] >> 6;
    ok[j] = pid[j] >= 0 && tok0[j] >= 0 && blk < max_blocks;
    base[j] = ok[j] ? __ldg(bt + req * bt_s0 + blk * bt_s1) : 0;
  }
  // tail token j = t (t < 3)
  bool tok_ok = false;
  int tphys = 0;
  if (t < 3) {
    int pool_len = seq / KPOOL;
    int tail_start = pool_len * KPOOL;
    int tail_count = seq - tail_start;
    int tok = tail_start + t;
    if (t < tail_count && tok >= 0) {
      int blk = tok >> 6;
      if (blk < max_blocks) {
        tok_ok = true;
        int b = __ldg(bt + req * bt_s0 + blk * bt_s1);
        tphys = (int)((unsigned)b * (unsigned)bsr + (unsigned)(tok & 63));
      }
    }
  }
  int c = 0;
#pragma unroll
  for (int j = 0; j < PPT; ++j) c += ok[j] ? 4 : 0;
  int total_hist;
  int off = block_excl_scan<NT>(c, &total_hist, s_w);
  unsigned tb = __ballot_sync(FULL, tok_ok);  // only lanes 0..2 of warp 0 can be set
  if (t == 0) s_w[0] = __popc(tb);            // broadcast warp 0's tail count
  __syncthreads();
  const int ntail = s_w[0];
  const int valid = total_hist + ntail;
  const bool all_full = __syncthreads_and(c == 4 * PPT);
  if (all_full) {
#pragma unroll
    for (int j = 0; j < PPT; ++j) {
      int p0 = (int)((unsigned)base[j] * (unsigned)bsr + (unsigned)(tok0[j] & 63));
      reinterpret_cast<int4*>(out_row)[PPT * t + j] = make_int4(p0, p0 + 1, p0 + 2, p0 + 3);
    }
  } else {
    int d = off;
#pragma unroll
    for (int j = 0; j < PPT; ++j) {
      if (ok[j]) {
        int p0 = (int)((unsigned)base[j] * (unsigned)bsr + (unsigned)(tok0[j] & 63));
        out_row[d] = p0; out_row[d + 1] = p0 + 1; out_row[d + 2] = p0 + 2; out_row[d + 3] = p0 + 3;
        d += 4;
      }
    }
  }
  if (tok_ok) out_row[total_hist + __popc(tb & lanemask_lt())] = tphys;
  for (int col = valid + t; col < PT_W; col += NT) out_row[col] = -1;
  if (t == 0) *valid_out = valid;
}

template <int NT>
__global__ void __launch_bounds__(NT) e2a_kernel(const int* __restrict__ pool, int pool_s0, const int* __restrict__ seq_lens,
                                                 const int* __restrict__ req_id, const int* __restrict__ bt, long bt_s0,
                                                 long bt_s1, int max_blocks, int bsr, int* __restrict__ out,
                                                 int* __restrict__ valid) {
  constexpr int PPT = SELK / NT;
  __shared__ int s_w[NT / 32 + 2];
  const int r = blockIdx.x, t = threadIdx.x;
  int pid[PPT];
  const int* prow = pool + (long)r * pool_s0 + PPT * t;
  if constexpr (PPT == 4) {
    int4 p = __ldg(reinterpret_cast<const int4*>(prow));
    pid[0] = p.x; pid[1] = p.y; pid[2] = p.z; pid[3] = p.w;
  } else if constexpr (PPT == 2) {
    int2 p = __ldg(reinterpret_cast<const int2*>(prow));
    pid[0] = p.x; pid[1] = p.y;
  } else {
#pragma unroll
    for (int j = 0; j < PPT; ++j) pid[j] = __ldg(prow + j);
  }
  const int req = __ldg(req_id + r);
  const int seq = t < 3 ? __ldg(seq_lens + r) : 0;
  post_row<NT, PPT>(pid, req, seq, bt, bt_s0, bt_s1, max_blocks, bsr, out + (long)r * PT_W, valid + r, s_w);
}

// floor 0: same bytes, no dependent gather, no scan: read pool row, write the full output row
__global__ void __launch_bounds__(128) pf0_kernel(const int* __restrict__ pool, int pool_s0, const int* __restrict__ seq_lens,
                                                  int* __restrict__ out, int* __restrict__ valid) {
  const int r = blockIdx.x, t = threadIdx.x;
  int4 p = __ldg(reinterpret_cast<const int4*>(pool + (long)r * pool_s0) + t);
  int s = __ldg(seq_lens + r);
  int4* o = reinterpret_cast<int4*>(out + (long)r * PT_W);
  o[4 * t + 0] = make_int4(p.x, p.x + 1, p.x + 2, p.x + 3);
  o[4 * t + 1] = make_int4(p.y, p.y + 1, p.y + 2, p.y + 3);
  o[4 * t + 2] = make_int4(p.z, p.z + 1, p.z + 2, p.z + 3);
  o[4 * t + 3] = make_int4(p.w, p.w + 1, p.w + 2, p.w + 3);
  out[(long)r * PT_W + TOPK + t] = s + t;
  if (t == 0) valid[r] = s;
}

// floor 1: + the dependent block-table gather (one load per pool), still no scan / compaction
__global__ void __launch_bounds__(128) pf1_kernel(const int* __restrict__ pool, int pool_s0, const int* __restrict__ seq_lens,
                                                  const int* __restrict__ req_id, const int* __restrict__ bt, long bt_s0,
                                                  int bsr, int* __restrict__ out, int* __restrict__ valid) {
  const int r = blockIdx.x, t = threadIdx.x;
  int4 p = __ldg(reinterpret_cast<const int4*>(pool + (long)r * pool_s0) + t);
  int req = __ldg(req_id + r);
  int s = __ldg(seq_lens + r);
  const int* btr = bt + req * bt_s0;
  int b0 = __ldg(btr + (p.x >> 4)), b1 = __ldg(btr + (p.y >> 4)), b2 = __ldg(btr + (p.z >> 4)), b3 = __ldg(btr + (p.w >> 4));
  int4* o = reinterpret_cast<int4*>(out + (long)r * PT_W);
  int q0 = b0 * bsr + ((p.x * 4) & 63), q1 = b1 * bsr + ((p.y * 4) & 63), q2 = b2 * bsr + ((p.z * 4) & 63), q3 = b3 * bsr + ((p.w * 4) & 63);
  o[4 * t + 0] = make_int4(q0, q0 + 1, q0 + 2, q0 + 3);
  o[4 * t + 1] = make_int4(q1, q1 + 1, q1 + 2, q1 + 3);
  o[4 * t + 2] = make_int4(q2, q2 + 1, q2 + 2, q2 + 3);
  o[4 * t + 3] = make_int4(q3, q3 + 1, q3 + 2, q3 + 3);
  out[(long)r * PT_W + TOPK + t] = t < 3 ? s + t : -1;
  if (t == 0) valid[r] = s;
}

__global__ void empty_kernel() {}

// ------------------------------------------------------------------------------------------------
// top-k
__device__ __forceinline__ void hist_add(int* hist, int bin, bool active) {
  unsigned act = __ballot_sync(FULL, active);
  if (active) {
    unsigned peers = __match_any_sync(act, bin);
    if (lane_id() == __ffs(peers) - 1) atomicAdd(&hist[bin], __popc(peers));
  }
}

// bins are ordered high (nbins-1) to low (0). Find bin b with above(b) < R <= above(b) + hist[b].
// result -> s_res[0] = bin, s_res[1] = above. All threads must call.
template <int NT>
__device__ __forceinline__ void find_bin(const int* hist, int nbins, int R, int* s_res, int* s_w) {
  const int t = threadIdx.x;
  const int bpt = nbins >= NT ? nbins / NT : 1;
  const int hi = nbins - t * bpt;  // exclusive upper bin
  int sum = 0;
  if (hi > 0)
    for (int b = hi - 1; b >= hi - bpt; --b) sum += hist[b];
  int tot;
  int above = block_excl_scan<NT>(sum, &tot, s_w);
  if (hi > 0 && above < R && above + sum >= R) {
    int acc = above;
    for (int b = hi - 1; b >= hi - bpt; --b) {
      int h = hist[b];
      if (acc + h >= R) { s_res[0] = b; s_res[1] = acc; break; }
      acc += h;
    }
  }
  __syncthreads();
}

struct TopkSmem {
  uint32_t ck[CAP];
  int ci[CAP];
  int hist[2048];
  int sel[SELK];
  int w[40];
  int res[4];
  int cnt, nsel, neq, flag;
};

template <int NT>
__device__ __forceinline__ void zero_hist(int* h, int n) {
  for (int i = threadIdx.x; i < n; i += NT) h[i] = 0;
  __syncthreads();
}

// warp-aggregated append of idx to sm.sel when take
__device__ __forceinline__ void sel_append(TopkSmem& sm, bool take, int idx) {
  unsigned b = __ballot_sync(FULL, take);
  if (b) {
    int base = 0;
    if (lane_id() == 0) base = atomicAdd(&sm.nsel, __popc(b));
    base = __shfl_sync(FULL, base, 0);
    if (take) {
      int p = base + __popc(b & lanemask_lt());
      if (p < SELK) sm.sel[p] = idx;
    }
  }
}

// exact radix select over keys provided by a getter (candidates in smem or the global row).
// Returns threshold key T and R = how many keys == T to take, in sm.res[2], sm.res[3].
template <int NT, typename Get>
__device__ __forceinline__ void radix_select(TopkSmem& sm, int n, int R, Get get) {
  uint32_t prefix = 0, mask = 0;
#pragma unroll 1
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    zero_hist<NT>(sm.hist, 256);
    const int nit = (n + NT - 1) / NT;
    for (int it = 0; it < nit; ++it) {
      int i = it * NT + threadIdx.x;
      uint32_t k = i < n ? get(i) : 0u;
      bool act = i < n && (k & mask) == prefix;
      hist_add(sm.hist, (k >> shift) & 255, act);
    }
    __syncthreads();
    find_bin<NT>(sm.hist, 256, R, sm.res, sm.w);
    int d = sm.res[0], a = sm.res[1];
    prefix |= (uint32_t)d << shift;
    mask |= 255u << shift;
    R -= a;
    __syncthreads();
  }
  if (threadIdx.x == 0) { sm.res[2] = (int)prefix; sm.res[3] = R; }
  __syncthreads();
}

// gather: key > T always, key == T first R tickets
template <int NT, typename Get, typename Idx>
__device__ __forceinline__ void gather_sel(TopkSmem& sm, int n, uint32_t T, int R, Get get, Idx idxf) {
  const int nit = (n + NT - 1) / NT;
  for (int it = 0; it < nit; ++it) {
    int i = it * NT + threadIdx.x;
    uint32_t k = i < n ? get(i) : 0u;
    bool gt = i < n && k > T;
    bool eq = i < n && k == T;
    unsigned be = __ballot_sync(FULL, eq);
    bool take_eq = false;
    if (be) {
      int base = 0;
      if (lane_id() == 0) base = atomicAdd(&sm.neq, __popc(be));
      base = __shfl_sync(FULL, base, 0);
      take_eq = eq && (base + __popc(be & lanemask_lt())) < R;
    }
    sel_append(sm, gt || take_eq, i < n ? idxf(i) : -1);
  }
  __syncthreads();
}

template <int NT, int STAGE, int U>
__global__ void __launch_bounds__(NT) topk_kernel(const float* __restrict__ logits, long ls0, const int* __restrict__ row_lens,
                                                  int* __restrict__ pool_out, float* __restrict__ dummy,
                                                  int* __restrict__ stats,
                                                  // fused epilogue
                                                  const int* __restrict__ seq_lens, const int* __restrict__ req_id,
                                                  const int* __restrict__ bt, long bt_s0, long bt_s1, int max_blocks,
                                                  int bsr, int* __restrict__ out, int* __restrict__ valid) {
  extern __shared__ __align__(16) unsigned char smraw[];
  TopkSmem& sm = *reinterpret_cast<TopkSmem*>(smraw);
  const int r = blockIdx.x, t = threadIdx.x, lane = lane_id();
  const int L = row_lens[r];
  const float* row = logits + (long)r * ls0;
  const int keff = L < SELK ? (L > 0 ? L : 0) : SELK;
  if (t == 0) { sm.cnt = 0; sm.nsel = 0; sm.neq = 0; sm.flag = 0; }
  // issue the fused epilogue's independent loads early
  int req = 0, seq = 0;
  if constexpr (STAGE == 4) { req = __ldg(req_id + r); seq = t < 3 ? __ldg(seq_lens + r) : 0; }

  // ---------------- sample threshold
  uint32_t tau = 0;
  if constexpr (STAGE >= 1) {
    if (L > CAP) {
      zero_hist<NT>(sm.hist, 2048);  // also publishes sm.cnt = 0
      uint32_t sk[NSAMP_F4 / NT * 4];
#pragma unroll
      for (int j = 0; j < NSAMP_F4 / NT; ++j) {
        int s = j * NT + t;  // float4 slot: chunk s>>1, half s&1
        int chunk = s >> 1;
        long start = ((long)chunk * (L - 8) / 255) & ~3L;
        float4 v = __ldg(reinterpret_cast<const float4*>(row + start) + (s & 1));
        sk[4 * j + 0] = fkey(v.x); sk[4 * j + 1] = fkey(v.y); sk[4 * j + 2] = fkey(v.z); sk[4 * j + 3] = fkey(v.w);
      }
      // target rank in the sample: expected candidates ~ 3k
      int Rs = (int)((long)(3 * SELK) * (NSAMP_F4 * 4) / L);
      Rs = max(1, min(Rs, NSAMP_F4 * 4));
#pragma unroll
      for (int j = 0; j < NSAMP_F4 / NT * 4; ++j) hist_add(sm.hist, sk[j] >> 21, true);
      __syncthreads();
      find_bin<NT>(sm.hist, 2048, Rs, sm.res, sm.w);
      const uint32_t b1 = sm.res[0];
      const int R2 = Rs - sm.res[1];
      zero_hist<NT>(sm.hist, 2048);
#pragma unroll
      for (int j = 0; j < NSAMP_F4 / NT * 4; ++j) hist_add(sm.hist, (sk[j] >> 10) & 2047, (sk[j] >> 21) == b1);
      __syncthreads();
      find_bin<NT>(sm.hist, 2048, R2, sm.res, sm.w);
      tau = (b1 << 21) | ((uint32_t)sm.res[0] << 10);
    } else {
      __syncthreads();
    }
  } else {
    __syncthreads();
  }

  // ---------------- stream the row
  const int n4 = L >> 2;
  const int per_batch = U * NT;
  const int nb = (n4 + per_batch - 1) / per_batch;
  float acc = 0.f;
#pragma unroll 1
  for (int b = 0; b < nb; ++b) {
    float4 v[U];
#pragma unroll
    for (int j = 0; j < U; ++j) {
      int i4 = b * per_batch + j * NT + t;
      v[j] = i4 < n4 ? __ldg(reinterpret_cast<const float4*>(row) + i4) : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
    }
    if constexpr (STAGE <= 1) {
#pragma unroll
      for (int j = 0; j < U; ++j) acc += v[j].x + v[j].y + v[j].z + v[j].w;
    } else {
      int nh = 0;
#pragma unroll
      for (int j = 0; j < U; ++j) {
        int i4 = b * per_batch + j * NT + t;
        bool in = i4 < n4;
        nh += (in && fkey(v[j].x) >= tau) + (in && fkey(v[j].y) >= tau) + (in && fkey(v[j].z) >= tau) + (in && fkey(v[j].w) >= tau);
      }
      int wt = __reduce_add_sync(FULL, nh);
      if (wt) {
        int incl = nh;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(FULL, incl, o); if (lane >= o) incl += y; }
        int base = 0;
        if (lane == 31) base = atomicAdd(&sm.cnt, incl);
        base = __shfl_sync(FULL, base, 31);
        int p = base + incl - nh;
        if (nh) {
#pragma unroll
          for (int j = 0; j < U; ++j) {
            int i4 = b * per_batch + j * NT + t;
            if (i4 < n4) {
              float e[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
              for (int c = 0; c < 4; ++c) {
                uint32_t k = fkey(e[c]);
                if (k >= tau) {
                  if (p < CAP) { sm.ck[p] = k; sm.ci[p] = 4 * i4 + c; }
                  ++p;
                }
              }
            }
          }
        }
      }
    }
  }
  // remainder elements (L % 4)
  if constexpr (STAGE >= 2) {
    const int rem = L & 3;
    bool hit = false;
    uint32_t k = 0;
    int idx = 4 * n4 + t;
    if (t < rem) { float x = __ldg(row + idx); k = fkey(x); hit = k >= tau; }
    if (t < 32) {
      unsigned bm = __ballot_sync(FULL, hit);
      if (bm) {
        int base = 0;
        if (lane == 0) base = atomicAdd(&sm.cnt, __popc(bm));
        base = __shfl_sync(FULL, base, 0);
        if (hit) { int p = base + __popc(bm & lanemask_lt()); if (p < CAP) { sm.ck[p] = k; sm.ci[p] = idx; } }
      }
    }
  } else {
    if (t < (L & 3)) acc += __ldg(row + 4 * n4 + t);
  }
  __syncthreads();

  if constexpr (STAGE <= 2) {
    // prevent dead-code elimination; write one value per row
    float s = acc;
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
    if (lane == 0) atomicAdd(dummy + r, s + (float)sm.cnt * (STAGE == 2 ? 1.f : 0.f) + (float)tau * 0.f);
    if (STAGE >= 1 && t == 0 && stats) stats[r] = sm.cnt + (int)(tau & 1u) * 0;
    return;
  } else {
    // ---------------- exact select
    const int C = sm.cnt;
    const bool fallback = (C > CAP) || (C < keff);
    if (!fallback) {
      if (C == keff) {
        for (int i = t; i < C; i += NT) sm.sel[i] = sm.ci[i];
      } else {
        radix_select<NT>(sm, C, keff, [&](int i) { return sm.ck[i]; });
        const uint32_t T = (uint32_t)sm.res[2];
        const int R = sm.res[3];
        gather_sel<NT>(sm, C, T, R, [&](int i) { return sm.ck[i]; }, [&](int i) { return sm.ci[i]; });
      }
    } else if (keff > 0) {
      // exact select over the global row (slow path; correctness for any distribution)
      auto gk = [&](int i) { return fkey(__ldg(row + i)); };
      radix_select<NT>(sm, L, keff, gk);
      const uint32_t T = (uint32_t)sm.res[2];
      const int R = sm.res[3];
      gather_sel<NT>(sm, L, T, R, gk, [&](int i) { return i; });
    }
    for (int i = keff + t; i < SELK; i += NT) sm.sel[i] = -1;
    if (t == 0 && stats) stats[r] = fallback ? -C - 1 : C;
    __syncthreads();
    if constexpr (STAGE == 3) {
      for (int i = t; i < SELK; i += NT) pool_out[(long)r * SELK + i] = sm.sel[i];
    } else {
      constexpr int PPT = SELK / NT;
      int pid[PPT];
#pragma unroll
      for (int j = 0; j < PPT; ++j) pid[j] = sm.sel[PPT * t + j];
      post_row<NT, PPT>(pid, req, seq, bt, bt_s0, bt_s1, max_blocks, bsr, out + (long)r * PT_W, valid + r, sm.w);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// top-k v2 (job 2): float-compare + per-slot ballot filter (cuts instructions per element), first batch prefetched
// before the sample threshold, double-buffered stream, plain-atomic sample histograms, select variants
//   SELV 0: 4 x 8-bit passes, match_any aggregation (v1)    SELV 1: plain atomics + early exit
// stamps (int64 [M, 8], optional): globaltimer at start / tau ready / stream done / select done / end
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ float key2f(uint32_t k) {
  return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

template <int NT, int SELV, typename Get>
__device__ __forceinline__ void radix_select2(TopkSmem& sm, int n, int R, Get get) {
  uint32_t prefix = 0, mask = 0;
#pragma unroll 1
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    zero_hist<NT>(sm.hist, 256);
    const int nit = (n + NT - 1) / NT;
    for (int it = 0; it < nit; ++it) {
      int i = it * NT + threadIdx.x;
      uint32_t k = i < n ? get(i) : 0u;
      bool act = i < n && (k & mask) == prefix;
      if constexpr (SELV == 0) hist_add(sm.hist, (k >> shift) & 255, act);
      else if (act) atomicAdd(&sm.hist[(k >> shift) & 255], 1);
    }
    __syncthreads();
    find_bin<NT>(sm.hist, 256, R, sm.res, sm.w);
    const int d = sm.res[0], a = sm.res[1];
    const int h = sm.hist[d];
    prefix |= (uint32_t)d << shift;
    mask |= 255u << shift;
    R -= a;
    __syncthreads();
    if (SELV == 1 && h == R) { R = 0x7fffffff; break; }  // the whole bin is taken: select key >= prefix
  }
  if (threadIdx.x == 0) { sm.res[2] = (int)prefix; sm.res[3] = R; }
  __syncthreads();
}


// SELV 2: range-adaptive 11-bit passes over [lo, hi] of the candidate keys (bins spread over the live range), early exit
template <int NT, typename Get>
__device__ __forceinline__ void radix_select3(TopkSmem& sm, int n, int R, Get get) {
  if (threadIdx.x == 0) { sm.res[0] = -1; sm.res[1] = 0; }  // res[0]: min (as uint max), res[1]: max
  __syncthreads();
  uint32_t mn = 0xffffffffu, mx = 0u;
  for (int i = threadIdx.x; i < n; i += NT) { uint32_t k = get(i); if (k) { mn = min(mn, k); mx = max(mx, k); } }  // key 0 = empty slot
  mn = __reduce_min_sync(FULL, mn); mx = __reduce_max_sync(FULL, mx);
  if (lane_id() == 0) { atomicMin(reinterpret_cast<unsigned*>(&sm.res[0]), mn); atomicMax(reinterpret_cast<unsigned*>(&sm.res[1]), mx); }
  __syncthreads();
  uint32_t base = (uint32_t)sm.res[0], hi = (uint32_t)sm.res[1];
  __syncthreads();
#pragma unroll 1
  while (true) {
    const uint32_t range = hi - base;
    if (range == 0) break;  // all live keys equal base: T = base, R tickets
    const int nbits = 32 - __clz(range);
    const int shift = nbits > 11 ? nbits - 11 : 0;
    zero_hist<NT>(sm.hist, 2048);
    const int nit = (n + NT - 1) / NT;
    for (int it = 0; it < nit; ++it) {
      int i = it * NT + threadIdx.x;
      if (i < n) {
        uint32_t k = get(i);
        if (k && k >= base && k <= hi) atomicAdd(&sm.hist[(k - base) >> shift], 1);
      }
    }
    __syncthreads();
    find_bin<NT>(sm.hist, 2048, R, sm.res, sm.w);
    const uint32_t b = (uint32_t)sm.res[0];
    const int a = sm.res[1];
    const int h = sm.hist[b];
    __syncthreads();
    R -= a;
    const uint32_t nbase = base + (b << shift);
    const uint32_t top = (shift >= 32) ? 0xffffffffu : nbase + ((1u << shift) - 1u);
    hi = min(hi, top);
    base = nbase;
    if (shift == 0) break;                       // exact key
    if (h == R) { R = 0x7fffffff; break; }       // whole bin taken: select key >= base
  }
  if (threadIdx.x == 0) { sm.res[2] = (int)base; sm.res[3] = R; }
  __syncthreads();
}

template <int NT, int STAGE, int U, int SELV>
__global__ void __launch_bounds__(NT, 2) topk2_kernel(const float* __restrict__ logits, long ls0, const int* __restrict__ row_lens,
                                                   int* __restrict__ pool_out, float* __restrict__ dummy,
                                                   int* __restrict__ stats, long long* __restrict__ stamps,
                                                   const int* __restrict__ seq_lens, const int* __restrict__ req_id,
                                                   const int* __restrict__ bt, long bt_s0, long bt_s1, int max_blocks,
                                                   int bsr, int* __restrict__ out, int* __restrict__ valid) {
  extern __shared__ __align__(16) unsigned char smraw[];
  TopkSmem& sm = *reinterpret_cast<TopkSmem*>(smraw);
  const int r = blockIdx.x, t = threadIdx.x, lane = lane_id();
  long long* stp = stamps ? stamps + (long)r * 8 : nullptr;
  if (stp && t == 0) stp[0] = gtime();
  const int L = row_lens[r];
  const float* row = logits + (long)r * ls0;
  const float4* row4 = reinterpret_cast<const float4*>(row);
  const int keff = L < SELK ? (L > 0 ? L : 0) : SELK;
  if (t == 0) { sm.cnt = 0; sm.nsel = 0; sm.neq = 0; sm.flag = 0; }
  int req = 0, seq = 0;
  if constexpr (STAGE == 4) { req = __ldg(req_id + r); seq = t < 3 ? __ldg(seq_lens + r) : 0; }
  const int n4 = L >> 2;
  constexpr int PB = U * NT;
  const int nb = (n4 + PB - 1) / PB;
  const float NANF = __int_as_float(0x7fc00000);
  float4 v[U];
#pragma unroll
  for (int j = 0; j < U; ++j) { int i4 = j * NT + t; v[j] = i4 < n4 ? __ldg(row4 + i4) : make_float4(NANF, NANF, NANF, NANF); }

  // ---------------- sample threshold (overlaps the first batch's loads)
  float tf = -INFINITY;
  if constexpr (STAGE >= 1) {
    if (L > CAP) {
      zero_hist<NT>(sm.hist, 2048);
      constexpr int SJ = NSAMP_F4 / NT;
      uint32_t sk[SJ * 4];
#pragma unroll
      for (int j = 0; j < SJ; ++j) {
        int s = j * NT + t;
        long start = ((long)(s >> 1) * (L - 8) / 255) & ~3L;
        float4 q = __ldg(reinterpret_cast<const float4*>(row + start) + (s & 1));
        sk[4 * j + 0] = fkey(q.x); sk[4 * j + 1] = fkey(q.y); sk[4 * j + 2] = fkey(q.z); sk[4 * j + 3] = fkey(q.w);
      }
      int Rs = (int)((long)(2 * SELK) * (NSAMP_F4 * 4) / L);
      Rs = max(1, min(Rs, NSAMP_F4 * 4));
#pragma unroll
      for (int j = 0; j < SJ * 4; ++j) atomicAdd(&sm.hist[sk[j] >> 21], 1);
      __syncthreads();
      find_bin<NT>(sm.hist, 2048, Rs, sm.res, sm.w);
      const uint32_t b1 = sm.res[0];
      const int R2 = Rs - sm.res[1];
      zero_hist<NT>(sm.hist, 2048);
#pragma unroll
      for (int j = 0; j < SJ * 4; ++j)
        if ((sk[j] >> 21) == b1) atomicAdd(&sm.hist[(sk[j] >> 10) & 2047], 1);
      __syncthreads();
      find_bin<NT>(sm.hist, 2048, R2, sm.res, sm.w);
      tf = key2f((b1 << 21) | ((uint32_t)sm.res[0] << 10));
    } else {
      __syncthreads();
    }
  } else {
    __syncthreads();
  }
  if (stp && t == 0) stp[1] = gtime();

  // ---------------- stream (double-buffered: batch b+1 loads issued before batch b is filtered)
  float acc = 0.f;
#pragma unroll 1
  for (int b = 0; b < nb; ++b) {
    float4 w[U];
    const bool more = b + 1 < nb;
#pragma unroll
    for (int j = 0; j < U; ++j) {
      int i4 = (b + 1) * PB + j * NT + t;
      w[j] = (more && i4 < n4) ? __ldg(row4 + i4) : make_float4(NANF, NANF, NANF, NANF);
    }
    if constexpr (STAGE <= 1) {
#pragma unroll
      for (int j = 0; j < U; ++j) acc += fmaxf(fmaxf(v[j].x, v[j].y), fmaxf(v[j].z, v[j].w));
    } else {
#pragma unroll
      for (int j = 0; j < U; ++j) {
        const int i4 = b * PB + j * NT + t;
        const float e[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const bool hit = e[c] >= tf;
          const unsigned bm = __ballot_sync(FULL, hit);
          if (bm) {
            int base = 0;
            if (lane == 0) base = atomicAdd(&sm.cnt, __popc(bm));
            base = __shfl_sync(FULL, base, 0);
            if (hit) {
              int p = base + __popc(bm & lanemask_lt());
              if (p < CAP) { sm.ck[p] = fkey(e[c]); sm.ci[p] = 4 * i4 + c; }
            }
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < U; ++j) v[j] = w[j];
  }
  // remainder (L % 4)
  {
    const int rem = L & 3;
    float x = NANF;
    if (t < rem) x = __ldg(row + 4 * n4 + t);
    if constexpr (STAGE >= 2) {
      if (t < 32) {
        bool hit = x >= tf;
        unsigned bm = __ballot_sync(FULL, hit);
        if (bm) {
          int base = 0;
          if (lane == 0) base = atomicAdd(&sm.cnt, __popc(bm));
          base = __shfl_sync(FULL, base, 0);
          if (hit) { int p = base + __popc(bm & lanemask_lt()); if (p < CAP) { sm.ck[p] = fkey(x); sm.ci[p] = 4 * n4 + t; } }
        }
      }
    } else if (t < rem) acc += x;
  }
  __syncthreads();
  if (stp && t == 0) stp[2] = gtime();

  if constexpr (STAGE <= 2) {
    float s = acc;
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
    if (lane == 0) atomicAdd(dummy + r, s + (float)sm.cnt + tf * 0.f);
    if (t == 0 && stats) stats[r] = sm.cnt;
    if (stp && t == 0) { stp[3] = stp[4] = gtime(); }
    return;
  } else {
    const int C = sm.cnt;
    const bool fallback = (C > CAP) || (C < keff);
    if (!fallback) {
      if (C == keff) {
        for (int i = t; i < C; i += NT) sm.sel[i] = sm.ci[i];
      } else {
        if constexpr (SELV == 2) radix_select3<NT>(sm, C, keff, [&](int i) { return sm.ck[i]; });
        else radix_select2<NT, SELV>(sm, C, keff, [&](int i) { return sm.ck[i]; });
        const uint32_t T = (uint32_t)sm.res[2];
        const int R = sm.res[3];
        gather_sel<NT>(sm, C, T, R, [&](int i) { return sm.ck[i]; }, [&](int i) { return sm.ci[i]; });
      }
    } else if (keff > 0) {
      auto gk = [&](int i) { return fkey(__ldg(row + i)); };
      radix_select2<NT, 0>(sm, L, keff, gk);
      const uint32_t T = (uint32_t)sm.res[2];
      const int R = sm.res[3];
      gather_sel<NT>(sm, L, T, R, gk, [&](int i) { return i; });
    }
    for (int i = keff + t; i < SELK; i += NT) sm.sel[i] = -1;
    if (t == 0 && stats) stats[r] = fallback ? -C - 1 : C;
    __syncthreads();
    if (stp && t == 0) stp[3] = gtime();
    if constexpr (STAGE == 3) {
      for (int i = t; i < SELK; i += NT) pool_out[(long)r * SELK + i] = sm.sel[i];
    } else {
      constexpr int PPT = SELK / NT;
      int pid[PPT];
#pragma unroll
      for (int j = 0; j < PPT; ++j) pid[j] = sm.sel[PPT * t + j];
      post_row<NT, PPT>(pid, req, seq, bt, bt_s0, bt_s1, max_blocks, bsr, out + (long)r * PT_W, valid + r, sm.w);
    }
    if (stp && t == 0) stp[4] = gtime();
  }
}

template <int NT, int STAGE, int U, int SELV>
static void launch_topk2(int M, const float* lg, long ls0, const int* lens, int* pout, float* dummy, int* stats, long long* stamps,
                         const int* seq, const int* req, const int* bt, long bs0, long bs1, int maxb, int bsr, int* out, int* valid) {
  auto k = topk2_kernel<NT, STAGE, U, SELV>;
  int smem = sizeof(TopkSmem);
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  k<<<M, NT, smem, at::cuda::getCurrentCUDAStream()>>>(lg, ls0, lens, pout, dummy, stats, stamps, seq, req, bt, bs0, bs1, maxb, bsr, out, valid);
}

void topk2(torch::Tensor logits, torch::Tensor lens, int64_t stage, int64_t selv, torch::Tensor pool_out, torch::Tensor dummy,
           torch::Tensor stats, torch::Tensor stamps, torch::Tensor seq, torch::Tensor req, torch::Tensor bt, int64_t bsr,
           torch::Tensor out, torch::Tensor valid) {
  int M = lens.numel();
  TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) % 4 == 0);
  long long* sp = stamps.numel() ? reinterpret_cast<long long*>(stamps.data_ptr<int64_t>()) : nullptr;
  int* st = stats.numel() ? stats.data_ptr<int>() : nullptr;
#define L2_(S_, V_)                                                                                                         \
  if (stage == S_ && selv == V_) {                                                                                          \
    launch_topk2<512, S_, 4, V_>(M, logits.data_ptr<float>(), logits.stride(0), lens.data_ptr<int>(), pool_out.data_ptr<int>(), \
                                 dummy.data_ptr<float>(), st, sp, seq.data_ptr<int>(), req.data_ptr<int>(), bt.data_ptr<int>(),  \
                                 bt.stride(0), bt.stride(1), bt.size(1), bsr, out.data_ptr<int>(), valid.data_ptr<int>());     \
    return;                                                                                                                 \
  }
  L2_(0, 0) L2_(1, 0) L2_(2, 0) L2_(3, 0) L2_(4, 0) L2_(3, 1) L2_(4, 1) L2_(3, 2) L2_(4, 2)
#undef L2_
  TORCH_CHECK(false, "bad topk2 config");
}

// ------------------------------------------------------------------------------------------------
// top-k v3 (job 3):
//  - chunk-spread stream order: batch 0 = 64 warp chunks (32 float4 each) spread evenly over the row, so batch 0 IS the
//    threshold sample (no extra loads, no extra DRAM round trip in front of the stream)
//  - warp-private candidate segments (SEG = CAP/16 per warp, counter in a register): no atomics in the stream
//  - select: range-adaptive 11-bit passes (SELV 2 of v2) over all CAP slots, empty slots hold key 0
//  - clk (int64 [M, 16], optional): clock64 at phase boundaries (thread 0), plus globaltimer start/end
constexpr int NW3 = 16;
constexpr int SEG = CAP / NW3;

template <int NT, int STAGE, int U>
__global__ void __launch_bounds__(NT, 2) topk3_kernel(const float* __restrict__ logits, long ls0, const int* __restrict__ row_lens,
                                                      int* __restrict__ pool_out, float* __restrict__ dummy,
                                                      int* __restrict__ stats, long long* __restrict__ clk,
                                                      const int* __restrict__ seq_lens, const int* __restrict__ req_id,
                                                      const int* __restrict__ bt, long bt_s0, long bt_s1, int max_blocks,
                                                      int bsr, int* __restrict__ out, int* __restrict__ valid) {
  static_assert(NT == 512, "16 warps");
  extern __shared__ __align__(16) unsigned char smraw[];
  TopkSmem& sm = *reinterpret_cast<TopkSmem*>(smraw);
  __shared__ int s_wcnt[NW3];
  const int r = blockIdx.x, t = threadIdx.x, lane = lane_id(), w = t >> 5;
  long long* ck_ = clk ? clk + (long)r * 16 : nullptr;
  if (ck_ && t == 0) { ck_[0] = gtime(); ck_[1] = clock64(); }
  const int L = row_lens[r];
  const float4* row4 = reinterpret_cast<const float4*>(logits + (long)r * ls0);
  const int keff = L < SELK ? (L > 0 ? L : 0) : SELK;
  int req = 0, seq = 0;
  if constexpr (STAGE == 4) { req = __ldg(req_id + r); seq = t < 3 ? __ldg(seq_lens + r) : 0; }
  const int n4 = L >> 2;
  constexpr int G = U * NW3;                 // warp chunks per batch
  const int nch = (n4 + 31) >> 5;
  const int Q = (nch + G - 1) / G;           // batches
  const float NANF = __int_as_float(0x7fc00000);
  auto i4_of = [&](int b, int j) { int c = (j * NW3 + w) * Q + b; return c < nch ? c * 32 + lane : n4; };
  float4 v[U];
#pragma unroll
  for (int j = 0; j < U; ++j) { int i4 = i4_of(0, j); v[j] = i4 < n4 ? __ldg(row4 + i4) : make_float4(NANF, NANF, NANF, NANF); }
  for (int i = t; i < CAP; i += NT) sm.ck[i] = 0u;
  if (t == 0) { sm.cnt = 0; sm.nsel = 0; sm.neq = 0; sm.flag = 0; }

  // ---------------- threshold from batch 0 (spread sample, already in registers)
  float tf = -INFINITY;
  if constexpr (STAGE >= 1) {
    if (L > CAP) {
      zero_hist<NT>(sm.hist, 2048);
      uint32_t sk[U * 4];
      int nv = 0;
#pragma unroll
      for (int j = 0; j < U; ++j) {
        const bool in = i4_of(0, j) < n4;
        sk[4 * j + 0] = in ? fkey(v[j].x) : 0u; sk[4 * j + 1] = in ? fkey(v[j].y) : 0u;
        sk[4 * j + 2] = in ? fkey(v[j].z) : 0u; sk[4 * j + 3] = in ? fkey(v[j].w) : 0u;
        nv += in ? 4 : 0;
      }
      if (ck_ && t == 0) ck_[2] = clock64();
      nv = __reduce_add_sync(FULL, nv);
      if (lane == 0) atomicAdd(&sm.cnt, nv);
#pragma unroll
      for (int j = 0; j < U * 4; ++j) hist_add(sm.hist, sk[j] >> 21, sk[j] != 0u);
      __syncthreads();
      const int ns = sm.cnt;
      int Rs = (int)((long)(2 * SELK) * ns / L);
      Rs = max(1, min(Rs, ns));
      find_bin<NT>(sm.hist, 2048, Rs, sm.res, sm.w);
      const uint32_t b1 = sm.res[0];
      const int R2 = Rs - sm.res[1];
      if (ck_ && t == 0) ck_[3] = clock64();
      zero_hist<NT>(sm.hist, 2048);
#pragma unroll
      for (int j = 0; j < U * 4; ++j) hist_add(sm.hist, (sk[j] >> 10) & 2047, sk[j] != 0u && (sk[j] >> 21) == b1);
      __syncthreads();
      find_bin<NT>(sm.hist, 2048, R2, sm.res, sm.w);
      tf = key2f((b1 << 21) | ((uint32_t)sm.res[0] << 10));
    } else {
      __syncthreads();
    }
  } else {
    __syncthreads();
  }
  if (ck_ && t == 0) ck_[4] = clock64();

  // ---------------- stream with warp-private segments
  float acc = 0.f;
  int wc = 0;                       // warp-uniform candidate count
  const int segb = w * SEG;
#pragma unroll 1
  for (int b = 0; b < Q; ++b) {
    float4 nx[U];
    const bool more = b + 1 < Q;
#pragma unroll
    for (int j = 0; j < U; ++j) {
      int i4 = more ? i4_of(b + 1, j) : n4;
      nx[j] = i4 < n4 ? __ldg(row4 + i4) : make_float4(NANF, NANF, NANF, NANF);
    }
    if constexpr (STAGE <= 1) {
#pragma unroll
      for (int j = 0; j < U; ++j) acc += fmaxf(fmaxf(v[j].x, v[j].y), fmaxf(v[j].z, v[j].w));
    } else {
#pragma unroll
      for (int j = 0; j < U; ++j) {
        const int i4 = i4_of(b, j);
        const float e[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const bool hit = e[c] >= tf;
          const unsigned bm = __ballot_sync(FULL, hit);
          if (bm) {
            const int p = wc + __popc(bm & lanemask_lt());
            if (hit && p < SEG) { sm.ck[segb + p] = fkey(e[c]); sm.ci[segb + p] = 4 * i4 + c; }
            wc += __popc(bm);
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < U; ++j) v[j] = nx[j];
  }
  // remainder (L % 4) -> warp 0's segment
  if constexpr (STAGE >= 2) {
    if (w == 0) {
      const int rem = L & 3;
      float x = NANF;
      if (lane < rem) x = __ldg(reinterpret_cast<const float*>(row4) + 4 * n4 + lane);
      const bool hit = x >= tf;
      const unsigned bm = __ballot_sync(FULL, hit);
      const int p = wc + __popc(bm & lanemask_lt());
      if (hit && p < SEG) { sm.ck[segb + p] = fkey(x); sm.ci[segb + p] = 4 * n4 + lane; }
      wc += __popc(bm);
    }
    if (lane == 0) s_wcnt[w] = wc;
  } else {
    if (t < (L & 3)) acc += __ldg(reinterpret_cast<const float*>(row4) + 4 * n4 + t);
  }
  __syncthreads();
  if (ck_ && t == 0) ck_[5] = clock64();

  if constexpr (STAGE <= 2) {
    float s = acc;
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
    int C = 0;
    if constexpr (STAGE == 2) for (int i = 0; i < NW3; ++i) C += s_wcnt[i];
    if (lane == 0) atomicAdd(dummy + r, s + (float)C + tf * 0.f);
    if (t == 0 && stats) stats[r] = C;
    if (ck_ && t == 0) { ck_[6] = ck_[7] = ck_[8] = clock64(); ck_[9] = gtime(); }
    return;
  } else {
    int C = 0, mx = 0;
#pragma unroll
    for (int i = 0; i < NW3; ++i) { C += s_wcnt[i]; mx = max(mx, s_wcnt[i]); }
    const bool fallback = (mx > SEG) || (C < keff);
    if (!fallback) {
      auto gk = [&](int i) { return sm.ck[i]; };
      auto gi = [&](int i) { return sm.ci[i]; };
      if (C == keff) {
        gather_sel<NT>(sm, CAP, 0u, 0, gk, gi);                  // every non-empty slot (key > 0)
        if (ck_ && t == 0) ck_[6] = clock64();
      } else {
        radix_select3<NT>(sm, CAP, keff, gk);
        if (ck_ && t == 0) ck_[6] = clock64();
        gather_sel<NT>(sm, CAP, (uint32_t)sm.res[2], sm.res[3], gk, gi);
      }
    } else if (keff > 0) {
      const float* row = reinterpret_cast<const float*>(row4);
      auto gk = [&](int i) { return fkey(__ldg(row + i)); };
      radix_select2<NT, 0>(sm, L, keff, gk);
      if (ck_ && t == 0) ck_[6] = clock64();
      gather_sel<NT>(sm, L, (uint32_t)sm.res[2], sm.res[3], gk, [&](int i) { return i; });
    }
    for (int i = keff + t; i < SELK; i += NT) sm.sel[i] = -1;
    if (t == 0 && stats) stats[r] = fallback ? -C - 1 : C;
    __syncthreads();
    if (ck_ && t == 0) ck_[7] = clock64();
    if constexpr (STAGE == 3) {
      for (int i = t; i < SELK; i += NT) pool_out[(long)r * SELK + i] = sm.sel[i];
    } else {
      int pid[1] = {sm.sel[t]};
      post_row<NT, 1>(pid, req, seq, bt, bt_s0, bt_s1, max_blocks, bsr, out + (long)r * PT_W, valid + r, sm.w);
    }
    if (ck_ && t == 0) { ck_[8] = clock64(); ck_[9] = gtime(); }
  }
}

template <int STAGE>
static void launch_topk3(int M, const float* lg, long ls0, const int* lens, int* pout, float* dummy, int* stats, long long* clk,
                         const int* seq, const int* req, const int* bt, long bs0, long bs1, int maxb, int bsr, int* out, int* valid) {
  auto k = topk3_kernel<512, STAGE, 4>;
  int smem = sizeof(TopkSmem);
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  k<<<M, 512, smem, at::cuda::getCurrentCUDAStream()>>>(lg, ls0, lens, pout, dummy, stats, clk, seq, req, bt, bs0, bs1, maxb, bsr, out, valid);
}

void topk3(torch::Tensor logits, torch::Tensor lens, int64_t stage, torch::Tensor pool_out, torch::Tensor dummy,
           torch::Tensor stats, torch::Tensor clk, torch::Tensor seq, torch::Tensor req, torch::Tensor bt, int64_t bsr,
           torch::Tensor out, torch::Tensor valid) {
  int M = lens.numel();
  TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) % 4 == 0);
  long long* cp = clk.numel() ? reinterpret_cast<long long*>(clk.data_ptr<int64_t>()) : nullptr;
  int* st = stats.numel() ? stats.data_ptr<int>() : nullptr;
#define L3_(S_)                                                                                                             \
  if (stage == S_) {                                                                                                        \
    launch_topk3<S_>(M, logits.data_ptr<float>(), logits.stride(0), lens.data_ptr<int>(), pool_out.data_ptr<int>(),         \
                     dummy.data_ptr<float>(), st, cp, seq.data_ptr<int>(), req.data_ptr<int>(), bt.data_ptr<int>(),        \
                     bt.stride(0), bt.stride(1), bt.size(1), bsr, out.data_ptr<int>(), valid.data_ptr<int>());             \
    return;                                                                                                                 \
  }
  L3_(0) L3_(1) L3_(2) L3_(3) L3_(4)
#undef L3_
  TORCH_CHECK(false, "bad topk3 stage");
}

// ------------------------------------------------------------------------------------------------
// top-k v4 (job 4): no shared-memory histograms anywhere (job 3: hot-bin atomics / match_any cost 6-11 µs in the sample
// threshold and ~5 µs in select+gather). Thresholds come from a block-wide 16-ary count search on keys held in registers:
// each step counts keys >= T | (j << shift), j = 1..15, per warp with redux, 16 warps summed through smem.
//  - tau: 2048 samples (4 keys per thread from batch 0), 3 steps (12-bit lower bound), target 2.2k candidates
//  - select: candidates from the warp's own segment into 8 registers (warp-uniform count), exact search from the common
//    prefix of min/max in 4-bit steps, count(k > T), one packed block scan places gt + tie picks (deterministic order)
struct CS {
  int c[NW3][16];
  int tot[16];
  int mn, mx;
};

template <int NT, int NK>
__device__ __forceinline__ int count_step16(const uint32_t (&k)[NK], int mlim, uint32_t T, int shift, int R, CS& cs) {
  const int lane = lane_id(), w = threadIdx.x >> 5;
  int c[15];
#pragma unroll
  for (int j = 0; j < 15; ++j) c[j] = 0;
#pragma unroll
  for (int m = 0; m < NK; ++m) {
    if (m < mlim) {
      const uint32_t km = k[m];
      const uint32_t d = km >= T ? min((km - T) >> shift, 15u) : 0u;   // T has no bits below shift + 4
#pragma unroll
      for (int j = 0; j < 15; ++j) c[j] += d > (uint32_t)j;
    }
  }
#pragma unroll
  for (int j = 0; j < 15; ++j) {
    int s = __reduce_add_sync(FULL, c[j]);
    if (lane == 0) cs.c[w][j] = s;
  }
  __syncthreads();
  if (threadIdx.x < 15) {
    int s = 0;
#pragma unroll
    for (int q = 0; q < NT / 32; ++q) s += cs.c[q][threadIdx.x];
    cs.tot[threadIdx.x] = s;
  }
  __syncthreads();
  int best = 0;
#pragma unroll
  for (int j = 0; j < 15; ++j) best = cs.tot[j] >= R ? j + 1 : best;   // counts are non-increasing in j
  __syncthreads();
  return best;
}

template <int NT, int NK>
__device__ __forceinline__ int block_count_gt(const uint32_t (&k)[NK], int mlim, uint32_t T, CS& cs) {
  int c = 0;
#pragma unroll
  for (int m = 0; m < NK; ++m) if (m < mlim) c += k[m] > T;
  c = __reduce_add_sync(FULL, c);
  if (lane_id() == 0) cs.c[threadIdx.x >> 5][15] = c;
  __syncthreads();
  int s = 0;
#pragma unroll
  for (int q = 0; q < NT / 32; ++q) s += cs.c[q][15];
  __syncthreads();
  return s;
}

template <int NT, int STAGE, int U>
__global__ void __launch_bounds__(NT, 2) topk4_kernel(const float* __restrict__ logits, long ls0, const int* __restrict__ row_lens,
                                                      int* __restrict__ pool_out, float* __restrict__ dummy,
                                                      int* __restrict__ stats, long long* __restrict__ clk,
                                                      const int* __restrict__ seq_lens, const int* __restrict__ req_id,
                                                      const int* __restrict__ bt, long bt_s0, long bt_s1, int max_blocks,
                                                      int bsr, int* __restrict__ out, int* __restrict__ valid) {
  static_assert(NT == 512, "16 warps");
  extern __shared__ __align__(16) unsigned char smraw[];
  TopkSmem& sm = *reinterpret_cast<TopkSmem*>(smraw);
  __shared__ int s_wcnt[NW3];
  __shared__ CS cs;
  const int r = blockIdx.x, t = threadIdx.x, lane = lane_id(), w = t >> 5;
  long long* ck_ = clk ? clk + (long)r * 16 : nullptr;
  if (ck_ && t == 0) { ck_[0] = gtime(); ck_[1] = clock64(); }
  const int L = row_lens[r];
  const float4* row4 = reinterpret_cast<const float4*>(logits + (long)r * ls0);
  const int keff = L < SELK ? (L > 0 ? L : 0) : SELK;
  int req = 0, seq = 0;
  if constexpr (STAGE == 4) { req = __ldg(req_id + r); seq = t < 3 ? __ldg(seq_lens + r) : 0; }
  const int n4 = L >> 2;
  constexpr int G = U * NW3;
  const int nch = (n4 + 31) >> 5;
  const int Q = (nch + G - 1) / G;
  const float NANF = __int_as_float(0x7fc00000);
  auto i4_of = [&](int b, int j) { int c = (j * NW3 + w) * Q + b; return c < nch ? c * 32 + lane : n4; };
  float4 v[U];
#pragma unroll
  for (int j = 0; j < U; ++j) { int i4 = i4_of(0, j); v[j] = i4 < n4 ? __ldg(row4 + i4) : make_float4(NANF, NANF, NANF, NANF); }
  if (t == 0) { sm.cnt = 0; sm.nsel = 0; sm.neq = 0; sm.flag = 0; }

  // ---------------- tau: 4 sample keys per thread (component lane&3 of each batch-0 float4)
  float tf = -INFINITY;
  if constexpr (STAGE >= 1) {
    if (L > CAP) {
      uint32_t sk[4];
      int nv = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const bool in = i4_of(0, j) < n4;
        const int q = lane & 3;
        const float x = q == 0 ? v[j].x : q == 1 ? v[j].y : q == 2 ? v[j].z : v[j].w;
        sk[j] = in ? fkey(x) : 0u;
        nv += in;
      }
      if (ck_ && t == 0) ck_[2] = clock64();
      const int ns = __syncthreads_count(nv > 0) + __syncthreads_count(nv > 1) + __syncthreads_count(nv > 2) + __syncthreads_count(nv > 3);
      int Rs = (int)((long)(2 * SELK + SELK / 5) * ns / L);
      Rs = max(1, min(Rs, ns));
      uint32_t T = 0;
      T |= (uint32_t)count_step16<NT, 4>(sk, 4, T, 28, Rs, cs) << 28;
      if (ck_ && t == 0) ck_[3] = clock64();
      T |= (uint32_t)count_step16<NT, 4>(sk, 4, T, 24, Rs, cs) << 24;
      T |= (uint32_t)count_step16<NT, 4>(sk, 4, T, 20, Rs, cs) << 20;
      tf = T ? key2f(T) : -INFINITY;
    } else {
      __syncthreads();
    }
  } else {
    __syncthreads();
  }
  if (ck_ && t == 0) ck_[4] = clock64();

  // ---------------- stream with warp-private segments (as v3)
  float acc = 0.f;
  int wc = 0;
  const int segb = w * SEG;
#pragma unroll 1
  for (int b = 0; b < Q; ++b) {
    float4 nx[U];
    const bool more = b + 1 < Q;
#pragma unroll
    for (int j = 0; j < U; ++j) {
      int i4 = more ? i4_of(b + 1, j) : n4;
      nx[j] = i4 < n4 ? __ldg(row4 + i4) : make_float4(NANF, NANF, NANF, NANF);
    }
    if constexpr (STAGE <= 1) {
#pragma unroll
      for (int j = 0; j < U; ++j) acc += fmaxf(fmaxf(v[j].x, v[j].y), fmaxf(v[j].z, v[j].w));
    } else {
#pragma unroll
      for (int j = 0; j < U; ++j) {
        const int i4 = i4_of(b, j);
        const float e[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const bool hit = e[c] >= tf;
          const unsigned bm = __ballot_sync(FULL, hit);
          if (bm) {
            const int p = wc + __popc(bm & lanemask_lt());
            if (hit && p < SEG) { sm.ck[segb + p] = fkey(e[c]); sm.ci[segb + p] = 4 * i4 + c; }
            wc += __popc(bm);
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < U; ++j) v[j] = nx[j];
  }
  if constexpr (STAGE >= 2) {
    if (w == 0) {
      const int rem = L & 3;
      float x = NANF;
      if (lane < rem) x = __ldg(reinterpret_cast<const float*>(row4) + 4 * n4 + lane);
      const bool hit = x >= tf;
      const unsigned bm = __ballot_sync(FULL, hit);
      const int p = wc + __popc(bm & lanemask_lt());
      if (hit && p < SEG) { sm.ck[segb + p] = fkey(x); sm.ci[segb + p] = 4 * n4 + lane; }
      wc += __popc(bm);
    }
    if (lane == 0) s_wcnt[w] = wc;
  } else {
    if (t < (L & 3)) acc += __ldg(reinterpret_cast<const float*>(row4) + 4 * n4 + t);
  }
  __syncthreads();
  if (ck_ && t == 0) ck_[5] = clock64();

  if constexpr (STAGE <= 2) {
    float s = acc;
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
    int C = 0;
    if constexpr (STAGE == 2) for (int i = 0; i < NW3; ++i) C += s_wcnt[i];
    if (lane == 0) atomicAdd(dummy + r, s + (float)C + tf * 0.f);
    if (t == 0 && stats) stats[r] = C;
    if (ck_ && t == 0) { ck_[6] = ck_[7] = ck_[8] = clock64(); ck_[9] = gtime(); }
    return;
  } else {
    int C = 0, mxc = 0;
#pragma unroll
    for (int i = 0; i < NW3; ++i) { C += s_wcnt[i]; mxc = max(mxc, s_wcnt[i]); }
    const bool fallback = (mxc > SEG) || (C < keff);
    if (!fallback) {
      const int cw = s_wcnt[w];
      const int mlim = (cw + 31) >> 5;   // warp-uniform
      uint32_t kk[SEG / 32];
      uint32_t mn = 0xffffffffu, mx = 0u;
#pragma unroll
      for (int m = 0; m < SEG / 32; ++m) {
        const int sl = lane + 32 * m;
        kk[m] = sl < cw ? sm.ck[segb + sl] : 0u;
        if (kk[m]) { mn = min(mn, kk[m]); mx = max(mx, kk[m]); }
      }
      mn = __reduce_min_sync(FULL, mn); mx = __reduce_max_sync(FULL, mx);
      if (t == 0) { cs.mn = -1; cs.mx = 0; }
      __syncthreads();
      if (lane == 0) { atomicMin(reinterpret_cast<unsigned*>(&cs.mn), mn); atomicMax(reinterpret_cast<unsigned*>(&cs.mx), mx); }
      __syncthreads();
      const uint32_t lo = (uint32_t)cs.mn, hi = (uint32_t)cs.mx;
      uint32_t T = lo;
      if (lo != hi) {
        const int d = 32 - __clz(lo ^ hi);
        T = lo & ~((d >= 32) ? 0xffffffffu : ((1u << d) - 1u));
#pragma unroll 1
        for (int sh = ((d + 3) / 4) * 4 - 4; sh >= 0; sh -= 4)
          T += (uint32_t)count_step16<NT, SEG / 32>(kk, mlim, T, sh, keff, cs) << sh;  // add: T may carry prefix bits above d
      }
      const int ngt = block_count_gt<NT, SEG / 32>(kk, mlim, T, cs);
      if (ck_ && t == 0) ck_[6] = clock64();
      const int req_eq = keff - ngt;
      int ng = 0, ne = 0;
#pragma unroll
      for (int m = 0; m < SEG / 32; ++m) if (m < mlim) { ng += kk[m] > T; ne += kk[m] == T; }
      int tot;
      const int ex = block_excl_scan<NT>((ng << 16) | ne, &tot, sm.w);
      int pg = ex >> 16, pe = ex & 0xffff;
      const int totg = tot >> 16;
#pragma unroll
      for (int m = 0; m < SEG / 32; ++m) {
        if (m < mlim) {
          const int sl = segb + lane + 32 * m;
          if (kk[m] > T) { sm.sel[pg++] = sm.ci[sl]; }
          else if (kk[m] == T) { if (pe < req_eq) sm.sel[totg + pe] = sm.ci[sl]; ++pe; }
        }
      }
    } else if (keff > 0) {
      const float* row = reinterpret_cast<const float*>(row4);
      auto gk = [&](int i) { return fkey(__ldg(row + i)); };
      radix_select2<NT, 0>(sm, L, keff, gk);
      if (ck_ && t == 0) ck_[6] = clock64();
      gather_sel<NT>(sm, L, (uint32_t)sm.res[2], sm.res[3], gk, [&](int i) { return i; });
    }
    for (int i = keff + t; i < SELK; i += NT) sm.sel[i] = -1;
    if (t == 0 && stats) stats[r] = fallback ? -C - 1 : C;
    __syncthreads();
    if (ck_ && t == 0) ck_[7] = clock64();
    if constexpr (STAGE == 3) {
      for (int i = t; i < SELK; i += NT) pool_out[(long)r * SELK + i] = sm.sel[i];
    } else {
      int pid[1] = {sm.sel[t]};
      post_row<NT, 1>(pid, req, seq, bt, bt_s0, bt_s1, max_blocks, bsr, out + (long)r * PT_W, valid + r, sm.w);
    }
    if (ck_ && t == 0) { ck_[8] = clock64(); ck_[9] = gtime(); }
  }
}

template <int STAGE>
static void launch_topk4(int M, const float* lg, long ls0, const int* lens, int* pout, float* dummy, int* stats, long long* clk,
                         const int* seq, const int* req, const int* bt, long bs0, long bs1, int maxb, int bsr, int* out, int* valid) {
  auto k = topk4_kernel<512, STAGE, 4>;
  int smem = sizeof(TopkSmem);
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  k<<<M, 512, smem, at::cuda::getCurrentCUDAStream()>>>(lg, ls0, lens, pout, dummy, stats, clk, seq, req, bt, bs0, bs1, maxb, bsr, out, valid);
}

void topk4(torch::Tensor logits, torch::Tensor lens, int64_t stage, torch::Tensor pool_out, torch::Tensor dummy,
           torch::Tensor stats, torch::Tensor clk, torch::Tensor seq, torch::Tensor req, torch::Tensor bt, int64_t bsr,
           torch::Tensor out, torch::Tensor valid) {
  int M = lens.numel();
  TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) % 4 == 0);
  long long* cp = clk.numel() ? reinterpret_cast<long long*>(clk.data_ptr<int64_t>()) : nullptr;
  int* st = stats.numel() ? stats.data_ptr<int>() : nullptr;
#define L4_(S_)                                                                                                             \
  if (stage == S_) {                                                                                                        \
    launch_topk4<S_>(M, logits.data_ptr<float>(), logits.stride(0), lens.data_ptr<int>(), pool_out.data_ptr<int>(),         \
                     dummy.data_ptr<float>(), st, cp, seq.data_ptr<int>(), req.data_ptr<int>(), bt.data_ptr<int>(),        \
                     bt.stride(0), bt.stride(1), bt.size(1), bsr, out.data_ptr<int>(), valid.data_ptr<int>());             \
    return;                                                                                                                 \
  }
  L4_(0) L4_(1) L4_(2) L4_(3) L4_(4)
#undef L4_
  TORCH_CHECK(false, "bad topk4 stage");
}

// ------------------------------------------------------------------------------------------------
// top-k v5 (job 5): v3 structure + the cheapest piece found for each phase so far
//  - PF: warp 0 issues cp.async.bulk.prefetch.L2 for the whole row at kernel start, so DRAM streams into L2 while the
//    CTA computes its threshold; batch 1 is also issued before the threshold
//  - tau: 4 sample keys per thread (2048), 4-ary count search (3 thresholds per step, 6 steps, 12-bit lower bound)
//  - stream: warp-private segments (v3)
//  - select: keys of the warp's own segment in registers, range-adaptive 8-bit radix (256 shared bins, plain atomics,
//    early exit), placement by one packed block scan (no atomics, deterministic order)
template <int NT, int NK>
__device__ __forceinline__ int count_step4(const uint32_t (&k)[NK], uint32_t T, int shift, int R, CS& cs, int par) {
  const int lane = lane_id(), w = threadIdx.x >> 5;
  int c0 = 0, c1 = 0, c2 = 0;
#pragma unroll
  for (int m = 0; m < NK; ++m) {
    const uint32_t km = k[m];
    const uint32_t d = km >= T ? min((km - T) >> shift, 3u) : 0u;
    c0 += d > 0u; c1 += d > 1u; c2 += d > 2u;
  }
  c0 = __reduce_add_sync(FULL, c0); c1 = __reduce_add_sync(FULL, c1); c2 = __reduce_add_sync(FULL, c2);
  int* slot = &cs.c[w][par * 4];
  if (lane == 0) { slot[0] = c0; slot[1] = c1; slot[2] = c2; }
  __syncthreads();
  int s0 = 0, s1 = 0, s2 = 0;
#pragma unroll
  for (int q = 0; q < NT / 32; ++q) { s0 += cs.c[q][par * 4]; s1 += cs.c[q][par * 4 + 1]; s2 += cs.c[q][par * 4 + 2]; }
  return s2 >= R ? 3 : s1 >= R ? 2 : s0 >= R ? 1 : 0;   // parity double-buffers cs.c: no trailing barrier
}

__device__ __forceinline__ float pick4(const float4& a, int q) { return q == 0 ? a.x : q == 1 ? a.y : q == 2 ? a.z : a.w; }

template <int NT, int STAGE, int U, int PF>
__global__ void __launch_bounds__(NT, 2) topk5_kernel(const float* __restrict__ logits, long ls0, const int* __restrict__ row_lens,
                                                      int* __restrict__ pool_out, float* __restrict__ dummy,
                                                      int* __restrict__ stats, long long* __restrict__ clk,
                                                      const int* __restrict__ seq_lens, const int* __restrict__ req_id,
                                                      const int* __restrict__ bt, long bt_s0, long bt_s1, int max_blocks,
                                                      int bsr, int* __restrict__ out, int* __restrict__ valid) {
  static_assert(NT == 512 && U == 4, "16 warps, 4 float4 per batch");
  extern __shared__ __align__(16) unsigned char smraw[];
  TopkSmem& sm = *reinterpret_cast<TopkSmem*>(smraw);
  __shared__ int s_wcnt[NW3];
  __shared__ CS cs;
  const int r = blockIdx.x, t = threadIdx.x, lane = lane_id(), w = t >> 5;
  long long* ck_ = clk ? clk + (long)r * 16 : nullptr;
  if (ck_ && t == 0) { ck_[0] = gtime(); ck_[1] = clock64(); }
  const int L = row_lens[r];
  const float* rowf = logits + (long)r * ls0;
  const float4* row4 = reinterpret_cast<const float4*>(rowf);
  if constexpr (PF) {
    if (w == 0 && L > 0) {
      const unsigned bytes = ((unsigned)L * 4u + 15u) & ~15u;
      const unsigned per = ((bytes + 31u) / 32u + 15u) & ~15u;
      const unsigned off = per * lane;
      if (off < bytes) {
        const unsigned sz = min(per, bytes - off);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(reinterpret_cast<const char*>(rowf) + off), "r"(sz) : "memory");
      }
    }
  }
  const int keff = L < SELK ? (L > 0 ? L : 0) : SELK;
  int req = 0, seq = 0;
  if constexpr (STAGE == 4) { req = __ldg(req_id + r); seq = t < 3 ? __ldg(seq_lens + r) : 0; }
  const int n4 = L >> 2;
  constexpr int G = U * NW3;
  const int nch = (n4 + 31) >> 5;
  const int Q = (nch + G - 1) / G;
  const float NANF = __int_as_float(0x7fc00000);
  auto i4_of = [&](int b, int j) { int c = (j * NW3 + w) * Q + b; return c < nch ? c * 32 + lane : n4; };
  float4 v[U], nx[U];
#pragma unroll
  for (int j = 0; j < U; ++j) { int i4 = i4_of(0, j); v[j] = i4 < n4 ? __ldg(row4 + i4) : make_float4(NANF, NANF, NANF, NANF); }
#pragma unroll
  for (int j = 0; j < U; ++j) { int i4 = Q > 1 ? i4_of(1, j) : n4; nx[j] = i4 < n4 ? __ldg(row4 + i4) : make_float4(NANF, NANF, NANF, NANF); }
  if (t == 0) { sm.cnt = 0; sm.nsel = 0; sm.neq = 0; sm.flag = 0; }

  float tf = -INFINITY;
  if constexpr (STAGE >= 1) {
    if (L > CAP) {
      uint32_t sk[4];
      int nv = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const bool in = i4_of(0, j) < n4;
        sk[j] = in ? fkey(pick4(v[j], lane & 3)) : 0u;
        nv += in;
      }
      if (ck_ && t == 0) ck_[2] = clock64();
      const int ns = __syncthreads_count(nv > 0) + __syncthreads_count(nv > 1) + __syncthreads_count(nv > 2) + __syncthreads_count(nv > 3);
      int Rs = (int)((long)(2 * SELK + SELK / 5) * ns / L);
      Rs = max(1, min(Rs, ns));
      uint32_t T = 0;
#pragma unroll 1
      for (int sh = 30, par = 0; sh >= 20; sh -= 2, par ^= 1) {
        T += (uint32_t)count_step4<NT, 4>(sk, T, sh, Rs, cs, par) << sh;
        if (ck_ && t == 0 && sh == 30) ck_[3] = clock64();
      }
      tf = T ? key2f(T) : -INFINITY;
      __syncthreads();
    } else {
      __syncthreads();
    }
  } else {
    __syncthreads();
  }
  if (ck_ && t == 0) ck_[4] = clock64();

  float acc = 0.f;
  int wc = 0;
  const int segb = w * SEG;
#pragma unroll 1
  for (int b = 0; b < Q; ++b) {
    float4 nn[U];
    const bool more = b + 2 < Q;
#pragma unroll
    for (int j = 0; j < U; ++j) {
      int i4 = more ? i4_of(b + 2, j) : n4;
      nn[j] = i4 < n4 ? __ldg(row4 + i4) : make_float4(NANF, NANF, NANF, NANF);
    }
    if constexpr (STAGE <= 1) {
#pragma unroll
      for (int j = 0; j < U; ++j) acc += fmaxf(fmaxf(v[j].x, v[j].y), fmaxf(v[j].z, v[j].w));
    } else {
#pragma unroll
      for (int j = 0; j < U; ++j) {
        const int i4 = i4_of(b, j);
        const float e[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const bool hit = e[c] >= tf;
          const unsigned bm = __ballot_sync(FULL, hit);
          if (bm) {
            const int p = wc + __popc(bm & lanemask_lt());
            if (hit && p < SEG) { sm.ck[segb + p] = fkey(e[c]); sm.ci[segb + p] = 4 * i4 + c; }
            wc += __popc(bm);
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < U; ++j) { v[j] = nx[j]; nx[j] = nn[j]; }
  }
  if constexpr (STAGE >= 2) {
    if (w == 0) {
      const int rem = L & 3;
      float x = NANF;
      if (lane < rem) x = __ldg(rowf + 4 * n4 + lane);
      const bool hit = x >= tf;
      const unsigned bm = __ballot_sync(FULL, hit);
      const int p = wc + __popc(bm & lanemask_lt());
      if (hit && p < SEG) { sm.ck[segb + p] = fkey(x); sm.ci[segb + p] = 4 * n4 + lane; }
      wc += __popc(bm);
    }
    if (lane == 0) s_wcnt[w] = wc;
  } else {
    if (t < (L & 3)) acc += __ldg(rowf + 4 * n4 + t);
  }
  __syncthreads();
  if (ck_ && t == 0) ck_[5] = clock64();

  if constexpr (STAGE <= 2) {
    float s = acc;
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
    int C = 0;
    if constexpr (STAGE == 2) for (int i = 0; i < NW3; ++i) C += s_wcnt[i];
    if (lane == 0) atomicAdd(dummy + r, s + (float)C + tf * 0.f);
    if (t == 0 && stats) stats[r] = C;
    if (ck_ && t == 0) { ck_[6] = ck_[7] = ck_[8] = clock64(); ck_[9] = gtime(); }
    return;
  } else {
    int C = 0, mxc = 0;
#pragma unroll
    for (int i = 0; i < NW3; ++i) { C += s_wcnt[i]; mxc = max(mxc, s_wcnt[i]); }
    const bool fallback = (mxc > SEG) || (C < keff);
    if (!fallback) {
      const int cw = s_wcnt[w];
      const int mlim = (cw + 31) >> 5;
      uint32_t kk[SEG / 32];
      uint32_t mn = 0xffffffffu, mx = 0u;
#pragma unroll
      for (int m = 0; m < SEG / 32; ++m) {
        const int sl = lane + 32 * m;
        kk[m] = (m < mlim && sl < cw) ? sm.ck[segb + sl] : 0u;
        if (kk[m]) { mn = min(mn, kk[m]); mx = max(mx, kk[m]); }
      }
      mn = __reduce_min_sync(FULL, mn); mx = __reduce_max_sync(FULL, mx);
      if (t == 0) { cs.mn = -1; cs.mx = 0; }
      __syncthreads();
      if (lane == 0) { atomicMin(reinterpret_cast<unsigned*>(&cs.mn), mn); atomicMax(reinterpret_cast<unsigned*>(&cs.mx), mx); }
      __syncthreads();
      uint32_t base = (uint32_t)cs.mn, hi = (uint32_t)cs.mx;
      int R = keff;
#pragma unroll 1
      while (true) {
        const uint32_t range = hi - base;
        if (range == 0) break;
        const int nbits = 32 - __clz(range);
        const int shift = nbits > 8 ? nbits - 8 : 0;
        if (t < 256) sm.hist[t] = 0;
        __syncthreads();
#pragma unroll
        for (int m = 0; m < SEG / 32; ++m)
          if (m < mlim) { const uint32_t k = kk[m]; if (k && k >= base && k <= hi) atomicAdd(&sm.hist[(k - base) >> shift], 1); }
        __syncthreads();
        find_bin<NT>(sm.hist, 256, R, sm.res, sm.w);
        const uint32_t b = (uint32_t)sm.res[0];
        const int a = sm.res[1];
        const int h = sm.hist[b];
        __syncthreads();
        R -= a;
        const uint32_t nbase = base + (b << shift);
        hi = min(hi, nbase + ((1u << shift) - 1u));
        base = nbase;
        if (shift == 0) break;
        if (h == R) { R = 0x7fffffff; break; }
      }
      const uint32_t T = base;
      if (ck_ && t == 0) ck_[6] = clock64();
      int ng = 0, ne = 0;
#pragma unroll
      for (int m = 0; m < SEG / 32; ++m) if (m < mlim) { ng += kk[m] > T; ne += kk[m] == T; }
      int tot;
      const int ex = block_excl_scan<NT>((ng << 16) | ne, &tot, sm.w);
      int pg = ex >> 16, pe = ex & 0xffff;
      const int totg = tot >> 16;
#pragma unroll
      for (int m = 0; m < SEG / 32; ++m) {
        if (m < mlim) {
          const int sl = segb + lane + 32 * m;
          if (kk[m] > T) { sm.sel[pg++] = sm.ci[sl]; }
          else if (kk[m] == T && kk[m]) { if (pe < R) sm.sel[totg + pe] = sm.ci[sl]; ++pe; }
        }
      }
    } else if (keff > 0) {
      auto gk = [&](int i) { return fkey(__ldg(rowf + i)); };
      radix_select2<NT, 0>(sm, L, keff, gk);
      if (ck_ && t == 0) ck_[6] = clock64();
      gather_sel<NT>(sm, L, (uint32_t)sm.res[2], sm.res[3], gk, [&](int i) { return i; });
    }
    for (int i = keff + t; i < SELK; i += NT) sm.sel[i] = -1;
    if (t == 0 && stats) stats[r] = fallback ? -C - 1 : C;
    __syncthreads();
    if (ck_ && t == 0) ck_[7] = clock64();
    if constexpr (STAGE == 3) {
      for (int i = t; i < SELK; i += NT) pool_out[(long)r * SELK + i] = sm.sel[i];
    } else {
      int pid[1] = {sm.sel[t]};
      post_row<NT, 1>(pid, req, seq, bt, bt_s0, bt_s1, max_blocks, bsr, out + (long)r * PT_W, valid + r, sm.w);
    }
    if (ck_ && t == 0) { ck_[8] = clock64(); ck_[9] = gtime(); }
  }
}

template <int STAGE, int PF>
static void launch_topk5(int M, const float* lg, long ls0, const int* lens, int* pout, float* dummy, int* stats, long long* clk,
                         const int* seq, const int* req, const int* bt, long bs0, long bs1, int maxb, int bsr, int* out, int* valid) {
  auto k = topk5_kernel<512, STAGE, 4, PF>;
  int smem = sizeof(TopkSmem);
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  k<<<M, 512, smem, at::cuda::getCurrentCUDAStream()>>>(lg, ls0, lens, pout, dummy, stats, clk, seq, req, bt, bs0, bs1, maxb, bsr, out, valid);
}

void topk5(torch::Tensor logits, torch::Tensor lens, int64_t stage, int64_t pf, torch::Tensor pool_out, torch::Tensor dummy,
           torch::Tensor stats, torch::Tensor clk, torch::Tensor seq, torch::Tensor req, torch::Tensor bt, int64_t bsr,
           torch::Tensor out, torch::Tensor valid) {
  int M = lens.numel();
  TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) % 4 == 0);
  long long* cp = clk.numel() ? reinterpret_cast<long long*>(clk.data_ptr<int64_t>()) : nullptr;
  int* st = stats.numel() ? stats.data_ptr<int>() : nullptr;
#define L5_(S_, P_)                                                                                                         \
  if (stage == S_ && pf == P_) {                                                                                            \
    launch_topk5<S_, P_>(M, logits.data_ptr<float>(), logits.stride(0), lens.data_ptr<int>(), pool_out.data_ptr<int>(),     \
                         dummy.data_ptr<float>(), st, cp, seq.data_ptr<int>(), req.data_ptr<int>(), bt.data_ptr<int>(),    \
                         bt.stride(0), bt.stride(1), bt.size(1), bsr, out.data_ptr<int>(), valid.data_ptr<int>());         \
    return;                                                                                                                 \
  }
  L5_(0, 1) L5_(1, 1) L5_(2, 1) L5_(3, 1) L5_(4, 1) L5_(0, 0) L5_(3, 0) L5_(4, 0)
#undef L5_
  TORCH_CHECK(false, "bad topk5 config");
}

// ------------------------------------------------------------------------------------------------
// top-k v6 (job 6): v5 without the L2 prefetch (it delayed batch 0 by ~9 µs); tau by an 8-ary count search (4 steps, cheap
// partial sums); filter into per-lane predicated buffers (no ballot per element: FSETP + predicated STS.64 + IADD),
// compacted into warp segments after the stream; select with 11-bit range-adaptive bins (usually 2 passes).
constexpr int LSLOT = 16;
template <int NT, int NK>
__device__ __forceinline__ int count_step8(const uint32_t (&k)[NK], uint32_t T, int shift, int R, CS& cs, int par) {
  const int lane = lane_id(), w = threadIdx.x >> 5;
  int c[7];
#pragma unroll
  for (int j = 0; j < 7; ++j) c[j] = 0;
#pragma unroll
  for (int m = 0; m < NK; ++m) {
    const uint32_t km = k[m];
    const uint32_t d = km >= T ? min((km - T) >> shift, 7u) : 0u;
#pragma unroll
    for (int j = 0; j < 7; ++j) c[j] += d > (uint32_t)j;
  }
#pragma unroll
  for (int j = 0; j < 7; ++j) c[j] = __reduce_add_sync(FULL, c[j]);
  if (lane < 7) {
    int x = c[0];
#pragma unroll
    for (int j = 1; j < 7; ++j) x = lane == j ? c[j] : x;
    cs.c[w][par * 8 + lane] = x;
  }
  __syncthreads();
  // lanes 0..15 read warp q = lane's partials; warp-reduce
  int best = 0;
#pragma unroll
  for (int j = 0; j < 7; ++j) {
    int x = lane < NT / 32 ? cs.c[lane][par * 8 + j] : 0;
    x = __reduce_add_sync(FULL, x);
    best = x >= R ? j + 1 : best;
  }
  return best;
}

template <int NT, int STAGE, int U, int PF>
__global__ void __launch_bounds__(NT, 2) topk6_kernel(const float* __restrict__ logits, long ls0, const int* __restrict__ row_lens,
                                                      int* __restrict__ pool_out, float* __restrict__ dummy,
                                                      int* __restrict__ stats, long long* __restrict__ clk,
                                                      const int* __restrict__ seq_lens, const int* __restrict__ req_id,
                                                      const int* __restrict__ bt, long bt_s0, long bt_s1, int max_blocks,
                                                      int bsr, int* __restrict__ out, int* __restrict__ valid) {
  static_assert(NT == 512 && U == 4, "16 warps, 4 float4 per batch");
  extern __shared__ __align__(16) unsigned char smraw[];
  TopkSmem& sm = *reinterpret_cast<TopkSmem*>(smraw);
  __shared__ int s_wcnt[NW3];
  __shared__ CS cs;
  const int r = blockIdx.x, t = threadIdx.x, lane = lane_id(), w = t >> 5;
  long long* ck_ = clk ? clk + (long)r * 16 : nullptr;
  if (ck_ && t == 0) { ck_[0] = gtime(); ck_[1] = clock64(); }
  const int L = row_lens[r];
  const float* rowf = logits + (long)r * ls0;
  const float4* row4 = reinterpret_cast<const float4*>(rowf);
  if constexpr (false) {
    if (w == 0 && L > 0) {
      const unsigned bytes = ((unsigned)L * 4u + 15u) & ~15u;
      const unsigned per = ((bytes + 31u) / 32u + 15u) & ~15u;
      const unsigned off = per * lane;
      if (off < bytes) {
        const unsigned sz = min(per, bytes - off);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(reinterpret_cast<const char*>(rowf) + off), "r"(sz) : "memory");
      }
    }
  }
  const int keff = L < SELK ? (L > 0 ? L : 0) : SELK;
  int req = 0, seq = 0;
  if constexpr (STAGE == 4) { req = __ldg(req_id + r); seq = t < 3 ? __ldg(seq_lens + r) : 0; }
  const int n4 = L >> 2;
  constexpr int G = U * NW3;
  const int nch = (n4 + 31) >> 5;
  const int Q = (nch + G - 1) / G;
  const float NANF = __int_as_float(0x7fc00000);
  auto i4_of = [&](int b, int j) { int c = (j * NW3 + w) * Q + b; return c < nch ? c * 32 + lane : n4; };
  float4 v[U];
#pragma unroll
  for (int j = 0; j < U; ++j) { int i4 = i4_of(0, j); v[j] = i4 < n4 ? __ldg(row4 + i4) : make_float4(NANF, NANF, NANF, NANF); }

  if (t == 0) { sm.cnt = 0; sm.nsel = 0; sm.neq = 0; sm.flag = 0; }

  float tf = -INFINITY;
  if constexpr (STAGE >= 1) {
    if (L > CAP) {
      uint32_t sk[4];
      int nv = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const bool in = i4_of(0, j) < n4;
        sk[j] = in ? fkey(pick4(v[j], lane & 3)) : 0u;
        nv += in;
      }
      if (ck_ && t == 0) ck_[2] = clock64();
      const int ns = __syncthreads_count(nv > 0) + __syncthreads_count(nv > 1) + __syncthreads_count(nv > 2) + __syncthreads_count(nv > 3);
      int Rs = (int)((long)(2 * SELK + SELK / 5) * ns / L);
      Rs = max(1, min(Rs, ns));
      uint32_t T = 0;
#pragma unroll 1
      for (int sh = 29, par = 0; sh >= 20; sh -= 3, par ^= 1) {
        T += (uint32_t)count_step8<NT, 4>(sk, T, sh, Rs, cs, par) << sh;
        if (ck_ && t == 0 && sh == 29) ck_[3] = clock64();
      }
      tf = T ? key2f(T) : -INFINITY;
      __syncthreads();
    } else {
      __syncthreads();
    }
  } else {
    __syncthreads();
  }
  if (ck_ && t == 0) ck_[4] = clock64();

  float acc = 0.f;
  int lc = 0;                       // per-lane candidate count
  const int segb = w * SEG;
  uint2* lb = reinterpret_cast<uint2*>(smraw + sizeof(TopkSmem));
#pragma unroll 1
  for (int b = 0; b < Q; ++b) {
    float4 nn[U];
    const bool more = b + 1 < Q;
#pragma unroll
    for (int j = 0; j < U; ++j) {
      int i4 = more ? i4_of(b + 1, j) : n4;
      nn[j] = i4 < n4 ? __ldg(row4 + i4) : make_float4(NANF, NANF, NANF, NANF);
    }
    if constexpr (STAGE <= 1) {
#pragma unroll
      for (int j = 0; j < U; ++j) acc += fmaxf(fmaxf(v[j].x, v[j].y), fmaxf(v[j].z, v[j].w));
    } else {
#pragma unroll
      for (int j = 0; j < U; ++j) {
        const int ib = 4 * i4_of(b, j);
        const float e[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (e[c] >= tf) {
            if (lc < LSLOT) lb[lc * NT + t] = make_uint2(__float_as_uint(e[c]), (unsigned)(ib + c));
            ++lc;
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < U; ++j) v[j] = nn[j];
  }
  if constexpr (STAGE >= 2) {
    if (t < (L & 3)) {
      const float x = __ldg(rowf + 4 * n4 + t);
      if (x >= tf) { if (lc < LSLOT) lb[lc * NT + t] = make_uint2(__float_as_uint(x), (unsigned)(4 * n4 + t)); ++lc; }
    }
    // compact lane buffers into the warp segment (order: lane-major)
    const int lcc = min(lc, LSLOT);
    int incl = lcc;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(FULL, incl, o); if (lane >= o) incl += y; }
    const int wt = __shfl_sync(FULL, incl, 31);
    int p = incl - lcc;
    for (int m = 0; m < lcc; ++m, ++p) {
      const uint2 e = lb[m * NT + t];
      if (p < SEG) { sm.ck[segb + p] = fkey(__uint_as_float(e.x)); sm.ci[segb + p] = (int)e.y; }
    }
    const bool lover = __any_sync(FULL, lc > LSLOT);
    if (lane == 0) s_wcnt[w] = lover ? SEG + 1 : wt;
  } else {
    if (t < (L & 3)) acc += __ldg(rowf + 4 * n4 + t);
  }
  __syncthreads();
  if (ck_ && t == 0) ck_[5] = clock64();

  if constexpr (STAGE <= 2) {
    float s = acc;
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
    int C = 0;
    if constexpr (STAGE == 2) for (int i = 0; i < NW3; ++i) C += s_wcnt[i];
    if (lane == 0) atomicAdd(dummy + r, s + (float)C + tf * 0.f);
    if (t == 0 && stats) stats[r] = C;
    if (ck_ && t == 0) { ck_[6] = ck_[7] = ck_[8] = clock64(); ck_[9] = gtime(); }
    return;
  } else {
    int C = 0, mxc = 0;
#pragma unroll
    for (int i = 0; i < NW3; ++i) { C += s_wcnt[i]; mxc = max(mxc, s_wcnt[i]); }
    const bool fallback = (mxc > SEG) || (C < keff);
    if (!fallback) {
      const int cw = s_wcnt[w];
      const int mlim = (cw + 31) >> 5;
      uint32_t kk[SEG / 32];
      uint32_t mn = 0xffffffffu, mx = 0u;
#pragma unroll
      for (int m = 0; m < SEG / 32; ++m) {
        const int sl = lane + 32 * m;
        kk[m] = (m < mlim && sl < cw) ? sm.ck[segb + sl] : 0u;
        if (kk[m]) { mn = min(mn, kk[m]); mx = max(mx, kk[m]); }
      }
      mn = __reduce_min_sync(FULL, mn); mx = __reduce_max_sync(FULL, mx);
      if (t == 0) { cs.mn = -1; cs.mx = 0; }
      __syncthreads();
      if (lane == 0) { atomicMin(reinterpret_cast<unsigned*>(&cs.mn), mn); atomicMax(reinterpret_cast<unsigned*>(&cs.mx), mx); }
      __syncthreads();
      uint32_t base = (uint32_t)cs.mn, hi = (uint32_t)cs.mx;
      int R = keff;
#pragma unroll 1
      while (true) {
        const uint32_t range = hi - base;
        if (range == 0) break;
        const int nbits = 32 - __clz(range);
        const int shift = nbits > 11 ? nbits - 11 : 0;
#pragma unroll
        for (int i = t; i < 2048; i += NT) sm.hist[i] = 0;
        __syncthreads();
#pragma unroll
        for (int m = 0; m < SEG / 32; ++m)
          if (m < mlim) { const uint32_t k = kk[m]; if (k && k >= base && k <= hi) atomicAdd(&sm.hist[(k - base) >> shift], 1); }
        __syncthreads();
        find_bin<NT>(sm.hist, 2048, R, sm.res, sm.w);
        const uint32_t b = (uint32_t)sm.res[0];
        const int a = sm.res[1];
        const int h = sm.hist[b];
        __syncthreads();
        R -= a;
        const uint32_t nbase = base + (b << shift);
        hi = min(hi, nbase + ((1u << shift) - 1u));
        base = nbase;
        if (shift == 0) break;
        if (h == R) { R = 0x7fffffff; break; }
      }
      const uint32_t T = base;
      if (ck_ && t == 0) ck_[6] = clock64();
      int ng = 0, ne = 0;
#pragma unroll
      for (int m = 0; m < SEG / 32; ++m) if (m < mlim) { ng += kk[m] > T; ne += kk[m] == T; }
      int tot;
      const int ex = block_excl_scan<NT>((ng << 16) | ne, &tot, sm.w);
      int pg = ex >> 16, pe = ex & 0xffff;
      const int totg = tot >> 16;
#pragma unroll
      for (int m = 0; m < SEG / 32; ++m) {
        if (m < mlim) {
          const int sl = segb + lane + 32 * m;
          if (kk[m] > T) { sm.sel[pg++] = sm.ci[sl]; }
          else if (kk[m] == T && kk[m]) { if (pe < R) sm.sel[totg + pe] = sm.ci[sl]; ++pe; }
        }
      }
    } else if (keff > 0) {
      auto gk = [&](int i) { return fkey(__ldg(rowf + i)); };
      radix_select2<NT, 0>(sm, L, keff, gk);
      if (ck_ && t == 0) ck_[6] = clock64();
      gather_sel<NT>(sm, L, (uint32_t)sm.res[2], sm.res[3], gk, [&](int i) { return i; });
    }
    for (int i = keff + t; i < SELK; i += NT) sm.sel[i] = -1;
    if (t == 0 && stats) stats[r] = fallback ? -C - 1 : C;
    __syncthreads();
    if (ck_ && t == 0) ck_[7] = clock64();
    if constexpr (STAGE == 3) {
      for (int i = t; i < SELK; i += NT) pool_out[(long)r * SELK + i] = sm.sel[i];
    } else {
      int pid[1] = {sm.sel[t]};
      post_row<NT, 1>(pid, req, seq, bt, bt_s0, bt_s1, max_blocks, bsr, out + (long)r * PT_W, valid + r, sm.w);
    }
    if (ck_ && t == 0) { ck_[8] = clock64(); ck_[9] = gtime(); }
  }
}

template <int STAGE, int PF>
static void launch_topk6(int M, const float* lg, long ls0, const int* lens, int* pout, float* dummy, int* stats, long long* clk,
                         const int* seq, const int* req, const int* bt, long bs0, long bs1, int maxb, int bsr, int* out, int* valid) {
  auto k = topk6_kernel<512, STAGE, 4, PF>;
  int smem = sizeof(TopkSmem) + LSLOT * 512 * 8;
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  k<<<M, 512, smem, at::cuda::getCurrentCUDAStream()>>>(lg, ls0, lens, pout, dummy, stats, clk, seq, req, bt, bs0, bs1, maxb, bsr, out, valid);
}

void topk6(torch::Tensor logits, torch::Tensor lens, int64_t stage, int64_t pf, torch::Tensor pool_out, torch::Tensor dummy,
           torch::Tensor stats, torch::Tensor clk, torch::Tensor seq, torch::Tensor req, torch::Tensor bt, int64_t bsr,
           torch::Tensor out, torch::Tensor valid) {
  int M = lens.numel();
  TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) % 4 == 0);
  long long* cp = clk.numel() ? reinterpret_cast<long long*>(clk.data_ptr<int64_t>()) : nullptr;
  int* st = stats.numel() ? stats.data_ptr<int>() : nullptr;
#define L6_(S_, P_)                                                                                                         \
  if (stage == S_ && pf == P_) {                                                                                            \
    launch_topk6<S_, P_>(M, logits.data_ptr<float>(), logits.stride(0), lens.data_ptr<int>(), pool_out.data_ptr<int>(),     \
                         dummy.data_ptr<float>(), st, cp, seq.data_ptr<int>(), req.data_ptr<int>(), bt.data_ptr<int>(),    \
                         bt.stride(0), bt.stride(1), bt.size(1), bsr, out.data_ptr<int>(), valid.data_ptr<int>());         \
    return;                                                                                                                 \
  }
  L6_(0, 1) L6_(1, 1) L6_(2, 1) L6_(3, 1) L6_(4, 1) L6_(0, 0) L6_(4, 0)
#undef L6_
  TORCH_CHECK(false, "bad topk5 config");
}

// ------------------------------------------------------------------------------------------------ host
static cudaStream_t cur() { return at::cuda::getCurrentCUDAStream(); }

void e2a(torch::Tensor pool, torch::Tensor seq, torch::Tensor req, torch::Tensor bt, int64_t bsr, torch::Tensor out,
         torch::Tensor valid, int64_t nt) {
  int M = pool.size(0);
  TORCH_CHECK(pool.size(1) == SELK && pool.stride(1) == 1 && pool.stride(0) % 4 == 0);
  TORCH_CHECK(out.size(1) == PT_W && out.is_contiguous());
  auto a = std::make_tuple(pool.data_ptr<int>(), (int)pool.stride(0), seq.data_ptr<int>(), req.data_ptr<int>(),
                           bt.data_ptr<int>());
  if (nt == 128)
    e2a_kernel<128><<<M, 128, 0, cur()>>>(pool.data_ptr<int>(), pool.stride(0), seq.data_ptr<int>(), req.data_ptr<int>(),
                                          bt.data_ptr<int>(), bt.stride(0), bt.stride(1), bt.size(1), bsr, out.data_ptr<int>(),
                                          valid.data_ptr<int>());
  else if (nt == 256)
    e2a_kernel<256><<<M, 256, 0, cur()>>>(pool.data_ptr<int>(), pool.stride(0), seq.data_ptr<int>(), req.data_ptr<int>(),
                                          bt.data_ptr<int>(), bt.stride(0), bt.stride(1), bt.size(1), bsr, out.data_ptr<int>(),
                                          valid.data_ptr<int>());
  else
    e2a_kernel<512><<<M, 512, 0, cur()>>>(pool.data_ptr<int>(), pool.stride(0), seq.data_ptr<int>(), req.data_ptr<int>(),
                                          bt.data_ptr<int>(), bt.stride(0), bt.stride(1), bt.size(1), bsr, out.data_ptr<int>(),
                                          valid.data_ptr<int>());
  (void)a;
}

void pf0(torch::Tensor pool, torch::Tensor seq, torch::Tensor out, torch::Tensor valid) {
  pf0_kernel<<<pool.size(0), 128, 0, cur()>>>(pool.data_ptr<int>(), pool.stride(0), seq.data_ptr<int>(), out.data_ptr<int>(),
                                              valid.data_ptr<int>());
}

void pf1(torch::Tensor pool, torch::Tensor seq, torch::Tensor req, torch::Tensor bt, int64_t bsr, torch::Tensor out,
         torch::Tensor valid) {
  pf1_kernel<<<pool.size(0), 128, 0, cur()>>>(pool.data_ptr<int>(), pool.stride(0), seq.data_ptr<int>(), req.data_ptr<int>(),
                                              bt.data_ptr<int>(), bt.stride(0), bsr, out.data_ptr<int>(), valid.data_ptr<int>());
}

void empty() { empty_kernel<<<1, 32, 0, cur()>>>(); }

template <int NT, int STAGE, int U>
static void launch_topk(int M, const float* lg, long ls0, const int* lens, int* pout, float* dummy, int* stats, const int* seq,
                        const int* req, const int* bt, long bs0, long bs1, int maxb, int bsr, int* out, int* valid) {
  auto k = topk_kernel<NT, STAGE, U>;
  int smem = sizeof(TopkSmem);
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  k<<<M, NT, smem, cur()>>>(lg, ls0, lens, pout, dummy, stats, seq, req, bt, bs0, bs1, maxb, bsr, out, valid);
}

void topk(torch::Tensor logits, torch::Tensor lens, int64_t stage, int64_t nt, torch::Tensor pool_out, torch::Tensor dummy,
          torch::Tensor stats, torch::Tensor seq, torch::Tensor req, torch::Tensor bt, int64_t bsr, torch::Tensor out,
          torch::Tensor valid) {
  int M = lens.numel();
  TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) % 4 == 0);
  const float* lg = logits.data_ptr<float>();
  long ls0 = logits.stride(0);
  const int* ln = lens.data_ptr<int>();
  int* po = pool_out.data_ptr<int>();
  float* dm = dummy.data_ptr<float>();
  int* st = stats.numel() ? stats.data_ptr<int>() : nullptr;
  const int* sq = seq.data_ptr<int>();
  const int* rq = req.data_ptr<int>();
  const int* b = bt.data_ptr<int>();
  long bs0 = bt.stride(0), bs1 = bt.stride(1);
  int maxb = bt.size(1);
  int* o = out.data_ptr<int>();
  int* v = valid.data_ptr<int>();
#define L_(NT_, S_) \
  if (nt == NT_ && stage == S_) { launch_topk<NT_, S_, 8>(M, lg, ls0, ln, po, dm, st, sq, rq, b, bs0, bs1, maxb, bsr, o, v); return; }
  L_(512, 0) L_(512, 1) L_(512, 2) L_(512, 3) L_(512, 4)
  L_(256, 0) L_(256, 1) L_(256, 2) L_(256, 3) L_(256, 4)
#undef L_
  TORCH_CHECK(false, "bad topk config");
}

// flat read floor: grid-stride over the same rows (all SMs balanced), sum
__global__ void __launch_bounds__(256) rowread_flat_kernel(const float* __restrict__ logits, long ls0, int L, int M,
                                                           float* __restrict__ dummy) {
  const int n4 = L >> 2;
  const long total = (long)M * n4;
  float acc = 0.f;
  const long stride = (long)gridDim.x * blockDim.x;
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + 7 * stride < total; i += 8 * stride) {
    float4 v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      long q = i + j * stride;
      int rr = (int)(q / n4), c = (int)(q - (long)rr * n4);
      v[j] = __ldg(reinterpret_cast<const float4*>(logits + rr * ls0) + c);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) acc += v[j].x + v[j].y + v[j].z + v[j].w;
  }
  for (; i < total; i += stride) {
    int rr = (int)(i / n4), c = (int)(i - (long)rr * n4);
    float4 v = __ldg(reinterpret_cast<const float4*>(logits + rr * ls0) + c);
    acc += v.x + v.y + v.z + v.w;
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(FULL, acc, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(dummy + (blockIdx.x & 255), acc);
}

void rowread_flat(torch::Tensor logits, int64_t L, int64_t M, torch::Tensor dummy, int64_t grid) {
  rowread_flat_kernel<<<grid, 256, 0, cur()>>>(logits.data_ptr<float>(), logits.stride(0), L, M, dummy.data_ptr<float>());
}

int64_t topk_smem() { return sizeof(TopkSmem); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("e2a", &e2a);
  m.def("pf0", &pf0);
  m.def("pf1", &pf1);
  m.def("empty", &empty);
  m.def("topk", &topk);
  m.def("rowread_flat", &rowread_flat);
  m.def("topk_smem", &topk_smem);
  m.def("topk2", &topk2);
  m.def("topk3", &topk3);
  m.def("topk4", &topk4);
  m.def("topk5", &topk5);
  m.def("topk6", &topk6);
}
