// One-launch mHC boundary: post of the finished block, pre of the next block
// and its RMSNorm, with an optional sum of several partial block outputs in
// front (the MoE combine's reduce).
//
//   x            = bf16(sum_p parts[p][t])            (P >= 1 partial outputs)
//   x            = bf16(x + extra[t])                 (optional addend, e.g. the shared expert)
//   new_res[j]   = bf16(sum_i comb[t,i,j] res[t,i] + post[t,j] x)       (mhc_post)
//   mixes        = (flatten(new_res) @ fn^T) * rsqrt(mean(new_res^2) + rms_eps)
//   pre, post', comb' = sigmoid / sigmoid*mult / Sinkhorn(softmax) of mixes (mhc_pre)
//   layer_input  = bf16(sum_h pre[h] new_res[h])
//   y            = bf16(rms_norm(layer_input) * norm_weight)
//
// This is the math of vLLM's mhc_fused_post_pre_tilelang with the RMSNorm fused
// in (vllm/model_executor/kernels/mhc/torch.py: mhc_post_torch, mhc_pre_torch).
//
// Structure: one cooperative launch of G = hidden / 32 CTAs.
//   Phase A (CTA c owns hidden columns [32c, 32c+32) of every stream, i.e. a
//     128-wide K slice of the pre GEMM): post-mix its slice for every token,
//     store new_res, and run the K-slice of flatten(new_res) @ fn^T on BF16
//     tensor cores with fn split into bf16 hi + lo (two MMAs, fp32 accumulate),
//     writing 24 partial mixes + the partial sum of squares per (token, CTA).
//   grid.sync()
//   Phase B1 (packed: token t is warp t % 8 of CTA t / 8): the warp sums the G
//     partials, runs the gates and Sinkhorn, and writes the 4 pre weights to
//     the workspace tail. Only ceil(T / 8) CTAs work here.
//   grid.sync()
//   Phase B2 (token t belongs to CTA t % G): the CTA collapses the updated
//     streams with the pre weights, RMS-normalizes and writes y.
// Tokens beyond CHUNK_TOKENS run as further A / sync / B1 / sync / B2 rounds in
// the same launch, so the workspace is bounded.

#include <tvm/ffi/extra/module.h>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "tvm_ffi_utils.h"

namespace mhc_one_launch {

namespace cg = cooperative_groups;

constexpr int HC = 4;                  // residual streams
constexpr int MIX = HC * (HC + 2);     // 24 mixes: pre, post, comb
constexpr int SLICE = 32;              // hidden columns per CTA in phase A
constexpr int KC = HC * SLICE;         // 128 K values per CTA
constexpr int KPAD = KC + 8;           // smem row pitch (bf16) for conflict-free ldmatrix
constexpr int WS = 32;                 // floats per (token, CTA) partial: 24 mixes, sumsq at 24
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int TOK_TILE = 16;           // tokens per MMA tile (one warp)
constexpr int CHUNK_TOKENS = 1024;     // tokens per A/sync/B round
constexpr int MAX_PARTS = 8;

struct Params {
  const __nv_bfloat16* parts[MAX_PARTS];  // P x [T, H], row pitch = part_row_stride
  int num_parts;
  int64_t part_row_stride;                 // elements between token rows of a part
  const __nv_bfloat16* extra;              // optional [T, H] addend, nullptr if none
  const __nv_bfloat16* residual;           // [T, HC, H]
  const float* post_prev;                  // [T, HC]
  const float* comb_prev;                  // [T, HC, HC]
  const float* fn;                         // [MIX, HC*H]
  const float* hc_scale;                   // [3]
  const float* hc_base;                    // [MIX]
  const __nv_bfloat16* norm_weight;        // [H]
  __nv_bfloat16* new_residual;             // [T, HC, H]
  float* post_out;                         // [T, HC]
  float* comb_out;                         // [T, HC, HC]
  __nv_bfloat16* y;                        // [T, H]
  float* ws;                               // [n, G, WS] partial mixes, then [n, HC] pre mixes; n = min(T, CHUNK_TOKENS)
  int num_tokens;
  int hidden;
  float rms_eps, hc_pre_eps, sinkhorn_eps, post_mult, norm_eps;
  int sinkhorn_iters;
  int stop_after;  // ablation: 0 prologue, 1 +post, 2 +GEMM partials, 3 +grid sync, 4 +gates, 5 full
};

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ void unpack8(const uint4 v, float* out) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float* in) {
  uint4 v;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(in[2 * i], in[2 * i + 1]);
  return v;
}

__device__ __forceinline__ float round_bf16(const float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* smem_ptr) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t b0,
                                         const uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float sigmoidf(const float v) { return 1.0f / (1.0f + expf(-v)); }

// Phase A shared memory: fn slice (hi, lo) and one A tile per warp.
struct SmemA {
  __nv_bfloat16 b_hi[MIX][KPAD];
  __nv_bfloat16 b_lo[MIX][KPAD];
  __nv_bfloat16 a[WARPS][TOK_TILE][KPAD];
};

// Phase B shared memory: the round's pre mixes, group reductions, and the
// bf16 layer input of each group's token.
struct SmemBHeader {
  float pre[WARPS][HC];
  float red[WARPS];
};

constexpr size_t SMEM_B_OFFSET = offsetof(SmemA, a);
static_assert(SMEM_B_OFFSET % 16 == 0 && sizeof(SmemBHeader) % 16 == 0, "uint4 alignment");

__global__ void __launch_bounds__(THREADS, 1) mhc_post_pre_one_launch_kernel(const Params p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  SmemA& sa = *reinterpret_cast<SmemA*>(smem_raw);
  // Phase B reuses the A tiles' space only: the fn slice must survive for the
  // next chunk's phase A.
  unsigned char* smem_b = smem_raw + SMEM_B_OFFSET;
  SmemBHeader& sb = *reinterpret_cast<SmemBHeader*>(smem_b);
  __nv_bfloat16* layer_in = reinterpret_cast<__nv_bfloat16*>(smem_b + sizeof(SmemBHeader));

  cg::grid_group grid = cg::this_grid();
  const int G = gridDim.x;
  const int cta = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int H = p.hidden;
  const int64_t K = static_cast<int64_t>(HC) * H;
  const int c0 = cta * SLICE;

  // fn slice -> bf16 hi/lo in smem: row o, local k = s*32 + j <- fn[o, s*H + c0 + j].
  // Every thread issues its float4 loads before converting any (one memory
  // round trip, not one per element).
  {
    constexpr int VECS = MIX * KC / 4;  // 768 float4
    static_assert(VECS % THREADS == 0, "fn slice split");
    constexpr int PER = VECS / THREADS;
    float4 f[PER];
#pragma unroll
    for (int r = 0; r < PER; ++r) {
      const int i = tid + r * THREADS;
      const int o = i / (KC / 4);
      const int kl = (i % (KC / 4)) * 4;
      f[r] = __ldg(reinterpret_cast<const float4*>(
          p.fn + o * K + static_cast<int64_t>(kl / SLICE) * H + c0 + kl % SLICE));
    }
#pragma unroll
    for (int r = 0; r < PER; ++r) {
      const int i = tid + r * THREADS;
      const int o = i / (KC / 4);
      const int kl = (i % (KC / 4)) * 4;
      const float v[4] = {f[r].x, f[r].y, f[r].z, f[r].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const __nv_bfloat16 hi = __float2bfloat16_rn(v[e]);
        sa.b_hi[o][kl + e] = hi;
        sa.b_lo[o][kl + e] = __float2bfloat16_rn(v[e] - __bfloat162float(hi));
      }
    }
  }
  __syncthreads();
  if (p.stop_after <= 0) return;

  float hc_scale[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) hc_scale[i] = __ldg(p.hc_scale + i);

  for (int chunk0 = 0; chunk0 < p.num_tokens; chunk0 += CHUNK_TOKENS) {
    const int chunk_tokens = min(CHUNK_TOKENS, p.num_tokens - chunk0);
    const int num_tiles = (chunk_tokens + TOK_TILE - 1) / TOK_TILE;

    // ---------------- Phase A ----------------
    for (int tile = warp; tile < num_tiles; tile += WARPS) {
      const int r = lane >> 1;          // token row inside the tile
      const int j0 = (lane & 1) * 16;   // 16 of the slice's 32 columns
      const int tl = tile * TOK_TILE + r;  // token inside the chunk
      const int t = chunk0 + tl;
      const bool valid = tl < chunk_tokens;
      float sumsq = 0.0f;
      if (valid) {
        // Issue every load of this token's slice before using any: the
        // partial outputs (predicated up to MAX_PARTS), the addend, the
        // residual streams and the previous post / comb mixes.
        uint4 part_raw[MAX_PARTS][2];
#pragma unroll
        for (int q = 0; q < MAX_PARTS; ++q) {
          if (q < p.num_parts) {
            const uint4* src = reinterpret_cast<const uint4*>(
                p.parts[q] + static_cast<int64_t>(t) * p.part_row_stride + c0 + j0);
            part_raw[q][0] = __ldg(src);
            part_raw[q][1] = __ldg(src + 1);
          }
        }
        uint4 extra_raw[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if (p.extra != nullptr) {
          const uint4* src =
              reinterpret_cast<const uint4*>(p.extra + static_cast<int64_t>(t) * H + c0 + j0);
          extra_raw[0] = __ldg(src);
          extra_raw[1] = __ldg(src + 1);
        }
        uint4 res_raw[HC][2];
#pragma unroll
        for (int i = 0; i < HC; ++i) {
          const uint4* src = reinterpret_cast<const uint4*>(
              p.residual + (static_cast<int64_t>(t) * HC + i) * H + c0 + j0);
          res_raw[i][0] = __ldg(src);
          res_raw[i][1] = __ldg(src + 1);
        }
        const float4 post4 = __ldg(reinterpret_cast<const float4*>(p.post_prev + t * HC));
        float4 comb4[HC];
#pragma unroll
        for (int i = 0; i < HC; ++i)
          comb4[i] = __ldg(reinterpret_cast<const float4*>(p.comb_prev + (t * HC + i) * HC));

        // Block output x: the sum of the partial outputs (+ the extra addend).
        float x[16];
#pragma unroll
        for (int e = 0; e < 16; ++e) x[e] = 0.0f;
#pragma unroll
        for (int q = 0; q < MAX_PARTS; ++q) {
          if (q < p.num_parts) {
            float v[8];
            unpack8(part_raw[q][0], v);
#pragma unroll
            for (int e = 0; e < 8; ++e) x[e] += v[e];
            unpack8(part_raw[q][1], v);
#pragma unroll
            for (int e = 0; e < 8; ++e) x[8 + e] += v[e];
          }
        }
#pragma unroll
        for (int e = 0; e < 16; ++e) x[e] = round_bf16(x[e]);
        if (p.extra != nullptr) {
          float v[8];
          unpack8(extra_raw[0], v);
#pragma unroll
          for (int e = 0; e < 8; ++e) x[e] = round_bf16(x[e] + v[e]);
          unpack8(extra_raw[1], v);
#pragma unroll
          for (int e = 0; e < 8; ++e) x[8 + e] = round_bf16(x[8 + e] + v[e]);
        }
        const float post[HC] = {post4.x, post4.y, post4.z, post4.w};
        float comb[HC][HC];
#pragma unroll
        for (int i = 0; i < HC; ++i) {
          comb[i][0] = comb4[i].x;
          comb[i][1] = comb4[i].y;
          comb[i][2] = comb4[i].z;
          comb[i][3] = comb4[i].w;
        }
        float res[HC][16];
#pragma unroll
        for (int i = 0; i < HC; ++i) {
          unpack8(res_raw[i][0], res[i]);
          unpack8(res_raw[i][1], res[i] + 8);
        }
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          float nr[16];
#pragma unroll
          for (int e = 0; e < 16; ++e) {
            float acc = post[j] * x[e];
#pragma unroll
            for (int i = 0; i < HC; ++i) acc += comb[i][j] * res[i][e];
            nr[e] = round_bf16(acc);
            sumsq += nr[e] * nr[e];
          }
          const uint4 lo = pack8(nr);
          const uint4 hi = pack8(nr + 8);
          uint4* dst = reinterpret_cast<uint4*>(
              p.new_residual + (static_cast<int64_t>(t) * HC + j) * H + c0 + j0);
          dst[0] = lo;
          dst[1] = hi;
          uint4* as = reinterpret_cast<uint4*>(&sa.a[warp][r][j * SLICE + j0]);
          as[0] = lo;
          as[1] = hi;
        }
      } else {
        const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          uint4* as = reinterpret_cast<uint4*>(&sa.a[warp][r][j * SLICE + j0]);
          as[0] = z;
          as[1] = z;
        }
      }
      sumsq += __shfl_xor_sync(0xffffffffu, sumsq, 1);
      __syncwarp();
      if (p.stop_after <= 1) continue;

      // K-slice of flatten(new_res) @ fn^T: 16 tokens x 24 mixes, K = 128.
      float acc[3][4];
#pragma unroll
      for (int n = 0; n < 3; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[n][e] = 0.0f;
#pragma unroll
      for (int k0 = 0; k0 < KC; k0 += 16) {
        uint32_t a[4];
        ldmatrix_x4(a, &sa.a[warp][lane & 15][k0 + (lane >> 4) * 8]);
#pragma unroll
        for (int n = 0; n < 3; ++n) {
          // m0/m1: hi rows n*8.., k0 / k0+8; m2/m3: lo rows, k0 / k0+8.
          const int row = n * 8 + (lane & 7);
          const int col = k0 + ((lane >> 3) & 1) * 8;
          uint32_t b[4];
          ldmatrix_x4(b, lane < 16 ? &sa.b_hi[row][col] : &sa.b_lo[row][col]);
          mma_bf16(acc[n], a, b[0], b[1]);
          mma_bf16(acc[n], a, b[2], b[3]);
        }
      }
      // c0,c1 -> row lane/4, cols 2*(lane%4)+{0,1}; c2,c3 -> row lane/4+8.
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row_tl = tile * TOK_TILE + (lane >> 2) + half * 8;
        if (row_tl < chunk_tokens) {
          float* dst = p.ws + (static_cast<int64_t>(row_tl) * G + cta) * WS;
#pragma unroll
          for (int n = 0; n < 3; ++n)
            *reinterpret_cast<float2*>(dst + n * 8 + 2 * (lane & 3)) =
                make_float2(acc[n][2 * half], acc[n][2 * half + 1]);
        }
      }
      if (valid && (lane & 1) == 0) p.ws[(static_cast<int64_t>(tl) * G + cta) * WS + MIX] = sumsq;
      __syncwarp();
    }
    if (p.stop_after <= 2) return;
    grid.sync();
    if (p.stop_after <= 3) return;

    // ---------------- Phase B1 ----------------
    // Token tl's gates belong to warp tl % WARPS of CTA tl / WARPS: every warp
    // of a CTA busy and only ceil(tokens / WARPS) CTAs running this code. (One
    // token per CTA put all 128 CTAs on it at once and ran 2-3x slower; see
    // logs/20261112_0_flash_mhc_fuse/gpu11.) The pre mixes go through the
    // workspace to phase B2's token owners.
    float* pre_g = p.ws + static_cast<int64_t>(min(p.num_tokens, CHUNK_TOKENS)) * G * WS;
    for (int tl = cta * WARPS + warp; tl < chunk_tokens; tl += G * WARPS) {
        const int t = chunk0 + tl;
        float v[MIX + 1];
#pragma unroll
        for (int o = 0; o <= MIX; ++o) v[o] = 0.0f;
#pragma unroll 4
        for (int c = lane; c < G; c += 32) {
          const float4* src =
              reinterpret_cast<const float4*>(p.ws + (static_cast<int64_t>(tl) * G + c) * WS);
#pragma unroll
          for (int q = 0; q < 6; ++q) {
            const float4 f = __ldcg(src + q);
            v[4 * q] += f.x;
            v[4 * q + 1] += f.y;
            v[4 * q + 2] += f.z;
            v[4 * q + 3] += f.w;
          }
          v[MIX] += __ldcg(p.ws + (static_cast<int64_t>(tl) * G + c) * WS + MIX);
        }
#pragma unroll
        for (int o = 0; o <= MIX; ++o)
#pragma unroll
          for (int m = 16; m >= 1; m >>= 1) v[o] += __shfl_xor_sync(0xffffffffu, v[o], m);
        const float rr = rsqrtf(v[MIX] / static_cast<float>(K) + p.rms_eps);
        // Lane q (< 16) owns comb (i = q / 4, j = q % 4); lanes 16..31 mirror it.
        const int q = lane & 15;
        float logit = 0.0f;
#pragma unroll
        for (int e = 0; e < HC * HC; ++e)
          if (e == q) logit = v[2 * HC + e] * rr * hc_scale[2] + __ldg(p.hc_base + 2 * HC + e);
        float mx = fmaxf(logit, __shfl_xor_sync(0xffffffffu, logit, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        float c = expf(logit - mx);
        float s = c + __shfl_xor_sync(0xffffffffu, c, 1);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        c = c / s + p.sinkhorn_eps;
        float cs = c + __shfl_xor_sync(0xffffffffu, c, 4);
        cs += __shfl_xor_sync(0xffffffffu, cs, 8);
        c = c / (cs + p.sinkhorn_eps);
        for (int it = 1; it < p.sinkhorn_iters; ++it) {
          float rs = c + __shfl_xor_sync(0xffffffffu, c, 1);
          rs += __shfl_xor_sync(0xffffffffu, rs, 2);
          c = c / (rs + p.sinkhorn_eps);
          cs = c + __shfl_xor_sync(0xffffffffu, c, 4);
          cs += __shfl_xor_sync(0xffffffffu, cs, 8);
          c = c / (cs + p.sinkhorn_eps);
        }
        if (lane < HC * HC) p.comb_out[t * HC * HC + lane] = c;
        if (lane < HC) {
          float pre = 0.0f, post = 0.0f;
#pragma unroll
          for (int h = 0; h < HC; ++h)
            if (h == lane) {
              pre = sigmoidf(v[h] * rr * hc_scale[0] + __ldg(p.hc_base + h)) + p.hc_pre_eps;
              post = sigmoidf(v[HC + h] * rr * hc_scale[1] + __ldg(p.hc_base + HC + h)) *
                     p.post_mult;
            }
          p.post_out[t * HC + lane] = post;
          pre_g[tl * HC + lane] = pre;
        }
    }
    if (p.stop_after <= 4) return;
    grid.sync();

    // ---------------- Phase B2 ----------------
    const int n_local = cta < chunk_tokens ? (chunk_tokens - 1 - cta) / G + 1 : 0;
    for (int base = 0; base < n_local; base += WARPS) {
      const int round_tokens = min(WARPS, n_local - base);
      if (tid < round_tokens * HC)
        sb.pre[tid / HC][tid % HC] = __ldcg(pre_g + (cta + (base + tid / HC) * G) * HC + tid % HC);
      __syncthreads();
      // B2: groups of warps, one token each: collapse, RMSNorm, y.
      int groups = 1;
      while (groups * 2 <= round_tokens) groups *= 2;
      const int warps_per_group = WARPS / groups;
      const int group = warp / warps_per_group;
      const int gthreads = warps_per_group * 32;
      const int gtid = tid - group * gthreads;
      __nv_bfloat16* li = layer_in + static_cast<int64_t>(group) * H;
      const int n_chunks = H / 8;
      for (int r0 = 0; r0 < round_tokens; r0 += groups) {
        const int slot = r0 + group;
        const bool active = slot < round_tokens;
        float ss = 0.0f;
        if (active) {
          const int t = chunk0 + cta + (base + slot) * G;
          const float pre[HC] = {sb.pre[slot][0], sb.pre[slot][1], sb.pre[slot][2],
                                 sb.pre[slot][3]};
          for (int ch = gtid; ch < n_chunks; ch += gthreads) {
            float out[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int h = 0; h < HC; ++h) {
              float nr[8];
              unpack8(__ldcg(reinterpret_cast<const uint4*>(
                          p.new_residual + (static_cast<int64_t>(t) * HC + h) * H) +
                      ch),
                      nr);
#pragma unroll
              for (int e = 0; e < 8; ++e) out[e] += pre[h] * nr[e];
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              out[e] = round_bf16(out[e]);
              ss += out[e] * out[e];
            }
            reinterpret_cast<uint4*>(li)[ch] = pack8(out);
          }
        }
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
        if (lane == 0) sb.red[warp] = ss;
        __syncthreads();
        if (active) {
          const int t = chunk0 + cta + (base + slot) * G;
          float total = 0.0f;
          for (int w = group * warps_per_group; w < (group + 1) * warps_per_group; ++w)
            total += sb.red[w];
          const float rr = rsqrtf(total / static_cast<float>(H) + p.norm_eps);
          for (int ch = gtid; ch < n_chunks; ch += gthreads) {
            float v[8], w[8];
            unpack8(reinterpret_cast<const uint4*>(li)[ch], v);
            unpack8(__ldg(reinterpret_cast<const uint4*>(p.norm_weight) + ch), w);
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] = v[e] * rr * w[e];
            reinterpret_cast<uint4*>(p.y + static_cast<int64_t>(t) * H)[ch] = pack8(v);
          }
        }
        __syncthreads();
      }
    }
    if (chunk0 + CHUNK_TOKENS < p.num_tokens) grid.sync();  // ws reuse by the next chunk
  }
}

size_t smem_bytes(const int hidden) {
  const size_t a = sizeof(SmemA);
  const size_t b = SMEM_B_OFFSET + sizeof(SmemBHeader) +
                   static_cast<size_t>(WARPS) * hidden * sizeof(__nv_bfloat16);
  return a > b ? a : b;
}

int64_t num_ctas(const int hidden) { return hidden / SLICE; }

void launch(const Params& params, cudaStream_t stream) {
  const int G = static_cast<int>(num_ctas(params.hidden));
  const size_t smem = smem_bytes(params.hidden);
  static bool attr_set = false;
  if (!attr_set) {
    TVM_FFI_ICHECK_EQ(cudaFuncSetAttribute(mhc_post_pre_one_launch_kernel,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(smem)),
                      cudaSuccess);
    attr_set = true;
  }
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(G);
  config.blockDim = dim3(THREADS);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeCooperative;
  attrs[0].val.cooperative = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&config, mhc_post_pre_one_launch_kernel, params);
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "mhc_post_pre_one_launch: " << cudaGetErrorString(err);
}

// parts: [P, T, H] bf16 (P partial block outputs, P = 1 for a plain block
// output); extra: [T, H] bf16 or an empty tensor.
void run(tvm::ffi::TensorView parts, tvm::ffi::TensorView extra, tvm::ffi::TensorView residual,
         tvm::ffi::TensorView post_prev, tvm::ffi::TensorView comb_prev, tvm::ffi::TensorView fn,
         tvm::ffi::TensorView hc_scale, tvm::ffi::TensorView hc_base,
         tvm::ffi::TensorView norm_weight, tvm::ffi::TensorView new_residual,
         tvm::ffi::TensorView post_out, tvm::ffi::TensorView comb_out, tvm::ffi::TensorView y,
         tvm::ffi::TensorView ws, double rms_eps, double hc_pre_eps, double sinkhorn_eps,
         double post_mult, double norm_eps, int64_t sinkhorn_iters, int64_t stop_after) {
  TVM_FFI_ICHECK_EQ(parts.ndim(), 3) << "parts must be [P, T, H]";
  const int64_t P = parts.size(0);
  const int64_t T = parts.size(1);
  const int64_t H = parts.size(2);
  TVM_FFI_ICHECK(P >= 1 && P <= MAX_PARTS) << "1..8 partial outputs";
  TVM_FFI_ICHECK_EQ(H % SLICE, 0);
  TVM_FFI_ICHECK_EQ(residual.size(0), T);
  TVM_FFI_ICHECK_EQ(residual.size(1), HC);
  TVM_FFI_ICHECK_EQ(residual.size(2), H);
  TVM_FFI_ICHECK_EQ(fn.size(0), MIX);
  TVM_FFI_ICHECK_EQ(fn.size(1), HC * H);
  TVM_FFI_ICHECK_GE(ws.numel(), std::min<int64_t>(T, CHUNK_TOKENS) * ((H / SLICE) * WS + HC));
  Params p{};
  for (int64_t q = 0; q < P; ++q)
    p.parts[q] = static_cast<const __nv_bfloat16*>(parts.data_ptr()) + q * T * H;
  p.num_parts = static_cast<int>(P);
  p.part_row_stride = H;
  p.extra = extra.numel() > 0 ? static_cast<const __nv_bfloat16*>(extra.data_ptr()) : nullptr;
  p.residual = static_cast<const __nv_bfloat16*>(residual.data_ptr());
  p.post_prev = static_cast<const float*>(post_prev.data_ptr());
  p.comb_prev = static_cast<const float*>(comb_prev.data_ptr());
  p.fn = static_cast<const float*>(fn.data_ptr());
  p.hc_scale = static_cast<const float*>(hc_scale.data_ptr());
  p.hc_base = static_cast<const float*>(hc_base.data_ptr());
  p.norm_weight = static_cast<const __nv_bfloat16*>(norm_weight.data_ptr());
  p.new_residual = static_cast<__nv_bfloat16*>(new_residual.data_ptr());
  p.post_out = static_cast<float*>(post_out.data_ptr());
  p.comb_out = static_cast<float*>(comb_out.data_ptr());
  p.y = static_cast<__nv_bfloat16*>(y.data_ptr());
  p.ws = static_cast<float*>(ws.data_ptr());
  p.num_tokens = static_cast<int>(T);
  p.hidden = static_cast<int>(H);
  p.rms_eps = static_cast<float>(rms_eps);
  p.hc_pre_eps = static_cast<float>(hc_pre_eps);
  p.sinkhorn_eps = static_cast<float>(sinkhorn_eps);
  p.post_mult = static_cast<float>(post_mult);
  p.norm_eps = static_cast<float>(norm_eps);
  p.sinkhorn_iters = static_cast<int>(sinkhorn_iters);
  p.stop_after = static_cast<int>(stop_after);
  if (T == 0) return;
  launch(p, get_stream(parts.device()));
}

int64_t workspace_floats(int64_t num_tokens, int64_t hidden) {
  return std::min<int64_t>(num_tokens, CHUNK_TOKENS) * ((hidden / SLICE) * WS + HC);
}

}  // namespace mhc_one_launch

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_post_pre_one_launch, mhc_one_launch::run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_post_pre_one_launch_workspace_floats,
                              mhc_one_launch::workspace_floats);
