// Fused mHC boundary in one launch: the block-output sum, post mix, pre mixes, gates, Sinkhorn,
// collapse and RMSNorm. Built by ../mhc_one_launch_cuda.py for the cuda_one_launch backend of
// mhc_fused_post_pre_rms_norm and mhc_fused_sum_post_pre_rms_norm; this is the build for token
// tiles of TM <= 32. Written in this repository (no upstream source).
//   * warp per token, lane per 8 columns: 16-byte loads/stores of residual, x, new_res, y
//   * the 24 mixes on BF16 tensor cores (mma.sync m16n8k16): A = new_res tile from smem (ldmatrix),
//     B = fn split into bf16 hi + lo fragments formed from the CTA's fn slice in smem (cp.async)
//   * the 10 Gram terms (sum_h nr_j nr_k; trace = sqrsum) per token by a warp all-reduce
// Cluster of CS=16 CTAs, CTA rank r owns columns [256r, 256r+256) of every stream.
// Per token tile: A -> cluster barrier -> B (owner warp: DSMEM gather, gates, coef to all CTAs,
// Sinkhorn) -> cluster barrier -> C (collapse + RMSNorm, y). With late_sinkhorn bit 4 the Sinkhorn
// overlaps phase C: the owner warps are the last ones (NW-1, NW-2), they arrive on the second
// cluster barrier as soon as the coefficients are out, run Sinkhorn, then wait; phase C runs on
// the other warps.
#include <tvm/ffi/extra/module.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "tvm_ffi_utils.h"

namespace mfuse3o {
namespace cg = cooperative_groups;
constexpr int H = 4096, HC = 4, K = HC * H, NV = 34, NVP = 36, MAXP = 9, TMMAX = 32;
constexpr int CS = 16, NT = H / CS, NW = NT / 32;  // 256 columns, 256 threads, 8 warps
constexpr int TP = TMMAX + 1, RP = NT + 8;
constexpr int FP = NT + 4;  // fn smem row pitch (floats): 96 rows (m, j) x 256 columns                // nr smem: [4 streams][TP tokens][RP] bf16
constexpr unsigned FULL = 0xffffffffu;
constexpr int SK_FN = 1, SK_GATHER = 2, SK_COEF = 4, SK_SINK = 8, SK_NRSTORE = 16, SK_BAR1 = 32, SK_BAR2 = 64,
              SK_MMA = 128;

struct Params {
  const __nv_bfloat16* parts[MAXP];
  int np;
  const __nv_bfloat16* extra;
  const __nv_bfloat16* res;
  const float* post_prev;
  const float* comb_prev;
  const float* fn;
  const float* hc_scale;
  const float* hc_base;
  const __nv_bfloat16* norm_w;
  __nv_bfloat16* new_res;
  float* post_out;
  float* comb_out;
  __nv_bfloat16* y;
  unsigned* sink;
  int T, TM, ntiles, stage, skip, iters, late_sinkhorn, pdl;
  float rms_eps, pre_eps, sk_eps, post_mult, norm_eps;
};

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float sigm(const float v) { return 1.f / (1.f + __expf(-v)); }
__device__ __forceinline__ void unpack8(const uint4 u, float* f) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) { const float2 t = __bfloat1622float2(h[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}
__device__ __forceinline__ uint4 pack8(const float* f) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return u;
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ void sinkhorn_store(const float* cl, int iters, float eps, float* out) {
  float c[16];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float m = fmaxf(fmaxf(cl[i * 4], cl[i * 4 + 1]), fmaxf(cl[i * 4 + 2], cl[i * 4 + 3]));
    float e[4], s = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) { e[j] = __expf(cl[i * 4 + j] - m); s += e[j]; }
    const float r = 1.f / s;
#pragma unroll
    for (int j = 0; j < 4; ++j) c[i * 4 + j] = e[j] * r + eps;
  }
  auto colnorm = [&]() {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float r = 1.f / ((c[j] + c[4 + j]) + (c[8 + j] + c[12 + j]) + eps);
#pragma unroll
      for (int i = 0; i < 4; ++i) c[i * 4 + j] *= r;
    }
  };
  colnorm();
  for (int it = 1; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float r = 1.f / ((c[i * 4] + c[i * 4 + 1]) + (c[i * 4 + 2] + c[i * 4 + 3]) + eps);
#pragma unroll
      for (int j = 0; j < 4; ++j) c[i * 4 + j] *= r;
    }
    colnorm();
  }
#pragma unroll
  for (int k = 0; k < 16; k += 4) *reinterpret_cast<float4*>(out + k) = make_float4(c[k], c[k + 1], c[k + 2], c[k + 3]);
}

// Same recurrence with packed fp32x2 adds/muls (sm_100 FADD2/FMUL2): rows stored as float2 pairs.
__device__ void sinkhorn_store_fast(const float* cl, int iters, float eps, float* out) {
  float2 c[4][2];
  const float2 e2 = make_float2(eps, eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float m = fmaxf(fmaxf(cl[i * 4], cl[i * 4 + 1]), fmaxf(cl[i * 4 + 2], cl[i * 4 + 3]));
    const float a = __expf(cl[i * 4] - m), b = __expf(cl[i * 4 + 1] - m), d = __expf(cl[i * 4 + 2] - m),
                f = __expf(cl[i * 4 + 3] - m);
    const float r = 1.f / ((a + b) + (d + f));
    c[i][0] = __fadd2_rn(__fmul2_rn(make_float2(a, b), make_float2(r, r)), e2);
    c[i][1] = __fadd2_rn(__fmul2_rn(make_float2(d, f), make_float2(r, r)), e2);
  }
  auto colnorm = [&]() {
    float2 s0 = __fadd2_rn(__fadd2_rn(c[0][0], c[1][0]), __fadd2_rn(c[2][0], c[3][0]));
    float2 s1 = __fadd2_rn(__fadd2_rn(c[0][1], c[1][1]), __fadd2_rn(c[2][1], c[3][1]));
    s0 = __fadd2_rn(s0, e2);
    s1 = __fadd2_rn(s1, e2);
    const float2 r0 = make_float2(1.f / s0.x, 1.f / s0.y), r1 = make_float2(1.f / s1.x, 1.f / s1.y);
#pragma unroll
    for (int i = 0; i < 4; ++i) { c[i][0] = __fmul2_rn(c[i][0], r0); c[i][1] = __fmul2_rn(c[i][1], r1); }
  };
  colnorm();
  for (int it = 1; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 h = __fadd2_rn(c[i][0], c[i][1]);
      const float r = 1.f / (h.x + h.y + eps);
      c[i][0] = __fmul2_rn(c[i][0], make_float2(r, r));
      c[i][1] = __fmul2_rn(c[i][1], make_float2(r, r));
    }
    colnorm();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
    *reinterpret_cast<float4*>(out + 4 * i) = make_float4(c[i][0].x, c[i][0].y, c[i][1].x, c[i][1].y);
}

__global__ void __launch_bounds__(NT, 1) fused3_kernel(const __grid_constant__ Params p) {
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int cid = blockIdx.x / CS, ncl = gridDim.x / CS;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int col0 = rank * NT;  // first hidden column of this CTA
  extern __shared__ float4 smraw[];
  float* fnS = reinterpret_cast<float*>(smraw);                                    // [24*4][FP]
  __nv_bfloat16* nrs = reinterpret_cast<__nv_bfloat16*>(fnS + 96 * FP);            // [4][TP][RP]
  float* mixp = reinterpret_cast<float*>(nrs + 4 * TP * RP);                       // [TMMAX][20]
  float* red = mixp + TMMAX * 20;                                                  // [NW][TMMAX][24]
  float* part = red + NW * TMMAX * 24;                                                // [TMMAX][NVP]
  float* coef = part + TMMAX * NVP;                                                // [TMMAX][8]
  float* clog = coef + TMMAX * 8;                                                  // [TMMAX][16]
  const int stage = p.stage;
  unsigned sink = 0;

  // fn slice -> smem with cp.async (weights only: issued before the PDL wait)
  const int jw = warp >> 1, cw = (warp & 1) * 128;
  if (!(p.skip & SK_FN)) {
#pragma unroll 4
    for (int i = tid; i < 96 * 64; i += NT) {
      const int rr = i >> 6, c = (i & 63) * 4;  // row rr = m*4 + j
      const float* src = p.fn + static_cast<size_t>(rr >> 2) * K + (rr & 3) * H + col0 + c;
      const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(fnS + rr * FP + c));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  float wv[8];
  unpack8(__ldg(reinterpret_cast<const uint4*>(p.norm_w + col0 + lane * 8)), wv);
  if (p.pdl) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
  for (int tile = cid; tile < p.ntiles; tile += ncl) {
    const int t0 = tile * p.TM;
    const int tn = min(p.TM, p.T - t0);
    for (int i = tid; i < tn * 20; i += NT) {
      const int t = i / 20, k = i - t * 20;
      mixp[i] = k < 4 ? __ldg(p.post_prev + (t0 + t) * 4 + k) : __ldg(p.comb_prev + (t0 + t) * 16 + k - 4);
    }
    // ---- phase A: post (warp per token), Gram ----
    // issue all of this warp's loads first (<= 4 tokens per warp)
    uint4 rv[4][4], xv[4];
    float xs[4][8];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int t = warp + q * NW;
      if (t < tn) {
        const size_t row = static_cast<size_t>(t0 + t);
#pragma unroll
        for (int i = 0; i < 4; ++i) rv[q][i] = __ldg(reinterpret_cast<const uint4*>(p.res + row * K + i * H + col0 + lane * 8));
        xv[q] = __ldg(reinterpret_cast<const uint4*>(p.parts[0] + row * H + col0 + lane * 8));
      }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int t = warp + q * NW;
      if (t < tn) {
        const size_t row = static_cast<size_t>(t0 + t);
        unpack8(xv[q], xs[q]);
        if (p.np > 1) {
          uint4 pv[MAXP - 1];
#pragma unroll
          for (int k = 1; k < MAXP - 1; ++k)
            if (k < p.np) pv[k - 1] = __ldg(reinterpret_cast<const uint4*>(p.parts[k] + row * H + col0 + lane * 8));
#pragma unroll
          for (int k = 1; k < MAXP - 1; ++k)
            if (k < p.np) {
              float g[8];
              unpack8(pv[k - 1], g);
#pragma unroll
              for (int e = 0; e < 8; ++e) xs[q][e] += g[e];
            }
#pragma unroll
          for (int e = 0; e < 8; ++e) xs[q][e] = __bfloat162float(__float2bfloat16_rn(xs[q][e]));
        }
        if (p.extra != nullptr) {
          float g[8];
          unpack8(__ldg(reinterpret_cast<const uint4*>(p.extra + row * H + col0 + lane * 8)), g);
#pragma unroll
          for (int e = 0; e < 8; ++e) xs[q][e] = __bfloat162float(__float2bfloat16_rn(xs[q][e] + g[e]));
        }
      }
    }
    __syncthreads();  // mixp visible
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int t = warp + q * NW;
      if (t >= tn) continue;
      const size_t row = static_cast<size_t>(t0 + t);
      const float* mp = mixp + t * 20;
      float r[4][8], nr[4][8];
#pragma unroll
      for (int i = 0; i < 4; ++i) unpack8(rv[q][i], r[i]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float v = r[j][e];
          if (stage >= 1)
            v = mp[4 + j] * r[0][e] + mp[8 + j] * r[1][e] + mp[12 + j] * r[2][e] + mp[16 + j] * r[3][e] + mp[j] * xs[q][e];
          nr[j][e] = v;
        }
        const uint4 u = pack8(nr[j]);
        unpack8(u, nr[j]);  // rounded values feed the GEMM and Gram
        if (!(p.skip & SK_NRSTORE)) *reinterpret_cast<uint4*>(p.new_res + row * K + j * H + col0 + lane * 8) = u;
        *reinterpret_cast<uint4*>(nrs + (j * TP + t) * RP + lane * 8) = u;
      }
      if (stage < 5) *reinterpret_cast<uint4*>(p.y + row * H + col0 + lane * 8) = pack8(xs[q]);
      if (stage >= 2) {
        float g[10];
#pragma unroll
        for (int k = 0; k < 10; ++k) g[k] = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          g[0] += nr[0][e] * nr[0][e]; g[1] += nr[0][e] * nr[1][e]; g[2] += nr[0][e] * nr[2][e];
          g[3] += nr[0][e] * nr[3][e]; g[4] += nr[1][e] * nr[1][e]; g[5] += nr[1][e] * nr[2][e];
          g[6] += nr[1][e] * nr[3][e]; g[7] += nr[2][e] * nr[2][e]; g[8] += nr[2][e] * nr[3][e];
          g[9] += nr[3][e] * nr[3][e];
        }
#pragma unroll
        for (int s = 16; s >= 1; s >>= 1)
#pragma unroll
          for (int k = 0; k < 10; ++k) g[k] += __shfl_xor_sync(FULL, g[k], s);
        if (lane < 10) {
          float gv = g[0];
#pragma unroll
          for (int k = 1; k < 10; ++k) gv = lane == k ? g[k] : gv;
          part[t * NVP + 24 + lane] = gv;
        }
      }
    }
    if (stage >= 2) {
      asm volatile("cp.async.wait_all;" ::: "memory");
      __syncthreads();  // nrs and fnS complete
      // ---- mixes on tensor cores: warp w covers stream jw, columns cw..cw+127 ----
      float acc[2][3][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 3; ++nt)
#pragma unroll
          for (int k = 0; k < 4; ++k) acc[mt][nt][k] = 0.f;
      const int nmt = (tn + 15) >> 4;
      if (!(p.skip & SK_MMA)) {
        const uint32_t abase = static_cast<uint32_t>(__cvta_generic_to_shared(
            nrs + (jw * TP + (lane & 15)) * RP + cw + (lane >> 4) * 8));
#pragma unroll 2
        for (int ks = 0; ks < 8; ++ks) {
          uint32_t bh[3][2], bl[3][2];
#pragma unroll
          for (int nt = 0; nt < 3; ++nt)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              const int n = nt * 8 + (lane >> 2);
              const int kk = ks * 16 + hf * 8 + 2 * (lane & 3);
              const float2 v = *reinterpret_cast<const float2*>(fnS + (n * 4 + jw) * FP + cw + kk);
              const __nv_bfloat162 hi = __floats2bfloat162_rn(v.x, v.y);
              const float2 hf2 = __bfloat1622float2(hi);
              bh[nt][hf] = *reinterpret_cast<const uint32_t*>(&hi);
              bl[nt][hf] = pack2(v.x - hf2.x, v.y - hf2.y);
            }
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            if (mt >= nmt) break;
            uint32_t a0, a1, a2, a3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(abase + mt * 16 * RP * 2 + ks * 32));
#pragma unroll
            for (int nt = 0; nt < 3; ++nt) {
              asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                           : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]), "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                           : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bh[nt][0]), "r"(bh[nt][1]));
              asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                           : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]), "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                           : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bl[nt][0]), "r"(bl[nt][1]));
            }
          }
        }
      }
      {
        const int r0 = lane >> 2, c0 = 2 * (lane & 3);
        float* rw = red + warp * TMMAX * 24;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          if (mt >= nmt) break;
#pragma unroll
          for (int nt = 0; nt < 3; ++nt) {
            *reinterpret_cast<float2*>(rw + (mt * 16 + r0) * 24 + nt * 8 + c0) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
            *reinterpret_cast<float2*>(rw + (mt * 16 + r0 + 8) * 24 + nt * 8 + c0) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
          }
        }
      }
      __syncthreads();
      for (int i = tid; i < tn * 24; i += NT) {
        const int t = i / 24, m = i - t * 24;
        float s = 0.f;
#pragma unroll
        for (int ww = 0; ww < NW; ++ww) s += red[ww * TMMAX * 24 + t * 24 + m];
        part[t * NVP + m] = s;
      }
    }
    if (stage < 3) { __syncthreads(); continue; }
    if (p.skip & SK_BAR1) __syncthreads(); else cluster.sync();
    // ---- phase B ----
    for (int t = rank + (NW - 1 - warp) * CS; t < tn; t += CS * NW) {
      float vals[CS], vals2[CS];
#pragma unroll
      for (int rk = 0; rk < CS; ++rk) {
        const float* rp = (p.skip & SK_GATHER) ? part + t * NVP : cluster.map_shared_rank(part + t * NVP, rk);
        vals[rk] = rp[lane];
        vals2[rk] = rp[32 + (lane & 1)];
      }
      float s = 0.f, s2 = 0.f;
#pragma unroll
      for (int rk = 0; rk < CS; ++rk) { s += vals[rk]; s2 += vals2[rk]; }
      float G[10];
#pragma unroll
      for (int k = 0; k < 8; ++k) G[k] = __shfl_sync(FULL, s, 24 + k);
      G[8] = __shfl_sync(FULL, s2, 0);
      G[9] = __shfl_sync(FULL, s2, 1);
      const float rr = rsqrtf((G[0] + G[4] + G[7] + G[9]) / static_cast<float>(K) + p.rms_eps);
      const float mix = s * rr;
      const float base = lane < 24 ? __ldg(p.hc_base + lane) : 0.f;
      const float pre = sigm(mix * __ldg(p.hc_scale) + base) + p.pre_eps;
      float pj[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) pj[j] = __shfl_sync(FULL, pre, j);
      const size_t row = static_cast<size_t>(t0 + t);
      if (lane >= 4 && lane < 8) p.post_out[row * 4 + lane - 4] = sigm(mix * __ldg(p.hc_scale + 1) + base) * p.post_mult;
      const float y2 = pj[0] * pj[0] * G[0] + pj[1] * pj[1] * G[4] + pj[2] * pj[2] * G[7] + pj[3] * pj[3] * G[9] +
                       2.f * (pj[0] * (pj[1] * G[1] + pj[2] * G[2] + pj[3] * G[3]) + pj[1] * (pj[2] * G[5] + pj[3] * G[6]) +
                              pj[2] * pj[3] * G[8]);
      const float rstd = rsqrtf(y2 / static_cast<float>(H) + p.norm_eps);
      if ((p.skip & SK_COEF) ? lane == 0 : lane < CS) {
        float* cp = (p.skip & SK_COEF) ? coef + t * 8 : cluster.map_shared_rank(coef + t * 8, lane);
        *reinterpret_cast<float4*>(cp) = make_float4(pj[0], pj[1], pj[2], pj[3]);
        cp[4] = rstd;
      }
      const float logit = mix * __ldg(p.hc_scale + 2) + base;
      float cl[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) cl[k] = __shfl_sync(FULL, logit, 8 + k);
      if (p.skip & SK_SINK) {
        if (lane < 16) p.comb_out[row * 16 + lane] = cl[lane];
      } else if (p.late_sinkhorn & 5) {
        if (lane < 16) clog[t * 16 + lane] = cl[lane];
      } else if (lane == 0) {
        if (p.late_sinkhorn & 2) sinkhorn_store_fast(cl, stage >= 4 ? p.iters : 1, p.sk_eps, p.comb_out + row * 16);
        else sinkhorn_store(cl, stage >= 4 ? p.iters : 1, p.sk_eps, p.comb_out + row * 16);
      }
    }
    if (p.skip & SK_BAR2) {
      __syncthreads();
    } else if ((p.late_sinkhorn & 4) && !(p.skip & SK_SINK)) {
      // coefficients are out: arrive, run Sinkhorn on the owner warps, then wait
      asm volatile("barrier.cluster.arrive.release;" ::: "memory");
      for (int t = rank + (NW - 1 - warp) * CS; t < tn; t += CS * NW)
        if (lane == 0) sinkhorn_store_fast(clog + t * 16, stage >= 4 ? p.iters : 1, p.sk_eps, p.comb_out + static_cast<size_t>(t0 + t) * 16);
      asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
    } else {
      cluster.sync();
    }
    // ---- phase C ----
    if (stage >= 5) {
      // with the overlap, the owner warps (the last ceil(tn / CS)) skip phase C
      const int ncw = (p.late_sinkhorn & 4) ? NW - min(NW - 1, (tn + CS - 1) / CS) : NW;
      for (int t = warp; warp < ncw && t < tn; t += ncw) {
        const float4 pc = *reinterpret_cast<const float4*>(coef + t * 8);
        const float rs = coef[t * 8 + 4];
        float a[8], b[8];
        unpack8(*reinterpret_cast<const uint4*>(nrs + (0 * TP + t) * RP + lane * 8), b);
#pragma unroll
        for (int e = 0; e < 8; ++e) a[e] = pc.x * b[e];
        unpack8(*reinterpret_cast<const uint4*>(nrs + (1 * TP + t) * RP + lane * 8), b);
#pragma unroll
        for (int e = 0; e < 8; ++e) a[e] += pc.y * b[e];
        unpack8(*reinterpret_cast<const uint4*>(nrs + (2 * TP + t) * RP + lane * 8), b);
#pragma unroll
        for (int e = 0; e < 8; ++e) a[e] += pc.z * b[e];
        unpack8(*reinterpret_cast<const uint4*>(nrs + (3 * TP + t) * RP + lane * 8), b);
#pragma unroll
        for (int e = 0; e < 8; ++e) a[e] = __bfloat162float(__float2bfloat16_rn(a[e] + pc.w * b[e])) * rs * wv[e];
        *reinterpret_cast<uint4*>(p.y + static_cast<size_t>(t0 + t) * H + col0 + lane * 8) = pack8(a);
      }
    }
    if ((p.late_sinkhorn & 1) && !(p.skip & SK_SINK)) {
      for (int t = rank + (NW - 1 - warp) * CS; t < tn; t += CS * NW)
        if (lane == 0 && (p.late_sinkhorn & 2)) sinkhorn_store_fast(clog + t * 16, stage >= 4 ? p.iters : 1, p.sk_eps, p.comb_out + static_cast<size_t>(t0 + t) * 16);
        else if (lane == 0) sinkhorn_store(clog + t * 16, stage >= 4 ? p.iters : 1, p.sk_eps, p.comb_out + static_cast<size_t>(t0 + t) * 16);
    }
    __syncthreads();
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  if (sink == 0x9e3779b9u) *p.sink = sink;
}

constexpr size_t SMEM = sizeof(float) * 96 * FP + sizeof(__nv_bfloat16) * 4 * TP * RP +
                        sizeof(float) * (TMMAX * 20 + NW * TMMAX * 24 + TMMAX * NVP + TMMAX * 8 + TMMAX * 16);

cudaLaunchConfig_t make_cfg(int ncl, cudaStream_t st, cudaLaunchAttribute* at, bool pdl) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(fused3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(SMEM));
    cudaFuncSetAttribute(fused3_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    init = true;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(CS * ncl);
  cfg.blockDim = dim3(NT);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = st;
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = CS;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at;
  cfg.numAttrs = 2;
  return cfg;
}

int64_t max_clusters(int64_t) {
  cudaLaunchAttribute at[2];
  cudaLaunchConfig_t cfg = make_cfg(1, 0, at, false);
  cfg.numAttrs = 1;
  int n = 0;
  const cudaError_t e = cudaOccupancyMaxActiveClusters(&n, fused3_kernel, &cfg);
  TVM_FFI_ICHECK_EQ(e, cudaSuccess) << cudaGetErrorString(e);
  return n;
}

const __nv_bfloat16* b16(tvm::ffi::TensorView t) { return static_cast<const __nv_bfloat16*>(t.data_ptr()); }

void run(tvm::ffi::TensorView parts, tvm::ffi::TensorView extra, tvm::ffi::TensorView residual,
         tvm::ffi::TensorView post_prev, tvm::ffi::TensorView comb_prev, tvm::ffi::TensorView fn,
         tvm::ffi::TensorView hc_scale, tvm::ffi::TensorView hc_base, tvm::ffi::TensorView norm_weight,
         tvm::ffi::TensorView new_res, tvm::ffi::TensorView post_out, tvm::ffi::TensorView comb_out,
         tvm::ffi::TensorView y, tvm::ffi::TensorView sink, int64_t TM, int64_t cs, int64_t ncl_max,
         int64_t stage, int64_t skip, int64_t iters, int64_t late_sinkhorn, int64_t pdl, double rms_eps,
         double pre_eps, double sk_eps, double post_mult, double norm_eps) {
  Params p{};
  const int64_t P = parts.size(0);
  const int64_t T = parts.size(1);
  TVM_FFI_ICHECK(P >= 1 && P <= 8);
  TVM_FFI_ICHECK(parts.size(2) == H);
  TVM_FFI_ICHECK(TM >= 1 && TM <= TMMAX);
  TVM_FFI_ICHECK(cs == CS);
  for (int64_t q = 0; q < P; ++q) p.parts[q] = b16(parts) + q * T * H;
  p.np = static_cast<int>(P);
  p.extra = extra.numel() > 0 ? b16(extra) : nullptr;
  p.res = b16(residual);
  p.post_prev = static_cast<const float*>(post_prev.data_ptr());
  p.comb_prev = static_cast<const float*>(comb_prev.data_ptr());
  p.fn = static_cast<const float*>(fn.data_ptr());
  p.hc_scale = static_cast<const float*>(hc_scale.data_ptr());
  p.hc_base = static_cast<const float*>(hc_base.data_ptr());
  p.norm_w = b16(norm_weight);
  p.new_res = static_cast<__nv_bfloat16*>(new_res.data_ptr());
  p.post_out = static_cast<float*>(post_out.data_ptr());
  p.comb_out = static_cast<float*>(comb_out.data_ptr());
  p.y = static_cast<__nv_bfloat16*>(y.data_ptr());
  p.sink = static_cast<unsigned*>(sink.data_ptr());
  p.T = static_cast<int>(T);
  p.TM = static_cast<int>(TM);
  p.ntiles = static_cast<int>((T + TM - 1) / TM);
  p.stage = static_cast<int>(stage);
  p.skip = static_cast<int>(skip);
  p.iters = static_cast<int>(iters);
  p.late_sinkhorn = static_cast<int>(late_sinkhorn);
  p.pdl = static_cast<int>(pdl);
  p.rms_eps = static_cast<float>(rms_eps);
  p.pre_eps = static_cast<float>(pre_eps);
  p.sk_eps = static_cast<float>(sk_eps);
  p.post_mult = static_cast<float>(post_mult);
  p.norm_eps = static_cast<float>(norm_eps);
  const int ncl = static_cast<int>(std::min<int64_t>(p.ntiles, ncl_max));
  cudaLaunchAttribute at[2];
  cudaLaunchConfig_t cfg = make_cfg(ncl, get_stream(parts.device()), at, pdl != 0);
  const cudaError_t err = cudaLaunchKernelEx(&cfg, fused3_kernel, p);
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}
}  // namespace mfuse3o

TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_fused, mfuse3o::run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_max_clusters, mfuse3o::max_clusters);
