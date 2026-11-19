// Vendored kernel source for profiling/runners/attention/kda_recoverssm_commit_ws.py (backend
// cuda_persistent_ws_tf32_mma); remove it once the kernel ships in a callable package. Origin: the KDA RecoverSSM
// commit kernel study, commit/kcommit7.cu (source md5 e247804c7be0, 2026-11-13). Verbatim below this header.
// Lane K-5, third round, attempt 1 (v7): v6s3t (3xTF32 tensor-core fold) with the measured coefficient-side causes fixed.
// Ledger (ncu SourceCounters, v6s3t c128 L9, job 21895): coefficient warps execute 16.2M instructions (cl8) vs 13.1M
// (cl1) -- 440 per warp per item, 2.3x a fold warp's 190 -- so the 4 coefficient warps (one per SMSP) are the
// critical stage. What scales with n is their serial gate chain (cum += lb*sigmoid, 784 vs 228 samples, guarded
// loads that block ILP); what does not, but costs every role, is the per-item decode (integer div/mod + 3 dependent
// __ldg: 6.7M instructions, 715 + 317 + 199 samples), the 40-shuffle norm reduction (2.9M, 413 samples), the
// per-item __ldg of A_log / dt_bias on the coefficient path (204 samples), and the conv copy's runtime division.
// Fixes:
//   * the producer decodes each item once and publishes a stage header (item, head, b, l, n, slots, A = exp(A_log))
//     before arming kgfull; dt_bias rows arrive by bulk copy with the records;
//   * gate sigmoids are computed for all 8 tokens unconditionally (select, not branch), then prefix-summed;
//   * key norms: transpose-reduce (9 shuffles for 8 tokens instead of 40);
//   * conv rows copied row by row without division.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>

#define KD 128
#define QMAX 8
#define CVROW 1024            // bytes reserved per conv history row of one item (conv_dim / H * 2 <= 1024)
#define NPROD 32
#define NCOEF 128
#define NFOLD 256
#define NTHREADS (NPROD + NCOEF + NFOLD)

struct CA {
  const int64_t *st_addr, *st_stride, *co_addr, *co_stride, *kg_addr, *kg_stride;
  const float *a_log, *dtb;
  const int *src, *clen, *dst;
  const int64_t *cv_addr, *cv_bstride, *cv_dstride, *cv_tstride;
  int H, Q, conv_dim, hist;
  float lb;
  int null_slot;
  int B, L;
};

constexpr int PITCH = 256;
constexpr int ST_B = KD * KD * 2, ST_SM = ST_B, KG_B = QMAX * 2 * KD * 2, CO_B = QMAX * KD * 4, CV_B = 8 * CVROW;
constexpr int SB_B = 16 * 32 * 16, SE_B = KD * 4, BI_B = KD * 4;
constexpr int OFF_KG = ST_SM, OFF_CO = OFF_KG + KG_B, OFF_CV = OFF_CO + CO_B, OFF_B = OFF_CV + CV_B, OFF_E = OFF_B + SB_B, OFF_BI = OFF_E + SE_B;
constexpr int STAGE_B = OFF_BI + BI_B;

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(b)), "r"(c) : "memory"); }
__device__ __forceinline__ void mb_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void mb_arrive_tx(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}"
               :: "r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(b)) : "memory");
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma1688(float* c, const uint32_t* a, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(b0));
}
__device__ __forceinline__ uint32_t pk(float lo_elem, float hi_elem) {  // bf16x2: first element in the low half
  __nv_bfloat162 v = __floats2bfloat162_rn(lo_elem, hi_elem);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void mma_tf32(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t tf32(float v) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v)); return r; }
__device__ __forceinline__ void split32(float v, uint32_t& hi, uint32_t& lo) { hi = tf32(v); lo = tf32(v - __uint_as_float(hi)); }
__device__ __forceinline__ void split(float v, float& hi, float& lo) {
  hi = __bfloat162float(__float2bfloat16(v));
  lo = v - hi;
}


struct Hdr { int it, head, b, l, n, live; long long s, d; float A; };

template <int RUNG, int S>
__global__ void __launch_bounds__(NTHREADS, 1) k_commit7(CA a, int items) {
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ __align__(8) uint64_t kgfull[S], stfull[S], coefb[S], empty[S];
  __shared__ float sN[4][QMAX];
  __shared__ Hdr sH[S];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int H = a.H, per = a.conv_dim / H, hist = a.hist;
  if (tid == 0) {
    for (int s = 0; s < S; ++s) { mb_init(&kgfull[s], 1); mb_init(&stfull[s], 1); mb_init(&coefb[s], NCOEF); mb_init(&empty[s], NFOLD); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  auto stage = [&](int s) { return smem + s * STAGE_B; };

  if (warp == 0) {
    // ---------------------------------------------------------------- producer: decode once, header + copies
    for (int k = 0;; ++k) {
      const int s = k % S;
      if (k >= S) mb_wait(&empty[s], ((k / S) & 1) ^ 1);
      const int it = blockIdx.x + k * gridDim.x;
      if (it >= items) {
        if (lane == 0) { sH[s].it = -1; mb_arrive(&kgfull[s]); }
        break;
      }
      Hdr x;
      x.it = it; x.head = it % H; const int r = it / H; x.b = r % a.B; x.l = r / a.B;
      x.n = __ldg(a.clen + x.b); x.s = __ldg(a.src + x.b); x.d = __ldg(a.dst + x.b);
      x.live = x.n > 0 && x.s > a.null_slot && x.d > a.null_slot;
      x.A = RUNG >= 2 ? expf(__ldg(a.a_log + x.l * H + x.head)) : 0.f;
      unsigned char* p = stage(s);
      if (lane == 0) {
        sH[s] = x;
        if (x.live && RUNG >= 1) {
          const char* kg = reinterpret_cast<const char*>(__ldg(a.kg_addr + x.l))
                           + ((long long)x.b * __ldg(a.kg_stride + x.l) + (long long)x.head * a.Q * 2 * KD) * 2;
          const char* co = reinterpret_cast<const char*>(__ldg(a.co_addr + x.l))
                           + ((long long)x.b * __ldg(a.co_stride + x.l) + (long long)x.head * a.Q * KD) * 4;
          const float* bi = a.dtb + ((long long)x.l * H + x.head) * KD;
          mb_arrive_tx(&kgfull[s], 2 * x.n * 512 + BI_B);
          bulk_g2s(p + OFF_KG, kg, x.n * 512, &kgfull[s]);
          bulk_g2s(p + OFF_CO, co, x.n * 512, &kgfull[s]);
          bulk_g2s(p + OFF_BI, bi, BI_B, &kgfull[s]);
        } else {
          mb_arrive(&kgfull[s]);
        }
      }
      if (x.live) {
        const uint32_t cvb = RUNG >= 3 ? hist * per * 2 : 0;
        if (lane == 0) {
          mb_arrive_tx(&stfull[s], ST_B + cvb);
          const char* src = reinterpret_cast<const char*>(__ldg(a.st_addr + x.l)) + (x.s * __ldg(a.st_stride + x.l) + (long long)x.head * KD * KD) * 2;
          bulk_g2s(p, src, ST_B, &stfull[s]);
        }
        __syncwarp();
        if (RUNG >= 3 && lane < hist) {
          const long long bs = __ldg(a.cv_bstride + x.l), ts = __ldg(a.cv_tstride + x.l);
          const char* cs = reinterpret_cast<const char*>(__ldg(a.cv_addr + x.l)) + (x.s * bs + (long long)x.head * per) * 2;
          bulk_g2s(p + OFF_CV + lane * CVROW, cs + (long long)(x.n - 1 + lane) * ts * 2, per * 2, &stfull[s]);
        }
      } else if (lane == 0) {
        mb_arrive(&stfull[s]);
      }
    }
  } else if (warp <= 4) {
    // ---------------------------------------------------------------- coefficient warps: column c of Kd and e^L
    const int c = tid - NPROD, cw = warp - 1;
    const int q = c >> 3, jt = 4 * (q >> 2) + ((c & 7) >> 1), nB = 2 * (q & 3) + (c & 1);
    const int b4 = (lane >> 4) & 1, b3 = (lane >> 3) & 1, b2 = (lane >> 2) & 1;
    for (int k = 0;; ++k) {
      const int s = k % S;
      mb_wait(&kgfull[s], (k / S) & 1);
      const Hdr x = sH[s];
      if (x.it < 0) break;
      if (x.live && RUNG >= 2) {
        unsigned char* p = stage(s);
        const __nv_bfloat16* kg = reinterpret_cast<const __nv_bfloat16*>(p + OFF_KG);
        const int n = x.n;
        const float bias = reinterpret_cast<const float*>(p + OFF_BI)[c];
        float kv[QMAX], sg[QMAX], logs[QMAX];
#pragma unroll
        for (int t = 0; t < QMAX; ++t) {  // all tokens, independent; rows t >= n hold stale data and are selected out
          const float kraw = __bfloat162float(kg[t * 2 * KD + c]);
          const float g = __bfloat162float(kg[t * 2 * KD + KD + c]);
          sg[t] = a.lb * __fdividef(1.0f, 1.0f + __expf(-(x.A * (g + bias))));
          kv[t] = t < n ? kraw : 0.f;
        }
        float cum = 0.f;
#pragma unroll
        for (int t = 0; t < QMAX; ++t) { cum = t < n ? cum + sg[t] : cum; logs[t] = cum; }
        // key norms: transpose-reduce 8 token values over 32 lanes; lane ends with token 4*b4 + 2*b3 + b2
        float v4[4], v2[2], v1;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float lo = kv[i] * kv[i], hi = kv[i + 4] * kv[i + 4];
          const float send = b4 ? lo : hi, keep = b4 ? hi : lo;
          v4[i] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const float send = b3 ? v4[i] : v4[i + 2], keep = b3 ? v4[i + 2] : v4[i];
          v2[i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
        }
        {
          const float send = b2 ? v2[0] : v2[1], keep = b2 ? v2[1] : v2[0];
          v1 = keep + __shfl_xor_sync(0xffffffffu, send, 4);
        }
        v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
        v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
        if ((lane & 3) == 0) sN[cw][4 * b4 + 2 * b3 + b2] = v1;
        asm volatile("bar.sync 1, 128;" ::: "memory");
        float kd[QMAX];
#pragma unroll
        for (int t = 0; t < QMAX; ++t) {
          const float tot = sN[0][t] + sN[1][t] + sN[2][t] + sN[3][t];
          kd[t] = kv[t] * rsqrtf(tot + 1e-6f) * expf(cum - logs[t]);
        }
        reinterpret_cast<float*>(p + OFF_E)[c] = expf(cum);
#pragma unroll
        for (int tq = 0; tq < 4; ++tq) {
          const int ln = nB * 4 + tq;
          uint32_t h0, l0, h1, l1;
          split32(kd[tq], h0, l0); split32(kd[tq + 4], h1, l1);
          *reinterpret_cast<uint4*>(p + OFF_B + (jt * 32 + ln) * 16) = make_uint4(h0, h1, l0, l1);
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");  // sN reused by the next item
      }
      mb_arrive(&coefb[s]);
    }
  } else {
    // ---------------------------------------------------------------- fold warps: 16 rows x 128 cols each, 3xTF32
    const int ft = tid - NPROD - NCOEF, fw = warp - 5, g = lane >> 2, tq = lane & 3;
    const int r0 = 16 * fw + g, r1 = r0 + 8;
    for (int k = 0;; ++k) {
      const int s = k % S;
      const uint32_t par = (k / S) & 1;
      mb_wait(&kgfull[s], par);
      const Hdr x = sH[s];
      if (x.it < 0) break;
      mb_wait(&stfull[s], par);
      mb_wait(&coefb[s], par);
      if (x.live) {
        const unsigned char* p = stage(s);
        uint4 w0[4], w1[4];
        {
          const int od = g & 1;
          uint4 t0[4], t1[4];
#pragma unroll
          for (int h = 0; h < 4; ++h) {
            t0[h] = *reinterpret_cast<const uint4*>(p + r0 * PITCH + (tq + 4 * (h ^ od)) * 16);
            t1[h] = *reinterpret_cast<const uint4*>(p + r1 * PITCH + (tq + 4 * (h ^ od)) * 16);
          }
#pragma unroll
          for (int h = 0; h < 4; ++h) { w0[h] = od ? t0[h ^ 1] : t0[h]; w1[h] = od ? t1[h ^ 1] : t1[h]; }
        }
        if (RUNG >= 2) {
          const float* sE = reinterpret_cast<const float*>(p + OFF_E);
          const float* sU = reinterpret_cast<const float*>(p + OFF_CO);
          const int n = x.n;
          uint32_t Th[4], Tl[4];
          split32(tq < n ? sU[tq * KD + r0] : 0.f, Th[0], Tl[0]);
          split32(tq < n ? sU[tq * KD + r1] : 0.f, Th[1], Tl[1]);
          split32(tq + 4 < n ? sU[(tq + 4) * KD + r0] : 0.f, Th[2], Tl[2]);
          split32(tq + 4 < n ? sU[(tq + 4) * KD + r1] : 0.f, Th[3], Tl[3]);
          float acc[16][4];
#pragma unroll
          for (int h = 0; h < 4; ++h) {
            const float4 ea = *reinterpret_cast<const float4*>(&sE[8 * (tq + 4 * h)]);
            const float4 eb = *reinterpret_cast<const float4*>(&sE[8 * (tq + 4 * h) + 4]);
            const float2 E[4] = {make_float2(ea.x, ea.y), make_float2(ea.z, ea.w), make_float2(eb.x, eb.y), make_float2(eb.z, eb.w)};
            const __nv_bfloat162* q0 = reinterpret_cast<const __nv_bfloat162*>(&w0[h]);
            const __nv_bfloat162* q1 = reinterpret_cast<const __nv_bfloat162*>(&w1[h]);
#pragma unroll
            for (int wq = 0; wq < 4; ++wq) {
              const float2 v0 = __fmul2_rn(__bfloat1622float2(q0[wq]), E[wq]);
              const float2 v1 = __fmul2_rn(__bfloat1622float2(q1[wq]), E[wq]);
              acc[4 * h + wq][0] = v0.x; acc[4 * h + wq][1] = v0.y; acc[4 * h + wq][2] = v1.x; acc[4 * h + wq][3] = v1.y;
            }
          }
          const uint4* sB = reinterpret_cast<const uint4*>(p + OFF_B);
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            const uint4 bb = sB[j * 32 + lane];
            mma_tf32(acc[j], Tl[0], Tl[1], Tl[2], Tl[3], bb.x, bb.y);
            mma_tf32(acc[j], Th[0], Th[1], Th[2], Th[3], bb.z, bb.w);
            mma_tf32(acc[j], Th[0], Th[1], Th[2], Th[3], bb.x, bb.y);
          }
#pragma unroll
          for (int h = 0; h < 4; ++h) {
            uint32_t* o0 = reinterpret_cast<uint32_t*>(&w0[h]);
            uint32_t* o1 = reinterpret_cast<uint32_t*>(&w1[h]);
#pragma unroll
            for (int wq = 0; wq < 4; ++wq) {
              o0[wq] = pk(acc[4 * h + wq][0], acc[4 * h + wq][1]);
              o1[wq] = pk(acc[4 * h + wq][2], acc[4 * h + wq][3]);
            }
          }
        }
        if (RUNG >= 3) {
          const long long bs = __ldg(a.cv_bstride + x.l), ts = __ldg(a.cv_tstride + x.l);
          char* cd = reinterpret_cast<char*>(__ldg(a.cv_addr + x.l)) + (x.d * bs + (long long)x.head * per) * 2;
          const int cpr = per * 2 / 16;
          for (int h = 0; h < hist; ++h)
            for (int u = ft; u < cpr; u += NFOLD)
              *reinterpret_cast<uint4*>(cd + (long long)h * ts * 2 + u * 16) = *reinterpret_cast<const uint4*>(p + OFF_CV + h * CVROW + u * 16);
        }
        mb_arrive(&empty[s]);
        char* dstp = reinterpret_cast<char*>(__ldg(a.st_addr + x.l)) + (x.d * __ldg(a.st_stride + x.l) + (long long)x.head * KD * KD) * 2;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          *reinterpret_cast<uint4*>(dstp + r0 * KD * 2 + (tq + 4 * h) * 16) = w0[h];
          *reinterpret_cast<uint4*>(dstp + r1 * KD * 2 + (tq + 4 * h) * 16) = w1[h];
        }
      } else {
        mb_arrive(&empty[s]);
      }
    }
  }
}

void commit_fold(int64_t rung, int64_t var, torch::Tensor st_addr, torch::Tensor st_stride, torch::Tensor co_addr,
                 torch::Tensor co_stride, torch::Tensor kg_addr, torch::Tensor kg_stride, torch::Tensor a_log,
                 torch::Tensor dtb, torch::Tensor src, torch::Tensor clen, torch::Tensor dst, torch::Tensor cv_addr,
                 torch::Tensor cv_bstride, torch::Tensor cv_dstride, torch::Tensor cv_tstride, int64_t H, int64_t Q,
                 int64_t conv_dim, int64_t hist, double lb, int64_t null_slot, int64_t batch) {
  TORCH_CHECK(Q >= 1 && Q <= QMAX && hist <= 8);
  TORCH_CHECK(conv_dim % H == 0 && (conv_dim / H) * 2 % 16 == 0 && (conv_dim / H) * 2 <= CVROW, "conv slice per head");
  CA a;
  a.st_addr = st_addr.data_ptr<int64_t>(); a.st_stride = st_stride.data_ptr<int64_t>();
  a.co_addr = co_addr.data_ptr<int64_t>(); a.co_stride = co_stride.data_ptr<int64_t>();
  a.kg_addr = kg_addr.data_ptr<int64_t>(); a.kg_stride = kg_stride.data_ptr<int64_t>();
  a.a_log = a_log.data_ptr<float>(); a.dtb = dtb.data_ptr<float>();
  a.src = src.data_ptr<int>(); a.clen = clen.data_ptr<int>(); a.dst = dst.data_ptr<int>();
  a.cv_addr = cv_addr.data_ptr<int64_t>(); a.cv_bstride = cv_bstride.data_ptr<int64_t>();
  a.cv_dstride = cv_dstride.data_ptr<int64_t>(); a.cv_tstride = cv_tstride.data_ptr<int64_t>();
  a.H = H; a.Q = Q; a.conv_dim = conv_dim; a.hist = hist; a.lb = (float)lb; a.null_slot = null_slot;
  a.B = batch; a.L = st_addr.size(0);
  const int items = H * batch * a.L;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  auto s = c10::cuda::getCurrentCUDAStream();
  // var = ring stages (2 or 3)
#define K7(RG, S_) if (rung == RG && var == S_) { \
    const int smem = S_ * STAGE_B; \
    static bool init = false; \
    if (!init) { cudaFuncSetAttribute(k_commit7<RG, S_>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; } \
    k_commit7<RG, S_><<<std::min(items, sms), NTHREADS, smem, s>>>(a, items); return; }
  K7(0, 3) K7(1, 3) K7(2, 3) K7(3, 3) K7(0, 2) K7(1, 2) K7(2, 2) K7(3, 2)
  TORCH_CHECK(false, "commit_fold7 variant ", rung, " ", var);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("commit_fold", &commit_fold); }
