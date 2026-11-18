// Vendored kernel source for profiling/runners/attention/kda_recoverssm_commit_ws.py (backend
// cuda_persistent_ws); remove it once the kernel ships in a callable package. Origin: the KDA RecoverSSM
// commit kernel study, commit/kcommit4.cu (source md5 7cb1b5dd1793, 2026-11-13). Verbatim below this header.
// Lane K-5, attempt 3: persistent, warp-specialized KDA RecoverSSM commit (same contract and rungs as kcommit.cu).
// Measured cause (ncu k_commit3, job 21847): 4x the instructions of TileLang's fold per item and barrier stalls
// dominate: the 384 fold threads wait at __syncthreads while 128 threads build the coefficients, and the same
// warps issue the next item's loads, so per item the chain is prologue -> fold -> store with one item of lookahead.
// Here the roles are split and run concurrently over an S-stage shared-memory ring:
//   warp 0       producer: per item, bulk (TMA) copies of the key/gate and correction rows (t < n) into the stage
//                (barrier kgfull), then the 32 KB state page and the conv history slice (barrier stfull);
//   warps 1..4   coefficient warps: thread c builds column c's decayed keys and total decay for the item into the
//                stage's coefficient tile (barrier coef);
//   warps 5..12  fold warps: 256 threads, 8 rows x 8 keys each, fold on packed FFMA2, store the state and the conv
//                rows straight to global, then release the stage (barrier empty).
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

constexpr int ST_B = KD * KD * 2, KG_B = QMAX * 2 * KD * 2, CO_B = QMAX * KD * 4, CV_B = 8 * CVROW, SC_B = (QMAX + 1) * KD * 4;
constexpr int STAGE_B = ST_B + KG_B + CO_B + CV_B + SC_B;

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

struct Item { int head, b, l, n; long long s, d; bool live; };
__device__ __forceinline__ Item decode(const CA& a, int it) {
  Item x;
  x.head = it % a.H; const int r = it / a.H; x.b = r % a.B; x.l = r / a.B;
  x.n = __ldg(a.clen + x.b); x.s = __ldg(a.src + x.b); x.d = __ldg(a.dst + x.b);
  x.live = x.n > 0 && x.s > a.null_slot && x.d > a.null_slot;
  return x;
}

template <int RUNG, int S>
__global__ void __launch_bounds__(NTHREADS, 1) k_commit4(CA a, int items) {
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ __align__(8) uint64_t kgfull[S], stfull[S], coefb[S], empty[S];
  __shared__ float sN[4][QMAX];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int H = a.H, per = a.conv_dim / H, hist = a.hist;
  if (tid == 0) {
    for (int s = 0; s < S; ++s) { mb_init(&kgfull[s], 1); mb_init(&stfull[s], 1); mb_init(&coefb[s], NCOEF); mb_init(&empty[s], NFOLD); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  auto stage = [&](int s) { return smem + s * STAGE_B; };

  if (warp == 0) {
    // ---------------------------------------------------------------- producer
    if (lane == 0) {
      for (int k = 0;; ++k) {
        const int it = blockIdx.x + k * gridDim.x;
        if (it >= items) break;
        const int s = k % S;
        if (k >= S) mb_wait(&empty[s], ((k / S) & 1) ^ 1);
        const Item x = decode(a, it);
        unsigned char* p = stage(s);
        if (x.live && RUNG >= 1) {
          const char* kg = reinterpret_cast<const char*>(__ldg(a.kg_addr + x.l))
                           + ((long long)x.b * __ldg(a.kg_stride + x.l) + (long long)x.head * a.Q * 2 * KD) * 2;
          const char* co = reinterpret_cast<const char*>(__ldg(a.co_addr + x.l))
                           + ((long long)x.b * __ldg(a.co_stride + x.l) + (long long)x.head * a.Q * KD) * 4;
          mb_arrive_tx(&kgfull[s], 2 * x.n * 512);
          bulk_g2s(p + ST_B, kg, x.n * 512, &kgfull[s]);
          bulk_g2s(p + ST_B + KG_B, co, x.n * 512, &kgfull[s]);
        } else {
          mb_arrive(&kgfull[s]);
        }
        if (x.live) {
          const long long sstr = __ldg(a.st_stride + x.l);
          const char* src = reinterpret_cast<const char*>(__ldg(a.st_addr + x.l)) + (x.s * sstr + (long long)x.head * KD * KD) * 2;
          const uint32_t cvb = RUNG >= 3 ? hist * per * 2 : 0;
          mb_arrive_tx(&stfull[s], ST_B + cvb);
          bulk_g2s(p, src, ST_B, &stfull[s]);
          if (RUNG >= 3) {
            const long long bs = __ldg(a.cv_bstride + x.l), ts = __ldg(a.cv_tstride + x.l);
            const char* cs = reinterpret_cast<const char*>(__ldg(a.cv_addr + x.l)) + (x.s * bs + (long long)x.head * per) * 2;
            for (int h = 0; h < hist; ++h)
              bulk_g2s(p + ST_B + KG_B + CO_B + h * CVROW, cs + (long long)(x.n - 1 + h) * ts * 2, per * 2, &stfull[s]);
          }
        } else {
          mb_arrive(&stfull[s]);
        }
      }
    }
  } else if (warp <= 4) {
    // ---------------------------------------------------------------- coefficient warps
    const int c = tid - NPROD, cw = warp - 1;
    for (int k = 0;; ++k) {
      const int it = blockIdx.x + k * gridDim.x;
      if (it >= items) break;
      const int s = k % S;
      const Item x = decode(a, it);
      mb_wait(&kgfull[s], (k / S) & 1);
      if (x.live && RUNG >= 2) {
        const __nv_bfloat16* kg = reinterpret_cast<const __nv_bfloat16*>(stage(s) + ST_B);
        float* sC = reinterpret_cast<float*>(stage(s) + ST_B + KG_B + CO_B + CV_B);
        const int n = x.n;
        const float bias = __ldg(a.dtb + ((long long)x.l * H + x.head) * KD + c);
        const float A = expf(__ldg(a.a_log + x.l * H + x.head));
        float kv[QMAX], logs[QMAX], cum = 0.f;
#pragma unroll
        for (int t = 0; t < QMAX; ++t) {
          kv[t] = 0.f;
          if (t < n) {
            kv[t] = __bfloat162float(kg[t * 2 * KD + c]);
            const float g = __bfloat162float(kg[t * 2 * KD + KD + c]);
            cum = cum + a.lb * __fdividef(1.0f, 1.0f + __expf(-(A * (g + bias))));
          }
          logs[t] = cum;
        }
#pragma unroll
        for (int t = 0; t < QMAX; ++t) {
          float q = kv[t] * kv[t];
#pragma unroll
          for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
          if (lane == 0) sN[cw][t] = q;
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
#pragma unroll
        for (int t = 0; t < QMAX; ++t) {
          const float tot = sN[0][t] + sN[1][t] + sN[2][t] + sN[3][t];
          sC[t * KD + c] = kv[t] * rsqrtf(tot + 1e-6f) * expf(cum - logs[t]);
        }
        sC[QMAX * KD + c] = expf(cum);
        asm volatile("bar.sync 1, 128;" ::: "memory");  // sN reused by the next item
      }
      mb_arrive(&coefb[s]);
    }
  } else {
    // ---------------------------------------------------------------- fold warps
    const int ft = tid - NPROD - NCOEF, cc = ft & 15, rg = ft >> 4;
    constexpr int RJ = 8, RSTEP = 16;
    for (int k = 0;; ++k) {
      const int it = blockIdx.x + k * gridDim.x;
      if (it >= items) break;
      const int s = k % S;
      const Item x = decode(a, it);
      const uint32_t par = (k / S) & 1;
      mb_wait(&stfull[s], par);
      mb_wait(&kgfull[s], par);
      mb_wait(&coefb[s], par);
      if (x.live) {
        const unsigned char* p = stage(s);
        const __nv_bfloat16* stile = reinterpret_cast<const __nv_bfloat16*>(p);
        uint4 w[RJ];
#pragma unroll
        for (int j = 0; j < RJ; ++j) w[j] = *reinterpret_cast<const uint4*>(stile + (rg + RSTEP * j) * KD + cc * 8);
        if (RUNG >= 2) {
          const float* sU = reinterpret_cast<const float*>(p + ST_B + KG_B);
          const float* sC = reinterpret_cast<const float*>(p + ST_B + KG_B + CO_B + CV_B);
          float2 acc[RJ][4];
          const float4 e0 = *reinterpret_cast<const float4*>(&sC[QMAX * KD + cc * 8]);
          const float4 e1 = *reinterpret_cast<const float4*>(&sC[QMAX * KD + cc * 8 + 4]);
          const float2 E[4] = {make_float2(e0.x, e0.y), make_float2(e0.z, e0.w), make_float2(e1.x, e1.y), make_float2(e1.z, e1.w)};
#pragma unroll
          for (int j = 0; j < RJ; ++j) {
            const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(&w[j]);
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[j][i] = __fmul2_rn(__bfloat1622float2(q[i]), E[i]);
          }
          for (int t = 0; t < x.n; ++t) {
            const float4 c0 = *reinterpret_cast<const float4*>(&sC[t * KD + cc * 8]);
            const float4 c1 = *reinterpret_cast<const float4*>(&sC[t * KD + cc * 8 + 4]);
            const float2 C[4] = {make_float2(c0.x, c0.y), make_float2(c0.z, c0.w), make_float2(c1.x, c1.y), make_float2(c1.z, c1.w)};
#pragma unroll
            for (int j = 0; j < RJ; ++j) {
              const float u = sU[t * KD + rg + RSTEP * j];
              const float2 uu = make_float2(u, u);
#pragma unroll
              for (int i = 0; i < 4; ++i) acc[j][i] = __ffma2_rn(uu, C[i], acc[j][i]);
            }
          }
#pragma unroll
          for (int j = 0; j < RJ; ++j) {
            __nv_bfloat162* q = reinterpret_cast<__nv_bfloat162*>(&w[j]);
#pragma unroll
            for (int i = 0; i < 4; ++i) q[i] = __float22bfloat162_rn(acc[j][i]);
          }
        }
        __nv_bfloat16* dstp = reinterpret_cast<__nv_bfloat16*>(__ldg(a.st_addr + x.l))
                              + x.d * __ldg(a.st_stride + x.l) + (long long)x.head * KD * KD;
        if (RUNG >= 3) {
          const long long bs = __ldg(a.cv_bstride + x.l), ts = __ldg(a.cv_tstride + x.l);
          char* cd = reinterpret_cast<char*>(__ldg(a.cv_addr + x.l)) + (x.d * bs + (long long)x.head * per) * 2;
          const int cpr = per * 2 / 16;
          for (int u = ft; u < hist * cpr; u += NFOLD) {
            const int h = u / cpr, cch = u % cpr;
            *reinterpret_cast<uint4*>(cd + (long long)h * ts * 2 + cch * 16) =
                *reinterpret_cast<const uint4*>(p + ST_B + KG_B + CO_B + h * CVROW + cch * 16);
          }
        }
        mb_arrive(&empty[s]);  // everything this item needs from the stage is in registers or stored
#pragma unroll
        for (int j = 0; j < RJ; ++j) *reinterpret_cast<uint4*>(dstp + (rg + RSTEP * j) * KD + cc * 8) = w[j];
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
  // var = number of ring stages (3 or 4)
#define K4(RG, S_) if (rung == RG && var == S_) { \
    const int smem = S_ * STAGE_B; \
    static bool init = false; \
    if (!init) { cudaFuncSetAttribute(k_commit4<RG, S_>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; } \
    k_commit4<RG, S_><<<std::min(items, sms), NTHREADS, smem, s>>>(a, items); return; }
  K4(0, 3) K4(1, 3) K4(2, 3) K4(3, 3) K4(0, 4) K4(1, 4) K4(2, 4) K4(3, 4)
  TORCH_CHECK(false, "commit_fold4 variant ", rung, " ", var);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("commit_fold", &commit_fold); }
