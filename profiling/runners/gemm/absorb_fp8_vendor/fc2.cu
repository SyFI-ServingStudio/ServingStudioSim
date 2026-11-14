// VENDORED (temporary, delete once the kernel ships as a callable package): wt-glm53-decode-investigate/tmp/dsa-fusion/front/fc2.cu
// MoE-session lane F cut C kernel "fc2" (jobs 22107 / 22122 / 22147, xchat from_moe22.md); origin md5 1779132c92e3aec06ea3ed67a482f4c3,
// mtime 2026-11-13 18:10:47 PST, origin tree HEAD d3563d86 (file untracked). Verbatim below this header except three
// marked lines (// PATCH ragged M): the origin needs M % 128 == 0; here the last token tile reads TMA zero fill past M
// and skips the stores of its rows past M, so any M runs. Rows below M compute exactly as in the origin.
// fc2: cut C with one token tile per CTA, 256 CTAs, 2 CTAs per SM so one CTA's epilogue overlaps another's
// main loop. NN = n per CTA (128 or 256).
// Cut C (GLM-5.3-Flash DSA decode, B200 sm_100a): q_absorb + static fp8 quant in one launch.
//   ql[t, h, n] = fp8( bf16( sum_k q[t, h*256 + k] * W_UK_T[h, k, n] ) * (1 / scale) ),  n < 512
// W is pre-transposed offline to Wt[h, n, k] (K-major), a weight-load-time layout change.
// One CTA per unit (head h, n-half nh): MMA M = 128 tokens (MTT token tiles), N = 256 n, K = 256 (4 k-blocks).
//   warp 0: TMA (Wt box 64 k x 256 n, q box 64 k x 128 tokens per token tile), warp 1: TMEM + MMA,
//   warps 2-5: epilogue (TMEM lane = token, 16 columns per tcgen05.ld -> bf16 round -> * inv_scale ->
//   clamp +-448 -> e4m3 -> 16 B store).
// RUNG: 0 = W only, 1 = + q, 2 = + MMA, 3 = + epilogue.
#include "common.cuh"


struct PrmC {
  int mtiles;
  int M;  // PATCH ragged M
  CUtensorMap tmW, tmQ, tmO;  // tmO: out viewed as uint8 [M, 64*512], box 128 B x 128 tokens, SWIZZLE_128B
  uint8_t* out;        // [M, 64, 512] fp8
  const float* scale;  // [1]
  int div;             // 1: x / scale, 0: x * (1 / scale)
};

__device__ __forceinline__ uint32_t pack_fp8x4(float a, float b, float c, float d) {
  __nv_fp8x4_e4m3 v(make_float4(a, b, c, d));
  return *reinterpret_cast<uint32_t*>(&v);
}

template <int NN, int ST, int RUNG>
__global__ void __launch_bounds__(192, 2) k_c(const __grid_constant__ PrmC p) {
  constexpr int MTT = 1;
  constexpr uint32_t WB = NN * BK * 2, QB = 128 * BK * 2, SB = RUNG >= 1 ? WB + MTT * QB : WB;
  constexpr uint32_t IDESC = (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(NN >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
  constexpr int KB = 256 / BK;
  constexpr uint32_t TCOLS = MTT * NN;
  extern __shared__ __align__(1024) uint8_t dsm[];
  uint8_t* base = (uint8_t*)(((uintptr_t)dsm + 1023) & ~(uintptr_t)1023);
  uint64_t* full = (uint64_t*)(base + ST * SB);
  uint64_t* empty = full + ST;
  uint64_t* tfull = empty + ST;
  uint32_t* tslot = (uint32_t*)(tfull + 1);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // unit = (head h, n-chunk nh of NN, token tile mt); consecutive CTAs share the head (W in L2)
  const int nper = 512 / NN, mtiles = p.mtiles;
  const int h = blockIdx.x / (nper * mtiles), nh = (blockIdx.x / mtiles) % nper, mt0 = blockIdx.x % mtiles;
  if (threadIdx.x == 0) {
    for (int s = 0; s < ST; ++s) { mb_init(full + s, 1); mb_init(empty + s, 1); }
    mb_init(tfull, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  if (RUNG >= 2 && warp == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(sa(tslot)), "r"(TCOLS));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tc_before();
  __syncthreads();
  tc_after();
  const uint32_t tbase = RUNG >= 2 ? *tslot : 0;

  if (warp == 0) {
    if (lane == 0) {
      int s = 0; uint32_t ph = 0;
      for (int kb = 0; kb < KB; ++kb) {
        mb_wait(empty + s, ph ^ 1);
        mb_expect(full + s, SB);
        uint32_t dst = sa(base + s * SB);
        tma2d(dst, &p.tmW, kb * BK, h * 512 + nh * NN, full + s);
        if (RUNG >= 1)
          for (int m = 0; m < MTT; ++m) tma2d(dst + WB + m * QB, &p.tmQ, h * 256 + kb * BK, (mt0 + m) * 128, full + s);
        if (++s == ST) { s = 0; ph ^= 1; }
      }
    }
    __syncwarp();
  } else if (warp == 1) {
    if (lane == 0) {
      int s = 0; uint32_t ph = 0;
      for (int kb = 0; kb < KB; ++kb) {
        mb_wait(full + s, ph);
        tc_after();
        if (RUNG >= 2) {
          uint32_t a = sa(base + s * SB);
          uint64_t bd = sdesc(a);
#pragma unroll
          for (int m = 0; m < MTT; ++m) {
            uint64_t ad = sdesc(a + WB + m * QB);
#pragma unroll
            for (int j = 0; j < BK / 16; ++j) mma(tbase + m * NN, ad + 2 * j, bd + 2 * j, IDESC, (kb > 0 || j > 0) ? 1u : 0u);
          }
          commit(empty + s);
        } else {
          mb_arrive(empty + s);
        }
        if (++s == ST) { s = 0; ph ^= 1; }
      }
      if (RUNG >= 2) commit(tfull); else mb_arrive(tfull);
    }
    __syncwarp();
  } else {
    mb_wait(tfull, 0);
    tc_after();
    if (RUNG == 3) {
      const int q = warp & 3;
      const float sc = *p.scale, inv = 1.0f / sc;
#pragma unroll 1
      for (int m = 0; m < MTT; ++m) {
        const int tok = (mt0 + m) * 128 + q * 32 + lane;
        uint8_t* orow = p.out + ((size_t)tok * 64 + h) * 512 + nh * NN;
#pragma unroll 1
        for (int c0 = 0; c0 < NN / 16; c0 += 4) {
         uint32_t vv[4][16];
         tld4(tbase + ((uint32_t)(q * 32) << 16) + m * NN + c0 * 16, vv);
#pragma unroll
         for (int k = 0; k < 4; ++k) {
          const int c = c0 + k;
          const uint32_t* v = vv[k];
          float x[16];
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            float b = bfr(__uint_as_float(v[i]));
            b = p.div ? __fdiv_rn(b, sc) : b * inv;
            x[i] = fminf(fmaxf(b, -448.f), 448.f);
          }
          uint4 u;
          u.x = pack_fp8x4(x[0], x[1], x[2], x[3]);
          u.y = pack_fp8x4(x[4], x[5], x[6], x[7]);
          u.z = pack_fp8x4(x[8], x[9], x[10], x[11]);
          u.w = pack_fp8x4(x[12], x[13], x[14], x[15]);
          if (tok < p.M) *reinterpret_cast<uint4*>(orow + c * 16) = u;  // PATCH ragged M
         }
        }
      }
    }
    if (RUNG == 4) {  // smem staging (128B-swizzled) + TMA tensor stores; the ring is idle after tfull
      const int q = warp & 3, r = q * 32 + lane;
      const float sc = *p.scale, inv = 1.0f / sc;
#pragma unroll 1
      for (int m = 0; m < MTT; ++m) {
        uint8_t* stg = base + m * (NN * 128);   // [2 halves][128 rows][128 B]
#pragma unroll 1
        for (int c = 0; c < NN / 16; ++c) {
          uint32_t v[16];
          tld16(tbase + ((uint32_t)(q * 32) << 16) + m * NN + c * 16, v);
          float x[16];
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            float b = bfr(__uint_as_float(v[i]));
            b = p.div ? __fdiv_rn(b, sc) : b * inv;
            x[i] = fminf(fmaxf(b, -448.f), 448.f);
          }
          uint4 u;
          u.x = pack_fp8x4(x[0], x[1], x[2], x[3]);
          u.y = pack_fp8x4(x[4], x[5], x[6], x[7]);
          u.z = pack_fp8x4(x[8], x[9], x[10], x[11]);
          u.w = pack_fp8x4(x[12], x[13], x[14], x[15]);
          const int half = c >> 3, j = c & 7;  // NN/128 halves
          *reinterpret_cast<uint4*>(stg + half * (128 * 128) + r * 128 + ((j ^ (r & 7)) << 4)) = u;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (threadIdx.x == 64) {
          for (int hf = 0; hf < NN / 128; ++hf)
            asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                         ::"l"((uint64_t)&p.tmO), "r"(h * 512 + nh * NN + hf * 128), "r"((mt0 + m) * 128),
                           "r"(sa(stg + hf * 128 * 128)) : "memory");
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
      }
      if (threadIdx.x == 64) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
  }
  tc_before();
  __syncthreads();
  if (RUNG >= 2 && warp == 1) {
    tc_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tbase), "r"(TCOLS));
  }
}

template <int NN, int ST, int RUNG>
static void go(const PrmC& p) {
  constexpr int SB = RUNG >= 1 ? (NN + 128) * BK * 2 : NN * BK * 2;
  int smem = 1024 + ST * SB + 2 * ST * 8 + 8 + 16;
  auto k = k_c<NN, ST, RUNG>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  k<<<64 * (512 / NN) * p.mtiles, 192, smem, c10::cuda::getCurrentCUDAStream()>>>(p);
}

// q [M, 64*256] bf16 (row stride may be larger), wt [64*512, 256] bf16, out [M, 64, 512] fp8 (uint8 view)
void run_c(torch::Tensor q, torch::Tensor wt, torch::Tensor out, torch::Tensor scale, int64_t div, int64_t rung,
           int64_t stages, int64_t nn) {
  PrmC p;
  int M = q.size(0);
  mkmap(&p.tmW, wt.data_ptr(), 256, wt.size(0), 256, BK, (int)nn);
  mkmap(&p.tmQ, q.data_ptr(), 64 * 256, M, q.stride(0), BK, 128);
  p.out = (uint8_t*)out.data_ptr();
  {
    cuuint64_t dims[2] = {64 * 512, (cuuint64_t)M};
    cuuint64_t strides[1] = {64 * 512};
    cuuint32_t box[2] = {128, 128};
    cuuint32_t es[2] = {1, 1};
    CUresult r = encodeFn()(&p.tmO, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, out.data_ptr(), dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "tmO encode failed ", (int)r);
  }
  p.scale = scale.data_ptr<float>();
  p.div = (int)div;
  p.mtiles = (M + 127) / 128; p.M = M;  // PATCH ragged M
#define C(NN_, ST_, R_) if (nn == NN_ && stages == ST_ && rung == R_) return go<NN_, ST_, R_>(p);
#define CR(NN_, ST_) C(NN_, ST_, 0) C(NN_, ST_, 1) C(NN_, ST_, 2) C(NN_, ST_, 3) C(NN_, ST_, 4)
  CR(128, 2) CR(128, 3) CR(256, 2)
  TORCH_CHECK(false, "run_c: unsupported M/stages/rung ", M, " ", stages, " ", rung);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run_c2", &run_c); }
