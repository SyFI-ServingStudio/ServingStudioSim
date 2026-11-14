// VENDORED (temporary, with fc2.cu): wt-glm53-decode-investigate/tmp/dsa-fusion/front/common.cuh, origin md5
// 557308223273ee89cb9aab346f032245, origin tree HEAD d3563d86 (file untracked). Verbatim below this header; no changes.
// helpers shared by fa.cu / fc.cu / fb.cu (copied from fa.cu)
#pragma once
#include <torch/extension.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>
#define BK 64
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(b)), "r"(c));
}
__device__ __forceinline__ void mb_expect(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sa(b)) : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t par) {
  uint32_t a = sa(b);
  asm volatile(
      "{\n.reg .pred p;\nW%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra W%=;\n}" ::"r"(a), "r"(par) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(dst), "l"((uint64_t)tm), "r"(c0), "r"(c1), "r"(sa(bar)) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(sa(bar)) : "memory");
}
__device__ __forceinline__ uint64_t sdesc(uint32_t addr) {
  uint64_t d = 0;
  d |= (uint64_t)((addr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)(1024 >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
__device__ __forceinline__ void mma(uint32_t tmem, uint64_t ad, uint64_t bd, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(tmem), "l"(ad), "l"(bd), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void commit(uint64_t* b) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(sa(b)) : "memory");
}
__device__ __forceinline__ void tc_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
// tld4: four 32x32b.x16 loads (64 columns) in flight, one wait; the empty asm ties the registers to the wait
__device__ __forceinline__ void tld16nw(uint32_t taddr, uint32_t* v) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
        "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
      : "r"(taddr));
}
__device__ __forceinline__ void tpin16(uint32_t* v) {
  asm volatile("" : "+r"(v[0]), "+r"(v[1]), "+r"(v[2]), "+r"(v[3]), "+r"(v[4]), "+r"(v[5]), "+r"(v[6]), "+r"(v[7]),
               "+r"(v[8]), "+r"(v[9]), "+r"(v[10]), "+r"(v[11]), "+r"(v[12]), "+r"(v[13]), "+r"(v[14]), "+r"(v[15]));
}
__device__ __forceinline__ void tld4(uint32_t taddr, uint32_t (*vv)[16]) {
#pragma unroll
  for (int k = 0; k < 4; ++k) tld16nw(taddr + k * 16, vv[k]);
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int k = 0; k < 4; ++k) tpin16(vv[k]);
}
__device__ __forceinline__ void tld16(uint32_t taddr, uint32_t* v) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
        "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
      : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fence_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ void st_bf4(__nv_bfloat16* dst, float a, float b, float c, float d) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(a, b), hi = __floats2bfloat162_rn(c, d);
  uint2 u;
  u.x = *reinterpret_cast<uint32_t*>(&lo);
  u.y = *reinterpret_cast<uint32_t*>(&hi);
  *reinterpret_cast<uint2*>(dst) = u;
}

static PFN_cuTensorMapEncodeTiled_v12000 encodeFn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* f = nullptr;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &f, cudaEnableDefault, &q) == cudaSuccess && f, "entry");
    fn = (PFN_cuTensorMapEncodeTiled_v12000)f;
  }
  return fn;
}
static void mkmap(CUtensorMap* m, const void* ptr, uint64_t inner, uint64_t outer, uint64_t stride_elems, uint32_t box_in,
                  uint32_t box_out) {
  cuuint64_t dims[2] = {inner, outer};
  cuuint64_t strides[1] = {stride_elems * 2};
  cuuint32_t box[2] = {box_in, box_out};
  cuuint32_t es[2] = {1, 1};
  CUresult r = encodeFn()(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, es,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
}

