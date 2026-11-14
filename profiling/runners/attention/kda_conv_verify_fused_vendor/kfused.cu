// VENDORED (temporary, delete once the kernel ships as a callable package): wt-glm53-decode-investigate/tmp/kda-floor/fused/kfused.cu
// MoE-session lane V fused conv + RecoverSSM verify, window a compile-time parameter (xchat from_moe17.md, job 22037): window 8 = default build,
// rung 2 / minb 4 / ABL 6064 (job 22006); window 4 = -DSQ=4, rung 2 / minb 4 / ABL 22448. Source md5 245b0847323b2e144b8fb843607970d3,
// mtime 2026-11-13 16:35:56 PST, source tree HEAD d3563d86 (file untracked). Verbatim below this header; no changes.
// Window (tokens per request) is a compile-time parameter: -DSQ=8 (default, byte-identical SASS to the SQ-8 file,
// md5 33ad67b49997) or -DSQ=4 (HiSparse, conv state 6 rows). The window-parameterized copy loops follow the
// decode session's kfused_window.cu (reviewed: correct). SQ 4 adds its own geometry under ABL 16384 (see there):
// tokens as the mma N dimension instead of padded M rows. Without 16384 an SQ-4 build is the padded
// re-parameterization (token slots 4..7 of the m16 / n8 tiles zero), kept as the comparison.
// Lane V: fused KDA short conv + RecoverSSM verify for GLM-5.3-Flash decode (B200), grown from its floor.
//
// Geometry: one CTA (128 threads, 4 warps) per (request b, head h). Thread j owns key / value index j.
//   - state [V=128][K=128] bf16 of slot[b], head h: loaded straight into registers in the B-operand
//     fragment order of mma.m16n8k16 (warp w owns V rows 32w..32w+31; per 8-row n-tile and 32-K chunk one
//     16-byte load per lane). No smem copy of the state.
//   - raw x rows of the head's q/k/v channels (8 tokens), the 3 history rows at acc-1 of the SD conv page,
//     and the raw gate rows: cp.async 16 B into smem (one round trip, issued with the state loads).
//   - taps (fp32 [C,4]): one 16-byte load per channel into registers.
// Rungs (compile-time):
//   0: byte pattern only: every load above, then every store: rolled conv rows (copied from smem), the
//      key/gate record (raw x k-section as stand-in key, real raw gate), correction and out from a fold of
//      the loaded words. No math.
//   1: + conv math (fp32 taps, SiLU, bf16 rounding) -> real conv rows, real key half of the record.
//   2: + norms, gates, WY form of the 8-step recurrence (tensor-core mma, bf16 hi/lo split of W), forward
//      substitution -> real correction and out.
//   3: (rung 2 already writes every output; rung 3 = rung 2 with stores vectorized)
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>
#include <algorithm>

#define KD 128
#ifndef SQ
#define SQ 8  // window (tokens per request); build with -DSQ=4 or -DSQ=8
#endif
#define NSLOT 8  // token slots of the tensor-core tiles (m16 = 8 k rows + 8 q rows, n8 = 8 columns); tokens >= SQ are zero
static_assert(SQ == 4 || SQ == 8, "window 4 or 8");
#define CWID 4
#define NTH 128

struct Args {
  const char* pool;      // KV pool base (bytes); block s at pool + s*page
  long long page;        // bytes per block
  long long conv_bytes;  // conv part of a block: L*C*2 (SD [L][C] bf16), state follows
  const int* slots;      // [B]
  const int* acc;        // [B] accepted tokens 1..8
  const __nv_bfloat16* x;  // raw qkv rows [T, ldx]
  long long ldx;           // elements
  const __nv_bfloat16* gate;  // raw gate [T, ldg]
  long long ldg;
  const __nv_bfloat16* beta;  // raw beta [T, ldb]
  long long ldb;
  const float* a_log;  // [H]
  const float* dtb;    // [H*K]
  const float* w;      // conv taps [C, 4] fp32
  __nv_bfloat16* out;  // [T, H*K]
  float* corr;         // [rows, H, S, V]
  __nv_bfloat16* kg;   // [rows, H, S, 2K]
  float lb;
  unsigned long long* stamps;  // [grid][5]: t_start, t_loaded, t_math_done, t_end, smid (STAMP builds)
  int B, H, L;
};

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ float4 ldg_f4(const void* p) {
  float4 r;
  asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  unsigned s = (unsigned)__cvta_generic_to_shared(sdst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gsrc));
}
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// smem image per CTA
struct Smem {
  __nv_bfloat16 xs[2][3 + SQ][3 * KD];  // window rows: 3 history + 8 tokens, q|k|v channels of head h (x2 buffers)
  __nv_bfloat16 gs[2][SQ][KD];          // raw gate rows (x2 buffers)
  float bet[SQ];
  float sbet[SQ];                       // sigmoid(beta) (ABL 256)
  __nv_bfloat16 ks[SQ][KD];          // conv output, key section (record)
  float red[4][16];                  // per-warp partial sums of |k_t|^2, |q_t|^2
  uint4 wa[2][8][32];                // W = [D_t*kn_t ; D_t*qn_t] (16 x K) as mma A fragments, [hi/lo][kstep][lane]
  uint2 ab[2][8][32];                // a_s = kn_s / D_s (K x 8) as mma B fragments, [hi/lo][kstep][lane]
  float pq[16][KD];                  // P_t = h0 (D_t kn_t), Q_t = h0 (D_t qn_t)
  __align__(16) float mm[16][8];                   // Mk[t][s] (rows 0-7), Mq[t][s] (rows 8-15)
};

__device__ __forceinline__ void mma16816(float* c, const uint4& a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float ex2f(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rcpf(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
#define L2E 1.4426950408889634f
__device__ __forceinline__ float sigm_ftz(float x) { return rcpf(1.f + ex2f(-x * L2E)); }
__device__ __forceinline__ void split(float v, __nv_bfloat16& hi, __nv_bfloat16& lo) {
  hi = __float2bfloat16(v);
  lo = __float2bfloat16(v - __bfloat162float(hi));
}
// ABL 2048: packed f32x2 helpers (Blackwell FFMA2/FMUL2/FADD2; each lane rounds exactly like the scalar op)
typedef unsigned long long f2_t;
__device__ __forceinline__ f2_t pk2(float a, float b) { f2_t r; asm("mov.b64 %0, {%1,%2};" : "=l"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ void upk2(f2_t r, float& a, float& b) { asm("mov.b64 {%0,%1}, %2;" : "=f"(a), "=f"(b) : "l"(r)); }
__device__ __forceinline__ f2_t fma2(f2_t a, f2_t b, f2_t c) { f2_t d; asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d; }
__device__ __forceinline__ f2_t mul2(f2_t a, f2_t b) { f2_t d; asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
__device__ __forceinline__ f2_t add2(f2_t a, f2_t b) { f2_t d; asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
// ABL 4096: hi by truncation (no F2F), exact fp32 residual rounded once to bf16
__device__ __forceinline__ void split_t(float v, __nv_bfloat16& hi, __nv_bfloat16& lo) {
  const unsigned u = __float_as_uint(v);
  hi = __ushort_as_bfloat16((unsigned short)(u >> 16));
  lo = __float2bfloat16(v - __uint_as_float(u & 0xffff0000u));
}

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned smid() {
  unsigned r;
  asm volatile("mov.u32 %0, %smid;" : "=r"(r));
  return r;
}
// ABL (ablation / fix bits, compile time): 1 no big mma (state folded instead), 2 no small mma, 4 no forward
// substitution (c = beta (v - P)), 8 no norms/gates (kn = k, D = 1), 16 fast SiLU / sigmoid (__fdividef)
//   32 persistent grid (SMs x occupancy, static item stride) + L2 prefetch of the CTA's next item
//   64 persistent grid without the prefetch
// cp.async the window rows (3 history + 8 x rows, q|k|v of head h) and the 8 raw gate rows into buffer buf
template <bool FC>
__device__ __forceinline__ void issue_rows(const Args& a, Smem& sm, const int buf, const int b, const int h,
                                           const __nv_bfloat16* conv, const int ac) {
  const int tid = threadIdx.x, HK = a.H * KD, C = 3 * HK;
  const long long tok0 = (long long)b * SQ;
  if constexpr (FC) {  // fixed (section, chunk) per thread, rows by stride: no divisions in the loop
    if (tid < 96) {
      const int sc = tid % 48, sec = sc >> 4, ch = sc & 15, r0 = tid / 48;
      const int off = sec * HK + h * KD + ch * 8;
      __nv_bfloat16* dst = &sm.xs[buf][0][sec * KD + ch * 8];
      const __nv_bfloat16* hsrc = conv + (long long)(ac - 1) * C + off;
      const __nv_bfloat16* xsrc = a.x + tok0 * a.ldx + off;
#pragma unroll
      for (int k = 0; k < (3 + SQ + 1) / 2; k++) {
        const int r = r0 + 2 * k;
        if (r < 3) cp16(dst + r * (3 * KD), hsrc + (long long)r * C);
        else if (r < 3 + SQ) cp16(dst + r * (3 * KD), xsrc + (long long)(r - 3) * a.ldx);
      }
    } else {
      const int u = tid - 96, ch = u & 15, t0 = u >> 4;
      const __nv_bfloat16* gsrc = a.gate + tok0 * a.ldg + h * KD + ch * 8;
#pragma unroll
      for (int k = 0; k < SQ / 2; k++) cp16(&sm.gs[buf][t0 + 2 * k][ch * 8], gsrc + (long long)(t0 + 2 * k) * a.ldg);
    }
    asm volatile("cp.async.commit_group;");
    return;
  }
  for (int c = tid; c < 144 + 48 * SQ + 16 * SQ; c += NTH) {
    if (c < 144) {
      int r = c / 48, rem = c % 48, sec = rem >> 4, ch = rem & 15;
      cp16(&sm.xs[buf][r][sec * KD + ch * 8], conv + (long long)(ac - 1 + r) * C + sec * HK + h * KD + ch * 8);
    } else if (c < 144 + 48 * SQ) {
      int c2 = c - 144, t = c2 / 48, rem = c2 % 48, sec = rem >> 4, ch = rem & 15;
      cp16(&sm.xs[buf][3 + t][sec * KD + ch * 8], a.x + (tok0 + t) * a.ldx + sec * HK + h * KD + ch * 8);
    } else {
      int c3 = c - (144 + 48 * SQ), t = c3 >> 4, ch = c3 & 15;
      cp16(&sm.gs[buf][t][ch * 8], a.gate + (tok0 + t) * a.ldg + h * KD + ch * 8);
    }
  }
  asm volatile("cp.async.commit_group;");
}

//  128 (with 32): the next item's rows go by cp.async into the other smem buffer and its state by L2 prefetch,
//      both issued after this item's own data landed (no competition with the first load)
//  256 math trims: reduce-scatter norm sums (16 shuffles), sigmoid(beta) once per item, float4 Mk/Mq reads
template <int RUNG, int ABL, int STAMP>
__device__ __forceinline__ bool item_body(const Args& a, Smem& sm, const int item, const int nxt, const int buf,
                                          const bool pre) {
  bool issued = false;  // this item issued the next item's rows (ABL 128)
  unsigned long long st0 = 0, st1 = 0, st2 = 0;
  if constexpr (STAMP) st0 = gtime();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gid = lane >> 2, tq = lane & 3;
  const int b = item / a.H, h = item % a.H;
  const int HK = a.H * KD, C = 3 * HK;
  const int slot = a.slots[b];
  const int ac = a.acc[b];
  if (slot <= 0) {  // NULL block: zero output, nothing else (fork Triton verify)
#pragma unroll
    for (int t = 0; t < SQ; t++) a.out[((long long)b * SQ + t) * HK + h * KD + tid] = __float2bfloat16(0.f);
    return false;
  }
  const char* blk = a.pool + (long long)slot * a.page;
  const __nv_bfloat16* conv = (const __nv_bfloat16*)blk;  // [L][C]
  const char* st = blk + a.conv_bytes + (long long)h * (KD * KD * 2);

  // ---- loads: state into registers (B-fragment order) ----
  uint4 sv[4][4];  // [n-tile][32-K chunk]
#pragma unroll
  for (int nt = 0; nt < 4; nt++)
#pragma unroll
    for (int q = 0; q < 4; q++)
      sv[nt][q] = ldg_stream(st + (warp * 32 + nt * 8 + gid) * 256 + q * 64 + tq * 16);
  // ---- window rows + gate rows into smem ----
  const long long tok0 = (long long)b * SQ;
  if (!((ABL & 128) && pre)) issue_rows<(ABL & 512) != 0>(a, sm, buf, b, h, conv, ac);
  if constexpr ((ABL & 32) != 0 && (ABL & 128) == 0) {
    if (nxt >= 0) {  // pull the next item's bytes into L2 while this item computes
      const int b2 = nxt / a.H, h2 = nxt % a.H;
      const int slot2 = a.slots[b2];
      if (slot2 > 0) {
        const char* blk2 = a.pool + (long long)slot2 * a.page;
        if (tid == NTH - 1) {
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], 32768;" ::"l"(blk2 + a.conv_bytes + (long long)h2 * (KD * KD * 2))
                       : "memory");
        } else if (tid < 18 + 6 * SQ + 2 * SQ) {
          const char* p;
          if (tid < 18) {
            const int r = tid / 6, rem = tid % 6, sec = rem >> 1, half = rem & 1;
            const int ac2 = a.acc[b2];
            p = (const char*)((const __nv_bfloat16*)blk2 + (long long)(ac2 - 1 + r) * C + sec * HK + h2 * KD) + half * 128;
          } else if (tid < 18 + 6 * SQ) {
            const int c2 = tid - 18, t = c2 / 6, rem = c2 % 6, sec = rem >> 1, half = rem & 1;
            p = (const char*)(a.x + ((long long)b2 * SQ + t) * a.ldx + sec * HK + h2 * KD) + half * 128;
          } else {
            const int c3 = tid - (18 + 6 * SQ), t = c3 >> 1, half = c3 & 1;
            p = (const char*)(a.gate + ((long long)b2 * SQ + t) * a.ldg + h2 * KD) + half * 128;
          }
          asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
        }
      }
    }
  }
  float4 tp[3];
#pragma unroll
  for (int sec = 0; sec < 3; sec++) tp[sec] = ldg_f4(a.w + (long long)(sec * HK + h * KD + tid) * 4);
  const float dtb = a.dtb[h * KD + tid];
  const float alog = a.a_log[h];
  if (tid < SQ) {
    const float rb = __bfloat162float(a.beta[(tok0 + tid) * a.ldb + h]);
    sm.bet[tid] = rb;
    if constexpr ((ABL & 256) != 0) sm.sbet[tid] = sigm_ftz(rb);
  }
  cp_wait_all();
  __syncthreads();
  if constexpr (STAMP) st1 = gtime();
  if constexpr ((ABL & 128) != 0) {  // next item: rows -> other smem buffer, state -> L2
    if (nxt >= 0) {
      const int b2 = nxt / a.H, h2 = nxt % a.H;
      const int slot2 = a.slots[b2];
      if (slot2 > 0) {
        const char* blk2 = a.pool + (long long)slot2 * a.page;
        issue_rows<(ABL & 512) != 0>(a, sm, buf ^ 1, b2, h2, (const __nv_bfloat16*)blk2, a.acc[b2]);
        issued = true;
        if (tid == NTH - 1)
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], 32768;" ::"l"(blk2 + a.conv_bytes + (long long)h2 * (KD * KD * 2))
                       : "memory");
      }
    }
  }

  __nv_bfloat16* convw = (__nv_bfloat16*)blk;  // rolled rows written in place
  if constexpr (RUNG == 0) {
    // rolled conv rows = window rows 1..10
    for (int c = tid; c < 48 * (SQ + 2); c += NTH) {
      int r = c / 48, rem = c % 48, sec = rem >> 4, ch = rem & 15;
      *(uint4*)(convw + (long long)r * C + sec * HK + h * KD + ch * 8) = *(const uint4*)&sm.xs[buf][r + 1][sec * KD + ch * 8];
    }
    // key/gate record: [rows][H][S][2K]
    __nv_bfloat16* kgp = a.kg + ((long long)b * a.H + h) * SQ * 2 * KD;
    for (int c = tid; c < 32 * SQ; c += NTH) {
      int t = c >> 5, part = c & 31;
      uint4 v = part < 16 ? *(const uint4*)&sm.xs[buf][3 + t][KD + part * 8] : *(const uint4*)&sm.gs[buf][t][(part - 16) * 8];
      *(uint4*)(kgp + t * 2 * KD + part * 8) = v;
    }
    unsigned f = 0;
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
      for (int q = 0; q < 4; q++) f ^= sv[nt][q].x ^ sv[nt][q].y ^ sv[nt][q].z ^ sv[nt][q].w;
    float fv = __uint_as_float(f & 0x3f7fffffu) + tp[0].x + tp[1].y + tp[2].z + tp[0].w + dtb + alog;
    float* cp = a.corr + ((long long)b * a.H + h) * SQ * KD;
#pragma unroll
    for (int t = 0; t < SQ; t++) {
      cp[t * KD + tid] = fv + sm.bet[t];
      a.out[(tok0 + t) * HK + h * KD + tid] = __float2bfloat16(fv * sm.bet[t]);
    }
  } else {
    // rolled conv rows = window rows 1..10 (raw x / history; independent of the math)
    if constexpr ((ABL & 512) != 0) {
      if (tid < 96) {
        const int sc = tid % 48, sec = sc >> 4, ch = sc & 15, r0 = tid / 48;
        __nv_bfloat16* dst = convw + sec * HK + h * KD + ch * 8;
        const __nv_bfloat16* src = &sm.xs[buf][1][sec * KD + ch * 8];
#pragma unroll
        for (int k = 0; k < (SQ + 2) / 2; k++) {
          const int r = r0 + 2 * k;
          *(uint4*)(dst + (long long)r * C) = *(const uint4*)(src + r * (3 * KD));
        }
      }
    } else
    for (int c = tid; c < 48 * (SQ + 2); c += NTH) {
      int r = c / 48, rem = c % 48, sec = rem >> 4, ch = rem & 15;
      *(uint4*)(convw + (long long)r * C + sec * HK + h * KD + ch * 8) = *(const uint4*)&sm.xs[buf][r + 1][sec * KD + ch * 8];
    }
    // ---- conv: thread j = channel j of q, k, v ----
    float cq[SQ], ck[SQ], cv[SQ];
#pragma unroll
    for (int sec = 0; sec < 3; sec++) {
      float win[3 + SQ];
#pragma unroll
      for (int r = 0; r < 3 + SQ; r++) win[r] = __bfloat162float(sm.xs[buf][r][sec * KD + tid]);
      const float w4[4] = {tp[sec].x, tp[sec].y, tp[sec].z, tp[sec].w};
      if constexpr ((ABL & 2048) != 0) {  // token pairs (t, t+1) in f32x2 lanes; same per-lane op order as the scalar path
        const f2_t nl2 = pk2(-L2E, -L2E), one2 = pk2(1.f, 1.f);
#pragma unroll
        for (int p = 0; p < SQ / 2; p++) {
          f2_t y2 = mul2(pk2(win[2 * p], win[2 * p + 1]), pk2(w4[0], w4[0]));
#pragma unroll
          for (int i = 1; i < CWID; i++) y2 = fma2(pk2(win[2 * p + i], win[2 * p + 1 + i]), pk2(w4[i], w4[i]), y2);
          float y0, y1, e0, e1;
          upk2(mul2(y2, nl2), e0, e1);
          upk2(add2(pk2(ex2f(e0), ex2f(e1)), one2), e0, e1);
          upk2(mul2(y2, pk2(rcpf(e0), rcpf(e1))), y0, y1);
          const float yy[2] = {y0, y1};
#pragma unroll
          for (int k = 0; k < 2; k++) {
            const int t = 2 * p + k;
            const __nv_bfloat16 yb = __float2bfloat16(yy[k]);
            const float yr = __bfloat162float(yb);
            if (sec == 0) cq[t] = yr;
            if (sec == 1) { ck[t] = yr; sm.ks[t][tid] = yb; }
            if (sec == 2) cv[t] = yr;
          }
        }
      } else
#pragma unroll
      for (int t = 0; t < SQ; t++) {
        float y = 0.f;
#pragma unroll
        for (int i = 0; i < CWID; i++) y = fmaf(win[t + i], w4[i], y);
        if constexpr (ABL & 256) y = y * rcpf(1.f + ex2f(-y * L2E));
        else if constexpr (ABL & 16) y = __fdividef(y, 1.f + __expf(-y));
        else y = y / (1.f + __expf(-y));
        const __nv_bfloat16 yb = __float2bfloat16(y);
        const float yr = __bfloat162float(yb);
        if (sec == 0) cq[t] = yr;
        if (sec == 1) { ck[t] = yr; sm.ks[t][tid] = yb; }
        if (sec == 2) cv[t] = yr;
      }
    }
    if constexpr (RUNG == 1) {
      __syncthreads();
      if constexpr (STAMP) st2 = gtime();
      __nv_bfloat16* kgp = a.kg + ((long long)b * a.H + h) * SQ * 2 * KD;
      for (int c = tid; c < 32 * SQ; c += NTH) {
        int t = c >> 5, part = c & 31;
        uint4 v = part < 16 ? *(const uint4*)&sm.ks[t][part * 8] : *(const uint4*)&sm.gs[buf][t][(part - 16) * 8];
        *(uint4*)(kgp + t * 2 * KD + part * 8) = v;
      }
      unsigned f = 0;
#pragma unroll
      for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int q = 0; q < 4; q++) f ^= sv[nt][q].x ^ sv[nt][q].y ^ sv[nt][q].z ^ sv[nt][q].w;
      float fv = __uint_as_float(f & 0x3f7fffffu) + dtb + alog;
      float* cp = a.corr + ((long long)b * a.H + h) * SQ * KD;
#pragma unroll
      for (int t = 0; t < SQ; t++) {
        cp[t * KD + tid] = fv + sm.bet[t] + cq[t];
        a.out[(tok0 + t) * HK + h * KD + tid] = __float2bfloat16(fv * sm.bet[t] + cv[t]);
      }
#if SQ == 4
    } else if constexpr ((ABL & 16384) != 0) {
      // ==== window-4 geometry (ABL 16384, SQ 4 only) ====
      // 4 tokens give 8 W rows (D_t k_t, D_t q_t; unnormalized, norm factors applied to the accumulators).
      // Instead of padding the m16 tile with 8 zero rows, its two halves carry the bf16 hi and lo parts of the
      // same 8 rows: A = [W_hi ; W_lo], so one mma per (n-tile, k-step) gives h0 W_hi^T in rows 0..7 and
      // h0 W_lo^T in rows 8..15 of the same thread (c0 + c2, c1 + c3): 32 HMMA per warp instead of 64, one A
      // fragment read per k-step, the state stays the B operand in its load order (no register shuffles).
      // Small GEMM likewise: B = [a_hi | a_lo] (n = s, 4 + s), so C = W_hi a_hi | W_hi a_lo over W_lo a_hi in
      // one mma per k-step (8 HMMA instead of 24); M = C[t][s] + C[8+t][s] + C[t][4+s].
      static_assert((ABL & 1280) == 1280, "window-4 geometry assumes ABL 256 | 1024");
      // ---- norms: 8 sums (|k_t|^2 at t, |q_t|^2 at 4 + t), reduce-scatter (3 halving steps + 2 xor) ----
      float ss[8];
#pragma unroll
      for (int t = 0; t < 4; t++) { ss[t] = ck[t] * ck[t]; ss[4 + t] = cq[t] * cq[t]; }
#pragma unroll
      for (int w = 4, o = 16; w >= 1; w >>= 1, o >>= 1) {
        const bool up = (lane & o) != 0;
#pragma unroll
        for (int i = 0; i < w; i++) {
          const float send = up ? ss[i] : ss[i + w];
          const float keep = up ? ss[i + w] : ss[i];
          ss[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
      }
      ss[0] += __shfl_xor_sync(0xffffffffu, ss[0], 2);
      ss[0] += __shfl_xor_sync(0xffffffffu, ss[0], 1);
      if ((lane & 3) == 0) sm.red[warp][((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1)] = ss[0];
      // ---- gates ----
      const float A = __expf(alog);
      float G[4];
      float run = 0.f;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        const float gi = __bfloat162float(sm.gs[buf][t][tid]) + dtb;
        run += a.lb * sigm_ftz(A * gi);
        G[t] = run;
      }
      // ---- W (A fragments [hi rows 0..7 ; lo rows 8..15], sm.wa[0]) and a (B fragments [hi n 0..3 | lo n 4..7], sm.ab[0]) ----
      {
        const int fq = tid >> 5, fr = tid & 31, ftq = fr >> 3, fw = fr & 7, fsub = fw >> 2, fe = fw & 3;
        const int ks_ = 2 * fq + fsub, areg = (fe >= 2 ? 2 : 0), ahalf = fe & 1;
        __nv_bfloat16* wA = (__nv_bfloat16*)&sm.wa[0][ks_][0];
        __nv_bfloat16* aB = (__nv_bfloat16*)&sm.ab[0][ks_][0];
#pragma unroll
        for (int t = 0; t < 4; t++) {
          const float D = ex2f(G[t] * L2E), Di = rcpf(D);
          __nv_bfloat16 hi, lo;
          const int lk = t * 4 + ftq, lq = (4 + t) * 4 + ftq;  // A lane of rows t (k) and 4 + t (q)
          if constexpr (ABL & 4096) split_t(D * ck[t], hi, lo); else split(D * ck[t], hi, lo);
          wA[(lk * 4 + areg + 0) * 2 + ahalf] = hi; wA[(lk * 4 + areg + 1) * 2 + ahalf] = lo;
          if constexpr (ABL & 4096) split_t(D * cq[t], hi, lo); else split(D * cq[t], hi, lo);
          wA[(lq * 4 + areg + 0) * 2 + ahalf] = hi; wA[(lq * 4 + areg + 1) * 2 + ahalf] = lo;
          if constexpr (ABL & 4096) split_t(ck[t] * Di, hi, lo); else split(ck[t] * Di, hi, lo);
          aB[(lk * 2 + (fe >= 2)) * 2 + ahalf] = hi; aB[(lq * 2 + (fe >= 2)) * 2 + ahalf] = lo;
        }
      }
      __syncthreads();  // (3) red, ks, W^T, a complete
      // ---- key/gate record (key half from ks, gate half from gs): one 16-byte chunk per thread ----
      {
        __nv_bfloat16* kgp = a.kg + ((long long)b * a.H + h) * SQ * 2 * KD;
        const int part = tid & 31, t0 = tid >> 5;
        const __nv_bfloat16* src = part < 16 ? &sm.ks[t0][part * 8] : &sm.gs[buf][t0][(part - 16) * 8];
        *(uint4*)(kgp + t0 * 2 * KD + part * 8) = *(const uint4*)src;
      }
      // norm factors: lane i (mod 8) holds r_i = rsqrt(sum_i) (x K^-1/2 for the q columns i >= 4)
      float rnv;
      {
        const int i = lane & 7;
        rnv = rsqrtf(sm.red[0][i] + sm.red[1][i] + sm.red[2][i] + sm.red[3][i] + 1e-6f);
        if (i >= 4) rnv *= 0.08838834764831845f;
      }
      const float rg = __shfl_sync(0xffffffffu, rnv, gid);  // factor of row n = gid
      {
        float acc[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
          for (int i = 0; i < 4; i++) acc[nt][i] = 0.f;
#pragma unroll
        for (int s = 0; s < 8; s++) {
          const uint4 aw = sm.wa[0][s][lane];
          const int q = s >> 1, sub = s & 1;
#pragma unroll
          for (int nt = 0; nt < 4; nt++) {
            const unsigned b0 = sub ? sv[nt][q].z : sv[nt][q].x;
            const unsigned b1 = sub ? sv[nt][q].w : sv[nt][q].y;
            mma16816(acc[nt], aw, b0, b1);
          }
        }
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          const int v = warp * 32 + nt * 8 + 2 * tq;
          *(float2*)&sm.pq[gid][v] = make_float2((acc[nt][0] + acc[nt][2]) * rg, (acc[nt][1] + acc[nt][3]) * rg);
        }
      }
      if (warp == 0) {  // C = [W_hi ; W_lo] [a_hi | a_lo]; M[n][s] = C[n][s] + C[8+n][s] + C[n][4+s]
        float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < 8; s++) {
          const uint2 bb = sm.ab[0][s][lane];
          mma16816(c, sm.wa[0][s][lane], bb.x, bb.y);
        }
        const float x0 = __shfl_down_sync(0xffffffffu, c[0], 2), x1 = __shfl_down_sync(0xffffffffu, c[1], 2);
        const float r0 = __shfl_sync(0xffffffffu, rnv, (2 * tq) & 3), r1 = __shfl_sync(0xffffffffu, rnv, (2 * tq + 1) & 3);
        if (tq < 2)
          *(float2*)&sm.mm[gid][2 * tq] = make_float2((c[0] + c[2] + x0) * (rg * r0), (c[1] + c[3] + x1) * (rg * r1));
      }
      __syncthreads();  // (4) P, Q, M complete
      if constexpr (STAMP) st2 = gtime();
      // ---- forward substitution, thread v = tid (same per-row op order as the SQ 8 ABL-256 path) ----
      {
        float mk[4][4], mq[4][4];  // Mk[t][s] = mm[t][s], Mq[t][s] = mm[4 + t][s]
#pragma unroll
        for (int t = 0; t < 4; t++) {
          const float4 k4 = *(const float4*)&sm.mm[t][0], q4 = *(const float4*)&sm.mm[4 + t][0];
          mk[t][0] = k4.x; mk[t][1] = k4.y; mk[t][2] = k4.z; mk[t][3] = k4.w;
          mq[t][0] = q4.x; mq[t][1] = q4.y; mq[t][2] = q4.z; mq[t][3] = q4.w;
        }
        float P[4], Q[4];
#pragma unroll
        for (int t = 0; t < 4; t++) { P[t] = sm.pq[t][tid]; Q[t] = sm.pq[4 + t][tid]; }
        float cc[4];
        float* cp = a.corr + ((long long)b * a.H + h) * SQ * KD;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          float u = cv[t] - P[t];
#pragma unroll
          for (int s2 = 0; s2 < t; s2++) u = fmaf(-mk[t][s2], cc[s2], u);
          cc[t] = u * sm.sbet[t];
          float o = Q[t];
#pragma unroll
          for (int s2 = 0; s2 <= t; s2++) o = fmaf(mq[t][s2], cc[s2], o);
          cp[t * KD + tid] = cc[t];
          a.out[(tok0 + t) * HK + h * KD + tid] = __float2bfloat16(o);
        }
      }
#endif
    } else {
      // ---- norms: |k_t|^2, |q_t|^2 over the 128 channels ----
      float ss[16];
#pragma unroll
      for (int t = 0; t < SQ; t++) { ss[t] = ck[t] * ck[t]; ss[8 + t] = cq[t] * cq[t]; }
#pragma unroll
      for (int t = SQ; t < NSLOT; t++) { ss[t] = 0.f; ss[8 + t] = 0.f; }
      if constexpr ((ABL & 256) != 0) {
        // reduce-scatter: after the 4 halving steps lane holds index ((lane>>4)&1)*8+((lane>>3)&1)*4+((lane>>2)&1)*2+((lane>>1)&1)
#pragma unroll
        for (int w = 8, o = 16; w >= 1; w >>= 1, o >>= 1) {
          const bool up = (lane & o) != 0;
#pragma unroll
          for (int i = 0; i < w; i++) {
            const float send = up ? ss[i] : ss[i + w];
            const float keep = up ? ss[i + w] : ss[i];
            ss[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
          }
        }
        ss[0] += __shfl_xor_sync(0xffffffffu, ss[0], 1);
        if ((lane & 1) == 0) sm.red[warp][((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1)] = ss[0];
      } else {
#pragma unroll
        for (int i = 0; i < 16; i++)
#pragma unroll
          for (int o = 16; o; o >>= 1) ss[i] += __shfl_xor_sync(0xffffffffu, ss[i], o);
        if (lane == 0) {
#pragma unroll
          for (int i = 0; i < 16; i++) sm.red[warp][i] = ss[i];
        }
      }
      // ---- gates (per key channel j): g_t = lb * sigmoid(A (raw_g + dt_bias)), G_t = sum_{u<=t} g_u ----
      const float A = __expf(alog);
      float G[SQ];
      float run = 0.f;
#pragma unroll
      for (int t = 0; t < SQ; t++) {
        float gi = __bfloat162float(sm.gs[buf][t][tid]) + dtb;
        if constexpr (ABL & 8) run += gi;
        else if constexpr (ABL & 256) run += a.lb * sigm_ftz(A * gi);
        else if constexpr (ABL & 16) run += a.lb * __fdividef(1.f, 1.f + __expf(-A * gi));
        else run += a.lb * sigm(A * gi);
        G[t] = run;
      }
      if constexpr (!(ABL & 1024)) __syncthreads();  // (2) red, ks complete
      // key/gate record (key half from ks, gate half from gs)
#define KG_RECORD() do { \
        __nv_bfloat16* kgp = a.kg + ((long long)b * a.H + h) * SQ * 2 * KD; \
        if constexpr ((ABL & 512) != 0) { \
          const int part = tid & 31, t0 = tid >> 5; \
          const __nv_bfloat16* src = part < 16 ? &sm.ks[t0][part * 8] : &sm.gs[buf][t0][(part - 16) * 8]; \
_Pragma("unroll") \
          for (int k = 0; k < SQ / 4; k++) *(uint4*)(kgp + (t0 + 4 * k) * 2 * KD + part * 8) = *(const uint4*)(src + 4 * k * KD); \
        } else \
        for (int c = tid; c < 32 * SQ; c += NTH) { \
          int t = c >> 5, part = c & 31; \
          uint4 v = part < 16 ? *(const uint4*)&sm.ks[t][part * 8] : *(const uint4*)&sm.gs[buf][t][(part - 16) * 8]; \
          *(uint4*)(kgp + t * 2 * KD + part * 8) = v; \
        } \
      } while (0)
      if constexpr (!(ABL & 1024)) KG_RECORD();
      // fragment coordinates of key channel j (consistent K permutation, see header)
      const int fq = tid >> 5, fr = tid & 31, ftq = fr >> 3, fw = fr & 7, fsub = fw >> 2, fe = fw & 3;
      const int ks_ = 2 * fq + fsub;
      __nv_bfloat16* waH = (__nv_bfloat16*)&sm.wa[0][ks_][0];
      __nv_bfloat16* waL = (__nv_bfloat16*)&sm.wa[1][ks_][0];
      __nv_bfloat16* abH = (__nv_bfloat16*)&sm.ab[0][ks_][0];
      __nv_bfloat16* abL = (__nv_bfloat16*)&sm.ab[1][ks_][0];
      const int areg = (fe >= 2 ? 2 : 0), ahalf = fe & 1;
#pragma unroll
      for (int t = 0; t < SQ; t++) {
        const float sk = (ABL & 1024) ? 0.f : sm.red[0][t] + sm.red[1][t] + sm.red[2][t] + sm.red[3][t];
        const float sq = (ABL & 1024) ? 0.f : sm.red[0][8 + t] + sm.red[1][8 + t] + sm.red[2][8 + t] + sm.red[3][8 + t];
        float kn, qn, D, Di;
        if constexpr (ABL & 1024) {  // deferred norm scaling: W, a built from unnormalized ck, cq
          (void)sk; (void)sq;
          kn = ck[t]; qn = cq[t];
          D = ex2f(G[t] * L2E); Di = rcpf(D);
        } else if constexpr (ABL & 8) { kn = ck[t] + sk; qn = cq[t] + sq; D = G[t]; Di = G[t]; }
        else {
          kn = ck[t] * rsqrtf(sk + 1e-6f);
          qn = cq[t] * (rsqrtf(sq + 1e-6f) * 0.08838834764831845f);
          if constexpr (ABL & 256) { D = ex2f(G[t] * L2E); Di = rcpf(D); }
          else { D = __expf(G[t]); Di = __expf(-G[t]); }
        }
        __nv_bfloat16 hi, lo;
        // A fragment: row m = t (b_t) and m = 8 + t (e_t); lane = (m & 7) * 4 + ftq
        const int ln = t * 4 + ftq;
        if constexpr (ABL & 4096) split_t(D * kn, hi, lo); else split(D * kn, hi, lo);
        waH[(ln * 4 + areg + 0) * 2 + ahalf] = hi; waL[(ln * 4 + areg + 0) * 2 + ahalf] = lo;
        if constexpr (ABL & 4096) split_t(D * qn, hi, lo); else split(D * qn, hi, lo);
        waH[(ln * 4 + areg + 1) * 2 + ahalf] = hi; waL[(ln * 4 + areg + 1) * 2 + ahalf] = lo;
        // B fragment of the small GEMM: column n = s = t; lane = t * 4 + ftq, reg = fe >= 2
        if constexpr (ABL & 4096) split_t(kn * Di, hi, lo); else split(kn * Di, hi, lo);
        abH[(ln * 2 + (fe >= 2)) * 2 + ahalf] = hi; abL[(ln * 2 + (fe >= 2)) * 2 + ahalf] = lo;
      }
      // token slots >= SQ: zero rows of W and columns of a (their P, Q, Mk, Mq entries are 0 and never read)
#pragma unroll
      for (int t = SQ; t < NSLOT; t++) {
        const __nv_bfloat16 z = __float2bfloat16(0.f);
        const int ln = t * 4 + ftq;
        waH[(ln * 4 + areg + 0) * 2 + ahalf] = z; waL[(ln * 4 + areg + 0) * 2 + ahalf] = z;
        waH[(ln * 4 + areg + 1) * 2 + ahalf] = z; waL[(ln * 4 + areg + 1) * 2 + ahalf] = z;
        abH[(ln * 2 + (fe >= 2)) * 2 + ahalf] = z; abL[(ln * 2 + (fe >= 2)) * 2 + ahalf] = z;
      }
      __syncthreads();  // (3) W, a fragments complete (ABL 1024: also red, ks)
      if constexpr ((ABL & 1024) != 0) KG_RECORD();
      // ABL 1024: per-warp norm factors; lane i (mod 16) holds r_i = rsqrt(|.|^2) (x K^-1/2 for q rows i >= 8)
      float rnv = 1.f, rkg = 1.f, rqg = 1.f;
      if constexpr (ABL & 1024) {
        const int i = lane & 15;
        const float s4 = sm.red[0][i] + sm.red[1][i] + sm.red[2][i] + sm.red[3][i];
        rnv = rsqrtf(s4 + 1e-6f);
        if (i >= 8) rnv *= 0.08838834764831845f;
        rkg = __shfl_sync(0xffffffffu, rnv, gid);
        rqg = __shfl_sync(0xffffffffu, rnv, 8 + gid);
      }
      // ---- P/Q = W h0^T : warp w owns V rows 32w..32w+31 (4 n-tiles) ----
      {
        float acc[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
          for (int i = 0; i < 4; i++) acc[nt][i] = 0.f;
        if constexpr (ABL & 1) {
          unsigned f = 0;
#pragma unroll
          for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int q = 0; q < 4; q++) f ^= sv[nt][q].x ^ sv[nt][q].y ^ sv[nt][q].z ^ sv[nt][q].w;
          acc[0][0] = __uint_as_float(f & 0x3f7fffffu) + __uint_as_float(sm.wa[0][lane & 7][lane].x & 0x3f7fffffu);
        } else {
#pragma unroll
        for (int s = 0; s < 8; s++) {
          const uint4 ah = sm.wa[0][s][lane], al = sm.wa[1][s][lane];
          const int q = s >> 1, sub = s & 1;
#pragma unroll
          for (int nt = 0; nt < 4; nt++) {
            const unsigned b0 = sub ? sv[nt][q].z : sv[nt][q].x;
            const unsigned b1 = sub ? sv[nt][q].w : sv[nt][q].y;
            mma16816(acc[nt], ah, b0, b1);
            mma16816(acc[nt], al, b0, b1);
          }
        }
        }
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          const int v = warp * 32 + nt * 8 + 2 * tq;
          if constexpr (ABL & 1024) {
            acc[nt][0] *= rkg; acc[nt][1] *= rkg; acc[nt][2] *= rqg; acc[nt][3] *= rqg;
          }
          *(float2*)&sm.pq[gid][v] = make_float2(acc[nt][0], acc[nt][1]);
          *(float2*)&sm.pq[gid + 8][v] = make_float2(acc[nt][2], acc[nt][3]);
        }
      }
      if (!(ABL & 2) && warp == 0) {  // Mk/Mq = W a^T (3-term hi/lo)
        float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < 8; s++) {
          const uint4 ah = sm.wa[0][s][lane], al = sm.wa[1][s][lane];
          const uint2 bh = sm.ab[0][s][lane], bl = sm.ab[1][s][lane];
          mma16816(c, ah, bh.x, bh.y);
          mma16816(c, al, bh.x, bh.y);
          mma16816(c, ah, bl.x, bl.y);
        }
        if constexpr (ABL & 1024) {
          const float r0 = __shfl_sync(0xffffffffu, rnv, 2 * tq), r1 = __shfl_sync(0xffffffffu, rnv, 2 * tq + 1);
          c[0] *= rkg * r0; c[1] *= rkg * r1; c[2] *= rqg * r0; c[3] *= rqg * r1;
        }
        if constexpr ((ABL & 8192) != 0) {  // transposed: mT[s][t] (Mk, t < 8) and mT[s][8 + t] (Mq)
          float* mT = &sm.mm[0][0];
          mT[(2 * tq) * 16 + gid] = c[0]; mT[(2 * tq + 1) * 16 + gid] = c[1];
          mT[(2 * tq) * 16 + 8 + gid] = c[2]; mT[(2 * tq + 1) * 16 + 8 + gid] = c[3];
        } else {
        *(float2*)&sm.mm[gid][2 * tq] = make_float2(c[0], c[1]);
        *(float2*)&sm.mm[gid + 8][2 * tq] = make_float2(c[2], c[3]);
        }
      }
      __syncthreads();  // (4) P, Q, M complete
      if constexpr (STAMP) st2 = gtime();
      // ---- forward substitution, thread v = tid ----
      float cc[SQ];
      float* cp = a.corr + ((long long)b * a.H + h) * SQ * KD;
      if constexpr ((ABL & 8192) != 0) {
        // right-looking substitution on token pairs (t, t+1) in f32x2 lanes; per-row accumulation order
        // (s = 0, 1, ...) is the same as the left-looking scalar loop, so results match it bit for bit
        const float* mT = &sm.mm[0][0];
        f2_t U[SQ / 2], O[SQ / 2];
#pragma unroll
        for (int p = 0; p < SQ / 2; p++) {
          U[p] = pk2(cv[2 * p] - sm.pq[2 * p][tid], cv[2 * p + 1] - sm.pq[2 * p + 1][tid]);
          O[p] = pk2(sm.pq[8 + 2 * p][tid], sm.pq[8 + 2 * p + 1][tid]);
        }
#pragma unroll
        for (int s2 = 0; s2 < SQ; s2++) {
          float ul, uh;
          upk2(U[s2 >> 1], ul, uh);
          const float c_ = ((s2 & 1) ? uh : ul) * sm.sbet[s2];
          cp[s2 * KD + tid] = c_;
          const f2_t c2 = pk2(c_, c_), nc2 = pk2(-c_, -c_);
#pragma unroll
          for (int p = s2 >> 1; p < SQ / 2; p++) {
            const f2_t mk2 = *(const f2_t*)&mT[s2 * 16 + 2 * p];
            const f2_t mq2 = *(const f2_t*)&mT[s2 * 16 + 8 + 2 * p];
            if (2 * p >= s2) O[p] = fma2(mq2, c2, O[p]);
            else {  // 2p + 1 == s2: only row t = s2 (high lane)
              float ol, oh, ml, mh; upk2(O[p], ol, oh); upk2(mq2, ml, mh);
              O[p] = pk2(ol, fmaf(mh, c_, oh));
            }
            if (2 * p > s2) U[p] = fma2(mk2, nc2, U[p]);
            else if (2 * p == s2) {  // only row t = s2 + 1 (high lane)
              float vl, vh, ml, mh; upk2(U[p], vl, vh); upk2(mk2, ml, mh);
              U[p] = pk2(vl, fmaf(-mh, c_, vh));
            }
          }
        }
#pragma unroll
        for (int p = 0; p < SQ / 2; p++) {
          float ol, oh; upk2(O[p], ol, oh);
          a.out[(tok0 + 2 * p) * HK + h * KD + tid] = __float2bfloat16(ol);
          a.out[(tok0 + 2 * p + 1) * HK + h * KD + tid] = __float2bfloat16(oh);
        }
      } else if constexpr ((ABL & 256) != 0) {
#pragma unroll
        for (int t = 0; t < SQ; t++) {
          const float4 k0 = *(const float4*)&sm.mm[t][0], k1 = *(const float4*)&sm.mm[t][4];
          const float4 q0 = *(const float4*)&sm.mm[8 + t][0], q1 = *(const float4*)&sm.mm[8 + t][4];
          const float mk[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
          const float mq[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
          float u = cv[t] - sm.pq[t][tid];
#pragma unroll
          for (int s2 = 0; s2 < t; s2++) u = fmaf(-mk[s2], cc[s2], u);
          cc[t] = u * sm.sbet[t];
          float o = sm.pq[8 + t][tid];
#pragma unroll
          for (int s2 = 0; s2 <= t; s2++) o = fmaf(mq[s2], cc[s2], o);
          cp[t * KD + tid] = cc[t];
          a.out[(tok0 + t) * HK + h * KD + tid] = __float2bfloat16(o);
        }
      } else
#pragma unroll
      for (int t = 0; t < SQ; t++) {
        float u = cv[t] - sm.pq[t][tid];
        if constexpr (!(ABL & 4)) {
#pragma unroll
          for (int s2 = 0; s2 < t; s2++) u = fmaf(-sm.mm[t][s2], cc[s2], u);
        }
        if constexpr (ABL & 16) cc[t] = u * __fdividef(1.f, 1.f + __expf(-sm.bet[t]));
        else cc[t] = u * sigm(sm.bet[t]);
        float o = sm.pq[8 + t][tid];
        if constexpr (!(ABL & 4)) {
#pragma unroll
          for (int s2 = 0; s2 <= t; s2++) o = fmaf(sm.mm[8 + t][s2], cc[s2], o);
        } else o += cc[t];
        cp[t * KD + tid] = cc[t];
        a.out[(tok0 + t) * HK + h * KD + tid] = __float2bfloat16(o);
      }
    }
  }
  if constexpr (STAMP) {
    if (tid == 0 && item == (int)blockIdx.x) {
      unsigned long long* sp = a.stamps + (long long)blockIdx.x * 5;
      sp[0] = st0; sp[1] = st1; sp[2] = (RUNG == 0 ? st1 : st2); sp[3] = gtime(); sp[4] = smid();
    }
  }
  return issued;
}

template <int RUNG, int MINB, int ABL = 0, int STAMP = 0>
__global__ void __launch_bounds__(NTH, MINB) k_fused(Args a) {
  __shared__ __align__(16) Smem sm;
  if constexpr ((ABL & 96) != 0) {
    const int n = a.B * a.H;
    int it = 0;
    bool pre = false;
    for (int item = blockIdx.x; item < n; item += gridDim.x) {
      const int nxt = item + (int)gridDim.x < n ? item + (int)gridDim.x : -1;
      pre = item_body<RUNG, ABL, STAMP>(a, sm, item, nxt, (ABL & 128) ? (it & 1) : 0, pre);
      it++;
      __syncthreads();  // smem image reused by the next item
    }
    if constexpr (STAMP) {  // end of the CTA's last item
      if (threadIdx.x == 0 && (int)blockIdx.x < n) a.stamps[(long long)blockIdx.x * 5 + 3] = gtime();
    }
  } else {
    item_body<RUNG, ABL, STAMP>(a, sm, blockIdx.x, -1, 0, false);
  }
}

__global__ void k_empty() {}

void fused(int64_t rung, int64_t minb, torch::Tensor pool, int64_t page, int64_t conv_bytes, torch::Tensor slots,
           torch::Tensor acc, torch::Tensor x, int64_t ldx, torch::Tensor gate, int64_t ldg, torch::Tensor beta,
           int64_t ldb, torch::Tensor a_log, torch::Tensor dtb, torch::Tensor w, torch::Tensor out, torch::Tensor corr,
           torch::Tensor kg, double lb, int64_t B, int64_t H, int64_t L, int64_t abl, int64_t stamp,
           torch::Tensor stamps) {
  Args a;
  a.pool = (const char*)pool.data_ptr();
  a.page = page;
  a.conv_bytes = conv_bytes;
  a.slots = slots.data_ptr<int>();
  a.acc = acc.data_ptr<int>();
  a.x = (const __nv_bfloat16*)x.data_ptr();
  a.ldx = ldx;
  a.gate = (const __nv_bfloat16*)gate.data_ptr();
  a.ldg = ldg;
  a.beta = (const __nv_bfloat16*)beta.data_ptr();
  a.ldb = ldb;
  a.a_log = a_log.data_ptr<float>();
  a.dtb = dtb.data_ptr<float>();
  a.w = w.data_ptr<float>();
  a.out = (__nv_bfloat16*)out.data_ptr();
  a.corr = corr.data_ptr<float>();
  a.kg = (__nv_bfloat16*)kg.data_ptr();
  a.lb = (float)lb;
  a.B = (int)B;
  a.H = (int)H;
  a.L = (int)L;
  a.stamps = stamp ? (unsigned long long*)stamps.data_ptr() : nullptr;
  auto s = c10::cuda::getCurrentCUDAStream();
  dim3 g((unsigned)(B * H));
#define L_(R, M, AB, ST)                                                                              \
  if (rung == R && minb == M && abl == AB && stamp == ST) {                                           \
    if (AB & 96) {                                                                                    \
      static int cap = 0;                                                                             \
      if (!cap) {                                                                                     \
        int dev, nsm, occ;                                                                            \
        cudaGetDevice(&dev);                                                                          \
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);                            \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_fused<R, M, AB, ST>, NTH, 0);           \
        cap = nsm * occ;                                                                              \
      }                                                                                               \
      g = dim3((unsigned)std::min<int64_t>(B * H, cap));                                              \
    }                                                                                                 \
    k_fused<R, M, AB, ST><<<g, NTH, 0, s>>>(a);                                                       \
    return;                                                                                           \
  }
  L_(0, 4, 0, 0) L_(1, 4, 0, 0) L_(2, 1, 0, 0) L_(2, 4, 0, 0)
  L_(0, 4, 0, 1) L_(1, 4, 0, 1) L_(2, 4, 0, 1)
  L_(1, 4, 16, 0) L_(2, 4, 16, 0)
  L_(2, 4, 1, 0) L_(2, 4, 2, 0) L_(2, 4, 4, 0) L_(2, 4, 8, 0) L_(2, 4, 15, 0)
  L_(0, 4, 32, 0) L_(0, 4, 64, 0) L_(1, 4, 48, 0) L_(2, 4, 48, 0) L_(2, 4, 80, 0) L_(2, 1, 48, 0)
  L_(2, 4, 16, 1) L_(2, 4, 48, 1)
  L_(0, 4, 160, 0) L_(2, 4, 176, 0) L_(2, 4, 272, 0) L_(2, 4, 432, 0) L_(2, 4, 176, 1)
  L_(2, 4, 784, 0) L_(2, 4, 944, 0) L_(2, 4, 944, 1)
  L_(2, 4, 1968, 0) L_(2, 4, 5040, 0) L_(2, 4, 6064, 0) L_(2, 4, 5904, 0) L_(2, 4, 6064, 1)
  L_(2, 4, 2992, 0) L_(2, 4, 8112, 0) L_(2, 4, 7952, 0) L_(2, 4, 8112, 1)
  L_(2, 4, 16304, 0) L_(2, 4, 16144, 0) L_(2, 4, 16304, 1)
#if SQ == 4
  L_(2, 4, 22448, 0) L_(2, 5, 22448, 0) L_(2, 4, 22448, 1) L_(2, 5, 6064, 0) L_(0, 5, 0, 0) L_(0, 5, 160, 0)
#endif
#undef L_
  TORCH_CHECK(false, "unsupported rung/minb");
}

void empty() { k_empty<<<1, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused", &fused);
  m.def("empty", &empty);
}
