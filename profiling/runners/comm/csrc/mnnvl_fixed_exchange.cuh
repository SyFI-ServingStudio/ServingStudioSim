// Fixed-pattern MoE exchange over an NVLink-mapped workspace (MNNVL fabric memory).
//
// Expert-parallel MoE with data-parallel attention, no routing-dependent row selection:
//
//   dispatch  every rank sends all of its token rows to every rank (all-gather semantics):
//             NVFP4 activations, e4m3 block scales, fp32 top-k weights and int32 top-k ids,
//             the four tensors vLLM's naive DP/EP prepare gathers (prepare_finalize/naive_dp_ep.py).
//   combine   every rank holds bf16 partial outputs of all gathered tokens (its local experts
//             only) and every owner gets the sum of the ranks' partials for its own tokens
//             (reduce-scatter semantics).
//
// Transport: Lamport arrival, the scheme of FlashInfer's MNNVL all-reduce and of its NVLink
// one-sided MoE all-to-all (include/flashinfer/comm/moe_alltoall_lamport.cuh). Every landing word
// is armed with kSentinel; a sender stores its rows straight into each peer's landing region with
// 16-byte stores, after rewriting the few words that could equal the sentinel; the receiver polls
// the data itself (a 16-byte vector has arrived when none of its 32-bit words is the sentinel;
// 32-bit stores are single-copy atomic), copies it out and re-arms it. One launch per leg, no
// completion flags and no fences.
//
// Wire rewrites (lossless for every value these tensors carry):
//   NVFP4 payload  every -0 nibble becomes +0, then the word is XORed with kNvfp4WireXor
//                  (FlashInfer's rule: the only raw word mapping to the sentinel is eight -0s).
//   e4m3 scales    0xFFFFFFFF -> 0x7FFFFFFF (four NaNs; NVFP4 block scales are non-negative).
//   fp32 weights   0xFFFFFFFF (a NaN) -> 0x7FFFFFFF.
//   int32 ids      XORed with kIdsWireXor (only id 0x7FFFFFFF would collide).
//   bf16 partials  every 0xFFFF lane (a NaN) -> 0x7FFF.
//
// Rounds. Each leg's landing has two sets, chosen by the leg's epoch (device memory, so CUDA
// graphs replay it); the last block to finish advances it. A sender writes set n % 2 in round n.
// It reaches round n + 2 only after its round n + 1 received something from every peer, which
// every peer sends after finishing round n (stream order), so a set is never written while still
// being read. A peer's rows alone do not guarantee that (a zero-count peer sends none, and a rank
// receiving nothing would lap it), so every round also carries one 16-byte marker per peer and
// set (markers region), pushed and polled like a row vector.
//
// Pull combine (the fused exchange's consumer side). Every rank's expert output lives in its own
// mapped "partials" region (the MoE kernel writes it there; no copy). The consumer of round n
// announces its partials with a release store of n + 1 into every peer's ready flag, waits for
// all peers' flags, then reads its own tokens' rows from all ranks over NVLink and sums them in
// fp32. The partials have two sets, chosen by the caller (the producer's buffer, fixed when a
// CUDA graph is captured; e.g. the MoE layer's parity); the producer may rewrite a set only after
// a group-wide step that every consumer of its previous round has passed (in a MoE stack, the
// next layer's dispatch). `pull_sum_vector` is the device-side entry for a consumer kernel that fuses the sum.
//
// Layout of one rank's mapped slice (byte offsets in ExchangeLayout):
//   state     int32 round state (kStateWords), this rank's only
//   flags     int32 ready flag per source rank (pull combine)
//   markers   per leg 2 sets x [world source] 16-byte round markers (Lamport), at kMarkerOffset
//   dispatch  2 sets x [world source][max_rows][dispatch row bytes]
//   combine   2 sets x [world source][max_rows][hidden * 2]
//   partials  2 sets x [world * max_rows][hidden * 2] bf16, gathered-token order

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace vibesim {
namespace fixed_exchange {

constexpr int kMaxRanks = 8;
constexpr uint32_t kSentinel = 0xFFFFFFFFu;
constexpr uint32_t kNvfp4WireXor = 0x77777777u;
constexpr uint32_t kIdsWireXor = 0x80000000u;
// Spin for at most this long (ns) before reporting a missing peer and trapping.
constexpr uint64_t kTimeoutNs = 20ull * 1000 * 1000 * 1000;

enum StateWord : int {
  kDispatchEpoch = 0,
  kDispatchFinished = 1,
  kCombineEpoch = 2,
  kCombineFinished = 3,
  kPullRound = 4,
  kPullFinished = 5,
  kTimeoutReported = 6,
  kStateWords = 16,
};

// Round markers: byte offset in a slice (inside the first 4 KiB, after the flags; the host arms
// them with the landing) and one 2 x kMaxRanks x 16-byte block per Lamport leg.
constexpr int64_t kMarkerOffset = 1024;
enum MarkerLeg : int { kDispatchMarkers = 0, kCombineMarkers = 1 };

// Dispatch row planes, in wire order.
enum Plane : int { kPayload = 0, kScales = 1, kWeights = 2, kIds = 3, kNumPlanes = 4 };

struct ExchangeLayout {
  int64_t state_offset;
  int64_t flags_offset;
  int64_t dispatch_offset;
  int64_t dispatch_set_bytes;
  int64_t combine_offset;
  int64_t combine_set_bytes;
  int64_t partials_offset;
  int64_t partials_set_bytes;
};

struct Group {
  uint8_t* slices[kMaxRanks];  // every rank's mapped slice, as mapped in this process
  ExchangeLayout layout;
  int world;
  int rank;
  int max_rows;  // landing rows per source rank
};

struct DispatchPlanes {
  int vectors[kNumPlanes];  // 16-byte vectors per row of each plane
  int bytes[kNumPlanes];    // bytes per row of each plane
  int row_vectors;          // sum of vectors
  int row_bytes;            // sum of bytes (the landing row)
};

// =========================================================================== PTX helpers
__device__ __forceinline__ uint4 load_volatile(void const* ptr) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}

__device__ __forceinline__ int load_volatile_int(int const* ptr) {
  int v;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
  return v;
}

__device__ __forceinline__ int load_acquire_int(int const* ptr) {
  int v;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
  return v;
}

__device__ __forceinline__ void store_release_int(int* ptr, int v) {
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");
}

__device__ __forceinline__ void store_volatile(void* ptr, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w)
               : "memory");
}

__device__ __forceinline__ uint4 load_nc(void const* ptr) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint64_t global_timer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Diagnostic timeline (built only with -DFIXED_EXCHANGE_TIMELINE, a separate JIT module): per
// round (epoch mod kTimelineSlots), the earliest / latest block entry and the latest block's
// push end, receive end and exit, in globaltimer ns. The marks add a __syncthreads per phase.
#ifdef FIXED_EXCHANGE_TIMELINE
constexpr int kTimelineSlots = 1024;
enum TimelinePhase : int {
  kEntryMin = 0,
  kEntryMax = 1,
  kPushEnd = 2,
  kReceiveEnd = 3,
  kExit = 4,
  kTimelinePhases = 8,
};
__device__ unsigned long long g_timeline[kTimelineSlots][kTimelinePhases];
__device__ __forceinline__ void timeline_mark(int phase, int epoch) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long const t = global_timer_ns();
    unsigned long long* slot = g_timeline[epoch & (kTimelineSlots - 1)];
    if (phase == kEntryMax) atomicMin(slot + kEntryMin, t);
    atomicMax(slot + phase, t);
  }
}
#define FX_TIMELINE(phase, epoch) timeline_mark(phase, epoch)
#else
#define FX_TIMELINE(phase, epoch) ((void)0)
#endif

__device__ __forceinline__ uint4 splat(uint32_t w) { return make_uint4(w, w, w, w); }

__device__ __forceinline__ bool has_arrived(uint4 v) {
  return v.x != kSentinel && v.y != kSentinel && v.z != kSentinel && v.w != kSentinel;
}

[[noreturn]] static __device__ __noinline__ void report_and_trap(int32_t* state, int rank,
                                                                 char const* what) {
  if (atomicExch(state + kTimeoutReported, 1) == 0)
    printf("fixed exchange: rank %d timed out waiting for %s\n", rank, what);
  __threadfence_system();
  __trap();
}

// ============================================================================ wire rewrites
__device__ __forceinline__ uint32_t canonicalize_fp4x8(uint32_t word) {
  uint32_t const magnitude = word & 0x77777777u;
  uint32_t const nonzero = (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11111111u;
  return word & (0x77777777u | (nonzero << 3));
}

__device__ __forceinline__ uint32_t canonicalize_word(uint32_t w) {
  return w == kSentinel ? 0x7FFFFFFFu : w;
}

__device__ __forceinline__ uint32_t canonicalize_bf16_lanes(uint32_t w) {
  if ((w & 0xFFFFu) == 0xFFFFu) w = (w & 0xFFFF0000u) | 0x7FFFu;
  if ((w >> 16) == 0xFFFFu) w = (w & 0x0000FFFFu) | 0x7FFF0000u;
  return w;
}

__device__ __forceinline__ uint4 to_wire(uint4 v, int plane) {
  switch (plane) {
    case kPayload:
      return make_uint4(canonicalize_fp4x8(v.x) ^ kNvfp4WireXor,
                        canonicalize_fp4x8(v.y) ^ kNvfp4WireXor,
                        canonicalize_fp4x8(v.z) ^ kNvfp4WireXor,
                        canonicalize_fp4x8(v.w) ^ kNvfp4WireXor);
    case kIds:
      return make_uint4(v.x ^ kIdsWireXor, v.y ^ kIdsWireXor, v.z ^ kIdsWireXor,
                        v.w ^ kIdsWireXor);
    default:
      return make_uint4(canonicalize_word(v.x), canonicalize_word(v.y), canonicalize_word(v.z),
                        canonicalize_word(v.w));
  }
}

__device__ __forceinline__ uint4 from_wire(uint4 v, int plane) {
  switch (plane) {
    case kPayload:
      return make_uint4(v.x ^ kNvfp4WireXor, v.y ^ kNvfp4WireXor, v.z ^ kNvfp4WireXor,
                        v.w ^ kNvfp4WireXor);
    case kIds:
      return make_uint4(v.x ^ kIdsWireXor, v.y ^ kIdsWireXor, v.z ^ kIdsWireXor,
                        v.w ^ kIdsWireXor);
    default:
      return v;
  }
}

__device__ __forceinline__ uint4 canonicalize_partial(uint4 v) {
  return make_uint4(canonicalize_bf16_lanes(v.x), canonicalize_bf16_lanes(v.y),
                    canonicalize_bf16_lanes(v.z), canonicalize_bf16_lanes(v.w));
}

// ===================================================================== shared block prologue
struct RoundInfo {
  int epoch;
  int counts[kMaxRanks];
  int offsets[kMaxRanks];  // first gathered row of each rank
  int total;
};

// Every thread gets the leg's epoch and the per-rank token counts (device array, so a CUDA
// graph replays with whatever counts the step wrote).
__device__ __forceinline__ void load_round(Group const& g, int epoch_word, int32_t const* counts,
                                           RoundInfo& out) {
  __shared__ int s_epoch;
  __shared__ int s_counts[kMaxRanks];
  int32_t* state = reinterpret_cast<int32_t*>(g.slices[g.rank] + g.layout.state_offset);
  if (threadIdx.x == 0) s_epoch = load_volatile_int(state + epoch_word);
  if (threadIdx.x < g.world) s_counts[threadIdx.x] = counts[threadIdx.x];
  __syncthreads();
  out.epoch = s_epoch;
  int running = 0;
#pragma unroll
  for (int r = 0; r < kMaxRanks; ++r) {
    int const c = r < g.world ? s_counts[r] : 0;
    out.counts[r] = c;
    out.offsets[r] = running;
    running += c;
  }
  out.total = running;
}

// The last block to finish advances the leg's epoch.
__device__ __forceinline__ void finish_round(Group const& g, int epoch_word, int finished_word,
                                             int epoch) {
  __syncthreads();
  if (threadIdx.x == 0) {
    int32_t* state = reinterpret_cast<int32_t*>(g.slices[g.rank] + g.layout.state_offset);
    __threadfence();
    int const prev = atomicAdd(state + finished_word, 1);
    if (prev == static_cast<int>(gridDim.x) - 1) {
      state[finished_word] = 0;
      atomicExch(state + epoch_word, epoch + 1);
      __threadfence();
    }
  }
}

__device__ __forceinline__ uint8_t* round_marker(Group const& g, int leg, int dest, int set,
                                                 int source) {
  return g.slices[dest] + kMarkerOffset + ((leg * 2 + set) * kMaxRanks + source) * 16;
}

// Block 0, one thread per peer: push this rank's marker of the round to the peer.
__device__ __forceinline__ void push_round_markers(Group const& g, int leg, int set, int epoch) {
  if (blockIdx.x == 0 && threadIdx.x < g.world && threadIdx.x != g.rank) {
    uint32_t const w = static_cast<uint32_t>(epoch) & 0x7FFFFFFFu;  // never the sentinel
    store_volatile(round_marker(g, leg, threadIdx.x, set, g.rank), splat(w));
  }
}

// Block 0, one thread per peer: wait for the peer's marker of the round, then re-arm it.
__device__ __forceinline__ void receive_round_markers(Group const& g, int leg, int set) {
  if (blockIdx.x == 0 && threadIdx.x < g.world && threadIdx.x != g.rank) {
    uint8_t* const addr = round_marker(g, leg, g.rank, set, threadIdx.x);
    int32_t* state = reinterpret_cast<int32_t*>(g.slices[g.rank] + g.layout.state_offset);
    uint64_t const start = global_timer_ns();
    for (int spins = 0; !has_arrived(load_volatile(addr)); ++spins)
      if ((spins & 255) == 255 && global_timer_ns() - start > kTimeoutNs)
        report_and_trap(state, g.rank, "round markers");
    store_volatile(addr, splat(kSentinel));
  }
}

__device__ __forceinline__ void locate_plane(DispatchPlanes const& planes, int vec, int& plane,
                                             int& plane_vec, int& byte_in_row) {
  int base = 0, bytes = 0;
  plane = kNumPlanes - 1;
#pragma unroll
  for (int p = 0; p < kNumPlanes; ++p) {
    if (vec < base + planes.vectors[p]) {
      plane = p;
      break;
    }
    base += planes.vectors[p];
    bytes += planes.bytes[p];
  }
  plane_vec = vec - base;
  byte_in_row = bytes + plane_vec * 16;
}

__device__ __forceinline__ uint8_t* dispatch_landing(Group const& g, DispatchPlanes const& planes,
                                                     int dest, int set, int source, int row) {
  return g.slices[dest] + g.layout.dispatch_offset + set * g.layout.dispatch_set_bytes +
         (static_cast<int64_t>(source) * g.max_rows + row) * planes.row_bytes;
}

__device__ __forceinline__ uint8_t* combine_landing(Group const& g, int row_bytes, int dest,
                                                    int set, int source, int row) {
  return g.slices[dest] + g.layout.combine_offset + set * g.layout.combine_set_bytes +
         (static_cast<int64_t>(source) * g.max_rows + row) * row_bytes;
}

// Pushes one wire vector to every peer's landing row (rank, row) of the set, peers in rotated
// order so the ranks' first stores spread over the links.
__device__ __forceinline__ void push_to_peers(Group const& g, int64_t landing_byte, uint4 wire) {
  for (int k = 1; k < g.world; ++k) {
    int dest = g.rank + k;
    if (dest >= g.world) dest -= g.world;
    store_volatile(g.slices[dest] + landing_byte, wire);
  }
}

// ======================================================================== dispatch receive
// Copies every peer's rows of this round out of the local landing set into the gathered output
// planes, re-arming each vector. Each thread keeps kUnroll polls in flight.
template <int kUnroll>
__device__ __forceinline__ void dispatch_receive(Group const& g, DispatchPlanes const& planes,
                                                 RoundInfo const& round, int set,
                                                 uint8_t* const* out) {
  int const self = g.rank;
  int const remote_rows = round.total - round.counts[self];
  int64_t const tasks = static_cast<int64_t>(remote_rows) * planes.row_vectors;
  int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int32_t* state = reinterpret_cast<int32_t*>(g.slices[self] + g.layout.state_offset);
  for (int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; first < tasks;
       first += stride * kUnroll) {
    uint8_t* addr[kUnroll];
    uint4 v[kUnroll];
    int plane[kUnroll], plane_vec[kUnroll], out_row[kUnroll];
    uint32_t pending = 0;
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      int64_t const task = first + u * stride;
      if (task >= tasks) continue;
      int const vec = static_cast<int>(task % planes.row_vectors);
      int remote_row = static_cast<int>(task / planes.row_vectors);
      // Remote rows in source order, skipping this rank.
      int source = 0;
      for (; source < g.world; ++source) {
        if (source == self) continue;
        if (remote_row < round.counts[source]) break;
        remote_row -= round.counts[source];
      }
      int byte_in_row;
      locate_plane(planes, vec, plane[u], plane_vec[u], byte_in_row);
      addr[u] = dispatch_landing(g, planes, self, set, source, remote_row) + byte_in_row;
      out_row[u] = round.offsets[source] + remote_row;
      pending |= 1u << u;
    }
    uint64_t const start = global_timer_ns();
    for (int spins = 0; pending; ++spins) {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u)
        if ((pending >> u) & 1u) v[u] = load_volatile(addr[u]);
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (((pending >> u) & 1u) && has_arrived(v[u])) {
          pending &= ~(1u << u);
          *reinterpret_cast<uint4*>(out[plane[u]] +
                                    static_cast<int64_t>(out_row[u]) * planes.bytes[plane[u]] +
                                    plane_vec[u] * 16) = from_wire(v[u], plane[u]);
          store_volatile(addr[u], splat(kSentinel));
        }
      }
      if (pending && (spins & 255) == 255 && global_timer_ns() - start > kTimeoutNs)
        report_and_trap(state, self, "dispatch rows");
    }
  }
}

// ============================================================================ K1 dispatch
struct DispatchArgs {
  Group group;
  DispatchPlanes planes;
  uint8_t const* in[kNumPlanes];  // this rank's rows, [counts[rank], planes.bytes[p]]
  uint8_t* out[kNumPlanes];       // gathered, [sum(counts), planes.bytes[p]]
  int32_t const* counts;          // [world] device
};

template <int kThreads, int kUnroll>
__global__ void __launch_bounds__(kThreads) fixed_dispatch_kernel(DispatchArgs const a) {
  Group const& g = a.group;
  RoundInfo round;
  load_round(g, kDispatchEpoch, a.counts, round);
  FX_TIMELINE(kEntryMax, round.epoch);
  int const set = round.epoch & 1;
  int const self = g.rank;
  push_round_markers(g, kDispatchMarkers, set, round.epoch);
  int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t const push_tasks = static_cast<int64_t>(round.counts[self]) * a.planes.row_vectors;
  int64_t const landing_base =
      g.layout.dispatch_offset + set * g.layout.dispatch_set_bytes +
      static_cast<int64_t>(self) * g.max_rows * a.planes.row_bytes;
  for (int64_t task = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < push_tasks; task += stride) {
    int const row = static_cast<int>(task / a.planes.row_vectors);
    int const vec = static_cast<int>(task % a.planes.row_vectors);
    int plane, plane_vec, byte_in_row;
    locate_plane(a.planes, vec, plane, plane_vec, byte_in_row);
    uint4 const raw = load_nc(a.in[plane] + static_cast<int64_t>(row) * a.planes.bytes[plane] +
                              plane_vec * 16);
    uint4 const wire = to_wire(raw, plane);
    push_to_peers(g, landing_base + static_cast<int64_t>(row) * a.planes.row_bytes + byte_in_row,
                  wire);
    // The own rows take the same canonical form the peers receive.
    *reinterpret_cast<uint4*>(
        a.out[plane] + static_cast<int64_t>(round.offsets[self] + row) * a.planes.bytes[plane] +
        plane_vec * 16) = from_wire(wire, plane);
  }
  FX_TIMELINE(kPushEnd, round.epoch);
  dispatch_receive<kUnroll>(g, a.planes, round, set, const_cast<uint8_t* const*>(a.out));
  FX_TIMELINE(kReceiveEnd, round.epoch);
  receive_round_markers(g, kDispatchMarkers, set);
  finish_round(g, kDispatchEpoch, kDispatchFinished, round.epoch);
  FX_TIMELINE(kExit, round.epoch);
}

// ====================================================================== K2 quant + dispatch
// NVFP4 quantization of bf16 rows (16-element blocks, e4m3 block scales in the linear layout,
// one fp32 global scale) fused with the dispatch push. The arithmetic is the default branch of
// TensorRT-LLM's cvt_warp_fp16_to_fp4 (FlashInfer nv_internal quantization_utils.cuh), which the
// runner checks bit for bit against vLLM's own quantization before timing.
__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ uint32_t fp32x8_to_e2m1x8(float const (&f)[8]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0, byte1, byte2, byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(f[0]), "f"(f[1]), "f"(f[2]), "f"(f[3]), "f"(f[4]), "f"(f[5]), "f"(f[6]), "f"(f[7]));
  return val;
}

// One 16-element block: 16 bf16 values (two uint4) -> 8 packed bytes and the e4m3 scale.
__device__ __forceinline__ uint2 quantize_block16(uint4 lo, uint4 hi, float global_scale,
                                                  float rcp_global_scale, uint8_t& scale_byte) {
  __nv_bfloat162 const* h_lo = reinterpret_cast<__nv_bfloat162 const*>(&lo);
  __nv_bfloat162 const* h_hi = reinterpret_cast<__nv_bfloat162 const*>(&hi);
  __nv_bfloat162 m = __habs2(h_lo[0]);
#pragma unroll
  for (int i = 1; i < 4; ++i) m = __hmax2(m, __habs2(h_lo[i]));
#pragma unroll
  for (int i = 0; i < 4; ++i) m = __hmax2(m, __habs2(h_hi[i]));
  float const vec_max = fmaxf(__bfloat162float(m.x), __bfloat162float(m.y));
  float sf = global_scale * (vec_max * rcp_approx_ftz(6.0f));
  __nv_fp8_e4m3 const sf8 = __nv_fp8_e4m3(sf);
  scale_byte = sf8.__x;
  sf = static_cast<float>(sf8);
  float const out_scale = scale_byte != 0 ? rcp_approx_ftz(sf * rcp_global_scale) : 0.0f;
  float f[8];
  uint2 packed;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 const x = __bfloat1622float2(h_lo[i]);
    f[2 * i] = x.x * out_scale;
    f[2 * i + 1] = x.y * out_scale;
  }
  packed.x = fp32x8_to_e2m1x8(f);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 const x = __bfloat1622float2(h_hi[i]);
    f[2 * i] = x.x * out_scale;
    f[2 * i + 1] = x.y * out_scale;
  }
  packed.y = fp32x8_to_e2m1x8(f);
  return packed;
}

struct QuantDispatchArgs {
  Group group;
  DispatchPlanes planes;          // payload hidden/2, scales hidden/16, weights, ids
  __nv_bfloat16 const* hidden;    // [counts[rank], hidden]
  float const* global_scale;      // device scalar (a1_gscale)
  uint8_t const* topk_weights;    // [counts[rank], top_k] fp32
  uint8_t const* topk_ids;        // [counts[rank], top_k] int32
  uint8_t* out[kNumPlanes];       // gathered planes, as K1
  int32_t const* counts;
  int hidden_size;
};

// Thread task (row, payload vector j): 32 elements -> 16 payload bytes and 2 scale bytes. Eight
// consecutive lanes hold one 16-byte scale vector, which their first lane pushes. A row's
// payload tasks are hidden/32 consecutive tasks, a multiple of 32, so a warp never straddles
// rows. The weights and ids vectors of each row are extra tasks after the payload tasks.
template <int kThreads, int kUnroll>
__global__ void __launch_bounds__(kThreads) quant_dispatch_kernel(QuantDispatchArgs const a) {
  Group const& g = a.group;
  RoundInfo round;
  load_round(g, kDispatchEpoch, a.counts, round);
  FX_TIMELINE(kEntryMax, round.epoch);
  int const set = round.epoch & 1;
  int const self = g.rank;
  push_round_markers(g, kDispatchMarkers, set, round.epoch);
  int const rows = round.counts[self];
  int const payload_vectors = a.planes.vectors[kPayload];  // hidden / 32
  float const global_scale = *a.global_scale;
  float const rcp_global_scale = rcp_approx_ftz(global_scale);
  int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t const landing_base =
      g.layout.dispatch_offset + set * g.layout.dispatch_set_bytes +
      static_cast<int64_t>(self) * g.max_rows * a.planes.row_bytes;
  int const lane = threadIdx.x & 31;
  // Payload tasks; every lane of a warp runs the same number of iterations (the shuffles).
  int64_t const payload_tasks = static_cast<int64_t>(rows) * payload_vectors;
  int64_t const warp_first = static_cast<int64_t>(blockIdx.x) * blockDim.x + (threadIdx.x & ~31);
  for (int64_t base = warp_first; base < payload_tasks; base += stride) {
    int64_t const task = base + lane;
    bool const active = task < payload_tasks;
    int const row = active ? static_cast<int>(task / payload_vectors) : 0;
    int const j = active ? static_cast<int>(task % payload_vectors) : 0;
    uint4 x[4] = {};
    if (active) {
      uint4 const* src = reinterpret_cast<uint4 const*>(
          a.hidden + static_cast<int64_t>(row) * a.hidden_size + j * 32);
#pragma unroll
      for (int i = 0; i < 4; ++i) x[i] = load_nc(src + i);
    }
    uint8_t s0 = 0, s1 = 0;
    uint2 const p0 = quantize_block16(x[0], x[1], global_scale, rcp_global_scale, s0);
    uint2 const p1 = quantize_block16(x[2], x[3], global_scale, rcp_global_scale, s1);
    uint4 const payload = make_uint4(p0.x, p0.y, p1.x, p1.y);
    uint32_t const my_scales = static_cast<uint32_t>(s0) | (static_cast<uint32_t>(s1) << 8);
    // Scale vector of lanes 8q..8q+7: word w = scales of lanes 2w and 2w + 1.
    uint32_t words[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      uint32_t const a0 = __shfl_sync(0xFFFFFFFFu, my_scales, (lane & ~7) + 2 * w);
      uint32_t const a1 = __shfl_sync(0xFFFFFFFFu, my_scales, (lane & ~7) + 2 * w + 1);
      words[w] = a0 | (a1 << 16);
    }
    if (!active) continue;
    int64_t const row_landing = landing_base + static_cast<int64_t>(row) * a.planes.row_bytes;
    int64_t const out_row = round.offsets[self] + row;
    uint4 const payload_wire = to_wire(payload, kPayload);
    push_to_peers(g, row_landing + j * 16, payload_wire);
    *reinterpret_cast<uint4*>(a.out[kPayload] + out_row * a.planes.bytes[kPayload] + j * 16) =
        from_wire(payload_wire, kPayload);
    if ((lane & 7) == 0) {
      uint4 const scales = make_uint4(words[0], words[1], words[2], words[3]);
      int const sv = j / 8;
      uint4 const scales_wire = to_wire(scales, kScales);
      push_to_peers(g, row_landing + a.planes.bytes[kPayload] + sv * 16, scales_wire);
      *reinterpret_cast<uint4*>(a.out[kScales] + out_row * a.planes.bytes[kScales] + sv * 16) =
          from_wire(scales_wire, kScales);
    }
  }
  // Weights and ids vectors.
  int const extra_vectors = a.planes.vectors[kWeights] + a.planes.vectors[kIds];
  int64_t const extra_tasks = static_cast<int64_t>(rows) * extra_vectors;
  for (int64_t task = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < extra_tasks; task += stride) {
    int const row = static_cast<int>(task / extra_vectors);
    int const e = static_cast<int>(task % extra_vectors);
    int const plane = e < a.planes.vectors[kWeights] ? kWeights : kIds;
    int const pv = plane == kWeights ? e : e - a.planes.vectors[kWeights];
    uint8_t const* src = plane == kWeights ? a.topk_weights : a.topk_ids;
    uint4 const raw = load_nc(src + static_cast<int64_t>(row) * a.planes.bytes[plane] + pv * 16);
    int const byte_in_row = a.planes.bytes[kPayload] + a.planes.bytes[kScales] +
                            (plane == kIds ? a.planes.bytes[kWeights] : 0) + pv * 16;
    uint4 const wire = to_wire(raw, plane);
    push_to_peers(g, landing_base + static_cast<int64_t>(row) * a.planes.row_bytes + byte_in_row,
                  wire);
    *reinterpret_cast<uint4*>(a.out[plane] +
                              static_cast<int64_t>(round.offsets[self] + row) *
                                  a.planes.bytes[plane] +
                              pv * 16) = from_wire(wire, plane);
  }
  FX_TIMELINE(kPushEnd, round.epoch);
  dispatch_receive<kUnroll>(g, a.planes, round, set, const_cast<uint8_t* const*>(a.out));
  FX_TIMELINE(kReceiveEnd, round.epoch);
  receive_round_markers(g, kDispatchMarkers, set);
  finish_round(g, kDispatchEpoch, kDispatchFinished, round.epoch);
  FX_TIMELINE(kExit, round.epoch);
}

// ============================================================================= K1 combine
struct CombineArgs {
  Group group;
  __nv_bfloat16 const* partials;  // [sum(counts), hidden], this rank's experts only
  __nv_bfloat16* out;             // [counts[rank], hidden]
  int32_t const* counts;
  int hidden_size;
};

__device__ __forceinline__ void accumulate_bf16x8(float (&acc)[8], uint4 v) {
  __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 const f = __bfloat1622float2(h[i]);
    acc[2 * i] += f.x;
    acc[2 * i + 1] += f.y;
  }
}

__device__ __forceinline__ uint4 pack_bf16x8(float const (&acc)[8]) {
  uint4 r;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
  return r;
}

template <int kThreads>
__global__ void __launch_bounds__(kThreads) fixed_combine_kernel(CombineArgs const a) {
  Group const& g = a.group;
  RoundInfo round;
  load_round(g, kCombineEpoch, a.counts, round);
  int const set = round.epoch & 1;
  int const self = g.rank;
  push_round_markers(g, kCombineMarkers, set, round.epoch);
  int const row_vectors = a.hidden_size / 8;  // bf16 16-byte vectors
  int const row_bytes = a.hidden_size * 2;
  int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  // Push: peers' tokens, in destination order starting after this rank.
  int64_t const push_tasks =
      static_cast<int64_t>(round.total - round.counts[self]) * row_vectors;
  for (int64_t task = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < push_tasks; task += stride) {
    int const vec = static_cast<int>(task % row_vectors);
    int remote_row = static_cast<int>(task / row_vectors);
    int dest = self;
    for (int k = 1; k < g.world; ++k) {
      dest = self + k;
      if (dest >= g.world) dest -= g.world;
      if (remote_row < round.counts[dest]) break;
      remote_row -= round.counts[dest];
    }
    uint4 const raw = load_nc(reinterpret_cast<uint8_t const*>(a.partials) +
                              static_cast<int64_t>(round.offsets[dest] + remote_row) * row_bytes +
                              vec * 16);
    store_volatile(combine_landing(g, row_bytes, dest, set, self, remote_row) + vec * 16,
                   canonicalize_partial(raw));
  }
  // Reduce this rank's tokens over every source, in rank order.
  int32_t* state = reinterpret_cast<int32_t*>(g.slices[self] + g.layout.state_offset);
  int64_t const reduce_tasks = static_cast<int64_t>(round.counts[self]) * row_vectors;
  for (int64_t task = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < reduce_tasks; task += stride) {
    int const row = static_cast<int>(task / row_vectors);
    int const vec = static_cast<int>(task % row_vectors);
    uint4 v[kMaxRanks];
    uint8_t* addr[kMaxRanks];
    uint32_t pending = 0;
#pragma unroll
    for (int s = 0; s < kMaxRanks; ++s) {
      if (s >= g.world) continue;
      if (s == self) {
        v[s] = load_nc(reinterpret_cast<uint8_t const*>(a.partials) +
                       static_cast<int64_t>(round.offsets[self] + row) * row_bytes + vec * 16);
      } else {
        addr[s] = combine_landing(g, row_bytes, self, set, s, row) + vec * 16;
        pending |= 1u << s;
      }
    }
    uint64_t const start = global_timer_ns();
    for (int spins = 0; pending; ++spins) {
#pragma unroll
      for (int s = 0; s < kMaxRanks; ++s)
        if ((pending >> s) & 1u) v[s] = load_volatile(addr[s]);
#pragma unroll
      for (int s = 0; s < kMaxRanks; ++s)
        if (((pending >> s) & 1u) && has_arrived(v[s])) pending &= ~(1u << s);
      if (pending && (spins & 255) == 255 && global_timer_ns() - start > kTimeoutNs)
        report_and_trap(state, self, "combine partials");
    }
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int s = 0; s < kMaxRanks; ++s) {
      if (s >= g.world) continue;
      accumulate_bf16x8(acc, v[s]);
      if (s != self) store_volatile(addr[s], splat(kSentinel));
    }
    *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(a.out) +
                              static_cast<int64_t>(row) * row_bytes + vec * 16) = pack_bf16x8(acc);
  }
  receive_round_markers(g, kCombineMarkers, set);
  finish_round(g, kCombineEpoch, kCombineFinished, round.epoch);
}

// ====================================================================== K2 pull combine
// Device entry for a consumer that fuses the sum: vector `vec` (8 bf16) of this rank's token
// `row`, summed over every rank's partials of the round in rank order. Call only after
// `pull_wait_ready` in the same kernel.
__device__ __forceinline__ void pull_sum_vector(Group const& g, int set, int gathered_row,
                                                int row_bytes, int vec, float (&acc)[8]) {
  uint4 v[kMaxRanks];
  int64_t const offset = g.layout.partials_offset + set * g.layout.partials_set_bytes +
                         static_cast<int64_t>(gathered_row) * row_bytes + vec * 16;
#pragma unroll
  for (int s = 0; s < kMaxRanks; ++s)
    if (s < g.world) v[s] = load_volatile(g.slices[s] + offset);
#pragma unroll
  for (int s = 0; s < kMaxRanks; ++s)
    if (s < g.world) accumulate_bf16x8(acc, v[s]);
}

// Announce this rank's partials of `round` to every peer (one thread per peer after a system
// fence), then wait until every peer has announced the same round. Every block calls it; block
// 0 announces. Returns round parity; a consumer inside a CUDA graph should instead read the set
// its producer wrote (fixed at capture), as pull_combine_kernel does.
__device__ __forceinline__ int pull_wait_ready(Group const& g, int round) {
  int32_t* state = reinterpret_cast<int32_t*>(g.slices[g.rank] + g.layout.state_offset);
  if (blockIdx.x == 0 && threadIdx.x < g.world) {
    __threadfence_system();
    store_release_int(
        reinterpret_cast<int*>(g.slices[threadIdx.x] + g.layout.flags_offset) + g.rank,
        round + 1);
  }
  if (threadIdx.x < g.world) {
    int const* flag = reinterpret_cast<int const*>(g.slices[g.rank] + g.layout.flags_offset) +
                      threadIdx.x;
    uint64_t const start = global_timer_ns();
    for (int spins = 0; load_acquire_int(flag) < round + 1; ++spins)
      if ((spins & 255) == 255 && global_timer_ns() - start > kTimeoutNs)
        report_and_trap(state, g.rank, "pull ready flags");
  }
  __syncthreads();
  return round & 1;
}

struct PullCombineArgs {
  Group group;
  __nv_bfloat16* out;  // [counts[rank], hidden]
  int32_t const* counts;
  int hidden_size;
  // Partials set to read, fixed by the caller (e.g. the MoE layer's parity) so a CUDA graph's
  // producer pointer and this read agree whatever the round count; the ready flags use rounds.
  int set;
};

template <int kThreads>
__global__ void __launch_bounds__(kThreads) pull_combine_kernel(PullCombineArgs const a) {
  Group const& g = a.group;
  RoundInfo round;
  load_round(g, kPullRound, a.counts, round);
  pull_wait_ready(g, round.epoch);
  int const set = a.set;
  int const self = g.rank;
  int const row_vectors = a.hidden_size / 8;
  int const row_bytes = a.hidden_size * 2;
  int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t const tasks = static_cast<int64_t>(round.counts[self]) * row_vectors;
  for (int64_t task = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; task < tasks;
       task += stride) {
    int const row = static_cast<int>(task / row_vectors);
    int const vec = static_cast<int>(task % row_vectors);
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    pull_sum_vector(g, set, round.offsets[self] + row, row_bytes, vec, acc);
    *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(a.out) +
                              static_cast<int64_t>(row) * row_bytes + vec * 16) = pack_bf16x8(acc);
  }
  finish_round(g, kPullRound, kPullFinished, round.epoch);
}

// =================================================================== bf16 pull all-gather
// The gather's reader side: every rank's rows already sit in its own mapped partials buffer of
// `set` at their gathered positions (the producer wrote them there); after the ready flags, each
// rank copies every gathered row (its own included) from the owner's buffer into `out`.
struct PullGatherArgs {
  Group group;
  __nv_bfloat16* out;  // [sum(counts), hidden]
  int32_t const* counts;
  int hidden_size;
  int set;
};

template <int kThreads>
__global__ void __launch_bounds__(kThreads) pull_gather_kernel(PullGatherArgs const a) {
  Group const& g = a.group;
  RoundInfo round;
  load_round(g, kPullRound, a.counts, round);
  pull_wait_ready(g, round.epoch);
  int const row_vectors = a.hidden_size / 8;
  int64_t const row_bytes = static_cast<int64_t>(a.hidden_size) * 2;
  int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t const base = g.layout.partials_offset + a.set * g.layout.partials_set_bytes;
  int64_t const tasks = static_cast<int64_t>(round.total) * row_vectors;
  for (int64_t task = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; task < tasks;
       task += stride) {
    int const row = static_cast<int>(task / row_vectors);
    int const vec = static_cast<int>(task % row_vectors);
    int owner = 0;
    while (owner + 1 < g.world && row >= round.offsets[owner + 1]) ++owner;
    int64_t const offset = base + row * row_bytes + vec * 16;
    *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(a.out) + row * row_bytes + vec * 16) =
        load_volatile(g.slices[owner] + offset);
  }
  finish_round(g, kPullRound, kPullFinished, round.epoch);
}

// Empty kernel of the exchange's grid shape: the graph-replay launch floor (diagnostic).
__global__ void noop_kernel() {}

}  // namespace fixed_exchange
}  // namespace vibesim
