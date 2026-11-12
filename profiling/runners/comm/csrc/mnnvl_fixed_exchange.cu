// TVM-FFI launchers for the fixed-pattern MoE exchange kernels (mnnvl_fixed_exchange.cuh).
//
// Every launcher takes a host int64 `config` tensor describing the group (see
// profiling/runners/comm/mnnvl_fixed_exchange.py `_config_words`), raw device pointers as int64,
// the block count and the CUDA stream handle, so a CUDA graph captures exactly one launch.

#include <tvm/ffi/extra/module.h>

#include <cstdint>

#include "mnnvl_fixed_exchange.cuh"
#include "tvm_ffi_utils.h"

namespace vibesim::fixed_exchange {

namespace {

constexpr int kThreads = 512;
constexpr int kUnroll = 4;

// config words: world, rank, max_rows, 8 layout offsets, then kMaxRanks slice pointers.
constexpr int kConfigWords = 3 + 8 + kMaxRanks;

Group make_group(tvm::ffi::TensorView config) {
  CHECK_CPU(config);
  CHECK_CONTIGUOUS(config);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(config.dtype()), int64_code) << "config must be int64";
  TVM_FFI_ICHECK_EQ(config.numel(), kConfigWords) << "config must hold " << kConfigWords
                                                  << " words";
  int64_t const* w = static_cast<int64_t const*>(config.data_ptr());
  Group g{};
  g.world = static_cast<int>(w[0]);
  g.rank = static_cast<int>(w[1]);
  g.max_rows = static_cast<int>(w[2]);
  TVM_FFI_ICHECK(g.world >= 2 && g.world <= kMaxRanks) << "world size " << g.world;
  TVM_FFI_ICHECK(g.rank >= 0 && g.rank < g.world) << "rank " << g.rank;
  g.layout.state_offset = w[3];
  g.layout.flags_offset = w[4];
  g.layout.dispatch_offset = w[5];
  g.layout.dispatch_set_bytes = w[6];
  g.layout.combine_offset = w[7];
  g.layout.combine_set_bytes = w[8];
  g.layout.partials_offset = w[9];
  g.layout.partials_set_bytes = w[10];
  for (int r = 0; r < kMaxRanks; ++r) {
    g.slices[r] = reinterpret_cast<uint8_t*>(w[11 + r]);
    if (r < g.world) TVM_FFI_ICHECK(g.slices[r] != nullptr) << "slice " << r << " is null";
  }
  return g;
}

DispatchPlanes make_planes(int64_t payload, int64_t scales, int64_t weights, int64_t ids) {
  DispatchPlanes p{};
  int64_t const bytes[kNumPlanes] = {payload, scales, weights, ids};
  for (int i = 0; i < kNumPlanes; ++i) {
    TVM_FFI_ICHECK(bytes[i] > 0 && bytes[i] % 16 == 0)
        << "plane " << i << " row bytes " << bytes[i] << " must be a positive multiple of 16";
    p.bytes[i] = static_cast<int>(bytes[i]);
    p.vectors[i] = static_cast<int>(bytes[i] / 16);
    p.row_vectors += p.vectors[i];
    p.row_bytes += p.bytes[i];
  }
  return p;
}

template <typename T>
T* ptr(int64_t value) {
  return reinterpret_cast<T*>(value);
}

void check_launch(char const* what) {
  cudaError_t const err = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << what << " launch failed: " << cudaGetErrorString(err);
}

}  // namespace

// Largest resident grid of each kernel: every block must be resident at once, because a block
// spinning on a peer's rows may be waiting for a peer block that is waiting in turn.
int64_t max_resident_blocks(int64_t which) {
  int device = 0, sms = 0, per_sm = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  switch (which) {
    case 0:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &per_sm, fixed_dispatch_kernel<kThreads, kUnroll>, kThreads, 0);
      break;
    case 1:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &per_sm, quant_dispatch_kernel<kThreads, kUnroll>, kThreads, 0);
      break;
    case 2:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fixed_combine_kernel<kThreads>,
                                                    kThreads, 0);
      break;
    default:
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, pull_combine_kernel<kThreads>,
                                                    kThreads, 0);
      break;
  }
  return static_cast<int64_t>(sms) * per_sm;
}

void fixed_dispatch(tvm::ffi::TensorView config, int64_t payload_bytes, int64_t scale_bytes,
                    int64_t weight_bytes, int64_t id_bytes, int64_t in_payload, int64_t in_scales,
                    int64_t in_weights, int64_t in_ids, int64_t out_payload, int64_t out_scales,
                    int64_t out_weights, int64_t out_ids, int64_t counts, int64_t blocks,
                    int64_t stream) {
  DispatchArgs a{};
  a.group = make_group(config);
  a.planes = make_planes(payload_bytes, scale_bytes, weight_bytes, id_bytes);
  int64_t const in[kNumPlanes] = {in_payload, in_scales, in_weights, in_ids};
  int64_t const out[kNumPlanes] = {out_payload, out_scales, out_weights, out_ids};
  for (int i = 0; i < kNumPlanes; ++i) {
    a.in[i] = ptr<uint8_t const>(in[i]);
    a.out[i] = ptr<uint8_t>(out[i]);
  }
  a.counts = ptr<int32_t const>(counts);
  fixed_dispatch_kernel<kThreads, kUnroll>
      <<<static_cast<int>(blocks), kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(a);
  check_launch("fixed_dispatch");
}

void quant_dispatch(tvm::ffi::TensorView config, int64_t hidden_size, int64_t top_k,
                    int64_t hidden, int64_t global_scale, int64_t topk_weights, int64_t topk_ids,
                    int64_t out_payload, int64_t out_scales, int64_t out_weights, int64_t out_ids,
                    int64_t counts, int64_t blocks, int64_t stream) {
  TVM_FFI_ICHECK(hidden_size % 256 == 0) << "hidden_size must be a multiple of 256";
  TVM_FFI_ICHECK(top_k % 4 == 0) << "top_k must be a multiple of 4";
  QuantDispatchArgs a{};
  a.group = make_group(config);
  a.planes = make_planes(hidden_size / 2, hidden_size / 16, top_k * 4, top_k * 4);
  a.hidden = ptr<__nv_bfloat16 const>(hidden);
  a.global_scale = ptr<float const>(global_scale);
  a.topk_weights = ptr<uint8_t const>(topk_weights);
  a.topk_ids = ptr<uint8_t const>(topk_ids);
  int64_t const out[kNumPlanes] = {out_payload, out_scales, out_weights, out_ids};
  for (int i = 0; i < kNumPlanes; ++i) a.out[i] = ptr<uint8_t>(out[i]);
  a.counts = ptr<int32_t const>(counts);
  a.hidden_size = static_cast<int>(hidden_size);
  quant_dispatch_kernel<kThreads, kUnroll>
      <<<static_cast<int>(blocks), kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(a);
  check_launch("quant_dispatch");
}

void fixed_combine(tvm::ffi::TensorView config, int64_t hidden_size, int64_t partials,
                   int64_t out, int64_t counts, int64_t blocks, int64_t stream) {
  TVM_FFI_ICHECK(hidden_size % 8 == 0) << "hidden_size must be a multiple of 8";
  CombineArgs a{};
  a.group = make_group(config);
  a.partials = ptr<__nv_bfloat16 const>(partials);
  a.out = ptr<__nv_bfloat16>(out);
  a.counts = ptr<int32_t const>(counts);
  a.hidden_size = static_cast<int>(hidden_size);
  fixed_combine_kernel<kThreads>
      <<<static_cast<int>(blocks), kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(a);
  check_launch("fixed_combine");
}

void pull_combine(tvm::ffi::TensorView config, int64_t hidden_size, int64_t out, int64_t counts,
                  int64_t set, int64_t blocks, int64_t stream) {
  TVM_FFI_ICHECK(set == 0 || set == 1) << "partials set must be 0 or 1";
  TVM_FFI_ICHECK(hidden_size % 8 == 0) << "hidden_size must be a multiple of 8";
  PullCombineArgs a{};
  a.group = make_group(config);
  a.out = ptr<__nv_bfloat16>(out);
  a.counts = ptr<int32_t const>(counts);
  a.hidden_size = static_cast<int>(hidden_size);
  a.set = static_cast<int>(set);
  pull_combine_kernel<kThreads>
      <<<static_cast<int>(blocks), kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(a);
  check_launch("pull_combine");
}

// Diagnostics: the empty kernel at a grid shape, and the timeline (zeros without
// -DFIXED_EXCHANGE_TIMELINE).
void noop(int64_t blocks, int64_t threads, int64_t stream) {
  noop_kernel<<<static_cast<int>(blocks), static_cast<int>(threads), 0,
                reinterpret_cast<cudaStream_t>(stream)>>>();
  check_launch("noop");
}

int64_t timeline_copy(int64_t dst, int64_t stream) {
#ifdef FIXED_EXCHANGE_TIMELINE
  auto s = reinterpret_cast<cudaStream_t>(stream);
  cudaMemcpyFromSymbolAsync(ptr<void>(dst), g_timeline, sizeof(g_timeline), 0,
                            cudaMemcpyDeviceToDevice, s);
  check_launch("timeline_copy");
  return static_cast<int64_t>(sizeof(g_timeline));
#else
  return 0;
#endif
}

void timeline_reset(int64_t stream) {
#ifdef FIXED_EXCHANGE_TIMELINE
  static unsigned long long init[kTimelineSlots][kTimelinePhases];
  for (auto& row : init) {
    for (int i = 0; i < kTimelinePhases; ++i) row[i] = 0;
    row[kEntryMin] = ~0ull;
  }
  cudaMemcpyToSymbolAsync(g_timeline, init, sizeof(init), 0, cudaMemcpyHostToDevice,
                          reinterpret_cast<cudaStream_t>(stream));
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  check_launch("timeline_reset");
#endif
}

}  // namespace vibesim::fixed_exchange

TVM_FFI_DLL_EXPORT_TYPED_FUNC(noop, vibesim::fixed_exchange::noop);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(timeline_copy, vibesim::fixed_exchange::timeline_copy);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(timeline_reset, vibesim::fixed_exchange::timeline_reset);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(max_resident_blocks,
                              vibesim::fixed_exchange::max_resident_blocks);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fixed_dispatch, vibesim::fixed_exchange::fixed_dispatch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quant_dispatch, vibesim::fixed_exchange::quant_dispatch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fixed_combine, vibesim::fixed_exchange::fixed_combine);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pull_combine, vibesim::fixed_exchange::pull_combine);
