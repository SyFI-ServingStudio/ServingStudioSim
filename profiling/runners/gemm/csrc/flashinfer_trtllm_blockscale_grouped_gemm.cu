// Launch-only binding for FlashInfer's vendored TensorRT-LLM FP8 block-scale
// GroupedWithOffset GEMM.
//
// The GEMM implementation and recipe selection remain owned by FlashInfer's
// vendored fp8_blockscale_gemm_kernel.cuh.  This binding only validates the
// physical tensor ABI and calls grouped_gemm_dispatch so L1 can profile the
// gate+up GEMM without synthetic routing or the surrounding fused-MoE kernels.

#include <tvm/ffi/extra/module.h>

#include <cstdint>
#include <limits>

#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh"
#include "tvm_ffi_utils.h"

namespace vibesim::fp8_blockscale_grouped_gemm {

namespace blockscale = tensorrt_llm::kernels::fp8_blockscale_gemm;

void run(tvm::ffi::TensorView activation, tvm::ffi::TensorView activation_scales,
         tvm::ffi::TensorView weight, tvm::ffi::TensorView weight_scales,
         tvm::ffi::TensorView output, tvm::ffi::TensorView problem_m_offsets,
         int64_t expected_m, int64_t max_shape_m, int64_t max_shape_m_padded,
         int64_t shape_n, int64_t shape_k, int64_t num_problems) {
  CHECK_INPUT(activation);
  CHECK_INPUT(activation_scales);
  CHECK_INPUT(weight);
  CHECK_INPUT(weight_scales);
  CHECK_INPUT(output);
  CHECK_INPUT(problem_m_offsets);

  TVM_FFI_ICHECK_EQ(activation.ndim(), 2)
      << "activation must be physical [max_shape_m, K]";
  TVM_FFI_ICHECK_EQ(activation_scales.ndim(), 2)
      << "activation_scales must be physical [K/128, max_shape_m_padded]";
  TVM_FFI_ICHECK_EQ(weight.ndim(), 3) << "weight must be physical [E, N, K]";
  TVM_FFI_ICHECK_EQ(weight_scales.ndim(), 3)
      << "weight_scales must be physical [E, N/128, K/128]";
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output must be physical [max_shape_m, N]";
  TVM_FFI_ICHECK_EQ(problem_m_offsets.ndim(), 1)
      << "problem_m_offsets must be a 1D tensor";

  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(activation.dtype()), float8_e4m3fn_code)
      << "activation must have FP8 E4M3 dtype";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(activation_scales.dtype()), float32_code)
      << "activation_scales must have FP32 dtype";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(weight.dtype()), float8_e4m3fn_code)
      << "weight must have FP8 E4M3 dtype";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(weight_scales.dtype()), float32_code)
      << "weight_scales must have FP32 dtype";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), bfloat16_code)
      << "output must have BF16 dtype";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(problem_m_offsets.dtype()), int64_code)
      << "problem_m_offsets must have int64 dtype";

  TVM_FFI_ICHECK_GT(expected_m, 0) << "expected_m must be positive";
  TVM_FFI_ICHECK_GT(max_shape_m, 0) << "max_shape_m must be positive";
  TVM_FFI_ICHECK_LE(expected_m, max_shape_m) << "expected_m cannot exceed max_shape_m";
  TVM_FFI_ICHECK_EQ(max_shape_m % 4, 0) << "max_shape_m must be aligned to 4 rows";
  TVM_FFI_ICHECK_GT(max_shape_m_padded, 0) << "max_shape_m_padded must be positive";
  TVM_FFI_ICHECK_GT(shape_n, 0) << "N must be positive";
  TVM_FFI_ICHECK_GT(shape_k, 0) << "K must be positive";
  TVM_FFI_ICHECK_EQ(shape_n % 128, 0) << "N must be divisible by 128";
  TVM_FFI_ICHECK_EQ(shape_k % 128, 0) << "K must be divisible by 128";
  TVM_FFI_ICHECK_GT(num_problems, 0) << "num_problems must be positive";
  TVM_FFI_ICHECK_LE(max_shape_m, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "max_shape_m exceeds the DeepGEMM uint32 ABI";
  TVM_FFI_ICHECK_LE(expected_m, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "expected_m exceeds the DeepGEMM uint32 ABI";
  TVM_FFI_ICHECK_LE(max_shape_m_padded,
                    static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "max_shape_m_padded exceeds the DeepGEMM uint32 ABI";
  TVM_FFI_ICHECK_LE(shape_n, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "N exceeds the DeepGEMM uint32 ABI";
  TVM_FFI_ICHECK_LE(shape_k, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "K exceeds the DeepGEMM uint32 ABI";
  TVM_FFI_ICHECK_LE(num_problems, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "num_problems exceeds the DeepGEMM uint32 ABI";

  int64_t const k_blocks = shape_k / 128;
  int64_t const n_blocks = shape_n / 128;
  int64_t const required_padded_rows =
      deep_gemm::compute_padded_offset(max_shape_m, num_problems);
  TVM_FFI_ICHECK_EQ(max_shape_m_padded, required_padded_rows)
      << "max_shape_m_padded does not match DeepGEMM grouped padding";

  TVM_FFI_ICHECK_EQ(activation.size(0), max_shape_m)
      << "activation row capacity does not match max_shape_m";
  TVM_FFI_ICHECK_EQ(activation.size(1), shape_k) << "activation K mismatch";
  TVM_FFI_ICHECK_EQ(activation_scales.size(0), k_blocks)
      << "activation scale K-block mismatch";
  TVM_FFI_ICHECK_EQ(activation_scales.size(1), max_shape_m_padded)
      << "activation scale padded-row mismatch";
  TVM_FFI_ICHECK_EQ(weight.size(0), num_problems) << "weight expert count mismatch";
  TVM_FFI_ICHECK_EQ(weight.size(1), shape_n) << "weight N mismatch";
  TVM_FFI_ICHECK_EQ(weight.size(2), shape_k) << "weight K mismatch";
  TVM_FFI_ICHECK_EQ(weight_scales.size(0), num_problems)
      << "weight-scale expert count mismatch";
  TVM_FFI_ICHECK_EQ(weight_scales.size(1), n_blocks)
      << "weight-scale N-block mismatch";
  TVM_FFI_ICHECK_EQ(weight_scales.size(2), k_blocks)
      << "weight-scale K-block mismatch";
  TVM_FFI_ICHECK_EQ(output.size(0), max_shape_m)
      << "output row capacity does not match max_shape_m";
  TVM_FFI_ICHECK_EQ(output.size(1), shape_n) << "output N mismatch";
  TVM_FFI_ICHECK_EQ(problem_m_offsets.size(0), num_problems + 1)
      << "problem_m_offsets must contain E + 1 boundaries";

  blockscale::grouped_gemm_dispatch(
      static_cast<__nv_fp8_e4m3*>(activation.data_ptr()),
      static_cast<__nv_fp8_e4m3*>(weight.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()), static_cast<uint32_t>(num_problems),
      static_cast<int64_t const*>(problem_m_offsets.data_ptr()),
      static_cast<uint32_t>(expected_m), static_cast<uint32_t>(max_shape_m),
      static_cast<uint32_t>(max_shape_m_padded), static_cast<uint32_t>(shape_n),
      static_cast<uint32_t>(shape_k), static_cast<float*>(activation_scales.data_ptr()),
      static_cast<float*>(weight_scales.data_ptr()), get_stream(activation.device()));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess)
      << "GroupedWithOffset FP8 block-scale GEMM launch failed";
}

}  // namespace vibesim::fp8_blockscale_grouped_gemm

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_fp8_blockscale_grouped_gemm,
                              vibesim::fp8_blockscale_grouped_gemm::run);
