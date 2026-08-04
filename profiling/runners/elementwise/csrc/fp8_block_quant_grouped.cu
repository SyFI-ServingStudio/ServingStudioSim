// Exact launch-only binding for TensorRT-LLM's grouped BF16 -> FP8 quantizer.
//
// The kernel body remains owned by the FlashInfer-vendored TensorRT-LLM
// header.  This file only reproduces the launch policy from
// fp8_grouped_gemm_run so L1 can measure the quantization launch without also
// launching the grouped GEMM that consumes its output.

#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh"
#include "tvm_ffi_utils.h"

namespace grouped_quant {

namespace kernels = tensorrt_llm::kernels::fp8_blockscale_gemm;

void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output, tvm::ffi::TensorView scales,
         tvm::ffi::TensorView problem_m_offsets, int64_t num_problems) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(scales);
  CHECK_INPUT(problem_m_offsets);

  TVM_FFI_ICHECK_EQ(input.ndim(), 2) << "input must be a 2D [M, K] tensor";
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output must be a 2D [M, K] tensor";
  TVM_FFI_ICHECK_EQ(scales.ndim(), 2) << "scales must be a 2D [K/128, padded_M] tensor";
  TVM_FFI_ICHECK_EQ(problem_m_offsets.ndim(), 1)
      << "problem_m_offsets must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(input.dtype()), bfloat16_code)
      << "input must have BF16 dtype";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), float8_e4m3fn_code)
      << "output must have FP8 E4M3 dtype";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(scales.dtype()), float32_code)
      << "scales must have FP32 dtype";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(problem_m_offsets.dtype()), int64_code)
      << "problem_m_offsets must have int64 dtype";
  TVM_FFI_ICHECK_GT(num_problems, 0) << "num_problems must be positive";
  TVM_FFI_ICHECK_LE(num_problems, static_cast<int64_t>(std::numeric_limits<int>::max()))
      << "num_problems is too large";
  TVM_FFI_ICHECK_EQ(problem_m_offsets.size(0), num_problems + 1)
      << "problem_m_offsets must contain num_problems + 1 boundaries";

  int64_t const shape_m = input.size(0);
  int64_t const shape_k = input.size(1);
  TVM_FFI_ICHECK_GT(shape_m, 0) << "M must be positive";
  TVM_FFI_ICHECK_GT(shape_k, 0) << "K must be positive";
  TVM_FFI_ICHECK_EQ(shape_k % 128, 0) << "K must be divisible by 128";
  TVM_FFI_ICHECK_EQ(output.size(0), shape_m) << "output M does not match input";
  TVM_FFI_ICHECK_EQ(output.size(1), shape_k) << "output K does not match input";

  int64_t const scales_dim_x = shape_k / 128;
  int64_t const scale_leading_dim =
      deep_gemm::compute_padded_offset(shape_m, num_problems);
  TVM_FFI_ICHECK_EQ(scales.size(0), scales_dim_x)
      << "scale K-block dimension does not match input K";
  TVM_FFI_ICHECK_GE(scales.size(1), scale_leading_dim)
      << "scale row dimension is smaller than the grouped padded layout";

  uint32_t scale_dim_x_mul;
  uint32_t scale_dim_x_shr;
  kernel_utils::find_divisor(scale_dim_x_mul, scale_dim_x_shr,
                             static_cast<int>(scales_dim_x));

  int device = 0;
  int num_device_sms = 0;
  TVM_FFI_ICHECK_EQ(cudaGetDevice(&device), cudaSuccess) << "cudaGetDevice failed";
  TVM_FFI_ICHECK_EQ(
      cudaDeviceGetAttribute(&num_device_sms, cudaDevAttrMultiProcessorCount, device), cudaSuccess)
      << "failed to query the device SM count";

  constexpr int NumThreads = 256;
  int const num_blocks = static_cast<int>(std::min<int64_t>(
      num_device_sms, (shape_m * scales_dim_x + NumThreads / 32 - 1) /
                          (NumThreads / 32)));
  int const dynamic_smem_size = static_cast<int>(num_problems * sizeof(int64_t));

  bool const use_binary_search =
      static_cast<double>(shape_m) * scales_dim_x /
          static_cast<double>(NumThreads * num_blocks / 32) <=
      static_cast<double>(num_problems) / std::log2(static_cast<double>(num_problems));
  auto kernel = use_binary_search
                    ? kernels::scale_1x128_kernel<true, __nv_bfloat16, __nv_fp8_e4m3>
                    : kernels::scale_1x128_kernel<false, __nv_bfloat16, __nv_fp8_e4m3>;
  TVM_FFI_ICHECK_EQ(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_size),
      cudaSuccess)
      << "failed to configure dynamic shared memory";

  kernel<<<num_blocks, NumThreads, dynamic_smem_size, get_stream(input.device())>>>(
      static_cast<__nv_fp8_e4m3*>(output.data_ptr()), static_cast<float*>(scales.data_ptr()),
      static_cast<__nv_bfloat16 const*>(input.data_ptr()),
      static_cast<int64_t const*>(problem_m_offsets.data_ptr()), static_cast<int>(num_problems),
      static_cast<int>(shape_k), scale_leading_dim, scale_dim_x_mul, scale_dim_x_shr);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess) << "scale_1x128_kernel launch failed";
}

}  // namespace grouped_quant

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_grouped_fp8_block_quant, grouped_quant::run);
