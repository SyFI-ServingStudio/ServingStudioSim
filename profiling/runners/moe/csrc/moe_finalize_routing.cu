// Launch-only binding for FlashInfer's vendored TensorRT-LLM MoE finalize.
//
// The kernel implementation remains in cutlass_fused_moe_kernels.cuh. This
// binding only validates the physical tensor ABI and invokes the production
// launcher in the same non-all-to-all mode used before vLLM's EP all-reduce.

#include <tvm/ffi/extra/module.h>

#include <cstdint>

#include "fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh"
#include "tvm_ffi_utils.h"

namespace vibesim::moe_finalize_routing {

namespace cutlass_moe = tensorrt_llm::kernels::cutlass_kernels;

void run(tvm::ffi::TensorView expanded_permuted_rows,
         tvm::ffi::TensorView reduced_unpermuted_output,
         tvm::ffi::TensorView final_scales,
         tvm::ffi::TensorView unpermuted_row_to_permuted_row,
         tvm::ffi::TensorView token_selected_experts, int64_t token_count,
         int64_t hidden_size, int64_t top_k, int64_t num_experts_per_rank) {
  CHECK_INPUT(expanded_permuted_rows);
  CHECK_INPUT(reduced_unpermuted_output);
  CHECK_INPUT(final_scales);
  CHECK_INPUT(unpermuted_row_to_permuted_row);
  CHECK_INPUT(token_selected_experts);

  TVM_FFI_ICHECK_EQ(expanded_permuted_rows.ndim(), 2)
      << "expanded_permuted_rows must be [token_count * top_k, hidden_size]";
  TVM_FFI_ICHECK_EQ(reduced_unpermuted_output.ndim(), 2)
      << "reduced_unpermuted_output must be [token_count, hidden_size]";
  TVM_FFI_ICHECK_EQ(final_scales.ndim(), 1) << "final_scales must be 1D";
  TVM_FFI_ICHECK_EQ(unpermuted_row_to_permuted_row.ndim(), 1)
      << "unpermuted_row_to_permuted_row must be 1D";
  TVM_FFI_ICHECK_EQ(token_selected_experts.ndim(), 1)
      << "token_selected_experts must be 1D";

  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(expanded_permuted_rows.dtype()), bfloat16_code)
      << "expanded_permuted_rows must be BF16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(reduced_unpermuted_output.dtype()), bfloat16_code)
      << "reduced_unpermuted_output must be BF16";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(final_scales.dtype()), float32_code)
      << "final_scales must be FP32";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(unpermuted_row_to_permuted_row.dtype()), int32_code)
      << "unpermuted_row_to_permuted_row must be int32";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(token_selected_experts.dtype()), int32_code)
      << "token_selected_experts must be int32";

  TVM_FFI_ICHECK_GT(token_count, 0) << "token_count must be positive";
  TVM_FFI_ICHECK_GT(hidden_size, 0) << "hidden_size must be positive";
  TVM_FFI_ICHECK_EQ(hidden_size % 8, 0)
      << "hidden_size must preserve the launcher's 128-bit BF16 vectors";
  TVM_FFI_ICHECK_GT(top_k, 0) << "top_k must be positive";
  TVM_FFI_ICHECK_GT(num_experts_per_rank, 0)
      << "num_experts_per_rank must be positive";

  int64_t const routed_capacity = token_count * top_k;
  TVM_FFI_ICHECK_EQ(expanded_permuted_rows.size(0), routed_capacity)
      << "expanded row count mismatch";
  TVM_FFI_ICHECK_EQ(expanded_permuted_rows.size(1), hidden_size)
      << "expanded hidden size mismatch";
  TVM_FFI_ICHECK_EQ(reduced_unpermuted_output.size(0), token_count)
      << "output token count mismatch";
  TVM_FFI_ICHECK_EQ(reduced_unpermuted_output.size(1), hidden_size)
      << "output hidden size mismatch";
  TVM_FFI_ICHECK_EQ(final_scales.size(0), routed_capacity)
      << "final_scales length mismatch";
  TVM_FFI_ICHECK_EQ(unpermuted_row_to_permuted_row.size(0), routed_capacity)
      << "unpermute map length mismatch";
  TVM_FFI_ICHECK_EQ(token_selected_experts.size(0), routed_capacity)
      << "selected-expert length mismatch";

  // Rank zero owns expert ids [0, num_experts_per_rank). ep_size=2 makes the
  // synthetic out-of-range id a legal remote expert while enable_alltoall=false
  // selects finalizeMoeRoutingKernel, matching the profiled vLLM path.
  cutlass_moe::MOEParallelismConfig parallelism_config(1, 0, 2, 0);
  cutlass_moe::finalizeMoeRoutingKernelLauncher<__nv_bfloat16, __nv_bfloat16,
                                                __nv_bfloat16>(
      static_cast<__nv_bfloat16 const*>(expanded_permuted_rows.data_ptr()),
      static_cast<__nv_bfloat16*>(reduced_unpermuted_output.data_ptr()),
      /*bias=*/nullptr, static_cast<float const*>(final_scales.data_ptr()),
      static_cast<int const*>(unpermuted_row_to_permuted_row.data_ptr()),
      /*permuted_row_to_unpermuted_row=*/nullptr,
      static_cast<int const*>(token_selected_experts.data_ptr()),
      /*expert_first_token_offset=*/nullptr, token_count, hidden_size, hidden_size,
      top_k, num_experts_per_rank, parallelism_config,
      /*enable_alltoall=*/false,
      /*enable_pdl=*/true, get_stream(expanded_permuted_rows.device()));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess)
      << "finalizeMoeRoutingKernel launch failed";
}

}  // namespace vibesim::moe_finalize_routing

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_moe_finalize_routing,
                              vibesim::moe_finalize_routing::run);
