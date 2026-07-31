#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

void persistent_topk(const torch::stable::Tensor& logits,
                     const torch::stable::Tensor& lengths,
                     torch::stable::Tensor& output,
                     torch::stable::Tensor& workspace, int64_t k,
                     int64_t max_seq_len);

// Torch 2.10 compatibility shim: the pinned kernel implementation is unchanged,
// but is registered in a private namespace so it can coexist with vLLM's wheel.
STABLE_TORCH_LIBRARY_FRAGMENT(_C_pinned_topk, ops) {
  ops.def(
      "persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
      "Tensor workspace, int k, int max_seq_len) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_pinned_topk, CUDA, ops) {
  ops.impl("persistent_topk", TORCH_BOX(&persistent_topk));
}
