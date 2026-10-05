//! DeepSeek-V4.1 FlashMLA mega attention over one mixed batch.
//!
//! vLLM's `DeepseekV4MegaAttnAttention` (fork `models/deepseek_v41/nvidia/
//! flash_mla_mega_attn.py:361` decode, `:465` prefill) splits a scheduled batch
//! into its decode rows and its prefill requests and launches one segment
//! method for each: the capture's mixed iterations carry two
//! `fused_norm_rope_attn_rope_cast_fwd` launches per layer, decode-only ones
//! carry one. Both segments share the layer's caches and the op boundary, so
//! they are one compound op with two fixed leaves (prefill, decode) over the
//! `compressed_sparse_mla_rope_cast` kind, which is keyed by `mode`.
//!
//! The prefill leaf is the whole chunk loop of the fork (compressed-cache
//! dequant/gather, SWA gather, `combine_topk_swa_indices`, one fused launch per
//! chunk); the decode leaf is one launch over every decode row. The global
//! top-k remap (`_compute_global_topk_indices_and_lens`) that precedes the
//! decode launch has no L1 kind and folds into this op (plan decision).

use std::sync::Arc;

use crate::timing::bridge::DType;
use crate::timing::kernels::{
    CompressedSparseMlaRopeCastKernel, CompressedSparseMlaRopeCastKernelConfig, CompressedSparseMlaRopeCastKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
};

/// Static identity of one layer's mega-attention call.
#[derive(Clone, Debug)]
pub struct DeepseekV41MegaAttnOpConfig {
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// 0 (SWA only), 1 or 2.
    pub compress_ratio: u32,
    pub window_size: u32,
    pub index_topk: u32,
    /// The kernel's padded Q head count (64 at TP4), not the live heads.
    pub padded_heads: Dim,
    pub head_dim: Dim,
    pub rope_dim: Dim,
    pub max_model_len: u32,
    pub max_num_batched_tokens: u32,
    pub prefill_chunk_size: u32,
    pub q_dtype: DType,
    pub swa_cache_format: String,
    /// `nvfp4` for the compressed cache; ignored (`none`) at ratio 0.
    pub compressed_cache_format: String,
    pub output_dtype: DType,
}

/// One rank's batch, split the way the fork splits it.
#[derive(Clone, Debug, Default)]
pub struct DeepseekV41MegaAttnOpInput {
    /// `(query_len, context_len)` per prefill request, context including the
    /// query.
    pub prefill_query_context_pairs: Vec<(u32, u32)>,
    /// Resident KV length before each one-token decode row.
    pub decode_kv_lens: Vec<u32>,
}

pub struct DeepseekV41MegaAttnOp {
    pub name: String,
    pub prefill: Arc<CompressedSparseMlaRopeCastKernel>,
    pub decode: Arc<CompressedSparseMlaRopeCastKernel>,
}

impl DeepseekV41MegaAttnOp {
    pub fn build(
        name: String,
        cfg: DeepseekV41MegaAttnOpConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let prefill_name = format!("{name}.prefill");
        let decode_name = format!("{name}.decode");
        Ok(Self {
            prefill: Arc::new(CompressedSparseMlaRopeCastKernel::build(
                prefill_name,
                kernel_config(&cfg, "prefill"),
                bridge,
            )?),
            decode: Arc::new(CompressedSparseMlaRopeCastKernel::build(
                decode_name,
                kernel_config(&cfg, "decode"),
                bridge,
            )?),
            name,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            builder.leaf(
                format!("{}.prefill", self.name),
                self.prefill.kind(),
                self.prefill.describe_config(),
            ),
            builder.leaf(
                format!("{}.decode", self.name),
                self.decode.kind(),
                self.decode.describe_config(),
            ),
        ])
    }

    pub fn eval(&self, input: &DeepseekV41MegaAttnOpInput, ev: &mut Evaluator) {
        let (prefill, decode) = split_input(input);
        push_segment(&self.prefill, prefill, ev);
        push_segment(&self.decode, decode, ev);
    }
}

fn push_segment(
    kernel: &CompressedSparseMlaRopeCastKernel,
    segment: CompressedSparseMlaRopeCastKernelInput,
    ev: &mut Evaluator,
) {
    let metrics = if segment.query_context_pairs.is_empty() {
        LeafMetrics::ZERO
    } else {
        kernel.eval(&segment)
    };
    ev.push(metrics, || segment.into());
}

/// Pure config expansion: the two leaves differ only in `mode`.
pub(crate) fn kernel_config(
    cfg: &DeepseekV41MegaAttnOpConfig,
    mode: &str,
) -> CompressedSparseMlaRopeCastKernelConfig {
    CompressedSparseMlaRopeCastKernelConfig {
        backends: cfg.backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        mode: mode.to_string(),
        compress_ratio: cfg.compress_ratio,
        window_size: cfg.window_size,
        index_topk: cfg.index_topk,
        num_heads: cfg.padded_heads.clone(),
        head_dim: cfg.head_dim.clone(),
        rope_dim: cfg.rope_dim.clone(),
        max_model_len: cfg.max_model_len,
        max_num_batched_tokens: cfg.max_num_batched_tokens,
        prefill_chunk_size: cfg.prefill_chunk_size,
        q_dtype: cfg.q_dtype,
        swa_cache_format: cfg.swa_cache_format.clone(),
        compressed_cache_format: if cfg.compress_ratio == 0 {
            "none".to_string()
        } else {
            cfg.compressed_cache_format.clone()
        },
        output_dtype: cfg.output_dtype,
    }
}

/// Pure input split: prefill requests pass through; each decode row becomes
/// its own `(1, kv + 1)` request, as the decode segment flattens them.
pub(crate) fn split_input(
    input: &DeepseekV41MegaAttnOpInput,
) -> (CompressedSparseMlaRopeCastKernelInput, CompressedSparseMlaRopeCastKernelInput) {
    let prefill = CompressedSparseMlaRopeCastKernelInput {
        query_context_pairs: input.prefill_query_context_pairs.clone(),
    };
    let decode = CompressedSparseMlaRopeCastKernelInput {
        query_context_pairs: input
            .decode_kv_lens
            .iter()
            .map(|&kv| {
                (
                    1,
                    kv.checked_add(1).expect("decode context must fit u32"),
                )
            })
            .collect(),
    };
    (prefill, decode)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(ratio: u32) -> DeepseekV41MegaAttnOpConfig {
        DeepseekV41MegaAttnOpConfig {
            backends: vec!["flashmla_mega"],
            gpu_name: "NVIDIA B200".into(),
            compress_ratio: ratio,
            window_size: 128,
            index_topk: 512,
            padded_heads: 64.into(),
            head_dim: 512.into(),
            rope_dim: 64.into(),
            max_model_len: 131_072,
            max_num_batched_tokens: 2048,
            prefill_chunk_size: 4,
            q_dtype: DType::Bf16,
            swa_cache_format: "mxfp8".into(),
            compressed_cache_format: "nvfp4".into(),
            output_dtype: DType::Fp8E4m3,
        }
    }

    #[test]
    fn mixed_batch_splits_into_prefill_requests_and_one_row_decode_requests() {
        let (prefill, decode) = split_input(&DeepseekV41MegaAttnOpInput {
            prefill_query_context_pairs: vec![(128, 128), (7, 300)],
            decode_kv_lens: vec![1303, 4244],
        });
        assert_eq!(prefill.query_context_pairs, vec![(128, 128), (7, 300)]);
        assert_eq!(decode.query_context_pairs, vec![(1, 1304), (1, 4245)]);
    }

    #[test]
    fn ratio_zero_drops_the_compressed_cache_and_modes_differ_only_in_mode() {
        let decode = kernel_config(&cfg(0), "decode");
        let prefill = kernel_config(&cfg(0), "prefill");
        assert_eq!(decode.compressed_cache_format, "none");
        assert_eq!((decode.mode.as_str(), prefill.mode.as_str()), ("decode", "prefill"));
        assert_eq!(kernel_config(&cfg(2), "decode").compressed_cache_format, "nvfp4");
        assert_eq!(decode.num_heads.get(), 64);
    }
}
