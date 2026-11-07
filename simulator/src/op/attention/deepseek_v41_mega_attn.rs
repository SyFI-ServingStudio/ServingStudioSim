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
//!
//! Layers vLLM marks `_use_swapab` (`_swapab_decode_supported`: a compressed
//! layer, 16 or 32 live heads, tiles that fit one cluster) get a third leaf,
//! `decode_swapab`: `_forward_decode_mega` sends decode segments of at most
//! `_SWAPAB_MAX_TOKENS` rows to the swap-AB kernel instead of the mega kernel
//! (`flash_mla_mega_attn.py:442-470`). The gate reads the segment's launch
//! rows (`q.shape[0]`), which a FULL CUDA graph fixes at its padded size; the
//! cost reads the live rows, as the mega decode leaf does. The other gate
//! terms are deployment facts this model takes as on: SM100, a CUDA 13.2+
//! CuTe DSL, `VLLM_BATCH_INVARIANT` off, and an SWA list as wide as the window
//! (only DSpark's non-causal draft steps widen it, and draft layers are ratio
//! 0, which never qualify).

use std::sync::Arc;

use crate::timing::bridge::DType;
use crate::timing::kernels::{
    swapab_layer, CompressedSparseMlaRopeCastKernel, CompressedSparseMlaRopeCastKernelConfig,
    CompressedSparseMlaRopeCastKernelInput, SWAPAB_MAX_TOKENS,
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
    /// Live Q heads per rank (16 at TP4): the swap-AB kernel's head count.
    pub live_heads: u32,
    /// The swap-AB decode backends; empty keeps every decode on the mega kernel.
    pub swapab_backends: Vec<&'static str>,
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
    /// Rows the decode segment launches over (`q.shape[0]`): the FULL graph's
    /// padded rows inside one, else the live decode rows. `None` = live rows.
    pub decode_launch_rows: Option<u32>,
}

pub struct DeepseekV41MegaAttnOp {
    pub name: String,
    pub prefill: Arc<CompressedSparseMlaRopeCastKernel>,
    pub decode: Arc<CompressedSparseMlaRopeCastKernel>,
    /// `Some` on layers vLLM marks `_use_swapab`.
    pub decode_swapab: Option<Arc<CompressedSparseMlaRopeCastKernel>>,
}

impl DeepseekV41MegaAttnOp {
    pub fn build(
        name: String,
        cfg: DeepseekV41MegaAttnOpConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let prefill_name = format!("{name}.prefill");
        let decode_name = format!("{name}.decode");
        let decode_swapab = swapab_config(&cfg)
            .map(|config| {
                CompressedSparseMlaRopeCastKernel::build(
                    format!("{name}.decode_swapab"),
                    config,
                    bridge,
                )
                .map(Arc::new)
            })
            .transpose()?;
        Ok(Self {
            decode_swapab,
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
        let mut leaves = vec![
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
        ];
        if let Some(swapab) = &self.decode_swapab {
            leaves.push(builder.leaf(
                format!("{}.decode_swapab", self.name),
                swapab.kind(),
                swapab.describe_config(),
            ));
        }
        CostNode::Sum(leaves)
    }

    pub fn eval(&self, input: &DeepseekV41MegaAttnOpInput, ev: &mut Evaluator) {
        let (prefill, decode) = split_input(input);
        push_segment(&self.prefill, prefill, ev);
        let Some(swapab) = &self.decode_swapab else {
            push_segment(&self.decode, decode, ev);
            return;
        };
        let empty = CompressedSparseMlaRopeCastKernelInput {
            query_context_pairs: Vec::new(),
        };
        if takes_swapab(input) {
            push_segment(&self.decode, empty, ev);
            push_segment(swapab, decode, ev);
        } else {
            push_segment(&self.decode, decode, ev);
            push_segment(swapab, empty, ev);
        }
    }
}

/// `_forward_decode_mega`'s token gate: `q.shape[0] <= _SWAPAB_MAX_TOKENS`.
fn takes_swapab(input: &DeepseekV41MegaAttnOpInput) -> bool {
    let live = input.decode_kv_lens.len() as u32;
    input.decode_launch_rows.unwrap_or(live) <= SWAPAB_MAX_TOKENS
}

/// The swap-AB decode leaf's config on layers vLLM marks `_use_swapab`.
pub(crate) fn swapab_config(
    cfg: &DeepseekV41MegaAttnOpConfig,
) -> Option<CompressedSparseMlaRopeCastKernelConfig> {
    let qualifies = !cfg.swapab_backends.is_empty()
        && swapab_layer(
            cfg.compress_ratio,
            cfg.live_heads,
            cfg.window_size,
            cfg.index_topk,
        );
    qualifies.then(|| CompressedSparseMlaRopeCastKernelConfig {
        backends: cfg.swapab_backends.clone(),
        num_heads: cfg.live_heads.into(),
        ..kernel_config(cfg, "decode")
    })
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
            live_heads: 16,
            swapab_backends: vec!["swapab_cutedsl"],
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
            decode_launch_rows: None,
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

    /// Catches a swap-AB leaf on a layer vLLM keeps on the mega kernel
    /// (ratio 0, padded-only heads, no backend), or one that is not the
    /// decode config at the live head count.
    #[test]
    fn swapab_leaf_exists_only_where_vllm_marks_use_swapab() {
        let swapab = swapab_config(&cfg(2)).unwrap();
        assert_eq!(
            (swapab.mode.as_str(), swapab.num_heads.get()),
            ("decode", 16)
        );
        assert_eq!(swapab.backends, vec!["swapab_cutedsl"]);
        assert!(swapab_config(&cfg(0)).is_none());
        let dp = DeepseekV41MegaAttnOpConfig {
            live_heads: 64,
            ..cfg(2)
        };
        assert!(swapab_config(&dp).is_none());
        let off = DeepseekV41MegaAttnOpConfig {
            swapab_backends: vec![],
            ..cfg(2)
        };
        assert!(swapab_config(&off).is_none());
    }

    /// Catches a token gate on the live rows when the FULL graph launches
    /// padded rows, or one off by one at `_SWAPAB_MAX_TOKENS`.
    #[test]
    fn token_gate_reads_the_launch_rows() {
        let input = |live: usize, launch: Option<u32>| DeepseekV41MegaAttnOpInput {
            prefill_query_context_pairs: vec![(100, 100)],
            decode_kv_lens: vec![500; live],
            decode_launch_rows: launch,
        };
        assert!(takes_swapab(&input(16, None)));
        assert!(!takes_swapab(&input(17, None)));
        assert!(takes_swapab(&input(12, Some(16))));
        assert!(!takes_swapab(&input(12, Some(24))));
    }
}
