//! GLM-5.3-Flash sparse MLA attention behind the pooled (kpool) indexer.
//!
//! One logical attention call is four launches in the vLLM fork: the MLA
//! latent cache append, the request-local to global index remap, and the
//! FlashInfer TRTLLM-gen token-sparse kernel, which serves prefill and decode
//! rows alike (the fork has no dense-MHA prefill backend for this model). The
//! compound op keeps the four as fixed leaves -- `mla_cache_append`,
//! `index_remap`, `prefill`, `decode` -- so a mixed batch prices its prefill
//! and decode rows against their own measured valid-count shapes.
//!
//! The pooled indexer keeps `index_topk` pools of `index_kpool` tokens plus the
//! trailing partial pool, so a row at causal position `n` sees
//! `min(n, index_topk + n mod index_kpool)` slots, never more than
//! `index_topk + index_kpool - 1` (2051), inside a `selected_k` = 2176 wide
//! page table. That law drives the remap's valid counts exactly.
//!
//! * Decode rows use `ValidCountsPattern::PooledUniformFull`, collapsed to the
//!   batch's MAX context (the measured pooled grid is uniform in context).
//! * Prefill rows use `ValidCountsPattern::CausalTail` per request, fanned into
//!   one slot. Its ramp is exact while a request's context is at most 2051; past
//!   that the measured causal encoding caps at `selected_k` (2176) instead of
//!   2051, overstating saturated rows by at most 6%. The Python encoder only
//!   accepts a clipped ramp whose cap equals `selected_k`, so an exact pooled
//!   prefill shape would need a new L1 encoding; this op does not add one.

use std::sync::Arc;

use crate::timing::bridge::DType;
use crate::timing::kernels::{
    DsaSparseIndexRemapKernel, DsaSparseIndexRemapKernelConfig, DsaSparseIndexRemapKernelInput,
    DsaSparseMlaAttentionKernel, DsaSparseMlaAttentionKernelConfig,
    DsaSparseMlaAttentionKernelInput, MlaCacheAppendKernel, MlaCacheAppendKernelConfig,
    MlaCacheAppendKernelInput, ValidCountsPattern,
};
use crate::timing::slot_input::{DsaSparseMlaDecodeLog, DsaSparseMlaPrefillLog};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
};

const OP_KIND: &str = "glm53_kpool_sparse_mla";
const SLOT_SUFFIXES: [&str; 4] = ["mla_cache_append", "index_remap", "prefill", "decode"];
/// vLLM's remap wrapper handles at most this many requests per launch.
const MAX_REMAP_REQUESTS: usize = 256;

/// Static identity expanded into the four L1 kernel configurations.
#[derive(Clone, Debug)]
pub struct Glm53KpoolSparseMlaConfig {
    pub sparse_attention_backends: Vec<&'static str>,
    pub mla_cache_append_backends: Vec<&'static str>,
    pub index_remap_backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Query heads on this rank.
    pub num_heads: Dim,
    pub latent_dim: Dim,
    /// Rope width of the latent cache; GLM-5.3-Flash has none.
    pub rope_dim: Dim,
    /// Page-table width: `round_up(index_topk + index_kpool - 1, 128)`.
    pub selected_k: u32,
    pub index_topk: u32,
    pub index_kpool: u32,
    pub softmax_scale_denominator: u32,
    /// Dtype of the latent rows handed to the cache append.
    pub activation_dtype: DType,
    pub q_dtype: DType,
    pub cache_dtype: DType,
    pub output_dtype: DType,
    pub index_dtype: String,
    pub index_distribution: String,
    pub cache_layout: String,
    pub mla_cache_block_size: u32,
    pub mla_cache_format: String,
    pub page_table_mapping: String,
    pub max_model_len: u32,
}

/// Per-rank inputs for one logical sparse MLA call.
#[derive(Clone, Debug, Default)]
pub struct Glm53KpoolSparseMlaInput {
    /// `(append, prefix + append)` per prefill request.
    pub prefill_query_cache_pairs: Vec<(u32, u32)>,
    /// One KV length per decode request, including the token being decoded.
    pub decode_context_lens: Vec<u32>,
}

pub struct Glm53KpoolSparseMlaOp {
    pub name: String,
    pub mla_cache_append: Arc<MlaCacheAppendKernel>,
    pub index_remap: Arc<DsaSparseIndexRemapKernel>,
    pub prefill: Arc<DsaSparseMlaAttentionKernel>,
    pub decode: Arc<DsaSparseMlaAttentionKernel>,
    index_topk: u32,
    index_kpool: u32,
}

/// The four L1 configurations one op identity expands into.
#[derive(Clone, Debug)]
pub struct Glm53KpoolSparseMlaResolved {
    pub mla_cache_append: MlaCacheAppendKernelConfig,
    pub index_remap: DsaSparseIndexRemapKernelConfig,
    pub prefill: DsaSparseMlaAttentionKernelConfig,
    pub decode: DsaSparseMlaAttentionKernelConfig,
}

impl Glm53KpoolSparseMlaOp {
    pub fn resolve_config(
        cfg: &Glm53KpoolSparseMlaConfig,
    ) -> Result<Glm53KpoolSparseMlaResolved, BuildError> {
        if cfg.index_topk == 0 || cfg.index_kpool == 0 {
            return Err(fit_failed("index_topk and index_kpool must be positive"));
        }
        let cap = cfg.index_topk + cfg.index_kpool - 1;
        if cap > cfg.selected_k {
            return Err(fit_failed(format!(
                "pooled cap {cap} exceeds page-table width {}",
                cfg.selected_k
            )));
        }
        if cfg.softmax_scale_denominator == 0 {
            return Err(fit_failed("softmax_scale_denominator must be positive"));
        }
        if cfg.mla_cache_block_size == 0 || cfg.max_model_len == 0 {
            return Err(fit_failed(
                "cache block size and max_model_len must be positive",
            ));
        }
        let attention = |valid_counts_pattern| DsaSparseMlaAttentionKernelConfig {
            backends: cfg.sparse_attention_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            num_heads: cfg.num_heads.clone(),
            num_kv_heads: Dim::param("num_kv_heads", 1),
            selected_k: cfg.selected_k,
            latent_dim: cfg.latent_dim.clone(),
            rope_dim: cfg.rope_dim.clone(),
            value_dim: cfg.latent_dim.clone(),
            softmax_scale_denominator: cfg.softmax_scale_denominator,
            q_dtype: cfg.q_dtype,
            cache_dtype: cfg.cache_dtype,
            index_dtype: cfg.index_dtype.clone(),
            output_dtype: cfg.output_dtype,
            valid_counts_pattern,
            index_distribution: cfg.index_distribution.clone(),
            cache_layout: cfg.cache_layout.clone(),
        };
        // The remap kernel names its locality by blocks rather than pages.
        let remap_distribution = match cfg.index_distribution.as_str() {
            "unique_scattered_pages" => "unique_scattered_blocks",
            "clustered_pages" => "clustered_blocks",
            other => other,
        };
        Ok(Glm53KpoolSparseMlaResolved {
            mla_cache_append: MlaCacheAppendKernelConfig {
                backends: cfg.mla_cache_append_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                kv_lora_rank: cfg.latent_dim.clone(),
                rope_dim: cfg.rope_dim.clone(),
                block_size: cfg.mla_cache_block_size,
                input_dtype: cfg.activation_dtype,
                kv_dtype: cfg.cache_dtype,
                cache_format: cfg.mla_cache_format.clone(),
            },
            index_remap: DsaSparseIndexRemapKernelConfig {
                backends: cfg.index_remap_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                selected_k: cfg.selected_k,
                block_size: cfg.mla_cache_block_size,
                max_blocks_per_request: cfg.max_model_len.div_ceil(cfg.mla_cache_block_size),
                index_distribution: remap_distribution.to_string(),
                page_table_mapping: cfg.page_table_mapping.clone(),
                return_valid_counts: true,
                index_dtype: cfg.index_dtype.clone(),
            },
            prefill: attention(ValidCountsPattern::CausalTail),
            decode: attention(ValidCountsPattern::PooledUniformFull {
                index_topk: cfg.index_topk,
                index_kpool: cfg.index_kpool,
            }),
        })
    }

    pub fn build(
        name: String,
        cfg: Glm53KpoolSparseMlaConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let resolved = Self::resolve_config(&cfg)?;
        let slot = |index: usize| format!("{name}.{}", SLOT_SUFFIXES[index]);
        Ok(Self {
            mla_cache_append: Arc::new(MlaCacheAppendKernel::build(
                slot(0),
                resolved.mla_cache_append,
                bridge,
            )?),
            index_remap: Arc::new(DsaSparseIndexRemapKernel::build(
                slot(1),
                resolved.index_remap,
                bridge,
            )?),
            prefill: Arc::new(DsaSparseMlaAttentionKernel::build(
                slot(2),
                resolved.prefill,
                bridge,
            )?),
            decode: Arc::new(DsaSparseMlaAttentionKernel::build(
                slot(3),
                resolved.decode,
                bridge,
            )?),
            index_topk: cfg.index_topk,
            index_kpool: cfg.index_kpool,
            name,
        })
    }

    /// Four fixed leaves, independent of request count (INV-1).
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let slot = |index: usize| format!("{}.{}", self.name, SLOT_SUFFIXES[index]);
        CostNode::Sum(vec![
            builder.leaf(
                slot(0),
                self.mla_cache_append.kind(),
                self.mla_cache_append.describe_config(),
            ),
            builder.leaf(
                slot(1),
                self.index_remap.kind(),
                self.index_remap.describe_config(),
            ),
            builder.leaf(slot(2), self.prefill.kind(), self.prefill.describe_config()),
            builder.leaf(slot(3), self.decode.kind(), self.decode.describe_config()),
        ])
    }

    /// Push the four slots in compile order (INV-2).
    pub fn eval(&self, input: &Glm53KpoolSparseMlaInput, ev: &mut Evaluator) {
        let shape = derive_shape(input, self.index_topk, self.index_kpool)
            .unwrap_or_else(|reason| panic!("invalid Glm53KpoolSparseMlaInput: {reason}"));

        let cache_append = MlaCacheAppendKernelInput {
            num_tokens: shape.num_rows,
        };
        let metrics = if shape.num_rows == 0 {
            LeafMetrics::ZERO
        } else {
            self.mla_cache_append.eval(&cache_append)
        };
        ev.push(metrics, || cache_append.clone().into());

        let metrics = shape
            .remap
            .as_ref()
            .map_or(LeafMetrics::ZERO, |remap| self.index_remap.eval(remap));
        ev.push(metrics, || {
            shape
                .remap
                .clone()
                .unwrap_or(DsaSparseIndexRemapKernelInput {
                    request_row_counts: Vec::new(),
                    local_span_lengths: Vec::new(),
                    valid_counts: Vec::new(),
                    workspace_partition: None,
                })
                .into()
        });

        let mut metrics = LeafMetrics::ZERO;
        for &(num_queries, num_cache_tokens) in &input.prefill_query_cache_pairs {
            metrics.add_fanin(self.prefill.eval(&DsaSparseMlaAttentionKernelInput {
                num_queries,
                num_cache_tokens,
            }));
        }
        ev.push(metrics, || {
            DsaSparseMlaPrefillLog {
                prefill_query_cache_pairs: input.prefill_query_cache_pairs.clone(),
            }
            .into()
        });

        let metrics = shape
            .decode
            .as_ref()
            .map_or(LeafMetrics::ZERO, |decode| self.decode.eval(decode));
        ev.push(metrics, || {
            DsaSparseMlaDecodeLog {
                context_lens: input.decode_context_lens.clone(),
                decode_next_n: 1,
                projected_context: shape
                    .decode
                    .as_ref()
                    .map_or(0, |decode| decode.num_cache_tokens),
            }
            .into()
        });
    }
}

struct Shape {
    num_rows: u32,
    remap: Option<DsaSparseIndexRemapKernelInput>,
    decode: Option<DsaSparseMlaAttentionKernelInput>,
}

/// Slots a row at causal position `span` (1-based) reads under the kpool law.
fn pooled_valid_count(span: u32, index_topk: u32, index_kpool: u32) -> u32 {
    span.min(index_topk + span % index_kpool)
}

fn derive_shape(
    input: &Glm53KpoolSparseMlaInput,
    index_topk: u32,
    index_kpool: u32,
) -> Result<Shape, String> {
    let mut request_row_counts = Vec::new();
    let mut local_span_lengths = Vec::new();
    let mut valid_counts = Vec::new();
    // vLLM lays decode rows out ahead of prefill rows in a mixed batch.
    for (request, &context) in input.decode_context_lens.iter().enumerate() {
        if context == 0 {
            return Err(format!("decode request {request} context must be positive"));
        }
        request_row_counts.push(1);
        local_span_lengths.push(context);
        valid_counts.push(pooled_valid_count(context, index_topk, index_kpool));
    }
    for (request, &(num_queries, context)) in input.prefill_query_cache_pairs.iter().enumerate() {
        if num_queries == 0 || num_queries > context {
            return Err(format!(
                "prefill request {request} requires 0 < Q <= S, got ({num_queries}, {context})"
            ));
        }
        request_row_counts.push(num_queries);
        for span in context - num_queries + 1..=context {
            local_span_lengths.push(span);
            valid_counts.push(pooled_valid_count(span, index_topk, index_kpool));
        }
    }
    let num_rows = u32::try_from(local_span_lengths.len())
        .map_err(|_| "query-row count exceeds u32".to_string())?;
    if request_row_counts.len() > MAX_REMAP_REQUESTS {
        return Err(format!(
            "index remap supports at most {MAX_REMAP_REQUESTS} requests, got {}",
            request_row_counts.len()
        ));
    }
    let max_queries = crate::timing::kernels::dsa_sparse_index_remap::MAX_QUERIES;
    if num_rows > max_queries {
        return Err(format!(
            "index remap supports at most {max_queries} query rows, got {num_rows}"
        ));
    }
    let decode = input
        .decode_context_lens
        .iter()
        .copied()
        .max()
        .map(|context| DsaSparseMlaAttentionKernelInput {
            num_queries: input.decode_context_lens.len() as u32,
            num_cache_tokens: context,
        });
    Ok(Shape {
        num_rows,
        remap: (num_rows > 0).then_some(DsaSparseIndexRemapKernelInput {
            request_row_counts,
            local_span_lengths,
            valid_counts,
            workspace_partition: None,
        }),
        decode,
    })
}

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: OP_KIND,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn cfg() -> Glm53KpoolSparseMlaConfig {
        Glm53KpoolSparseMlaConfig {
            sparse_attention_backends: vec!["flashinfer_trtllm_fp8_vllm_fork"],
            mla_cache_append_backends: vec!["vllm_cuda"],
            index_remap_backends: vec!["vllm_fork_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: Dim::param("local_attention_heads", 16),
            latent_dim: Dim::param("kv_lora_rank", 512),
            rope_dim: Dim::param("qk_rope_head_dim", 0),
            selected_k: 2176,
            index_topk: 2048,
            index_kpool: 4,
            softmax_scale_denominator: 16,
            activation_dtype: DType::Bf16,
            q_dtype: DType::Fp8E4m3,
            cache_dtype: DType::Fp8E4m3,
            output_dtype: DType::Bf16,
            index_dtype: "int32".to_string(),
            index_distribution: "unique_scattered_pages".to_string(),
            cache_layout: "hnd_paged_mqa_fp8_latent".to_string(),
            mla_cache_block_size: 64,
            mla_cache_format: "plain".to_string(),
            page_table_mapping: "interleaved_requests".to_string(),
            max_model_len: 8192,
        }
    }

    #[test]
    fn resolves_pooled_decode_causal_prefill_and_2176_remap() {
        let resolved = Glm53KpoolSparseMlaOp::resolve_config(&cfg()).unwrap();
        assert_eq!(
            resolved.decode.valid_counts_pattern,
            ValidCountsPattern::PooledUniformFull {
                index_topk: 2048,
                index_kpool: 4
            }
        );
        assert_eq!(
            resolved.prefill.valid_counts_pattern,
            ValidCountsPattern::CausalTail
        );
        assert_eq!(resolved.index_remap.selected_k, 2176);
        assert_eq!(resolved.index_remap.max_blocks_per_request, 128);
        assert_eq!(
            resolved.index_remap.index_distribution,
            "unique_scattered_blocks"
        );
        assert_eq!(resolved.mla_cache_append.rope_dim.get(), 0);
        assert_eq!(resolved.decode.value_dim.get(), 512);
    }

    #[test]
    fn pooled_law_saturates_at_topk_plus_tail() {
        assert_eq!(pooled_valid_count(1, 2048, 4), 1);
        assert_eq!(pooled_valid_count(2048, 2048, 4), 2048);
        assert_eq!(pooled_valid_count(2049, 2048, 4), 2049);
        assert_eq!(pooled_valid_count(2051, 2048, 4), 2051);
        assert_eq!(pooled_valid_count(2052, 2048, 4), 2048);
        assert_eq!(pooled_valid_count(4003, 2048, 4), 2051);
    }

    #[test]
    fn mixed_batch_puts_decode_rows_first_and_collapses_decode_to_max() {
        let input = Glm53KpoolSparseMlaInput {
            prefill_query_cache_pairs: vec![(3, 5)],
            decode_context_lens: vec![1000, 4003],
        };
        let shape = derive_shape(&input, 2048, 4).unwrap();
        assert_eq!(shape.num_rows, 5);
        let remap = shape.remap.unwrap();
        assert_eq!(remap.request_row_counts, [1, 1, 3]);
        assert_eq!(remap.local_span_lengths, [1000, 4003, 3, 4, 5]);
        assert_eq!(remap.valid_counts, [1000, 2051, 3, 4, 5]);
        let decode = shape.decode.unwrap();
        assert_eq!((decode.num_queries, decode.num_cache_tokens), (2, 4003));
    }

    #[test]
    fn empty_batch_has_no_remap_and_bad_pairs_fail() {
        let shape = derive_shape(&Glm53KpoolSparseMlaInput::default(), 2048, 4).unwrap();
        assert!(shape.remap.is_none() && shape.decode.is_none());
        let bad = Glm53KpoolSparseMlaInput {
            prefill_query_cache_pairs: vec![(6, 5)],
            decode_context_lens: Vec::new(),
        };
        assert!(derive_shape(&bad, 2048, 4).is_err());
    }

    #[test]
    fn compile_mints_four_fixed_slots() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let op = Glm53KpoolSparseMlaOp::build("m.attn.sparse_mla".into(), cfg(), &bridge).unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = op.compile(&mut builder);
        let tree = builder.finish(root);
        let names: Vec<_> = tree.slots.iter().map(|slot| slot.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "m.attn.sparse_mla.mla_cache_append",
                "m.attn.sparse_mla.index_remap",
                "m.attn.sparse_mla.prefill",
                "m.attn.sparse_mla.decode",
            ]
        );
    }
}
