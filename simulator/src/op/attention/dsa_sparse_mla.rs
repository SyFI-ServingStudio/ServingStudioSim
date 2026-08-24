//! GLM-5.2 sparse MLA attention compound op.
//!
//! One logical attention call is a fixed sequential pipeline:
//! query concat, MLA cache append, request-local index remap, sparse prefill,
//! then sparse decode. Query concat and index remap deliberately use the
//! measured `elementwise` placeholder until their dedicated L1 boundaries are
//! wired into this op.
//!
//! **v1 normalization.** As in [`super::FlashInferAttentionOp`], prefill
//! `(Q, S)` cells are evaluated per request and summed into one fixed slot;
//! decode is one optional cell already collapsed by L3. Mixed batches therefore
//! sum prefill and decode leaves even when production can combine them in one
//! `FlashMLA` launch. This intentionally approximates that shared launch overhead
//! while preserving a request-count-independent `CostTree`.

use std::sync::Arc;

use crate::timing::bridge::DType;
use crate::timing::kernels::{
    DsaSparseMlaAttentionKernel, DsaSparseMlaAttentionKernelConfig,
    DsaSparseMlaAttentionKernelInput, ElementwiseKernel, ElementwiseKernelConfig,
    ElementwiseKernelInput, MlaCacheAppendKernel, MlaCacheAppendKernelConfig,
    MlaCacheAppendKernelInput,
};
use crate::timing::slot_input::DsaSparseMlaPrefillLog;
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
};

const OP_KIND: &str = "dsa_sparse_mla_attention";
const REMAP_TILE_SIZE: u32 = 128;
const SLOT_SUFFIXES: [&str; 5] = [
    "query_concat",
    "mla_cache_append",
    "index_remap",
    "prefill",
    "decode",
];

/// Static op identity expanded into the five L1 kernel configurations.
#[derive(Clone, Debug)]
pub struct DsaSparseMlaAttentionConfig {
    pub sparse_attention_backends: Vec<&'static str>,
    pub mla_cache_append_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub num_kv_heads: Dim,
    pub selected_k: u32,
    pub latent_dim: Dim,
    pub rope_dim: Dim,
    pub value_dim: Dim,
    pub softmax_scale_denominator: u32,
    pub dtype: DType,
    pub index_dtype: String,
    pub index_distribution: String,
    pub sparse_cache_layout: String,
    pub mla_cache_block_size: u32,
    pub mla_cache_format: String,
    pub decode_next_n: u32,
}

/// Per-rank inputs for one logical sparse MLA attention call.
#[derive(Clone, Debug, Default)]
pub struct DsaSparseMlaAttentionInput {
    /// Number of rows written by the one MLA cache-append launch.
    pub num_new_tokens: u32,
    /// One physical sparse-attention `(Q, S)` cell per prefill request.
    pub prefill_query_cache_pairs: Vec<(u32, u32)>,
    /// Optional decode cell, already collapsed by the future L3 worklet.
    pub decode_query_cache: Option<(u32, u32)>,
}

pub struct DsaSparseMlaAttentionOp {
    pub name: String,
    pub query_concat: Arc<ElementwiseKernel>,
    pub mla_cache_append: Arc<MlaCacheAppendKernel>,
    pub index_remap: Arc<ElementwiseKernel>,
    pub prefill: Arc<DsaSparseMlaAttentionKernel>,
    pub decode: Arc<DsaSparseMlaAttentionKernel>,
    decode_next_n: u32,
}

impl DsaSparseMlaAttentionOp {
    pub fn build(
        name: String,
        cfg: DsaSparseMlaAttentionConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let subcfg = subkernel_configs(&cfg)?;
        let query_concat = Arc::new(ElementwiseKernel::build(
            format!("{name}.{}", SLOT_SUFFIXES[0]),
            subcfg.query_concat,
            bridge,
        )?);
        let mla_cache_append = Arc::new(MlaCacheAppendKernel::build(
            format!("{name}.{}", SLOT_SUFFIXES[1]),
            subcfg.mla_cache_append,
            bridge,
        )?);
        let index_remap = Arc::new(ElementwiseKernel::build(
            format!("{name}.{}", SLOT_SUFFIXES[2]),
            subcfg.index_remap,
            bridge,
        )?);
        let prefill = Arc::new(DsaSparseMlaAttentionKernel::build(
            format!("{name}.{}", SLOT_SUFFIXES[3]),
            subcfg.prefill,
            bridge,
        )?);
        let decode = Arc::new(DsaSparseMlaAttentionKernel::build(
            format!("{name}.{}", SLOT_SUFFIXES[4]),
            subcfg.decode,
            bridge,
        )?);
        Ok(Self {
            name,
            query_concat,
            mla_cache_append,
            index_remap,
            prefill,
            decode,
            decode_next_n: cfg.decode_next_n,
        })
    }

    /// Five fixed serial leaves, independent of request count (INV-1).
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            builder.leaf(
                format!("{}.{}", self.name, SLOT_SUFFIXES[0]),
                self.query_concat.kind(),
                self.query_concat.describe_config(),
            ),
            builder.leaf(
                format!("{}.{}", self.name, SLOT_SUFFIXES[1]),
                self.mla_cache_append.kind(),
                self.mla_cache_append.describe_config(),
            ),
            builder.leaf(
                format!("{}.{}", self.name, SLOT_SUFFIXES[2]),
                self.index_remap.kind(),
                self.index_remap.describe_config(),
            ),
            builder.leaf(
                format!("{}.{}", self.name, SLOT_SUFFIXES[3]),
                self.prefill.kind(),
                self.prefill.describe_config(),
            ),
            builder.leaf(
                format!("{}.{}", self.name, SLOT_SUFFIXES[4]),
                self.decode.kind(),
                self.decode.describe_config(),
            ),
        ])
    }

    /// Push the same five slots in compile order (INV-2).
    pub fn eval(&self, input: &DsaSparseMlaAttentionInput, ev: &mut Evaluator) {
        let normalized = normalize_input(input, self.decode_next_n)
            .unwrap_or_else(|reason| panic!("invalid DsaSparseMlaAttentionInput: {reason}"));

        let query_concat = ElementwiseKernelInput {
            num_tokens: normalized.query_rows,
        };
        let query_concat_metrics = eval_or_zero(&self.query_concat, &query_concat);
        ev.push(query_concat_metrics, || query_concat.clone().into());

        let cache_append = MlaCacheAppendKernelInput {
            num_tokens: input.num_new_tokens,
        };
        let cache_append_metrics = if input.num_new_tokens == 0 {
            LeafMetrics::ZERO
        } else {
            self.mla_cache_append.eval(&cache_append)
        };
        ev.push(cache_append_metrics, || cache_append.clone().into());

        let index_remap = ElementwiseKernelInput {
            num_tokens: normalized.query_rows,
        };
        let index_remap_metrics = eval_or_zero(&self.index_remap, &index_remap);
        ev.push(index_remap_metrics, || index_remap.clone().into());

        let mut prefill_metrics = LeafMetrics::ZERO;
        for &(num_queries, num_cache_tokens) in &input.prefill_query_cache_pairs {
            prefill_metrics.add_fanin(self.prefill.eval(&DsaSparseMlaAttentionKernelInput {
                num_queries,
                num_cache_tokens,
            }));
        }
        ev.push(prefill_metrics, || {
            DsaSparseMlaPrefillLog {
                prefill_query_cache_pairs: input.prefill_query_cache_pairs.clone(),
            }
            .into()
        });

        let decode_metrics = normalized
            .decode
            .as_ref()
            .map_or(LeafMetrics::ZERO, |shape| self.decode.eval(shape));
        ev.push(decode_metrics, || {
            normalized
                .decode
                .clone()
                .unwrap_or(DsaSparseMlaAttentionKernelInput {
                    num_queries: 0,
                    num_cache_tokens: 0,
                })
                .into()
        });
    }
}

struct SubkernelConfigs {
    query_concat: ElementwiseKernelConfig,
    mla_cache_append: MlaCacheAppendKernelConfig,
    index_remap: ElementwiseKernelConfig,
    prefill: DsaSparseMlaAttentionKernelConfig,
    decode: DsaSparseMlaAttentionKernelConfig,
}

struct NormalizedInput {
    query_rows: u32,
    decode: Option<DsaSparseMlaAttentionKernelInput>,
}

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: OP_KIND,
        reason: reason.into(),
    }
}

fn subkernel_configs(cfg: &DsaSparseMlaAttentionConfig) -> Result<SubkernelConfigs, BuildError> {
    let decode_pattern = match cfg.decode_next_n {
        1 => "uniform_full",
        2 => "speculative_pairs",
        value => {
            return Err(fit_failed(format!(
                "decode_next_n must be 1 or 2, got {value}"
            )))
        }
    };
    if cfg.softmax_scale_denominator == 0 {
        return Err(fit_failed("softmax_scale_denominator must be positive"));
    }

    Ok(SubkernelConfigs {
        query_concat: query_concat_config(cfg)?,
        mla_cache_append: mla_cache_append_config(cfg),
        index_remap: index_remap_config(cfg)?,
        prefill: sparse_attention_config(cfg, "causal_tail"),
        decode: sparse_attention_config(cfg, decode_pattern),
    })
}

fn query_concat_config(
    cfg: &DsaSparseMlaAttentionConfig,
) -> Result<ElementwiseKernelConfig, BuildError> {
    let bytes = u64::from(cfg.num_heads.get())
        .checked_mul(u64::from(
            cfg.latent_dim
                .get()
                .checked_add(cfg.rope_dim.get())
                .ok_or_else(|| fit_failed("query concat width overflows u32"))?,
        ))
        .and_then(|value| value.checked_mul(u64::from(cfg.dtype.size_bytes())))
        .ok_or_else(|| fit_failed("query concat byte rate overflows u64"))?;
    if bytes > u64::from(u32::MAX) {
        return Err(fit_failed("query concat byte rate exceeds u32"));
    }

    let bytes_per_row = cfg.num_heads.clone()
        * (cfg.latent_dim.clone() + cfg.rope_dim.clone())
        * cfg.dtype.size_bytes();
    Ok(ElementwiseKernelConfig {
        backends: cfg.elementwise_backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        input_bytes_per_token: bytes_per_row.clone(),
        output_bytes_per_token: bytes_per_row,
    })
}

fn mla_cache_append_config(cfg: &DsaSparseMlaAttentionConfig) -> MlaCacheAppendKernelConfig {
    MlaCacheAppendKernelConfig {
        backends: cfg.mla_cache_append_backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        kv_lora_rank: cfg.latent_dim.clone(),
        rope_dim: cfg.rope_dim.clone(),
        block_size: cfg.mla_cache_block_size,
        input_dtype: cfg.dtype,
        kv_dtype: cfg.dtype,
        cache_format: cfg.mla_cache_format.clone(),
    }
}

fn index_remap_config(
    cfg: &DsaSparseMlaAttentionConfig,
) -> Result<ElementwiseKernelConfig, BuildError> {
    if cfg.selected_k == 0 || !cfg.selected_k.is_multiple_of(REMAP_TILE_SIZE) {
        return Err(fit_failed(format!(
            "selected_k must be positive and divisible by {REMAP_TILE_SIZE}, got {}",
            cfg.selected_k
        )));
    }
    let tiles = cfg.selected_k / REMAP_TILE_SIZE;
    let input_bytes = cfg
        .selected_k
        .checked_mul(8)
        .and_then(|value| value.checked_add(tiles.checked_mul(4)?))
        .ok_or_else(|| fit_failed("index remap input byte rate overflows u32"))?;
    let output_bytes = cfg
        .selected_k
        .checked_mul(4)
        .ok_or_else(|| fit_failed("index remap output byte rate overflows u32"))?;

    Ok(ElementwiseKernelConfig {
        backends: cfg.elementwise_backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        input_bytes_per_token: input_bytes.into(),
        output_bytes_per_token: output_bytes.into(),
    })
}

fn sparse_attention_config(
    cfg: &DsaSparseMlaAttentionConfig,
    valid_counts_pattern: &str,
) -> DsaSparseMlaAttentionKernelConfig {
    DsaSparseMlaAttentionKernelConfig {
        backends: cfg.sparse_attention_backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        num_heads: cfg.num_heads.clone(),
        num_kv_heads: cfg.num_kv_heads.clone(),
        selected_k: cfg.selected_k,
        latent_dim: cfg.latent_dim.clone(),
        rope_dim: cfg.rope_dim.clone(),
        value_dim: cfg.value_dim.clone(),
        softmax_scale_denominator: cfg.softmax_scale_denominator,
        q_dtype: cfg.dtype,
        cache_dtype: cfg.dtype,
        index_dtype: cfg.index_dtype.clone(),
        output_dtype: cfg.dtype,
        valid_counts_pattern: valid_counts_pattern.to_string(),
        index_distribution: cfg.index_distribution.clone(),
        cache_layout: cfg.sparse_cache_layout.clone(),
    }
}

fn normalize_input(
    input: &DsaSparseMlaAttentionInput,
    decode_next_n: u32,
) -> Result<NormalizedInput, String> {
    if !matches!(decode_next_n, 1 | 2) {
        return Err(format!("decode_next_n must be 1 or 2, got {decode_next_n}"));
    }

    let mut query_rows = 0_u32;
    for (index, &(num_queries, num_cache_tokens)) in
        input.prefill_query_cache_pairs.iter().enumerate()
    {
        if num_queries == 0 || num_cache_tokens == 0 {
            return Err(format!(
                "prefill pair {index} must have nonzero Q and S, got ({num_queries}, {num_cache_tokens})"
            ));
        }
        if num_queries > num_cache_tokens {
            return Err(format!(
                "prefill pair {index} requires Q<=S, got ({num_queries}, {num_cache_tokens})"
            ));
        }
        query_rows = query_rows
            .checked_add(num_queries)
            .ok_or_else(|| "total query-row count overflows u32".to_string())?;
    }

    let decode = match input.decode_query_cache {
        Some((num_queries, num_cache_tokens)) => {
            if num_queries == 0 || num_cache_tokens == 0 {
                return Err(format!(
                    "decode cell must have nonzero Q and S, got ({num_queries}, {num_cache_tokens})"
                ));
            }
            if decode_next_n == 2 && num_queries % 2 != 0 {
                return Err(format!(
                    "speculative decode requires even Q, got {num_queries}"
                ));
            }
            query_rows = query_rows
                .checked_add(num_queries)
                .ok_or_else(|| "total query-row count overflows u32".to_string())?;
            Some(DsaSparseMlaAttentionKernelInput {
                num_queries,
                num_cache_tokens,
            })
        }
        None => None,
    };

    Ok(NormalizedInput { query_rows, decode })
}

fn eval_or_zero(kernel: &ElementwiseKernel, input: &ElementwiseKernelInput) -> LeafMetrics {
    if input.num_tokens == 0 {
        LeafMetrics::ZERO
    } else {
        kernel.eval(input)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        index_remap_config, normalize_input, query_concat_config, subkernel_configs,
        DsaSparseMlaAttentionConfig, DsaSparseMlaAttentionInput, SLOT_SUFFIXES,
    };
    use crate::timing::bridge::DType;
    use crate::timing::slot_input::DsaSparseMlaPrefillLog;
    use crate::timing::{BuildError, Dim, SlotInput};

    fn cfg(decode_next_n: u32) -> DsaSparseMlaAttentionConfig {
        DsaSparseMlaAttentionConfig {
            sparse_attention_backends: vec!["vllm_flashmla_bf16"],
            mla_cache_append_backends: vec!["vllm_cuda"],
            elementwise_backends: vec!["triton"],
            gpu_name: "NVIDIA H200".to_string(),
            num_heads: Dim::param("num_attention_heads", 64),
            num_kv_heads: Dim::param("num_kv_heads", 1),
            selected_k: 2048,
            latent_dim: Dim::param("kv_lora_rank", 512),
            rope_dim: Dim::param("qk_rope_head_dim", 64),
            value_dim: Dim::param("kv_lora_rank", 512),
            softmax_scale_denominator: 16,
            dtype: DType::Bf16,
            index_dtype: "int32".to_string(),
            index_distribution: "recent_contiguous".to_string(),
            sparse_cache_layout: "token_major_mqa_bf16_latent_rope".to_string(),
            mla_cache_block_size: 64,
            mla_cache_format: "plain".to_string(),
            decode_next_n,
        }
    }

    #[test]
    fn config_expands_into_the_frozen_five_subkernel_identities() {
        let configs = subkernel_configs(&cfg(1)).unwrap();

        assert_eq!(configs.query_concat.backends, vec!["triton"]);
        assert_eq!(configs.query_concat.input_bytes_per_token, 73_728);
        assert_eq!(configs.query_concat.output_bytes_per_token, 73_728);

        assert_eq!(configs.mla_cache_append.backends, vec!["vllm_cuda"]);
        assert_eq!(configs.mla_cache_append.kv_lora_rank, 512);
        assert_eq!(configs.mla_cache_append.rope_dim, 64);
        assert_eq!(configs.mla_cache_append.block_size, 64);
        assert_eq!(configs.mla_cache_append.input_dtype, DType::Bf16);
        assert_eq!(configs.mla_cache_append.kv_dtype, DType::Bf16);
        assert_eq!(configs.mla_cache_append.cache_format, "plain");

        assert_eq!(configs.index_remap.backends, vec!["triton"]);
        assert_eq!(configs.index_remap.input_bytes_per_token, 16_448);
        assert_eq!(configs.index_remap.output_bytes_per_token, 8_192);

        assert_eq!(configs.prefill.backends, vec!["vllm_flashmla_bf16"]);
        assert_eq!(configs.prefill.valid_counts_pattern, "causal_tail");
        assert_eq!(configs.decode.valid_counts_pattern, "uniform_full");
        for sparse in [&configs.prefill, &configs.decode] {
            assert_eq!(sparse.gpu_name, "NVIDIA H200");
            assert_eq!(sparse.num_heads, 64);
            assert_eq!(sparse.num_kv_heads, 1);
            assert_eq!(sparse.selected_k, 2048);
            assert_eq!(sparse.latent_dim, 512);
            assert_eq!(sparse.rope_dim, 64);
            assert_eq!(sparse.value_dim, 512);
            assert_eq!(sparse.softmax_scale_denominator, 16);
            assert_eq!(sparse.q_dtype, DType::Bf16);
            assert_eq!(sparse.cache_dtype, DType::Bf16);
            assert_eq!(sparse.output_dtype, DType::Bf16);
            assert_eq!(sparse.index_dtype, "int32");
            assert_eq!(sparse.index_distribution, "recent_contiguous");
            assert_eq!(sparse.cache_layout, "token_major_mqa_bf16_latent_rope");
        }
    }

    #[test]
    fn decode_next_n_two_selects_speculative_pairs() {
        let configs = subkernel_configs(&cfg(2)).unwrap();
        assert_eq!(configs.prefill.valid_counts_pattern, "causal_tail");
        assert_eq!(configs.decode.valid_counts_pattern, "speculative_pairs");
    }

    #[test]
    fn invalid_decode_next_n_and_scale_fail_at_config_expansion() {
        for decode_next_n in [0, 3] {
            let error = subkernel_configs(&cfg(decode_next_n))
                .err()
                .expect("invalid decode_next_n must fail");
            assert!(matches!(
                error,
                BuildError::FitFailed { reason, .. }
                    if reason.contains("decode_next_n must be 1 or 2")
            ));
        }

        let mut config = cfg(1);
        config.softmax_scale_denominator = 0;
        assert!(matches!(
            subkernel_configs(&config),
            Err(BuildError::FitFailed { reason, .. })
                if reason.contains("softmax_scale_denominator")
        ));
    }

    #[test]
    fn production_byte_formulas_and_selected_k_tile_guard_are_exact() {
        let config = cfg(1);
        let query = query_concat_config(&config).unwrap();
        let remap = index_remap_config(&config).unwrap();
        assert_eq!(query.input_bytes_per_token, 73_728);
        assert_eq!(query.output_bytes_per_token, 73_728);
        assert_eq!(remap.input_bytes_per_token, 16_448);
        assert_eq!(remap.output_bytes_per_token, 8_192);

        let mut invalid = config;
        invalid.selected_k = 2050;
        assert!(matches!(
            index_remap_config(&invalid),
            Err(BuildError::FitFailed { reason, .. }) if reason.contains("divisible by 128")
        ));
    }

    #[test]
    fn query_rows_collapse_prefill_and_optional_decode_with_zero_cases() {
        let empty = normalize_input(&DsaSparseMlaAttentionInput::default(), 1).unwrap();
        assert_eq!(empty.query_rows, 0);
        assert!(empty.decode.is_none());

        let input = DsaSparseMlaAttentionInput {
            num_new_tokens: 37,
            prefill_query_cache_pairs: vec![(8, 128), (16, 256)],
            decode_query_cache: Some((32, 8192)),
        };
        let normalized = normalize_input(&input, 1).unwrap();
        assert_eq!(normalized.query_rows, 56);
        let decode = normalized.decode.unwrap();
        assert_eq!(decode.num_queries, 32);
        assert_eq!(decode.num_cache_tokens, 8192);
        assert_eq!(input.num_new_tokens, 37);
    }

    #[test]
    fn malformed_prefill_decode_and_overflow_fail_clearly() {
        for pair in [(0, 1), (1, 0), (2, 1)] {
            let input = DsaSparseMlaAttentionInput {
                prefill_query_cache_pairs: vec![pair],
                ..Default::default()
            };
            assert!(normalize_input(&input, 1).is_err());
        }

        let zero_decode = DsaSparseMlaAttentionInput {
            decode_query_cache: Some((0, 1)),
            ..Default::default()
        };
        assert!(normalize_input(&zero_decode, 1).is_err());

        let odd_speculative = DsaSparseMlaAttentionInput {
            decode_query_cache: Some((3, 2048)),
            ..Default::default()
        };
        assert!(normalize_input(&odd_speculative, 2)
            .err()
            .expect("odd speculative Q must fail")
            .contains("even Q"));

        let overflowing = DsaSparseMlaAttentionInput {
            prefill_query_cache_pairs: vec![(u32::MAX, u32::MAX), (1, 1)],
            ..Default::default()
        };
        assert!(normalize_input(&overflowing, 1)
            .err()
            .expect("overflowing row sum must fail")
            .contains("overflows u32"));
    }

    #[test]
    fn aggregate_prefill_slot_input_serializes_the_complete_pair_vector() {
        let slot: SlotInput = DsaSparseMlaPrefillLog {
            prefill_query_cache_pairs: vec![(1, 1), (128, 2049)],
        }
        .into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"prefill_query_cache_pairs":[[1,1],[128,2049]]})
        );
    }

    #[test]
    fn compound_boundary_has_only_the_frozen_five_slots() {
        assert_eq!(
            SLOT_SUFFIXES,
            [
                "query_concat",
                "mla_cache_append",
                "index_remap",
                "prefill",
                "decode"
            ]
        );
        assert_eq!(SLOT_SUFFIXES.len(), 5);
        assert!(!SLOT_SUFFIXES.iter().any(|slot| slot.contains("topk")));
        assert!(!SLOT_SUFFIXES.iter().any(|slot| slot.contains("indexer")));
        assert!(!SLOT_SUFFIXES.iter().any(|slot| slot.contains("all_reduce")));
        assert!(!SLOT_SUFFIXES.iter().any(|slot| slot.contains("all_to_all")));
    }
}
