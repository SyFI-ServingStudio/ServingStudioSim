//! GLM-5.2 sparse MLA attention compound op.
//!
//! One logical attention call contains MLA cache append, sparse prefill, and
//! sparse decode. Providers with separate launch boundaries additionally emit
//! query concat and request-local index remap; SGLang fuses those operations
//! into adjacent launches and therefore has no synthetic slots for them. The
//! production B200 variant uses one exact-varlen sparse-prefill launch; the
//! inherited H200 variant keeps its accepted per-request prefill fallback.
//!
//! Backend-specific L1 selection and launch-graph identity are build-time
//! config; runtime request topology remains inside this compound op and never
//! leaks into L3.

use std::sync::Arc;

use crate::timing::bridge::DType;
use crate::timing::kernels::{
    DsaSparseIndexRemapKernel, DsaSparseIndexRemapKernelConfig, DsaSparseIndexRemapKernelInput,
    DsaSparseMlaAttentionKernel, DsaSparseMlaAttentionKernelConfig,
    DsaSparseMlaAttentionKernelInput, DsaSparseMlaPrefillKernel, DsaSparseMlaPrefillKernelConfig,
    DsaSparseMlaPrefillKernelInput, ElementwiseKernel, ElementwiseKernelConfig,
    ElementwiseKernelInput, MlaCacheAppendKernel, MlaCacheAppendKernelConfig,
    MlaCacheAppendKernelInput,
};
use crate::timing::slot_input::{DsaSparseMlaDecodeLog, DsaSparseMlaPrefillLog};
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
    pub attention_q_dtype: DType,
    pub attention_cache_dtype: DType,
    pub attention_output_dtype: DType,
    pub index_dtype: String,
    pub index_distribution: String,
    pub sparse_cache_layout: String,
    pub mla_cache_block_size: u32,
    pub mla_cache_format: String,
    pub decode_next_n: u32,
    pub launch_graph: DsaSparseMlaLaunchGraph,
    /// `Some` selects the production B200 remap + exact-varlen prefill leaves.
    /// `None` retains the accepted H200 fallback without changing its profile
    /// identities or request aggregation.
    pub exact_varlen: Option<DsaSparseMlaExactVarlenConfig>,
}

/// Measured launch boundary around sparse MLA. Provider worklets select one
/// graph; this is not a runtime/user tuning mode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DsaSparseMlaLaunchGraph {
    Separate {
        index_remap_backends: Vec<&'static str>,
    },
    /// SGLang's upstream fused RoPE produces the concatenated FP8 query and
    /// its sparse-attention preparation consumes local indices directly.
    FusedQueryConcatAndIndexRemap,
}

#[derive(Clone, Debug)]
pub struct DsaSparseMlaExactVarlenConfig {
    pub prefill_backends: Vec<&'static str>,
    pub max_model_len: u32,
    /// Production prefill indices are request-local and have a different
    /// locality identity from decode's page selection. Keep the two cache axes
    /// explicit instead of leaking `DsaSparseMlaAttentionConfig`'s decode
    /// distribution into the prefill lookup.
    pub prefill_index_distribution: String,
    /// Required only by a separate production index-remap launch. A fused
    /// provider graph must leave it absent rather than carrying an ignored
    /// remap identity.
    pub page_table_mapping: Option<String>,
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
    /// Exact request-local decode KV lengths when production metadata is
    /// available. The cache still uses one uniform-equivalent coordinate, but
    /// derives it as `mean + mean absolute deviation` instead of the batch
    /// maximum. Across 24 real-capture held-out batches this kept 21/24 points
    /// within 5 us while avoiding the mean projection's 17.2 us worst miss.
    pub decode_context_lens: Option<Vec<u32>>,
}

pub struct DsaSparseMlaAttentionOp {
    pub name: String,
    pub query_concat: Option<Arc<ElementwiseKernel>>,
    pub mla_cache_append: Arc<MlaCacheAppendKernel>,
    index_remap: Option<IndexRemapLeaf>,
    prefill: PrefillLeaf,
    pub decode: Arc<DsaSparseMlaAttentionKernel>,
    decode_next_n: u32,
    selected_k: u32,
}

enum IndexRemapLeaf {
    Legacy(Arc<ElementwiseKernel>),
    Production(Arc<DsaSparseIndexRemapKernel>),
}

impl IndexRemapLeaf {
    fn kind(&self) -> &'static str {
        match self {
            Self::Legacy(kernel) => kernel.kind(),
            Self::Production(kernel) => kernel.kind(),
        }
    }

    fn describe_config(&self) -> serde_json::Value {
        match self {
            Self::Legacy(kernel) => kernel.describe_config(),
            Self::Production(kernel) => kernel.describe_config(),
        }
    }
}

enum PrefillLeaf {
    Legacy(Arc<DsaSparseMlaAttentionKernel>),
    Production(Arc<DsaSparseMlaPrefillKernel>),
}

impl PrefillLeaf {
    fn kind(&self) -> &'static str {
        match self {
            Self::Legacy(kernel) => kernel.kind(),
            Self::Production(kernel) => kernel.kind(),
        }
    }

    fn describe_config(&self) -> serde_json::Value {
        match self {
            Self::Legacy(kernel) => kernel.describe_config(),
            Self::Production(kernel) => kernel.describe_config(),
        }
    }
}

impl DsaSparseMlaAttentionOp {
    pub fn build(
        name: String,
        cfg: DsaSparseMlaAttentionConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let subcfg = subkernel_configs(&cfg)?;
        let query_concat = subcfg
            .query_concat
            .map(|config| {
                ElementwiseKernel::build(format!("{name}.{}", SLOT_SUFFIXES[0]), config, bridge)
                    .map(Arc::new)
            })
            .transpose()?;
        let mla_cache_append = Arc::new(MlaCacheAppendKernel::build(
            format!("{name}.{}", SLOT_SUFFIXES[1]),
            subcfg.mla_cache_append,
            bridge,
        )?);
        let index_remap = match subcfg.index_remap {
            Some(IndexRemapConfig::Legacy(config)) => Some(IndexRemapLeaf::Legacy(Arc::new(
                ElementwiseKernel::build(format!("{name}.{}", SLOT_SUFFIXES[2]), config, bridge)?,
            ))),
            Some(IndexRemapConfig::Production(config)) => Some(IndexRemapLeaf::Production(
                Arc::new(DsaSparseIndexRemapKernel::build(
                    format!("{name}.{}", SLOT_SUFFIXES[2]),
                    config,
                    bridge,
                )?),
            )),
            None => None,
        };
        let prefill = match subcfg.prefill {
            PrefillConfig::Legacy(config) => {
                PrefillLeaf::Legacy(Arc::new(DsaSparseMlaAttentionKernel::build(
                    format!("{name}.{}", SLOT_SUFFIXES[3]),
                    config,
                    bridge,
                )?))
            }
            PrefillConfig::Production(config) => {
                PrefillLeaf::Production(Arc::new(DsaSparseMlaPrefillKernel::build(
                    format!("{name}.{}", SLOT_SUFFIXES[3]),
                    config,
                    bridge,
                )?))
            }
        };
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
            selected_k: cfg.selected_k,
        })
    }

    /// Emit only physical provider launches, independent of request count.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut slots = Vec::with_capacity(5);
        if let Some(query_concat) = &self.query_concat {
            slots.push(builder.leaf(
                format!("{}.{}", self.name, SLOT_SUFFIXES[0]),
                query_concat.kind(),
                query_concat.describe_config(),
            ));
        }
        slots.push(builder.leaf(
            format!("{}.{}", self.name, SLOT_SUFFIXES[1]),
            self.mla_cache_append.kind(),
            self.mla_cache_append.describe_config(),
        ));
        if let Some(index_remap) = &self.index_remap {
            slots.push(builder.leaf(
                format!("{}.{}", self.name, SLOT_SUFFIXES[2]),
                index_remap.kind(),
                index_remap.describe_config(),
            ));
        }
        slots.extend([
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
        ]);
        CostNode::Sum(slots)
    }

    /// Push the same provider-specific slots in compile order (INV-2).
    pub fn eval(&self, input: &DsaSparseMlaAttentionInput, ev: &mut Evaluator) {
        let normalized = normalize_input(input, self.decode_next_n)
            .unwrap_or_else(|reason| panic!("invalid DsaSparseMlaAttentionInput: {reason}"));

        if let Some(query_concat_kernel) = &self.query_concat {
            let query_concat = ElementwiseKernelInput {
                num_tokens: normalized.query_rows,
            };
            let query_concat_metrics = eval_or_zero(query_concat_kernel, &query_concat);
            ev.push(query_concat_metrics, || query_concat.clone().into());
        }

        let cache_append = MlaCacheAppendKernelInput {
            num_tokens: input.num_new_tokens,
        };
        let cache_append_metrics = if input.num_new_tokens == 0 {
            LeafMetrics::ZERO
        } else {
            self.mla_cache_append.eval(&cache_append)
        };
        ev.push(cache_append_metrics, || cache_append.clone().into());

        match &self.index_remap {
            None => {}
            Some(IndexRemapLeaf::Legacy(kernel)) => {
                let shape = ElementwiseKernelInput {
                    num_tokens: normalized.query_rows,
                };
                let metrics = eval_or_zero(kernel, &shape);
                ev.push(metrics, || shape.clone().into());
            }
            Some(IndexRemapLeaf::Production(kernel)) => {
                let shape =
                    production_index_remap_input(input, self.decode_next_n, self.selected_k)
                        .unwrap_or_else(|reason| {
                            panic!("invalid production index-remap input: {reason}")
                        });
                let metrics = shape
                    .as_ref()
                    .map_or(LeafMetrics::ZERO, |shape| kernel.eval(shape));
                ev.push(metrics, || {
                    shape
                        .unwrap_or(DsaSparseIndexRemapKernelInput {
                            request_row_counts: Vec::new(),
                            local_span_lengths: Vec::new(),
                            valid_counts: Vec::new(),
                            workspace_partition: None,
                        })
                        .into()
                });
            }
        }

        match &self.prefill {
            PrefillLeaf::Legacy(kernel) => {
                let mut metrics = LeafMetrics::ZERO;
                for &(num_queries, num_cache_tokens) in &input.prefill_query_cache_pairs {
                    metrics.add_fanin(kernel.eval(&DsaSparseMlaAttentionKernelInput {
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
            }
            PrefillLeaf::Production(kernel) => {
                let shape = (!input.prefill_query_cache_pairs.is_empty()).then(|| {
                    DsaSparseMlaPrefillKernelInput {
                        query_context_pairs: input.prefill_query_cache_pairs.clone(),
                    }
                });
                let metrics = shape
                    .as_ref()
                    .map_or(LeafMetrics::ZERO, |shape| kernel.eval(shape));
                ev.push(metrics, || {
                    shape
                        .unwrap_or(DsaSparseMlaPrefillKernelInput {
                            query_context_pairs: Vec::new(),
                        })
                        .into()
                });
            }
        }

        let decode_metrics = normalized
            .decode
            .as_ref()
            .map_or(LeafMetrics::ZERO, |shape| self.decode.eval(shape));
        ev.push(decode_metrics, || {
            DsaSparseMlaDecodeLog {
                context_lens: normalized.decode_context_lens.clone(),
                decode_next_n: self.decode_next_n,
                projected_context: normalized
                    .decode
                    .as_ref()
                    .map_or(0, |shape| shape.num_cache_tokens),
            }
            .into()
        });
    }
}

struct SubkernelConfigs {
    query_concat: Option<ElementwiseKernelConfig>,
    mla_cache_append: MlaCacheAppendKernelConfig,
    index_remap: Option<IndexRemapConfig>,
    prefill: PrefillConfig,
    decode: DsaSparseMlaAttentionKernelConfig,
}

enum IndexRemapConfig {
    Legacy(ElementwiseKernelConfig),
    Production(DsaSparseIndexRemapKernelConfig),
}

enum PrefillConfig {
    Legacy(DsaSparseMlaAttentionKernelConfig),
    Production(DsaSparseMlaPrefillKernelConfig),
}

struct NormalizedInput {
    query_rows: u32,
    decode: Option<DsaSparseMlaAttentionKernelInput>,
    decode_context_lens: Vec<u32>,
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

    let (max_blocks_per_request, prefill) = match &cfg.exact_varlen {
        None => (
            None,
            PrefillConfig::Legacy(sparse_attention_config(cfg, "causal_tail")),
        ),
        Some(exact) => {
            if exact.max_model_len == 0 {
                return Err(fit_failed("exact varlen max_model_len must be positive"));
            }
            if cfg.selected_k != 2048 {
                return Err(fit_failed(format!(
                    "exact varlen selected_k must be 2048, got {}",
                    cfg.selected_k
                )));
            }
            if cfg.mla_cache_block_size != 64 {
                return Err(fit_failed(format!(
                    "exact varlen cache block size must be 64, got {}",
                    cfg.mla_cache_block_size
                )));
            }
            let max_blocks_per_request = exact.max_model_len.div_ceil(cfg.mla_cache_block_size);
            if max_blocks_per_request > 16_384 {
                return Err(fit_failed(format!(
                    "exact varlen max_model_len requires {max_blocks_per_request} cache blocks; maximum is 16384"
                )));
            }
            (
                Some(max_blocks_per_request),
                PrefillConfig::Production(production_prefill_config(cfg, exact)),
            )
        }
    };

    let (query_concat, index_remap) = match &cfg.launch_graph {
        DsaSparseMlaLaunchGraph::Separate {
            index_remap_backends,
        } => {
            let remap = match (&cfg.exact_varlen, max_blocks_per_request) {
                (Some(exact), Some(max_blocks)) => {
                    if index_remap_backends.is_empty() {
                        return Err(fit_failed(
                            "exact-varlen separate graph requires an index-remap backend",
                        ));
                    }
                    if exact.page_table_mapping.is_none() {
                        return Err(fit_failed(
                            "exact-varlen separate graph requires page-table mapping",
                        ));
                    }
                    IndexRemapConfig::Production(production_index_remap_config(
                        cfg,
                        exact,
                        max_blocks,
                        index_remap_backends,
                    ))
                }
                (None, None) => {
                    if !index_remap_backends.is_empty() {
                        return Err(fit_failed(
                            "legacy separate graph must not configure a production index-remap backend",
                        ));
                    }
                    IndexRemapConfig::Legacy(legacy_index_remap_config(cfg)?)
                }
                _ => unreachable!("exact-varlen validation returns paired state"),
            };
            (Some(query_concat_config(cfg)?), Some(remap))
        }
        DsaSparseMlaLaunchGraph::FusedQueryConcatAndIndexRemap => {
            if cfg.exact_varlen.is_none() {
                return Err(fit_failed(
                    "fused sparse graph requires the production exact-varlen prefill",
                ));
            }
            if !cfg.elementwise_backends.is_empty() {
                return Err(fit_failed(
                    "fused sparse graph must not configure elementwise concat/remap backends",
                ));
            }
            if cfg
                .exact_varlen
                .as_ref()
                .is_some_and(|exact| exact.page_table_mapping.is_some())
            {
                return Err(fit_failed(
                    "fused sparse graph must not configure page-table remapping",
                ));
            }
            (None, None)
        }
    };

    Ok(SubkernelConfigs {
        query_concat,
        mla_cache_append: mla_cache_append_config(cfg),
        index_remap,
        prefill,
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
        input_dtype: match &cfg.launch_graph {
            DsaSparseMlaLaunchGraph::Separate { .. } => cfg.dtype,
            DsaSparseMlaLaunchGraph::FusedQueryConcatAndIndexRemap => cfg.attention_cache_dtype,
        },
        kv_dtype: cfg.attention_cache_dtype,
        cache_format: cfg.mla_cache_format.clone(),
    }
}

fn legacy_index_remap_config(
    cfg: &DsaSparseMlaAttentionConfig,
) -> Result<ElementwiseKernelConfig, BuildError> {
    if cfg.selected_k == 0 || cfg.selected_k % REMAP_TILE_SIZE != 0 {
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

fn production_index_remap_config(
    cfg: &DsaSparseMlaAttentionConfig,
    exact: &DsaSparseMlaExactVarlenConfig,
    max_blocks_per_request: u32,
    backends: &[&'static str],
) -> DsaSparseIndexRemapKernelConfig {
    let index_distribution = match cfg.index_distribution.as_str() {
        "unique_scattered_pages" => "unique_scattered_blocks",
        "clustered_pages" => "clustered_blocks",
        other => other,
    };
    DsaSparseIndexRemapKernelConfig {
        backends: backends.to_vec(),
        gpu_name: cfg.gpu_name.clone(),
        selected_k: cfg.selected_k,
        block_size: cfg.mla_cache_block_size,
        max_blocks_per_request,
        index_distribution: index_distribution.to_string(),
        page_table_mapping: exact
            .page_table_mapping
            .clone()
            .expect("separate exact-varlen graph validates page-table mapping"),
        return_valid_counts: true,
        index_dtype: cfg.index_dtype.clone(),
    }
}

fn production_prefill_config(
    cfg: &DsaSparseMlaAttentionConfig,
    exact: &DsaSparseMlaExactVarlenConfig,
) -> DsaSparseMlaPrefillKernelConfig {
    DsaSparseMlaPrefillKernelConfig {
        backends: exact.prefill_backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        max_model_len: exact.max_model_len,
        num_heads: cfg.num_heads.clone(),
        num_kv_heads: cfg.num_kv_heads.clone(),
        selected_k: cfg.selected_k,
        latent_dim: cfg.latent_dim.clone(),
        rope_dim: cfg.rope_dim.clone(),
        value_dim: cfg.value_dim.clone(),
        softmax_scale_denominator: cfg.softmax_scale_denominator,
        q_dtype: cfg.attention_q_dtype,
        cache_dtype: cfg.attention_cache_dtype,
        index_dtype: cfg.index_dtype.clone(),
        output_dtype: cfg.attention_output_dtype,
        index_distribution: exact.prefill_index_distribution.clone(),
        cache_layout: cfg.sparse_cache_layout.clone(),
    }
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
        q_dtype: cfg.attention_q_dtype,
        cache_dtype: cfg.attention_cache_dtype,
        index_dtype: cfg.index_dtype.clone(),
        output_dtype: cfg.attention_output_dtype,
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

    let mut normalized_decode_context_lens = Vec::new();
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
            let projected_context = if let Some(context_lens) = &input.decode_context_lens {
                let expected_requests = num_queries / decode_next_n;
                if context_lens.len() != expected_requests as usize {
                    return Err(format!(
                        "decode_context_lens has {} entries, expected {expected_requests}",
                        context_lens.len()
                    ));
                }
                let mut total = 0_u64;
                for (request, &context) in context_lens.iter().enumerate() {
                    if context < decode_next_n {
                        return Err(format!(
                            "decode request {request} context {context} must be at least decode_next_n {decode_next_n}"
                        ));
                    }
                    total = total
                        .checked_add(u64::from(context))
                        .ok_or_else(|| "decode context sum overflows u64".to_string())?;
                }
                let request_count = u64::from(expected_requests);
                let mean_context = total as f64 / request_count as f64;
                let mean_absolute_deviation = context_lens
                    .iter()
                    .map(|&context| (f64::from(context) - mean_context).abs())
                    .sum::<f64>()
                    / request_count as f64;
                normalized_decode_context_lens = context_lens.clone();
                u32::try_from((mean_context + mean_absolute_deviation).round() as u64)
                    .map_err(|_| "rounded decode context does not fit u32".to_string())?
            } else {
                normalized_decode_context_lens =
                    vec![num_cache_tokens; (num_queries / decode_next_n) as usize];
                num_cache_tokens
            };
            query_rows = query_rows
                .checked_add(num_queries)
                .ok_or_else(|| "total query-row count overflows u32".to_string())?;
            Some(DsaSparseMlaAttentionKernelInput {
                num_queries,
                num_cache_tokens: projected_context,
            })
        }
        None => {
            if input.decode_context_lens.is_some() {
                return Err("decode_context_lens requires decode_query_cache".to_string());
            }
            None
        }
    };

    Ok(NormalizedInput {
        query_rows,
        decode,
        decode_context_lens: normalized_decode_context_lens,
    })
}

/// Build the exact row topology consumed by vLLM's remap wrapper. Decode rows
/// precede prefill rows, matching vLLM's mixed-batch metadata layout. Every
/// row's local span is its causal position within that request; valid slots are
/// the same span capped by top-k. FlashInfer uses global cache slots, so this
/// path has no prefill workspace partition and requests share no fake merge.
fn production_index_remap_input(
    input: &DsaSparseMlaAttentionInput,
    decode_next_n: u32,
    selected_k: u32,
) -> Result<Option<DsaSparseIndexRemapKernelInput>, String> {
    let mut request_row_counts = Vec::new();
    let mut local_span_lengths = Vec::new();
    let mut valid_counts = Vec::new();

    if let Some((num_queries, uniform_context)) = input.decode_query_cache {
        if num_queries % decode_next_n != 0 {
            return Err(format!(
                "decode query rows {num_queries} must be divisible by decode_next_n {decode_next_n}"
            ));
        }
        let num_requests = num_queries / decode_next_n;
        let contexts = input
            .decode_context_lens
            .clone()
            .unwrap_or_else(|| vec![uniform_context; num_requests as usize]);
        if contexts.len() != num_requests as usize {
            return Err(format!(
                "decode_context_lens has {} entries, expected {num_requests}",
                contexts.len()
            ));
        }
        for context in contexts {
            if context < decode_next_n {
                return Err(format!(
                    "decode context {context} must be at least decode_next_n {decode_next_n}"
                ));
            }
            request_row_counts.push(decode_next_n);
            let first = context - decode_next_n + 1;
            for span in first..=context {
                local_span_lengths.push(span);
                valid_counts.push(span.min(selected_k));
            }
        }
    }

    for &(num_queries, context) in &input.prefill_query_cache_pairs {
        request_row_counts.push(num_queries);
        let first = context - num_queries + 1;
        for span in first..=context {
            local_span_lengths.push(span);
            valid_counts.push(span.min(selected_k));
        }
    }

    if request_row_counts.is_empty() {
        return Ok(None);
    }
    if request_row_counts.len() > 256 {
        return Err(format!(
            "production index remap supports at most 256 requests, got {}",
            request_row_counts.len()
        ));
    }
    if local_span_lengths.len() > 8192 {
        return Err(format!(
            "production index remap supports at most 8192 query rows, got {}",
            local_span_lengths.len()
        ));
    }

    Ok(Some(DsaSparseIndexRemapKernelInput {
        request_row_counts,
        local_span_lengths,
        valid_counts,
        workspace_partition: None,
    }))
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
        legacy_index_remap_config, normalize_input, production_index_remap_input,
        query_concat_config, subkernel_configs, DsaSparseMlaAttentionConfig,
        DsaSparseMlaAttentionInput, DsaSparseMlaExactVarlenConfig, DsaSparseMlaLaunchGraph,
        IndexRemapConfig, PrefillConfig, SLOT_SUFFIXES,
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
            attention_q_dtype: DType::Bf16,
            attention_cache_dtype: DType::Bf16,
            attention_output_dtype: DType::Bf16,
            index_dtype: "int32".to_string(),
            index_distribution: "recent_contiguous".to_string(),
            sparse_cache_layout: "token_major_mqa_bf16_latent_rope".to_string(),
            mla_cache_block_size: 64,
            mla_cache_format: "plain".to_string(),
            decode_next_n,
            launch_graph: DsaSparseMlaLaunchGraph::Separate {
                index_remap_backends: Vec::new(),
            },
            exact_varlen: None,
        }
    }

    fn exact_cfg(decode_next_n: u32) -> DsaSparseMlaAttentionConfig {
        let mut config = cfg(decode_next_n);
        config.sparse_attention_backends = vec!["flashinfer_trtllm_fp8"];
        config.gpu_name = "NVIDIA B200".to_string();
        config.num_heads = Dim::param("num_attention_heads", 16);
        config.attention_q_dtype = DType::Fp8E4m3;
        config.attention_cache_dtype = DType::Fp8E4m3;
        config.attention_output_dtype = DType::Bf16;
        config.index_distribution = "unique_scattered_pages".to_string();
        config.sparse_cache_layout = "hnd_paged_mqa_fp8_latent_rope".to_string();
        config.mla_cache_format = "plain".to_string();
        config.launch_graph = DsaSparseMlaLaunchGraph::Separate {
            index_remap_backends: vec!["vllm_triton"],
        };
        config.exact_varlen = Some(DsaSparseMlaExactVarlenConfig {
            prefill_backends: vec!["flashinfer_trtllm_fp8"],
            max_model_len: 8192,
            prefill_index_distribution: "recent_contiguous".to_string(),
            page_table_mapping: Some("request_contiguous".to_string()),
        });
        config
    }

    #[test]
    fn config_expands_into_the_frozen_five_subkernel_identities() {
        let configs = subkernel_configs(&cfg(1)).unwrap();

        let query_concat = configs.query_concat.as_ref().unwrap();
        assert_eq!(query_concat.backends, vec!["triton"]);
        assert_eq!(query_concat.input_bytes_per_token, 73_728);
        assert_eq!(query_concat.output_bytes_per_token, 73_728);

        assert_eq!(configs.mla_cache_append.backends, vec!["vllm_cuda"]);
        assert_eq!(configs.mla_cache_append.kv_lora_rank, 512);
        assert_eq!(configs.mla_cache_append.rope_dim, 64);
        assert_eq!(configs.mla_cache_append.block_size, 64);
        assert_eq!(configs.mla_cache_append.input_dtype, DType::Bf16);
        assert_eq!(configs.mla_cache_append.kv_dtype, DType::Bf16);
        assert_eq!(configs.mla_cache_append.cache_format, "plain");

        let Some(IndexRemapConfig::Legacy(index_remap)) = &configs.index_remap else {
            panic!("legacy config must retain the elementwise remap")
        };
        assert_eq!(index_remap.backends, vec!["triton"]);
        assert_eq!(index_remap.input_bytes_per_token, 16_448);
        assert_eq!(index_remap.output_bytes_per_token, 8_192);

        let PrefillConfig::Legacy(prefill) = &configs.prefill else {
            panic!("legacy config must retain per-request sparse attention")
        };
        assert_eq!(prefill.backends, vec!["vllm_flashmla_bf16"]);
        assert_eq!(prefill.valid_counts_pattern, "causal_tail");
        assert_eq!(configs.decode.valid_counts_pattern, "uniform_full");
        for sparse in [prefill, &configs.decode] {
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
        let PrefillConfig::Legacy(prefill) = &configs.prefill else {
            panic!("legacy config must retain per-request sparse attention")
        };
        assert_eq!(prefill.valid_counts_pattern, "causal_tail");
        assert_eq!(configs.decode.valid_counts_pattern, "speculative_pairs");
    }

    #[test]
    fn exact_varlen_expands_into_production_remap_and_prefill_configs() {
        let configs = subkernel_configs(&exact_cfg(1)).unwrap();

        let Some(IndexRemapConfig::Production(remap)) = &configs.index_remap else {
            panic!("exact varlen config must select the dedicated remap")
        };
        assert_eq!(remap.backends, vec!["vllm_triton"]);
        assert_eq!(remap.gpu_name, "NVIDIA B200");
        assert_eq!(remap.max_blocks_per_request, 128);
        assert_eq!(remap.index_distribution, "unique_scattered_blocks");
        assert_eq!(remap.page_table_mapping, "request_contiguous");
        assert!(remap.return_valid_counts);

        let PrefillConfig::Production(prefill) = &configs.prefill else {
            panic!("exact varlen config must select the one-launch prefill kernel")
        };
        assert_eq!(prefill.backends, vec!["flashinfer_trtllm_fp8"]);
        assert_eq!(prefill.max_model_len, 8192);
        assert_eq!(prefill.num_heads, 16);
        assert_eq!(prefill.q_dtype, DType::Fp8E4m3);
        assert_eq!(prefill.cache_dtype, DType::Fp8E4m3);
        assert_eq!(prefill.output_dtype, DType::Bf16);
        assert_eq!(prefill.index_distribution, "recent_contiguous");
        assert_eq!(prefill.cache_layout, "hnd_paged_mqa_fp8_latent_rope");

        assert_eq!(configs.mla_cache_append.kv_dtype, DType::Fp8E4m3);
        assert_eq!(configs.decode.backends, vec!["flashinfer_trtllm_fp8"]);
        assert_eq!(configs.decode.valid_counts_pattern, "uniform_full");
        assert_eq!(configs.decode.q_dtype, DType::Fp8E4m3);
        assert_eq!(configs.decode.cache_dtype, DType::Fp8E4m3);
        assert_eq!(configs.decode.output_dtype, DType::Bf16);
        assert_eq!(configs.decode.index_distribution, "unique_scattered_pages");
    }

    #[test]
    fn fused_sglang_graph_omits_concat_and_remap_and_appends_fp8_input() {
        let mut config = exact_cfg(1);
        config.launch_graph = DsaSparseMlaLaunchGraph::FusedQueryConcatAndIndexRemap;
        config.elementwise_backends.clear();
        config.exact_varlen.as_mut().unwrap().page_table_mapping = None;

        let configs = subkernel_configs(&config).unwrap();
        assert!(configs.query_concat.is_none());
        assert!(configs.index_remap.is_none());
        assert_eq!(configs.mla_cache_append.input_dtype, DType::Fp8E4m3);
        assert!(matches!(configs.prefill, PrefillConfig::Production(_)));
    }

    #[test]
    fn fused_graph_rejects_ignored_separate_launch_identity() {
        let mut missing_exact = cfg(1);
        missing_exact.launch_graph = DsaSparseMlaLaunchGraph::FusedQueryConcatAndIndexRemap;
        missing_exact.elementwise_backends.clear();
        assert!(subkernel_configs(&missing_exact).is_err());

        let mut config = exact_cfg(1);
        config.launch_graph = DsaSparseMlaLaunchGraph::FusedQueryConcatAndIndexRemap;
        assert!(subkernel_configs(&config).is_err());

        config.elementwise_backends.clear();
        assert!(subkernel_configs(&config).is_err());

        config.exact_varlen.as_mut().unwrap().page_table_mapping = None;
        assert!(subkernel_configs(&config).is_ok());
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
    fn legacy_byte_formulas_and_selected_k_tile_guard_are_exact() {
        let config = cfg(1);
        let query = query_concat_config(&config).unwrap();
        let remap = legacy_index_remap_config(&config).unwrap();
        assert_eq!(query.input_bytes_per_token, 73_728);
        assert_eq!(query.output_bytes_per_token, 73_728);
        assert_eq!(remap.input_bytes_per_token, 16_448);
        assert_eq!(remap.output_bytes_per_token, 8_192);

        let mut invalid = config;
        invalid.selected_k = 2050;
        assert!(matches!(
            legacy_index_remap_config(&invalid),
            Err(BuildError::FitFailed { reason, .. }) if reason.contains("divisible by 128")
        ));
    }

    #[test]
    fn production_remap_preserves_decode_first_request_and_row_topology() {
        let input = DsaSparseMlaAttentionInput {
            num_new_tokens: 28,
            prefill_query_cache_pairs: vec![(8, 128), (16, 256)],
            decode_query_cache: Some((4, 600)),
            decode_context_lens: None,
        };
        let remap = production_index_remap_input(&input, 2, 2048)
            .unwrap()
            .expect("nonempty input must produce remap work");

        assert_eq!(remap.request_row_counts, vec![2, 2, 8, 16]);
        assert_eq!(&remap.local_span_lengths[..4], &[599, 600, 599, 600]);
        assert_eq!(
            &remap.local_span_lengths[4..12],
            &(121..=128).collect::<Vec<_>>()
        );
        assert_eq!(
            &remap.local_span_lengths[12..],
            &(241..=256).collect::<Vec<_>>()
        );
        assert_eq!(remap.valid_counts, remap.local_span_lengths);
        assert!(remap.workspace_partition.is_none());
    }

    #[test]
    fn production_remap_handles_empty_and_enforces_launch_caps() {
        assert!(
            production_index_remap_input(&DsaSparseMlaAttentionInput::default(), 1, 2048)
                .unwrap()
                .is_none()
        );

        let too_many_requests = DsaSparseMlaAttentionInput {
            prefill_query_cache_pairs: vec![(1, 1); 257],
            ..Default::default()
        };
        assert!(production_index_remap_input(&too_many_requests, 1, 2048)
            .unwrap_err()
            .contains("at most 256 requests"));

        let supported_max_rows = DsaSparseMlaAttentionInput {
            prefill_query_cache_pairs: vec![(8192, 8192)],
            ..Default::default()
        };
        assert!(production_index_remap_input(&supported_max_rows, 1, 2048)
            .unwrap()
            .is_some());

        let too_many_rows = DsaSparseMlaAttentionInput {
            prefill_query_cache_pairs: vec![(8192, 8192), (1, 1)],
            ..Default::default()
        };
        assert!(production_index_remap_input(&too_many_rows, 1, 2048)
            .unwrap_err()
            .contains("at most 8192 query rows"));
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
            decode_context_lens: None,
        };
        let normalized = normalize_input(&input, 1).unwrap();
        assert_eq!(normalized.query_rows, 56);
        let decode = normalized.decode.unwrap();
        assert_eq!(decode.num_queries, 32);
        assert_eq!(decode.num_cache_tokens, 8192);
        assert_eq!(input.num_new_tokens, 37);
    }

    #[test]
    fn exact_decode_lengths_project_to_mean_plus_mad_and_remain_in_remap() {
        let input = DsaSparseMlaAttentionInput {
            num_new_tokens: 4,
            prefill_query_cache_pairs: Vec::new(),
            decode_query_cache: Some((4, 190)),
            decode_context_lens: Some(vec![12, 190, 12, 190]),
        };
        let normalized = normalize_input(&input, 1).unwrap();
        let decode = normalized.decode.unwrap();
        assert_eq!(decode.num_queries, 4);
        assert_eq!(decode.num_cache_tokens, 190);
        assert_eq!(normalized.decode_context_lens, vec![12, 190, 12, 190]);

        let remap = production_index_remap_input(&input, 1, 2048)
            .unwrap()
            .expect("decode rows must produce remap work");
        assert_eq!(remap.request_row_counts, vec![1, 1, 1, 1]);
        assert_eq!(remap.local_span_lengths, vec![12, 190, 12, 190]);
        assert_eq!(remap.valid_counts, vec![12, 190, 12, 190]);
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
