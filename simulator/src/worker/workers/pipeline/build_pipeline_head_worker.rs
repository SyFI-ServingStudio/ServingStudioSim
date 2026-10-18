//! Production pipeline-head recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{PoolId, SharedRequests, Time, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{
    PendingOrder, PipelinedChunkedPrefillAdmission, PrefixFetch, SessionPrefixTiers,
};
use crate::worker::config::{MicrobatchSizing, PrefixTierLoading};
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::{
    FullAttnKv, HybridGdnKv, PrefixCacheConfig, PrefixKv, ResumePoint, StateRetention,
};
use crate::worker::types::WorkerConfig;
use crate::worker::workers::iter_build_essentials::{
    full_attention_token_capacity, prepare_iter_build_essentials,
};

use super::pipeline_head_worker::{PipelineHeadWorker, PipelineLayout};
use super::{HybridPipelineHead, PipelineHead};

/// `stage_model` is stage 0's layers. KV capacity comes from `layout`, not the
/// stage model: every stage holds the same tokens, so the stage with the most KV
/// bytes per token bounds the pipeline.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pipeline_head_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    stage_model: Arc<M>,
    layout: PipelineLayout,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> PipelineHead<M> {
    let max_batch_tokens = config
        .max_batch_tokens
        .expect("pipeline head requires max_batch_tokens");
    assert_eq!(
        stage_model.gpus_per_replica(),
        1,
        "a pipeline stage runs on one GPU"
    );
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
    let mut prefix_fetch = head_prefix_fetch(
        &config,
        &layout,
        ResumePoint::AnyToken,
        cost_log_dir.as_deref(),
        id,
    );
    let kv_capacity = full_attention_token_capacity(1, layout.kv_bytes_per_token, &config);
    let prefix_cache =
        config
            .prefix_cache
            .resolve_tokens(kv_capacity, layout.kv_bytes_per_token, 1);
    let essentials = prepare_iter_build_essentials(
        id,
        pool_tag,
        requests,
        &config,
        cost_log_dir,
        pool,
        gpu_name,
        &cluster,
        1,
        stage_model.cost_log_manifest(),
        kv_capacity,
        1,
    );
    let send_gid =
        cluster
            .borrow_mut()
            .register_comm_group(essentials.allocation_base, 1, pool_tag, id.0);
    let kv_store = FullAttnKv::with_prefix_cache(
        1,
        essentials.kv_capacity,
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    let kv_store = match prefix_fetch
        .as_mut()
        .and_then(PrefixFetch::hbm_eviction_log)
    {
        Some(log) => kv_store.with_hbm_eviction_log(log),
        None => kv_store,
    };
    let mut admission = PipelinedChunkedPrefillAdmission::new(
        PendingOrder::new(config.pending_order),
        (),
        max_batch_tokens,
    );
    if config.balance_decode_microbatches {
        admission = admission.with_balanced_decodes(layout.depth);
    }
    if let MicrobatchSizing::Even { min_tokens } = config.microbatch_sizing {
        admission = admission.with_even_split(layout.depth, min_tokens);
    }
    if let Some(threshold) = config.long_prefill_token_threshold {
        admission = admission.with_long_prefill_threshold(threshold);
    }
    if let Some(load) = config.load_budget {
        admission = admission.with_load_budget(load);
    }
    if config.srpt {
        admission = admission.with_srpt();
    }
    if config.force_schedule_after_ms > 0.0 {
        admission = admission.with_force_after(config.force_schedule_after_ms);
    }
    let has_tiers = prefix_fetch.is_some();
    if let Some(fetch) = prefix_fetch {
        admission = admission.with_prefix_fetch(fetch);
    }

    with_head_options(
        PipelineHeadWorker::from_components(
            essentials.context,
            kv_store,
            admission,
            UnifiedIterExecution::new(stage_model, essentials.cost),
            layout,
            send_gid,
        ),
        has_tiers,
        config.external_decode,
    )
}

/// How a hybrid recurrent + full-attention pipeline shares one block pool,
/// following vLLM's hybrid KV cache manager.
///
/// Every cache group draws `block_tokens`-token blocks from one pool. The pool
/// holds as many blocks as the stage with the most bytes per block affords
/// (`PipelineLayout::kv_bytes_per_token` x `block_tokens`), less vLLM's null
/// block. A request holds one attention block per `block_tokens` of context
/// plus `state_blocks_per_request` fixed blocks (its recurrent-state groups and
/// any per-request scratch group).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineHybridState {
    pub block_tokens: u32,
    pub state_blocks_per_request: u32,
    /// vLLM's Mamba `align` mode: with prefix caching on, every non-final
    /// chunk ends on a `block_tokens` boundary, and a retained prefix keeps one
    /// state per block and resumes at the last one. `false` chunks plainly and
    /// keeps one state at a retained prefix's end: a hit is the whole prefix
    /// when the prompt extends it, else nothing.
    pub align_mode: bool,
}

impl PipelineHybridState {
    /// Pool capacity in attention-token units: usable blocks x block tokens.
    pub fn capacity_tokens(&self, layout: &PipelineLayout, attn_kv_bytes: u64) -> u64 {
        let block_tokens = u64::from(self.block_tokens.max(1));
        let bytes_per_block = layout.kv_bytes_per_token.max(1) * block_tokens;
        (attn_kv_bytes / bytes_per_block).saturating_sub(1).max(1) * block_tokens
    }

    /// One request's fixed blocks, in the same token units.
    pub fn state_tokens_per_request(&self) -> u64 {
        u64::from(self.state_blocks_per_request) * u64::from(self.block_tokens)
    }
}

/// [`build_pipeline_head_worker`] for a hybrid model: `HybridGdnKv` charges
/// each request its fixed state blocks on top of its context. With prefix
/// caching on and `hybrid.align_mode` set (vLLM's Mamba `align`
/// mode), every non-final chunk ends on a `block_tokens` boundary and prefix
/// hits floor to it; otherwise chunks are plain and a hit is a whole retained
/// context, resumed from the state at its end.
///
/// Context is charged by the token, not rounded up to whole blocks, so a
/// request's charge is low by less than one block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_hybrid_pipeline_head_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    stage_model: Arc<M>,
    layout: PipelineLayout,
    hybrid: PipelineHybridState,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> HybridPipelineHead<M> {
    let max_batch_tokens = config
        .max_batch_tokens
        .expect("pipeline head requires max_batch_tokens");
    assert_eq!(
        stage_model.gpus_per_replica(),
        1,
        "a pipeline stage runs on one GPU"
    );
    assert!(
        hybrid.block_tokens > 0,
        "hybrid block size must be positive"
    );
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
    let kv_capacity = hybrid.capacity_tokens(&layout, config.attn_kv_bytes);
    let state_tokens_per_request = hybrid.state_tokens_per_request();
    let mut prefix_fetch = head_prefix_fetch(
        &config,
        &layout,
        if hybrid.align_mode {
            ResumePoint::EveryInterval {
                tokens: hybrid.block_tokens,
            }
        } else {
            ResumePoint::ContextEnd
        },
        cost_log_dir.as_deref(),
        id,
    );
    let chunk_end_quantum = (hybrid.align_mode
        && !matches!(config.prefix_cache, PrefixCacheConfig::Disabled))
    .then_some(hybrid.block_tokens);
    tracing::info!(
        worker = id.0,
        pool_tag,
        kv_capacity_tokens = kv_capacity,
        block_tokens = hybrid.block_tokens,
        state_tokens_per_request,
        tier_state_bytes_per_request = layout.tier_state_bytes_per_request,
        tier_stored_state_bytes_per_request =
            config.tier_stored_state_bytes(layout.tier_state_bytes_per_request),
        chunk_end_quantum,
        max_batch_tokens,
        "hybrid pipeline head: one block pool sized by the most constrained stage"
    );
    let prefix_cache =
        config
            .prefix_cache
            .resolve_tokens(kv_capacity, layout.kv_bytes_per_token, 1);
    let essentials = prepare_iter_build_essentials(
        id,
        pool_tag,
        requests,
        &config,
        cost_log_dir,
        pool,
        gpu_name,
        &cluster,
        1,
        stage_model.cost_log_manifest(),
        kv_capacity,
        1,
    );
    let send_gid =
        cluster
            .borrow_mut()
            .register_comm_group(essentials.allocation_base, 1, pool_tag, id.0);
    let kv_store = HybridGdnKv::new(
        1,
        essentials.kv_capacity,
        state_tokens_per_request,
        hybrid.block_tokens,
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    assert!(
        config.state_retention().end_only() || !hybrid.align_mode,
        "recurrent_state_checkpoints > 1 or anchors need a state-at-end hybrid (plain chunking)"
    );
    let kv_store = if hybrid.align_mode {
        kv_store
    } else {
        kv_store.with_end_state_reuse(config.state_retention())
    };
    let mut kv_store = match prefix_fetch
        .as_mut()
        .and_then(PrefixFetch::hbm_eviction_log)
    {
        Some(log) => kv_store.with_hbm_eviction_log(log),
        None => kv_store,
    };
    match config.prefix_tier_loading {
        PrefixTierLoading::Held if config.prefix_tier_loading_area > 0.0 => {
            // HBM's loading area in the store's tokens, and in the tiers' for
            // the reads that fill it.
            let fraction = config.prefix_tier_loading_area;
            let area = (kv_store.capacity_tokens(0) as f64 * fraction) as u64;
            kv_store.set_loading_area(0, area);
            let tier_area = (area as f64 * layout.kv_bytes_per_token as f64
                / layout.tier_kv_bytes_per_token.max(1) as f64) as u64;
            prefix_fetch = prefix_fetch.map(|fetch| fetch.with_loading_area(tier_area, fraction));
        }
        PrefixTierLoading::Held => {}
        PrefixTierLoading::Bounded => {
            // pp_size microbatches of max_batch_tokens fill the pipeline.
            let ready_bound = u64::from(layout.depth) * u64::from(max_batch_tokens);
            let model = Arc::clone(&stage_model);
            let multiplier = config
                .prefill_gpu_time_multiplier
                .unwrap_or(config.gpu_time_multiplier);
            // Reads hold their HBM blocks like running requests; the
            // in-flight area only bounds the bytes in progress.
            prefix_fetch = prefix_fetch.map(|fetch| {
                fetch.with_bounded_loading(
                    move || prefill_batch_period(model.as_ref(), max_batch_tokens, multiplier),
                    ready_bound,
                )
            });
        }
    }
    let admission = PipelinedChunkedPrefillAdmission::new(
        PendingOrder::new(config.pending_order),
        (),
        max_batch_tokens,
    );
    let mut admission = match chunk_end_quantum {
        Some(quantum) => admission.with_chunk_end_quantum(quantum),
        None => admission,
    };
    if config.balance_decode_microbatches {
        admission = admission.with_balanced_decodes(layout.depth);
    }
    if let MicrobatchSizing::Even { min_tokens } = config.microbatch_sizing {
        admission = admission.with_even_split(layout.depth, min_tokens);
    }
    if let Some(threshold) = config.long_prefill_token_threshold {
        admission = admission.with_long_prefill_threshold(threshold);
    }
    if let Some(load) = config.load_budget {
        admission = admission.with_load_budget(load);
    }
    if config.srpt {
        admission = admission.with_srpt();
    }
    if config.force_schedule_after_ms > 0.0 {
        admission = admission.with_force_after(config.force_schedule_after_ms);
    }
    let has_tiers = prefix_fetch.is_some();
    if let Some(fetch) = prefix_fetch {
        admission = admission.with_prefix_fetch(fetch);
    }

    with_head_options(
        PipelineHeadWorker::from_components(
            essentials.context,
            kv_store,
            admission,
            UnifiedIterExecution::new(stage_model, essentials.cost),
            layout,
            send_gid,
        ),
        has_tiers,
        config.external_decode,
    )
}

/// One batch period of the pipeline: the stage time of a `max_batch_tokens`
/// prefill microbatch from an empty context, from `stage_model`'s cost model
/// with the prefill time multiplier. Stages are balanced and each holds one
/// microbatch, so the pipeline prefills `max_batch_tokens` per period.
fn prefill_batch_period<M: IterwiseUnifiedModel>(
    stage_model: &M,
    max_batch_tokens: u32,
    multiplier: f64,
) -> Time {
    let input = UnifiedArchInput {
        groups: vec![ArchGroupInput {
            batch_tokens: max_batch_tokens,
            prefill_tokens: max_batch_tokens,
            prefill_chunk_pairs: vec![(0, max_batch_tokens)],
            ..ArchGroupInput::default()
        }],
        tokens_per_source_rank: Vec::new(),
    };
    let ms = stage_model
        .eval_iter(&input, &mut Vec::new(), &mut Vec::new())
        .m
        .time_ms;
    Time::from_ms(f64::from(ms) * multiplier)
}

/// Reads from the DRAM/SSD tiers `config` asks for, in the pipeline's per-GPU
/// token units. A context's recurrent state is charged as
/// `layout.tier_state_bytes_per_request` in those units, stored compressed by
/// `config.tier_state_compression_ratio`.
fn head_prefix_fetch(
    config: &WorkerConfig,
    layout: &PipelineLayout,
    resume: ResumePoint,
    log_dir: Option<&std::path::Path>,
    id: WorkerId,
) -> Option<PrefixFetch> {
    let specs: Vec<_> = config.prefix_tiers.iter().flatten().copied().collect();
    let token_bytes = layout.tier_kv_bytes_per_token.max(1);
    (!specs.is_empty()).then(|| {
        let mut tiers = SessionPrefixTiers::new(
            &specs,
            layout.tier_kv_bytes_per_token,
            layout.tier_state_bytes_per_request.div_ceil(token_bytes),
            resume,
            if resume == ResumePoint::ContextEnd {
                config.state_retention()
            } else {
                StateRetention::Latest(1)
            },
            config.states_added_per_round(),
            log_dir,
            &format!("w{}", id.0),
        )
        .with_stored_state_tokens(
            config
                .tier_stored_state_bytes(layout.tier_state_bytes_per_request)
                .div_ceil(token_bytes),
        );
        if config.recurrent_state_anchors_in_place && resume == ResumePoint::ContextEnd {
            tiers = tiers.with_anchors_in_place();
        }
        PrefixFetch::new(vec![tiers], config.prefix_tier_warm_start).with_max_read_wait_ms(
            config.prefix_tier_max_read_wait_ms,
            config.prefix_tier_read_wait_policy,
        )
    })
}

fn with_head_options<K, A, E>(
    head: PipelineHeadWorker<K, A, E>,
    prefix_tiers: bool,
    external_decode: bool,
) -> PipelineHeadWorker<K, A, E>
where
    K: crate::worker::kv::IterWorkerKv + crate::worker::kv::PrefixKv,
    A: crate::worker::admission::MicrobatchAdmission<K>,
    E: crate::worker::execution::IterModelExecution<
        K,
        Input = crate::arch::contract::UnifiedArchInput,
    >,
{
    let head = if prefix_tiers {
        head.with_prefix_tiers()
    } else {
        head
    };
    if external_decode {
        head.with_external_decode()
    } else {
        head
    }
}
