//! Production pipeline-head recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{PendingOrder, PipelinedChunkedPrefillAdmission};
use crate::worker::config::MicrobatchSizing;
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::{FullAttnKv, HybridGdnKv, PrefixCacheConfig};
use crate::worker::types::WorkerConfig;
use crate::worker::workers::iter_build_essentials::{
    full_attention_token_capacity, prepare_iter_build_essentials,
};

use super::head_prefix_tiers::HeadPrefixTiers;
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
    let prefix_tiers = head_prefix_tiers(&config, &layout, 0, 1, cost_log_dir.as_deref(), id);
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
        admission = admission.with_load_budget(
            load.low_tokens,
            load.backlog_lo_tokens,
            load.backlog_hi_tokens,
        );
    }
    if config.srpt {
        admission = admission.with_srpt();
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
        prefix_tiers,
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
    /// reuses exactly: a retained prefix keeps the state at its end and resumes
    /// at any token.
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
/// hits floor to it; otherwise chunks are plain and reuse is exact.
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
    let prefix_tiers = head_prefix_tiers(
        &config,
        &layout,
        state_tokens_per_request,
        if hybrid.align_mode {
            hybrid.block_tokens
        } else {
            1
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
    let kv_store = if hybrid.align_mode {
        kv_store
    } else {
        kv_store.with_exact_prefix_reuse()
    };
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
        admission = admission.with_load_budget(
            load.low_tokens,
            load.backlog_lo_tokens,
            load.backlog_hi_tokens,
        );
    }
    if config.srpt {
        admission = admission.with_srpt();
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
        prefix_tiers,
        config.external_decode,
    )
}

/// The DRAM/SSD tiers `config` asks for, in the pipeline's per-GPU token units.
fn head_prefix_tiers(
    config: &WorkerConfig,
    layout: &PipelineLayout,
    state_tokens: u64,
    hit_quantum: u32,
    log_dir: Option<&std::path::Path>,
    id: WorkerId,
) -> Option<HeadPrefixTiers> {
    let specs: Vec<_> = config.prefix_tiers.iter().flatten().copied().collect();
    (!specs.is_empty()).then(|| {
        HeadPrefixTiers::new(
            &specs,
            layout.kv_bytes_per_token,
            state_tokens,
            hit_quantum,
            log_dir,
            id,
        )
    })
}

fn with_head_options<K, A, E>(
    head: PipelineHeadWorker<K, A, E>,
    prefix_tiers: Option<HeadPrefixTiers>,
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
    let head = match prefix_tiers {
        Some(tiers) => head.with_prefix_tiers(tiers),
        None => head,
    };
    if external_decode {
        head.with_external_decode()
    } else {
        head
    }
}
