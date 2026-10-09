//! Hard-capped chunked-prefill worker recipe for a hybrid recurrent +
//! full-attention arch.
//!
//! `build_chunked_prefill_worker` on every axis but KV: the store is
//! `HybridGdnKv`, so each request's recurrent state shares one capacity with the
//! per-token attention KV. With prefix caching on, vLLM runs such a model in its
//! Mamba `align` cache mode, which only checkpoints state at block boundaries and
//! therefore ends every non-final prefill chunk on one; the lifecycle gets the
//! arch's checkpoint interval as its chunk-end quantum to match, unless the
//! selector asks for `plain` chunking. `plain` also reuses prefixes exactly
//! (one state at a retained entry's end, `HybridGdnKv::with_exact_prefix_reuse`).
//!
//! The selector's DRAM/SSD tiers go to the admission (one set per attention DP
//! rank, read when a request reaches the head of its rank's queue; sessions
//! sticky to their rank), and the shell is wrapped in `SessionTierWorker`,
//! which writes finished contexts through and runs `external_decode`.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{
    ChunkedPrefillAdmission, LoadBalance, PendingOrder, PrefixFetch, SessionPrefixTiers,
};
use crate::worker::config::DpPlacement;
use crate::worker::config::PrefillChunkAlignment;
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::{HybridGdnKv, PrefixCacheConfig};
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{HybridChunkedPrefillWorker, IterBatchWorker};
use super::session_tier_worker::SessionTierWorker;
use crate::worker::workers::iter_build_essentials::{
    full_attention_token_capacity, prepare_iter_build_essentials,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_hybrid_chunked_prefill_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> HybridChunkedPrefillWorker<M> {
    let max_batch_tokens = config
        .max_batch_tokens
        .expect("chunked-prefill worker requires max_batch_tokens");
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
    let num_partitions = model.num_attn_dp_groups().max(1) as usize;
    let kv_capacity = full_attention_token_capacity(
        model.num_attn_shards(),
        model.total_kv_bytes_per_token(),
        &config,
    );
    // Whole-model quantities in one token space, as in `build_qwen36_hybrid_worker`.
    let state_tokens_per_request = model
        .recurrent_state_bytes_per_request()
        .div_ceil(model.total_kv_bytes_per_token().max(1));
    let checkpoint_interval_tokens = config
        .ssm_checkpoint_interval_tokens
        .unwrap_or_else(|| model.recurrent_checkpoint_interval_tokens());
    let chunk_end_quantum = (checkpoint_interval_tokens > 0
        && config.prefill_chunk_alignment == PrefillChunkAlignment::Checkpoint
        && !matches!(config.prefix_cache, PrefixCacheConfig::Disabled))
    .then_some(checkpoint_interval_tokens);
    tracing::info!(
        worker = id.0,
        pool_tag,
        state_tokens_per_request,
        checkpoint_interval_tokens,
        chunk_end_quantum,
        max_batch_tokens,
        kv_capacity_tokens = kv_capacity,
        "hybrid chunked prefill: recurrent state shares the attention capacity"
    );

    let exact_reuse = checkpoint_interval_tokens > 0
        && config.prefill_chunk_alignment == PrefillChunkAlignment::Plain;
    // One tier set per DP rank, sized and read per GPU of the rank.
    let tier_specs: Vec<_> = config.prefix_tiers.iter().flatten().copied().collect();
    let tiers: Vec<SessionPrefixTiers> = if tier_specs.is_empty() {
        Vec::new()
    } else {
        let bytes_per_gpu = model
            .total_kv_bytes_per_token()
            .div_ceil(u64::from(model.num_attn_shards().max(1)));
        let hit_quantum = if exact_reuse || checkpoint_interval_tokens == 0 {
            1
        } else {
            checkpoint_interval_tokens
        };
        (0..num_partitions)
            .map(|rank| {
                SessionPrefixTiers::new(
                    &tier_specs,
                    bytes_per_gpu,
                    state_tokens_per_request,
                    hit_quantum,
                    cost_log_dir.as_deref(),
                    &format!("w{}_p{rank}", id.0),
                )
            })
            .collect()
    };
    let sticky_sessions = !tiers.is_empty() || config.external_decode;

    let prefix_cache = config.prefix_cache.resolve_tokens(
        kv_capacity,
        model.total_kv_bytes_per_token(),
        model.num_attn_shards(),
    );
    let essentials = prepare_iter_build_essentials(
        id,
        pool_tag,
        requests,
        &config,
        cost_log_dir,
        pool,
        gpu_name,
        &cluster,
        model.gpus_per_replica(),
        model.cost_log_manifest(),
        kv_capacity,
        num_partitions,
    );
    let kv_store = HybridGdnKv::new(
        num_partitions,
        essentials.kv_capacity,
        state_tokens_per_request,
        checkpoint_interval_tokens,
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    let kv_store = if exact_reuse {
        kv_store.with_exact_prefix_reuse()
    } else {
        kv_store
    };
    let balance = match (num_partitions, config.dp_placement) {
        (1, _) => LoadBalance::Single,
        (_, DpPlacement::RoundRobin) => LoadBalance::RoundRobin { next: 0 },
        (_, DpPlacement::VllmLeastLoaded) => LoadBalance::LeastLoaded { next: 0 },
    };
    let admission = ChunkedPrefillAdmission::new(
        (0..num_partitions)
            .map(|_| (PendingOrder::new(config.pending_order), ()))
            .collect(),
        max_batch_tokens,
        config.batch_policy,
        config.kv_admission,
        balance,
    );
    let admission = match chunk_end_quantum {
        Some(quantum) => admission.with_chunk_end_quantum(quantum),
        None => admission,
    };
    let admission = match config.long_prefill_token_threshold {
        Some(threshold) => admission.with_long_prefill_threshold(threshold),
        None => admission,
    };
    let admission = if sticky_sessions {
        admission.with_sticky_sessions()
    } else {
        admission
    };
    let admission = if config.force_schedule_after_ms > 0.0 {
        admission.with_force_after(config.force_schedule_after_ms)
    } else {
        admission
    };
    let has_tiers = !tiers.is_empty();
    let admission = if has_tiers {
        admission.with_prefix_fetch(PrefixFetch::new(tiers, config.prefix_tier_warm_start))
    } else {
        admission
    };
    let shell = IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        UnifiedIterExecution::new(model, essentials.cost),
    );
    SessionTierWorker::new(shell, has_tiers, config.external_decode)
}
