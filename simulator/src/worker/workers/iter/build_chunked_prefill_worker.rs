//! Multi-partition hard-capped chunked-prefill worker recipe.
//!
//! When the arch's layout keeps a resumable recurrent state and prefix caching
//! is on, vLLM runs it in its Mamba `align` cache mode, which only checkpoints
//! state at block boundaries and therefore ends every non-final prefill chunk
//! on one. The lifecycle gets that block as its chunk-end quantum to match,
//! unless the selector asks for `plain` chunking.
//!
//! A model that reports a host tier (`kv_host_offload`, SGLang HiSparse) gets
//! it on the store: host memory bounds the summed context. Its fixed device
//! buffer per request is a part of the arch's KV layout.
//!
//! The selector's DRAM/SSD tiers go to the admission (one set per attention DP
//! rank, read when a request reaches the head of its rank's queue, bounded by
//! the read-wait bound and policy; sessions sticky to their rank).
//! [`build_session_chunked_prefill_worker`] wraps the shell in
//! `SessionTierWorker`, which writes finished contexts through and runs
//! `external_decode`; the bare [`build_chunked_prefill_worker`] refuses both.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::arch::kv_layout::KvCacheLayout;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{
    ChunkEndAlignment, ChunkedPrefillAdmission, LoadBalance, PendingOrder, PrefixFetch,
    SessionPrefixTiers,
};
use crate::worker::config::tier_state_bytes_per_request;
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{ChunkedPrefillWorker, IterBatchWorker, SessionChunkedPrefillWorker};
use super::session_tier_worker::SessionTierWorker;
use crate::worker::workers::iter_build_essentials::{
    host_kv_tier, kv_cache_store, prefill_chunk_end_quantum, prepare_iter_build_essentials,
    resolve_kv_accounting,
};

/// The chunked-prefill shell alone: no DRAM/SSD tiers, decode on this worker.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_chunked_prefill_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> ChunkedPrefillWorker<M> {
    assert!(
        config.prefix_tiers == [None, None] && !config.external_decode,
        "prefix tiers and external_decode need the session-tier chunked-prefill worker"
    );
    chunked_prefill_shell(
        id,
        pool_tag,
        model,
        requests,
        config,
        cost_log_dir,
        pool,
        gpu_name,
        cluster,
    )
    .0
}

/// The chunked-prefill shell under `SessionTierWorker`: with the selector's
/// DRAM/SSD tiers behind each DP rank's HBM cache and, with `external_decode`,
/// decode run elsewhere. Without either it behaves as the bare shell.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_session_chunked_prefill_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> SessionChunkedPrefillWorker<M> {
    let (shell, has_tiers) = chunked_prefill_shell(
        id,
        pool_tag,
        model,
        requests,
        config,
        cost_log_dir,
        pool,
        gpu_name,
        cluster,
    );
    SessionTierWorker::new(shell, has_tiers, config.external_decode)
}

/// One tier set per DP rank, sized and read per GPU of the rank. A tier hit
/// counts in `hit_quantum` tokens, the boundary HBM resumes a session on.
fn rank_prefix_tiers(
    config: &WorkerConfig,
    layout: &KvCacheLayout,
    num_attn_shards: u16,
    num_partitions: usize,
    hit_quantum: u32,
    log_dir: Option<&Path>,
    id: WorkerId,
) -> Vec<SessionPrefixTiers> {
    let tier_specs: Vec<_> = config.prefix_tiers.iter().flatten().copied().collect();
    if tier_specs.is_empty() {
        return Vec::new();
    }
    let bytes_per_token = layout.per_token_bytes().max(1);
    let bytes_per_gpu = bytes_per_token.div_ceil(u64::from(num_attn_shards.max(1)));
    let state_tokens_per_request = tier_state_bytes_per_request(layout).div_ceil(bytes_per_token);
    (0..num_partitions)
        .map(|rank| {
            SessionPrefixTiers::new(
                &tier_specs,
                bytes_per_gpu,
                state_tokens_per_request,
                hit_quantum,
                log_dir,
                &format!("w{}_p{rank}", id.0),
            )
        })
        .collect()
}

/// The shell, and whether its admission reads from DRAM/SSD tiers.
#[allow(clippy::too_many_arguments)]
fn chunked_prefill_shell<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> (ChunkedPrefillWorker<M>, bool) {
    let max_batch_tokens = config
        .max_batch_tokens
        .expect("chunked-prefill worker requires max_batch_tokens");
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
    let num_partitions = model.num_attn_dp_groups().max(1) as usize;
    let kv_cache_layout = model.kv_cache_layout();
    let kv_accounting = resolve_kv_accounting(&kv_cache_layout, model.num_attn_shards(), &config);
    let chunk_end_quantum = prefill_chunk_end_quantum(&kv_accounting, &config);
    tracing::info!(
        worker = id.0,
        pool_tag,
        kv_capacity_units = kv_accounting.capacity_units(),
        kv_bytes_per_unit = kv_accounting.bytes_per_unit(),
        chunk_end_quantum,
        max_batch_tokens,
        "chunked prefill KV"
    );
    let tiers = rank_prefix_tiers(
        &config,
        &kv_cache_layout,
        model.num_attn_shards(),
        num_partitions,
        chunk_end_quantum.unwrap_or(1),
        cost_log_dir.as_deref(),
        id,
    );
    let sticky_sessions = !tiers.is_empty() || config.external_decode;
    let host_tier = host_kv_tier(model.kv_host_offload(), &config);
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
        &kv_accounting,
        num_partitions,
    );
    let kv_capacity_units = kv_accounting.capacity_units();
    let kv_store = kv_cache_store(
        kv_accounting,
        num_partitions,
        model.num_attn_shards(),
        &config,
        essentials.sampler,
        prefix_cache_logger,
    );
    // A host-tier cache (HiSparse) bounds the summed context by host memory;
    // its fixed device buffer per request is a part of the arch's KV layout.
    let kv_store = match host_tier {
        Some(tier) => {
            tracing::info!(
                worker = id.0,
                pool_tag,
                kv_capacity_units,
                host_capacity_tokens = tier.host_capacity_tokens,
                "host-tier KV: host memory caps the summed context"
            );
            kv_store.with_host_tier(tier)
        }
        None => kv_store,
    };
    let balance = LoadBalance::for_dp(num_partitions, config.dp_placement);
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
        Some(quantum) => admission.with_chunk_end_alignment(ChunkEndAlignment::on_blocks(quantum)),
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
    let admission = match config.load_budget {
        Some(budget) => admission.with_load_budget(budget),
        None => admission,
    };
    let admission = if config.srpt {
        admission.with_srpt()
    } else {
        admission
    };
    let has_tiers = !tiers.is_empty();
    let admission = if has_tiers {
        admission.with_prefix_fetch(
            PrefixFetch::new(tiers, config.prefix_tier_warm_start).with_max_read_wait_ms(
                config.prefix_tier_max_read_wait_ms,
                config.prefix_tier_read_wait_policy,
            ),
        )
    } else {
        admission
    };

    let shell = IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        UnifiedIterExecution::new(model, essentials.cost),
    );
    (shell, has_tiers)
}
