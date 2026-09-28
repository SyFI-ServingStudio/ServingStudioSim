//! Hard-capped chunked-prefill worker recipe for a hybrid recurrent +
//! full-attention arch.
//!
//! `build_chunked_prefill_worker` on every axis but KV: the store is
//! `HybridGdnKv`, so each request's recurrent state shares one capacity with the
//! per-token attention KV. With prefix caching on, vLLM runs such a model in its
//! Mamba `align` cache mode, which only checkpoints state at block boundaries and
//! therefore ends every non-final prefill chunk on one; the lifecycle gets the
//! arch's checkpoint interval as its chunk-end quantum to match.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{ChunkedPrefillAdmission, LoadBalance, PendingOrder};
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::{HybridGdnKv, PrefixCacheConfig};
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{HybridChunkedPrefillWorker, IterBatchWorker};
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
    let balance = if num_partitions == 1 {
        LoadBalance::Single
    } else {
        LoadBalance::RoundRobin { next: 0 }
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
    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        UnifiedIterExecution::new(model, essentials.cost),
    )
}
