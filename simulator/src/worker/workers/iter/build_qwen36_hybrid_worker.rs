//! Whole-iteration worker recipe for a hybrid recurrent + full-attention arch.
//!
//! Identical to `build_barebone_worker` on every axis except KV: the same
//! `IterBatchWorker` shell, the same `LocalPrefillDecodeAdmission`, the same
//! `UnifiedIterExecution`. Only the store changes, to `HybridGdnKv`, so that the
//! per-request recurrent state shares one capacity with the per-token attention
//! KV instead of being invisible to admission.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{LocalPrefillDecodeAdmission, PendingOrder};
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::HybridGdnKv;
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{IterBatchWorker, Qwen36HybridWorker};
use crate::worker::workers::iter_build_essentials::{
    full_attention_token_capacity, prepare_iter_build_essentials,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_qwen36_hybrid_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> Qwen36HybridWorker<M> {
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
    let kv_capacity = full_attention_token_capacity(
        model.num_attn_shards(),
        model.total_kv_bytes_per_token(),
        &config,
    );
    let kv_bytes_per_token = model.total_kv_bytes_per_token().max(1);
    // Both quantities are whole-model (all attention ranks summed), matching the
    // unit `full_attention_token_capacity` produced, so one token-space capacity
    // governs the recurrent state and the attention KV alike. Round up: a
    // partially-filled token page is still a page the request holds.
    let state_tokens_per_request = model
        .recurrent_state_bytes_per_request()
        .div_ceil(kv_bytes_per_token);
    let checkpoint_interval_tokens = config
        .ssm_checkpoint_interval_tokens
        .unwrap_or_else(|| model.recurrent_checkpoint_interval_tokens());
    tracing::info!(
        worker = id.0,
        pool_tag,
        recurrent_state_bytes_per_request = model.recurrent_state_bytes_per_request(),
        state_tokens_per_request,
        checkpoint_interval_tokens,
        checkpoint_interval_source = if config.ssm_checkpoint_interval_tokens.is_some() {
            "preset"
        } else {
            "arch"
        },
        kv_capacity_tokens = kv_capacity,
        "hybrid KV: recurrent state shares the attention capacity"
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
        1,
    );
    let kv_store = HybridGdnKv::new(
        1,
        essentials.kv_capacity,
        state_tokens_per_request,
        checkpoint_interval_tokens,
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    let admission = LocalPrefillDecodeAdmission::new(
        PendingOrder::new(config.pending_order),
        (),
        config.max_batch_tokens,
        config.balance,
    );

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        UnifiedIterExecution::new(model, essentials.cost),
    )
}
