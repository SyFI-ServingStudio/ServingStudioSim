//! Multi-partition hard-capped chunked-prefill worker recipe.
//!
//! When the arch's layout keeps a resumable recurrent state and prefix caching
//! is on, vLLM runs it in its Mamba `align` cache mode, which only checkpoints
//! state at block boundaries and therefore ends every non-final prefill chunk
//! on one. The lifecycle gets that block as its chunk-end quantum to match,
//! unless the selector asks for `plain` chunking.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{
    ChunkEndAlignment, ChunkedPrefillAdmission, LoadBalance, PendingOrder,
};
use crate::worker::config::DpPlacement;
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{ChunkedPrefillWorker, IterBatchWorker};
use crate::worker::workers::iter_build_essentials::{
    kv_cache_store, prefill_chunk_end_quantum, prepare_iter_build_essentials, resolve_kv_accounting,
};

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
    let max_batch_tokens = config
        .max_batch_tokens
        .expect("chunked-prefill worker requires max_batch_tokens");
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
    let num_partitions = model.num_attn_dp_groups().max(1) as usize;
    let kv_accounting =
        resolve_kv_accounting(&model.kv_cache_layout(), model.num_attn_shards(), &config);
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
    let kv_store = kv_cache_store(
        kv_accounting,
        num_partitions,
        model.num_attn_shards(),
        &config,
        essentials.sampler,
        prefix_cache_logger,
    );
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
    )
    .with_request_limits(
        config.max_prefill_requests_per_iteration,
        config.max_resident_requests_per_partition,
    );
    let admission = match chunk_end_quantum {
        Some(quantum) => admission.with_chunk_end_alignment(ChunkEndAlignment::on_blocks(quantum)),
        None => admission,
    };

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        UnifiedIterExecution::new(model, essentials.cost),
    )
    .with_async_lookahead(config.async_lookahead)
}
