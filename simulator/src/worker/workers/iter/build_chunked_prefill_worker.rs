//! Multi-partition hard-capped chunked-prefill worker recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{ChunkedPrefillAdmission, LoadBalance, PendingOrder};
use crate::worker::config::DpPlacement;
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::FullAttnKv;
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{ChunkedPrefillWorker, IterBatchWorker};
use crate::worker::workers::iter_build_essentials::{
    full_attention_token_capacity, host_kv_tier, prepare_iter_build_essentials,
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
    let kv_capacity = full_attention_token_capacity(
        model.num_attn_shards(),
        model.total_kv_bytes_per_token(),
        &config,
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
    let host_tier = host_kv_tier(
        model.kv_host_offload(),
        model.total_kv_bytes_per_token(),
        &config,
    );
    let kv_store = FullAttnKv::with_prefix_cache(
        num_partitions,
        essentials.kv_capacity,
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    // A host-tier cache (HiSparse) charges a fixed device buffer per request
    // and bounds the summed context by host memory.
    let kv_store = match host_tier {
        Some(tier) => {
            tracing::info!(
                worker = id.0,
                pool_tag,
                kv_capacity_tokens = essentials.kv_capacity,
                device_tokens_per_request = tier.device_tokens_per_request,
                host_capacity_tokens = tier.host_capacity_tokens,
                "host-tier KV: per-request device buffer and host context cap"
            );
            kv_store.with_host_tier(tier)
        }
        None => kv_store,
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

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        UnifiedIterExecution::new(model, essentials.cost),
    )
}
