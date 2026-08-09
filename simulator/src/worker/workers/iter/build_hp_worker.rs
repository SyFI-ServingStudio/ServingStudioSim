//! Multi-partition HP/DP whole-iteration worker recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{LoadBalance, LocalPrefillDecodeAdmission, SessionStartOrder};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::FullAttnKv;
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{HpUnifiedWorker, IterBatchWorker};
use crate::worker::workers::unified_iter_build_essentials::{
    full_attention_token_capacity, prepare_unified_iter_build_essentials,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_hp_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> HpUnifiedWorker<M> {
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
    let num_partitions = model.num_attn_dp_groups().max(1) as usize;
    let kv_capacity = full_attention_token_capacity(model.as_ref(), &config);
    let prefix_cache = config.prefix_cache.resolve_tokens(
        kv_capacity,
        model.total_kv_bytes_per_token(),
        model.num_attn_shards(),
    );
    let essentials = prepare_unified_iter_build_essentials(
        id,
        pool_tag,
        model,
        requests,
        &config,
        cost_log_dir,
        pool,
        gpu_name,
        &cluster,
        kv_capacity,
        num_partitions,
    );
    let balance = match config.balance {
        LoadBalance::Single if num_partitions > 1 => LoadBalance::RoundRobin { next: 0 },
        configured => configured,
    };
    let kv_store = FullAttnKv::with_prefix_cache(
        num_partitions,
        essentials.kv_capacity,
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    let admission = LocalPrefillDecodeAdmission::new(
        SessionStartOrder::new(),
        (),
        config.max_batch_tokens,
        balance,
    );

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        essentials.execution,
    )
}
