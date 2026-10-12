//! Multi-partition HP/DP whole-iteration worker recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{LoadBalance, LocalPrefillDecodeAdmission, PendingOrder};
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{HpUnifiedWorker, IterBatchWorker};
use crate::worker::workers::iter_build_essentials::{
    kv_cache_store, prepare_iter_build_essentials, resolve_kv_accounting,
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
    let kv_accounting =
        resolve_kv_accounting(&model.kv_cache_layout(), model.num_attn_shards(), &config);
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
    let balance = match config.balance {
        LoadBalance::Single if num_partitions > 1 => LoadBalance::RoundRobin { next: 0 },
        configured => configured,
    };
    let kv_store = kv_cache_store(
        kv_accounting,
        num_partitions,
        model.num_attn_shards(),
        &config,
        essentials.sampler,
        prefix_cache_logger,
    );
    let admission = LocalPrefillDecodeAdmission::new(
        PendingOrder::new(config.pending_order),
        (),
        config.max_batch_tokens,
        balance,
    );

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        UnifiedIterExecution::new(model, essentials.cost),
    )
}
