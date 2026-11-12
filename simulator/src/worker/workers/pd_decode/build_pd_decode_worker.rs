//! Production PD-decode composition recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::admission::LoadBalance;
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::KvCacheStore;
use crate::worker::types::WorkerConfig;
use crate::worker::workers::iter_build_essentials::{
    prepare_iter_build_essentials, resolve_kv_accounting,
};

use super::pull_decode_worker::{PdDecodeWorker, PullDecodeWorker, PULL_BUDGET_FRACTION};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pd_decode_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> PdDecodeWorker<M> {
    let num_partitions = model.num_attn_dp_groups().max(1) as usize;
    let num_attn_shards = model.num_attn_shards().max(1);
    let partition_accounting =
        resolve_kv_accounting(&model.kv_cache_layout(), model.num_attn_shards(), &config);
    // A slice of the partition holds pulled KV until it enters a decode
    // partition; the rest is the decode store's.
    let pull_budget_units =
        ((partition_accounting.capacity_units() as f64 * PULL_BUDGET_FRACTION) as u64).max(1);
    let pull_budget_tokens = partition_accounting
        .tokens_fitting(pull_budget_units)
        .max(1);
    let kv_accounting = Arc::new(partition_accounting.reserving(pull_budget_units));

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
    let receive_group_id = cluster.borrow_mut().register_comm_group(
        essentials.allocation_base,
        num_attn_shards,
        pool_tag,
        id.0,
    );
    let balance = match config.balance {
        LoadBalance::Single if num_partitions > 1 => LoadBalance::RoundRobin { next: 0 },
        configured => configured,
    };
    let kv_store = KvCacheStore::without_prefix_cache(
        num_partitions,
        Arc::clone(&kv_accounting),
        essentials.sampler,
    );

    PullDecodeWorker::from_components(
        essentials.context,
        kv_store,
        UnifiedIterExecution::new(model, essentials.cost),
        balance,
        cluster,
        receive_group_id,
        pull_budget_tokens,
        kv_accounting,
    )
}
