use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::admission::policy::FifoOrder;
use super::super::super::admission::PrefixPrefillDecodeAdmission;
use super::super::super::execution::UnifiedIterExecution;
use super::super::super::kv::ModeledPrefixCacheKv;
use super::iter_batch_worker::IterBatchWorker;
use super::unified_iter_build_essentials::prepare_unified_iter_build_essentials;

/// Worker #12 — prefix-cache-aware dense/DP (F1, iter family). SAME shell +
/// `UnifiedIterExecution`; the KV is `ModeledPrefixCacheKv` (one cache probe per attention-DP partition)
/// and `PrefixPrefillDecodeAdmission` chooses the best feasible partition, whose ownership stays
/// sticky through decode. The scalar `prefix_hit_pct` models the same hit rate on
/// every partition; tests can supply distinct per-partition rates directly to `ModeledPrefixCacheKv`.
#[allow(clippy::too_many_arguments)]
pub fn build_prefix_cache_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
    prefix_hit_pct: u32,
) -> IterBatchWorker<
    ModeledPrefixCacheKv,
    PrefixPrefillDecodeAdmission<FifoOrder>,
    UnifiedIterExecution<M>,
> {
    let num_partitions = model.num_attn_dp_groups().max(1) as usize;
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
        num_partitions,
    );
    let kv_store = ModeledPrefixCacheKv::new(
        num_partitions,
        essentials.kv_capacity,
        config.admission,
        essentials.sampler,
        vec![prefix_hit_pct; num_partitions],
    );
    let admission = PrefixPrefillDecodeAdmission::new(FifoOrder::new(), ());

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        essentials.execution,
    )
}
