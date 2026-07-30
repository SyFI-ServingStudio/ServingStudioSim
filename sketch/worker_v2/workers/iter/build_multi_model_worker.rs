use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::KvSampler;
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::admission::policy::FifoOrder;
use super::super::super::admission::MultiModelAdmission;
use super::super::super::execution::MultiModelIterExecution;
use super::super::super::kv::ModelPartitionedKv;
use super::super::super::shared::context::WorkerContext;
use super::iter_batch_worker::IterBatchWorker;

/// Worker #17 — multi-model co-serve (multi-arch, iter family). SAME shell; the KV is
/// `ModelPartitionedKv` (one pool per co-resident model, `ModelSwitchKv`), the admission is
/// `MultiModelAdmission` (its `SwitchModel` message moves the active model), and the execution is
/// `MultiModelIterExecution` (per-partition model). `models` is the co-resident set (one pool +
/// one cost cache each). Proves the execution/admission constrain only on `ModelSwitchKv`, never
/// on the concrete KV impl.
#[allow(clippy::too_many_arguments)]
pub fn build_multi_model_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    models: Vec<Arc<M>>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> IterBatchWorker<ModelPartitionedKv, MultiModelAdmission<FifoOrder>, MultiModelIterExecution<M>>
{
    let num_models = models.len().max(1);
    let head = models[0].clone();
    cluster
        .borrow_mut()
        .allocate(pool.0, id.0, head.gpus_per_replica(), gpu_name, pool_tag);
    let group_kv_bytes = config
        .attn_kv_bytes
        .saturating_mul(head.num_attn_shards().max(1) as u64);
    let total_capacity = (group_kv_bytes / head.total_kv_bytes_per_token().max(1)).max(1);
    // Static budget split across the co-resident models (shared-budget refinement TODO).
    let per_model_capacity = (total_capacity / num_models as u64).max(1);
    {
        let mut gpu_cluster = cluster.borrow_mut();
        for model_partition in 0..num_models as u16 {
            gpu_cluster.register_kv_capacity(
                pool_tag,
                pool.0,
                id.0,
                model_partition,
                per_model_capacity,
            );
        }
    }
    let sampler = KvSampler::open_opt(
        cost_log_dir.as_deref(),
        pool_tag,
        id,
        num_models,
        config.kv_log_stride,
    );
    // One cost cache per model (the models keep separate cost timelines).
    let costs: Vec<CostBuffers> = models
        .iter()
        .map(|model| {
            CostBuffers::new_iter(
                cost_log_dir.clone(),
                pool_tag,
                id,
                model.as_ref(),
                config.gpu_time_multiplier,
            )
        })
        .collect();

    let context = WorkerContext {
        id,
        pool,
        requests,
        log_output_token_times: config.log_output_token_times,
        log_stage_transitions: config.log_stage_transitions,
    };
    let kv_store =
        ModelPartitionedKv::new(num_models, per_model_capacity, config.admission, sampler);
    let admission = MultiModelAdmission::new(FifoOrder::new(), ());
    let execution = MultiModelIterExecution::new(models, costs);

    IterBatchWorker::from_components(context, kv_store, admission, execution)
}
