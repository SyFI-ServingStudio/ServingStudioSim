use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::FfnLayerwiseModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::execution::FfnSectionExecutionAdapter;
use super::super::super::shared::context::WorkerContext;
use super::buffered_ffn_worker::BufferedFfnWorker;

/// Worker #8 — AFD ffn, dense (the old `DisaggFfnWorker`). Wires `FfnSectionExecutionAdapter<M>`;
/// the 9-arg signature matches `DisaggFfnWorker::new` so the L6 factory is untouched.
#[allow(clippy::too_many_arguments)]
pub fn build_afd_ffn_worker<M: FfnLayerwiseModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> BufferedFfnWorker<FfnSectionExecutionAdapter<M>> {
    // Allocate the ffn replica's GPU block + one comm group spanning the whole
    // replica (ep_size links), used as both pull-recv and QKV-send endpoint.
    let communication_group_id = {
        let mut gpu_cluster = cluster.borrow_mut();
        let allocation_base =
            gpu_cluster.allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name, pool_tag);
        gpu_cluster.register_comm_group(allocation_base, model.gpus_per_replica(), pool_tag, id.0)
    };
    let cost = CostBuffers::new(
        cost_log_dir,
        pool_tag,
        id,
        &model.cost_log_manifest(),
        config.gpu_time_multiplier,
    );

    let context = WorkerContext {
        id,
        pool,
        requests,
        log_output_token_times: config.log_output_token_times,
        log_stage_transitions: config.log_stage_transitions,
    };
    let execution = FfnSectionExecutionAdapter::new(model, cost);

    BufferedFfnWorker::from_components(context, execution, cluster, communication_group_id)
}
