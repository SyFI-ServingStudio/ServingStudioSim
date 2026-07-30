//! Production AFD-FFN composition recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::FfnLayerwiseModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::execution::FfnSectionExecutionAdapter;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::WorkerConfig;

use super::buffered_ffn_worker::{BufferedFfnWorker, DisaggFfnWorker};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_afd_ffn_worker<M: FfnLayerwiseModel>(
    id: WorkerId,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
    cost_log_dir: Option<PathBuf>,
    pool_tag: &'static str,
) -> DisaggFfnWorker<M> {
    let communication_group_id = {
        let mut cluster = cluster.borrow_mut();
        let allocation_base =
            cluster.allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name, pool_tag);
        cluster.register_comm_group(allocation_base, model.gpus_per_replica(), pool_tag, id.0)
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
    BufferedFfnWorker::from_components(
        context,
        FfnSectionExecutionAdapter::new(model, cost),
        cluster,
        communication_group_id,
    )
}
