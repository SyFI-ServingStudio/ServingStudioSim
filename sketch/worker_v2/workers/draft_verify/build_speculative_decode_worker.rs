use std::path::PathBuf;
use std::sync::Arc;

use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::KvSampler;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::admission::policy::FifoOrder;
use super::super::super::admission::{DraftVerifyAdmission, LocalPrefillDecodeAdmission};
use super::super::super::execution::{AcceptanceOracle, DraftVerifyExecution, DraftVerifyModel};
use super::super::super::kv::FullAttnKv;
use super::super::super::shared::context::WorkerContext;
use super::draft_verify_worker::DraftVerifyWorker;

/// Build the S6 speculative worker.
///
/// `proposal_tokens` is the draft width, not an accepted-length promise.
/// `acceptance_oracle` produces a request-local runtime result each iteration;
/// `DraftVerifyExecution` evaluates the matching draft+target batch shape.
#[allow(clippy::too_many_arguments)]
pub fn build_speculative_decode_worker<M, O>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    acceptance_oracle: O,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
    proposal_tokens: u32,
    num_partitions: usize,
) -> DraftVerifyWorker<FullAttnKv, DraftVerifyAdmission<FifoOrder>, DraftVerifyExecution<M, O>>
where
    M: DraftVerifyModel,
    O: AcceptanceOracle,
{
    let num_partitions = num_partitions.max(1);
    let execution =
        DraftVerifyExecution::new(Arc::clone(&model), acceptance_oracle, proposal_tokens);
    let model_kv_layout = execution.model_kv_layout();

    cluster.borrow_mut().allocate(
        pool.0,
        id.0,
        execution.gpus_per_replica(),
        gpu_name,
        pool_tag,
    );
    let group_kv_bytes = config
        .attn_kv_bytes
        .saturating_mul(model_kv_layout.num_attn_shards.max(1) as u64);
    let kv_capacity = (group_kv_bytes / model_kv_layout.total_kv_bytes_per_token.max(1)).max(1);
    {
        let mut cluster = cluster.borrow_mut();
        for partition in 0..num_partitions as u16 {
            cluster.register_kv_capacity(pool_tag, pool.0, id.0, partition, kv_capacity);
        }
    }
    let sampler = KvSampler::open_opt(
        cost_log_dir.as_deref(),
        pool_tag,
        id,
        num_partitions,
        config.kv_log_stride,
    );
    let context = WorkerContext {
        id,
        pool,
        requests,
        log_output_token_times: config.log_output_token_times,
        log_stage_transitions: config.log_stage_transitions,
    };
    let kv_store = FullAttnKv::new(num_partitions, kv_capacity, config.admission, sampler);
    let local_admission = LocalPrefillDecodeAdmission::new(
        FifoOrder::new(),
        (),
        config.max_batch_tokens,
        config.balance,
    );
    let admission = DraftVerifyAdmission::new(local_admission);

    DraftVerifyWorker::from_components(context, kv_store, admission, execution)
}
