use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::KvSampler;
use crate::worker::admission_helpers::LoadBalance;
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::execution::UnifiedIterExecution;
use super::super::super::kv::FullAttnKv;
use super::super::super::shared::context::WorkerContext;
use super::pull_decode_worker::{PullDecodeWorker, PULL_BUDGET_FRAC};

/// Worker #4 (decode half) — PD decode. Reuses `FullAttnKv` + `UnifiedIterExecution`; the
/// shell adds the KV-pull front-end. Registers the recv comm group (the prefill
/// side's KV lands here). `N = model.num_attn_dp_groups()`.
#[allow(clippy::too_many_arguments)]
pub fn build_pd_decode_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> PullDecodeWorker<FullAttnKv, UnifiedIterExecution<M>> {
    let receive_group_id = {
        let mut gpu_cluster = cluster.borrow_mut();
        let allocation_base =
            gpu_cluster.allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name, pool_tag);
        gpu_cluster.register_comm_group(
            allocation_base,
            model.num_attn_shards().max(1),
            pool_tag,
            id.0,
        )
    };
    let num_groups = model.num_attn_dp_groups().max(1) as usize;
    let group_kv_bytes = config
        .attn_kv_bytes
        .saturating_mul(model.num_attn_shards().max(1) as u64);
    let total_shard_tokens = (group_kv_bytes / model.total_kv_bytes_per_token().max(1)).max(1);
    // Carve the backlog slice out of the active-decode budget (no double-spend).
    let pull_budget_tokens = ((total_shard_tokens as f64 * PULL_BUDGET_FRAC) as u64).max(1);
    let kv_capacity = total_shard_tokens.saturating_sub(pull_budget_tokens).max(1);
    {
        let mut gpu_cluster = cluster.borrow_mut();
        for partition in 0..num_groups as u16 {
            gpu_cluster.register_kv_capacity(pool_tag, pool.0, id.0, partition, kv_capacity);
        }
    }
    let sampler = KvSampler::open_opt(
        cost_log_dir.as_deref(),
        pool_tag,
        id,
        num_groups,
        config.kv_log_stride,
    );
    let cost = CostBuffers::new_iter(
        cost_log_dir,
        pool_tag,
        id,
        model.as_ref(),
        config.gpu_time_multiplier,
    );

    // Round-robin handed-off requests across the decode shards.
    let balance = match config.balance {
        LoadBalance::Single if num_groups > 1 => LoadBalance::RoundRobin { next: 0 },
        other => other,
    };
    let kv_bytes_per_token = model.total_kv_bytes_per_token();

    let context = WorkerContext {
        id,
        pool,
        requests,
        log_output_token_times: config.log_output_token_times,
        log_stage_transitions: config.log_stage_transitions,
    };
    let kv_store = FullAttnKv::new(num_groups, kv_capacity, config.admission, sampler);
    let execution = UnifiedIterExecution::new(model, cost);

    PullDecodeWorker::from_components(
        context,
        kv_store,
        execution,
        balance,
        cluster,
        receive_group_id,
        pull_budget_tokens,
        kv_bytes_per_token,
    )
}
