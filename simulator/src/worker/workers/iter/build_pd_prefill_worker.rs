//! PD-prefill whole-iteration worker recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::admission::{FifoOrder, PrefillHandoffAdmission};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::FullAttnKv;
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{IterBatchWorker, PdPrefillWorker};
use crate::worker::workers::unified_iter_build_essentials::{
    full_attention_token_capacity, prefix_cache_token_capacity,
    prepare_unified_iter_build_essentials,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pd_prefill_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> PdPrefillWorker<M> {
    let num_attn_shards = model.num_attn_shards().max(1);
    let kv_capacity = full_attention_token_capacity(model.as_ref(), &config);
    let prefix_cache_capacity = prefix_cache_token_capacity(model.as_ref(), &config);
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
        1,
    );
    let send_group_id = cluster.borrow_mut().register_comm_group(
        essentials.allocation_base,
        num_attn_shards,
        pool_tag,
        id.0,
    );
    let kv_store = FullAttnKv::with_prefix_cache(
        1,
        essentials.kv_capacity,
        prefix_cache_capacity,
        config.prefix_cache_policy,
        essentials.sampler,
    );
    let admission = PrefillHandoffAdmission::new(FifoOrder::new(), (), send_group_id);

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        essentials.execution,
    )
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::common::{PrefixInput, RequestId, Time};
    use crate::test_helpers::{shared_with, test_cluster, FakeModel};
    use crate::worker::types::{PdPrefillEvent, PdPrefillMsg};

    #[test]
    fn prefill_emits_handoff() {
        let store = shared_with(&[(0, 16, 3)]);
        let mut worker = build_pd_prefill_worker(
            WorkerId(0),
            "prefill",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        worker.enqueue(PdPrefillMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..20 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            events,
            vec![PdPrefillEvent::PrefillDone {
                worker: WorkerId(0),
                req: RequestId(0),
                send_gid: 0,
                kv_tokens: 16,
            }]
        );
        assert_eq!(
            store.borrow()[RequestId(0)].progress.output_tokens_emitted,
            1
        );
    }

    #[test]
    fn cold_prefix_is_recomputed_and_handoff_carries_the_full_context() {
        let store = shared_with(&[(0, 16, 3)]);
        store.borrow_mut()[RequestId(0)].request.definition.prefix = PrefixInput::Session {
            session_id: 7,
            declared_prefix_tokens: 100,
        };
        let mut worker = build_pd_prefill_worker(
            WorkerId(0),
            "prefill",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        worker.enqueue(PdPrefillMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..20 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }

        assert_eq!(
            events,
            vec![PdPrefillEvent::PrefillDone {
                worker: WorkerId(0),
                req: RequestId(0),
                send_gid: 0,
                kv_tokens: 116,
            }]
        );
        assert_eq!(
            store.borrow()[RequestId(0)]
                .progress
                .prefill_tokens_processed,
            116
        );
    }

    #[test]
    fn handoff_ack_returns_session_kv_to_the_prefill_cache() {
        let store = shared_with(&[(0, 16, 3), (1, 20, 3)]);
        {
            let mut requests = store.borrow_mut();
            for request in [RequestId(0), RequestId(1)] {
                requests[request].request.definition.prefix = PrefixInput::Session {
                    session_id: 7,
                    declared_prefix_tokens: 100,
                };
            }
        }
        let config = WorkerConfig {
            attn_kv_bytes: 1_000,
            prefix_cache_capacity_bytes: 1_000,
            ..WorkerConfig::default()
        };
        let mut worker = build_pd_prefill_worker(
            WorkerId(0),
            "prefill",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            config,
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        let mut events = Vec::new();

        worker.enqueue(PdPrefillMsg::Request(RequestId(0)));
        for step in 0..20 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        worker.enqueue(PdPrefillMsg::ReleaseKv { req: RequestId(0) });
        worker.enqueue(PdPrefillMsg::Request(RequestId(1)));
        for step in 20..40 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }

        assert_eq!(
            events,
            vec![
                PdPrefillEvent::PrefillDone {
                    worker: WorkerId(0),
                    req: RequestId(0),
                    send_gid: 0,
                    kv_tokens: 116,
                },
                PdPrefillEvent::PrefillDone {
                    worker: WorkerId(0),
                    req: RequestId(1),
                    send_gid: 0,
                    kv_tokens: 120,
                },
            ]
        );
        assert_eq!(
            store.borrow()[RequestId(1)]
                .progress
                .prefill_tokens_processed,
            20
        );
    }

    #[test]
    fn one_token_request_completes_at_prefill() {
        let store = shared_with(&[(0, 16, 1)]);
        let mut worker = build_pd_prefill_worker(
            WorkerId(0),
            "prefill",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        worker.enqueue(PdPrefillMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..20 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            events,
            vec![PdPrefillEvent::RequestComplete {
                worker: WorkerId(0),
                req: RequestId(0),
            }]
        );
        assert!(store.borrow()[RequestId(0)].lifecycle.completed);
    }
}
