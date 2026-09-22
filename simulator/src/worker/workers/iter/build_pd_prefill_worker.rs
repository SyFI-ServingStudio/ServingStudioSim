//! PD-prefill whole-iteration worker recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{PendingOrder, PrefillHandoffAdmission};
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::FullAttnKv;
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{IterBatchWorker, PdPrefillWorker};
use crate::worker::workers::iter_build_essentials::{
    full_attention_token_capacity, prepare_iter_build_essentials,
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
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
    let num_attn_shards = model.num_attn_shards().max(1);
    let kv_capacity = full_attention_token_capacity(
        model.num_attn_shards(),
        model.total_kv_bytes_per_token(),
        &config,
    );
    let prefix_cache = config.prefix_cache.resolve_tokens(
        kv_capacity,
        model.total_kv_bytes_per_token(),
        model.num_attn_shards(),
    );
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
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    let admission =
        PrefillHandoffAdmission::new(PendingOrder::new(config.pending_order), (), send_group_id);

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        UnifiedIterExecution::new(model, essentials.cost),
    )
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::rc::Rc;

    use arrow_array::{StringArray, UInt32Array, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    use super::*;
    use crate::common::{RequestId, SessionInput, Time};
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
        store.borrow_mut()[RequestId(0)].request.definition.session = SessionInput::Session {
            session_id: 7,
            session_start_time: Time::ZERO,
            declared_prefix_tokens: 100,
            rounds_in_session: 1,
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
        let log_directory = tempdir().unwrap();
        let store = shared_with(&[(0, 16, 3), (1, 20, 3)]);
        {
            let mut requests = store.borrow_mut();
            for request in [RequestId(0), RequestId(1)] {
                requests[request].request.definition.session = SessionInput::Session {
                    session_id: 7,
                    session_start_time: Time::ZERO,
                    declared_prefix_tokens: 100,
                    rounds_in_session: 1,
                };
            }
        }
        let config = WorkerConfig {
            attn_kv_bytes: 1_000,
            ..WorkerConfig::default()
        };
        let mut worker = build_pd_prefill_worker(
            WorkerId(0),
            "prefill",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            config,
            Some(log_directory.path().to_path_buf()),
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        let mut events = Vec::new();

        worker.enqueue(PdPrefillMsg::Request(RequestId(0)));
        for step in 0..20 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        worker.enqueue(PdPrefillMsg::ReleaseKv {
            req: RequestId(0),
            at: Time::from_ms(20.0),
        });
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
        assert_eq!(
            store.borrow()[RequestId(0)]
                .telemetry
                .prefix_cache_hit_tokens,
            Some(0)
        );
        assert_eq!(
            store.borrow()[RequestId(1)]
                .telemetry
                .prefix_cache_hit_tokens,
            Some(100)
        );

        drop(worker);
        let path = log_directory
            .path()
            .join("raw/prefix_cache_event/worker_prefill_0.parquet");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3);

        let operations = batch
            .column_by_name("operation")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let reasons = batch
            .column_by_name("reason")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let requests = batch
            .column_by_name("request_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let sequence = batch
            .column_by_name("sequence")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(
            (0..3).map(|row| operations.value(row)).collect::<Vec<_>>(),
            ["activate", "retain", "activate"]
        );
        assert_eq!(
            (0..3).map(|row| reasons.value(row)).collect::<Vec<_>>(),
            ["miss", "handoff-complete", "hit"]
        );
        assert_eq!(
            (0..3).map(|row| requests.value(row)).collect::<Vec<_>>(),
            [0, 0, 1]
        );
        assert_eq!(
            (0..3).map(|row| sequence.value(row)).collect::<Vec<_>>(),
            [0, 1, 2]
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
