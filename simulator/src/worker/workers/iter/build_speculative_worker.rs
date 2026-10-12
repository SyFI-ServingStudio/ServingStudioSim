//! Speculating chunked-prefill worker recipe.
//!
//! Same composition as `build_chunked_prefill_worker` on every axis but two: the
//! lifecycle carries `SpeculativeDecodeCompletion` instead of the single-token
//! one, and execution lowers a verify batch through `SpeculativeIterExecution`.
//! Both differences are the same underlying fact — a resident decode moves by a
//! whole accepted chain per iteration rather than one token — split across the
//! axis that owns advancement and the axis that owns model input.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::SpeculativeUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{
    ChunkedPrefillAdmission, LoadBalance, PendingOrder, SpeculativeDecodeCompletion,
};
use crate::worker::config::DpPlacement;
use crate::worker::execution::SpeculativeIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::{FullAttnKv, HybridGdnKv};
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{HybridSpeculativeWorker, IterBatchWorker, SpeculativeWorker};
use crate::worker::workers::iter_build_essentials::{
    full_attention_token_capacity, host_kv_tier, prepare_iter_build_essentials,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_speculative_worker<M: SpeculativeUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> SpeculativeWorker<M> {
    let max_batch_tokens = config
        .max_batch_tokens
        .expect("speculative worker requires max_batch_tokens");
    let draft_tokens = config.speculative_draft_tokens;
    assert!(
        draft_tokens > 0,
        "speculative worker requires speculative_draft_tokens > 0; \
         zero drafts is the chunked-prefill worker"
    );
    let drafting_slots = model.drafting_slots_per_request();
    assert!(
        draft_tokens
            .checked_add(1)
            .and_then(|width| width.checked_add(drafting_slots))
            .is_some_and(|width| width <= max_batch_tokens),
        "speculative verify width must fit max_batch_tokens"
    );
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
    let num_partitions = model.num_attn_dp_groups().max(1) as usize;
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
        num_partitions,
    );
    let host_tier = host_kv_tier(
        model.kv_host_offload(),
        model.total_kv_bytes_per_token(),
        &config,
    );
    let kv_store = FullAttnKv::with_prefix_cache(
        num_partitions,
        essentials.kv_capacity,
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    // A host-tier cache (HiSparse) charges a fixed device buffer per request
    // and bounds the summed context by host memory.
    let kv_store = match host_tier {
        Some(tier) => {
            tracing::info!(
                worker = id.0,
                pool_tag,
                kv_capacity_tokens = essentials.kv_capacity,
                device_tokens_per_request = tier.device_tokens_per_request,
                host_capacity_tokens = tier.host_capacity_tokens,
                "host-tier KV: per-request device buffer and host context cap"
            );
            kv_store.with_host_tier(tier)
        }
        None => kv_store,
    };
    let balance = match (num_partitions, config.dp_placement) {
        (1, _) => LoadBalance::Single,
        (_, DpPlacement::RoundRobin) => LoadBalance::RoundRobin { next: 0 },
        (_, DpPlacement::VllmLeastLoaded) => LoadBalance::LeastLoaded { next: 0 },
        (_, DpPlacement::KvLeastLoaded) => LoadBalance::KvLeastLoaded { next: 0 },
        (_, DpPlacement::KvFitLeastLoaded) => LoadBalance::KvFitLeastLoaded { next: 0 },
    };
    let admission = ChunkedPrefillAdmission::with_decode_completion(
        (0..num_partitions)
            .map(|_| (PendingOrder::new(config.pending_order), ()))
            .collect(),
        max_batch_tokens,
        config.batch_policy,
        config.kv_admission,
        balance,
        SpeculativeDecodeCompletion::new(
            draft_tokens,
            config.speculative_acceptance_seed.unwrap_or(0),
        ),
        drafting_slots,
    );

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        SpeculativeIterExecution::new(
            model,
            essentials.cost,
            draft_tokens,
            config.speculative_acceptance_seed.unwrap_or(0),
        ),
    )
}

/// `build_speculative_worker` with a `HybridGdnKv` store: each request's
/// recurrent state (target linear-attention layers plus any per-request draft
/// state) shares one capacity with the per-token attention KV, as in
/// `build_hybrid_chunked_prefill_worker`. Prefill chunks are not checkpoint
/// aligned here; the verify step is what this worker models.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_hybrid_speculative_worker<M: SpeculativeUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> HybridSpeculativeWorker<M> {
    let max_batch_tokens = config
        .max_batch_tokens
        .expect("speculative worker requires max_batch_tokens");
    let draft_tokens = config.speculative_draft_tokens;
    assert!(
        draft_tokens > 0,
        "speculative worker requires speculative_draft_tokens > 0; \
         zero drafts is the chunked-prefill worker"
    );
    let drafting_slots = model.drafting_slots_per_request();
    assert!(
        draft_tokens
            .checked_add(1)
            .and_then(|width| width.checked_add(drafting_slots))
            .is_some_and(|width| width <= max_batch_tokens),
        "speculative verify width must fit max_batch_tokens"
    );
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
    let num_partitions = model.num_attn_dp_groups().max(1) as usize;
    let kv_capacity = full_attention_token_capacity(
        model.num_attn_shards(),
        model.total_kv_bytes_per_token(),
        &config,
    );
    let state_tokens_per_request = model
        .recurrent_state_bytes_per_request()
        .div_ceil(model.total_kv_bytes_per_token().max(1));
    let checkpoint_interval_tokens = config
        .ssm_checkpoint_interval_tokens
        .unwrap_or_else(|| model.recurrent_checkpoint_interval_tokens());
    tracing::info!(
        worker = id.0,
        pool_tag,
        state_tokens_per_request,
        checkpoint_interval_tokens,
        max_batch_tokens,
        kv_capacity_tokens = kv_capacity,
        "hybrid speculative: recurrent state shares the attention capacity"
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
        num_partitions,
    );
    let kv_store = HybridGdnKv::new(
        num_partitions,
        essentials.kv_capacity,
        state_tokens_per_request,
        checkpoint_interval_tokens,
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    let balance = match (num_partitions, config.dp_placement) {
        (1, _) => LoadBalance::Single,
        (_, DpPlacement::RoundRobin) => LoadBalance::RoundRobin { next: 0 },
        (_, DpPlacement::VllmLeastLoaded) => LoadBalance::LeastLoaded { next: 0 },
        (_, DpPlacement::KvLeastLoaded) => LoadBalance::KvLeastLoaded { next: 0 },
        (_, DpPlacement::KvFitLeastLoaded) => LoadBalance::KvFitLeastLoaded { next: 0 },
    };
    let admission = ChunkedPrefillAdmission::with_decode_completion(
        (0..num_partitions)
            .map(|_| (PendingOrder::new(config.pending_order), ()))
            .collect(),
        max_batch_tokens,
        config.batch_policy,
        config.kv_admission,
        balance,
        SpeculativeDecodeCompletion::new(
            draft_tokens,
            config.speculative_acceptance_seed.unwrap_or(0),
        ),
        drafting_slots,
    );

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        SpeculativeIterExecution::new(
            model,
            essentials.cost,
            draft_tokens,
            config.speculative_acceptance_seed.unwrap_or(0),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::arch::contract::SpeculativeArchInput;
    use crate::common::{AcceptanceProfile, DecodingStrategy, RequestId, Time};
    use crate::test_helpers::{lm, shared_with, test_cluster};
    use crate::timing::LeafMetrics;
    use crate::worker::config::{BoundedFutureKvAdmissionConfig, KvAdmissionConfig};
    use crate::worker::iter_worker::IterWorker;
    use crate::worker::types::{WorkerEventCommon, WorkerMsgCommon};
    use crate::worker::DecodeRetractionPolicy;

    const DRAFT_TOKENS: u32 = 5;
    const PROMPT_TOKENS: u32 = 8;
    const OUTPUT_TOKENS: u32 = 12;

    /// Fixed 1 ms per iteration, mirroring `FakeModel` on the ordinary face, so
    /// a 1 ms tick step drives exactly one iteration and a test can count them.
    struct FakeSpeculativeModel;

    impl SpeculativeUnifiedModel for FakeSpeculativeModel {
        fn eval_speculative_iter(
            &self,
            batch: &SpeculativeArchInput,
            slots: &mut Vec<LeafMetrics>,
            _scratch: &mut Vec<LeafMetrics>,
        ) -> LeafMetrics {
            slots.clear();
            assert_eq!(batch.draft_tokens, DRAFT_TOKENS);
            for group in &batch.groups {
                assert_eq!(
                    group.total_kv_len,
                    group
                        .decode_requests
                        .iter()
                        .map(|request| request.kv_len - request.query_len)
                        .sum::<u32>(),
                    "real KV admission stores computed context, excluding pending queries",
                );
            }
            lm(1.0)
        }

        fn total_kv_bytes_per_token(&self) -> u64 {
            1024
        }

        fn max_model_len(&self) -> u32 {
            8192
        }

        fn gpus_per_replica(&self) -> u16 {
            1
        }
    }

    fn config(kv_admission: KvAdmissionConfig) -> WorkerConfig {
        WorkerConfig {
            attn_kv_bytes: 1_000_000_000,
            max_batch_tokens: Some(2048),
            kv_admission,
            speculative_draft_tokens: DRAFT_TOKENS,
            speculative_acceptance_seed: Some(7),
            ..WorkerConfig::default()
        }
    }

    fn build(
        config: WorkerConfig,
        requests: SharedRequests,
    ) -> SpeculativeWorker<FakeSpeculativeModel> {
        build_speculative_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeSpeculativeModel),
            requests,
            config,
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    /// One request at `accept_rate`, run to completion. Returns the number of
    /// 1 ms iterations it took.
    fn iterations_to_complete(accept_rate: f32) -> u64 {
        let store = shared_with(&[(0, PROMPT_TOKENS, OUTPUT_TOKENS)]);
        store.borrow_mut()[RequestId(0)].request.definition.decoding =
            DecodingStrategy::Speculative {
                accept_rate: AcceptanceProfile::Uniform(accept_rate),
            };
        let mut worker = build(config(KvAdmissionConfig::FullFootprint), store);
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));

        let mut events = Vec::new();
        for step in 0..200u64 {
            worker.tick(Time::from_ms(step as f64), &mut events);
            if events.contains(&WorkerEventCommon::RequestComplete {
                worker: WorkerId(0),
                req: RequestId(0),
            }) {
                return step;
            }
        }
        panic!("the request never completed");
    }

    #[test]
    fn the_hybrid_recipe_charges_recurrent_state_against_capacity() {
        // The defect this catches: building a hybrid speculative arch on the
        // full-attention store, which would leave each request's recurrent
        // state out of capacity and admit several times too many decodes.
        use std::sync::atomic::{AtomicU32, Ordering};
        const CAPACITY: u64 = 1000;
        struct StatefulModel {
            state_bytes: u64,
            most_decodes: AtomicU32,
        }
        impl SpeculativeUnifiedModel for StatefulModel {
            fn eval_speculative_iter(
                &self,
                batch: &SpeculativeArchInput,
                slots: &mut Vec<LeafMetrics>,
                scratch: &mut Vec<LeafMetrics>,
            ) -> LeafMetrics {
                for group in &batch.groups {
                    self.most_decodes
                        .fetch_max(group.decode_requests.len() as u32, Ordering::Relaxed);
                }
                FakeSpeculativeModel.eval_speculative_iter(batch, slots, scratch)
            }
            fn total_kv_bytes_per_token(&self) -> u64 {
                1024
            }
            fn recurrent_state_bytes_per_request(&self) -> u64 {
                self.state_bytes
            }
            fn max_model_len(&self) -> u32 {
                8192
            }
            fn gpus_per_replica(&self) -> u16 {
                1
            }
        }
        let most_decodes = |state_bytes: u64| {
            let rows: Vec<_> = (0..16)
                .map(|id| (id, PROMPT_TOKENS, OUTPUT_TOKENS))
                .collect();
            let requests = shared_with(&rows);
            for id in 0..16 {
                requests.borrow_mut()[RequestId(id)]
                    .request
                    .definition
                    .decoding = DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::Uniform(0.0),
                };
            }
            let model = Arc::new(StatefulModel {
                state_bytes,
                most_decodes: AtomicU32::new(0),
            });
            let mut config = config(KvAdmissionConfig::FullFootprint);
            config.attn_kv_bytes = CAPACITY * 1024;
            let mut worker = build_hybrid_speculative_worker(
                WorkerId(0),
                "test",
                model.clone(),
                requests.clone(),
                config,
                None,
                PoolId(0),
                "test-gpu",
                test_cluster(),
            );
            for id in 0..16 {
                worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
            }
            for tick in 0..400 {
                worker.tick(Time::from_ms(tick as f64), &mut Vec::new());
            }
            for id in 0..16 {
                assert!(requests.borrow()[RequestId(id)].lifecycle.completed);
            }
            model.most_decodes.load(Ordering::Relaxed)
        };
        let stateless = most_decodes(0);
        let stateful = most_decodes(200 * 1024);
        assert_eq!(stateless, 16);
        assert!(
            stateful * 3 < stateless,
            "200 state tokens per request must cap 1000-token capacity at a few \
             decodes, got {stateful} against {stateless}"
        );
    }

    #[test]
    fn the_recipe_composes_a_worker_the_shell_accepts() {
        // Compile-time proof of the production tuple: the `IterWorker` bound is
        // what rejects a mismatched KV / lifecycle / execution combination.
        fn assert_iter_worker<W: IterWorker>() {}
        assert_iter_worker::<SpeculativeWorker<FakeSpeculativeModel>>();
    }

    #[test]
    fn a_fully_accepting_engine_finishes_in_a_fraction_of_the_decode_steps() {
        // The defect this catches: wiring `SingleTokenDecodeCompletion` into the
        // speculative recipe. The worker would still run and still emit the
        // right token count — it would just silently model no speedup at all.
        let accepting = iterations_to_complete(1.0);
        let rejecting = iterations_to_complete(0.0);
        assert!(
            accepting * 2 < rejecting,
            "a k={DRAFT_TOKENS} chain retires up to {} tokens per verify, but the \
             accepting run took {accepting} iterations against {rejecting}",
            DRAFT_TOKENS + 1
        );
    }

    #[test]
    fn every_acceptance_rate_emits_exactly_the_requested_output_length() {
        // Acceptance changes how fast tokens retire, never how many: an
        // over-long chain past `target_output_tokens` must be truncated.
        for accept_rate in [0.0, 0.5, 1.0] {
            let store = shared_with(&[(0, PROMPT_TOKENS, OUTPUT_TOKENS)]);
            store.borrow_mut()[RequestId(0)].request.definition.decoding =
                DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::Uniform(accept_rate),
                };
            let mut worker = build(config(KvAdmissionConfig::FullFootprint), store.clone());
            worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
            for step in 0..200u64 {
                worker.tick(Time::from_ms(step as f64), &mut Vec::new());
            }

            let store = store.borrow();
            let record = &store[RequestId(0)];
            assert!(record.lifecycle.completed, "accept_rate={accept_rate}");
            let observation = record.telemetry.speculative.as_ref().unwrap();
            assert_eq!(observation.query_width, 6);
            assert_eq!(observation.prefill_chunks, 1);
            assert_eq!(observation.emitted_tokens, 11);
            assert_eq!(observation.pending_decode, None);
            assert_eq!(observation.pending_prefill, None);
            if accept_rate == 0.0 {
                assert_eq!(observation.decode_rounds, 11);
                assert_eq!(observation.resident_kv_sum, 143);
            } else if accept_rate == 1.0 {
                assert_eq!(observation.decode_rounds, 2);
                assert_eq!(observation.resident_kv_sum, 22);
            }
            assert_eq!(
                record.progress.output_tokens_emitted, OUTPUT_TOKENS,
                "accept_rate={accept_rate}"
            );
        }
    }

    #[test]
    fn bounded_future_admission_retracts_before_a_verify_overflows_kv() {
        // The defect this catches: sizing a speculating engine's decode headroom
        // at one token per request. A verify advances up to k+1 positions, so
        // the store would overfill silently and retract only afterwards.
        use std::sync::atomic::{AtomicU32, Ordering};
        const CAPACITY: u64 = 76;
        const OUTPUT: u32 = 24;
        struct CapacityModel {
            most_decodes: AtomicU32,
        }
        impl SpeculativeUnifiedModel for CapacityModel {
            fn eval_speculative_iter(
                &self,
                batch: &SpeculativeArchInput,
                slots: &mut Vec<LeafMetrics>,
                scratch: &mut Vec<LeafMetrics>,
            ) -> LeafMetrics {
                for group in &batch.groups {
                    self.most_decodes
                        .fetch_max(group.decode_requests.len() as u32, Ordering::Relaxed);
                }
                FakeSpeculativeModel.eval_speculative_iter(batch, slots, scratch)
            }
            fn total_kv_bytes_per_token(&self) -> u64 {
                1024
            }
            fn max_model_len(&self) -> u32 {
                8192
            }
            fn gpus_per_replica(&self) -> u16 {
                1
            }
        }
        let rows: Vec<_> = (0..4).map(|id| (id, PROMPT_TOKENS, OUTPUT)).collect();
        let requests = shared_with(&rows);
        for id in 0..4 {
            requests.borrow_mut()[RequestId(id)]
                .request
                .definition
                .decoding = DecodingStrategy::Speculative {
                accept_rate: AcceptanceProfile::Uniform(1.0),
            };
        }
        let model = Arc::new(CapacityModel {
            most_decodes: AtomicU32::new(0),
        });
        let mut config = config(KvAdmissionConfig::BoundedFuture(
            BoundedFutureKvAdmissionConfig {
                page_size: 1,
                max_future_tokens: 8,
                initial_new_token_ratio: 0.0,
                minimum_new_token_ratio: 0.0,
                new_token_ratio_decay_steps: 1,
                retract_decode_steps: 1,
                retraction_policy: DecodeRetractionPolicy::Length,
            },
        ));
        config.attn_kv_bytes = CAPACITY * 1024;
        let mut worker = build_speculative_worker(
            WorkerId(0),
            "test",
            model.clone(),
            requests.clone(),
            config,
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        for id in 0..4 {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        for tick in 0..400 {
            // `FullAttnKv::advance` asserts the store never overfills.
            worker.tick(Time::from_ms(tick as f64), &mut Vec::new());
        }

        let requests = requests.borrow();
        for id in 0..4 {
            let record = &requests[RequestId(id)];
            assert!(record.lifecycle.completed, "request {id}");
            assert_eq!(
                record.progress.output_tokens_emitted, OUTPUT,
                "request {id}"
            );
        }
        assert!(
            (0..4).any(|id| requests[RequestId(id)].telemetry.retraction_count > 0),
            "the pressure this test builds must force a retraction"
        );
        // Full-footprint would hold two (2 x 32 of 76 tokens); bounded-future
        // admits on the near future and decodes more at once.
        assert!(model.most_decodes.load(Ordering::Relaxed) > 2);
    }

    #[test]
    #[should_panic(expected = "speculative_draft_tokens > 0")]
    fn a_worker_that_drafts_nothing_is_not_a_speculative_worker() {
        let mut config = config(KvAdmissionConfig::FullFootprint);
        config.speculative_draft_tokens = 0;
        build(config, shared_with(&[]));
    }

    #[test]
    #[should_panic(expected = "verify width must fit max_batch_tokens")]
    fn a_batch_must_fit_at_least_one_verify_window() {
        let mut config = config(KvAdmissionConfig::FullFootprint);
        config.max_batch_tokens = Some(DRAFT_TOKENS);
        build(config, shared_with(&[]));
    }

    #[test]
    fn drafting_slots_are_charged_to_every_scheduled_request() {
        // The defect this catches: budgeting a parallel drafter's requests at
        // the verify width alone. vLLM charges DFlash K more slots per scheduled
        // request, so at 33 tokens and k=5 it runs 3 requests per step, not 5.
        use std::sync::atomic::{AtomicU32, Ordering};
        const BUDGET: u32 = 33;
        struct DraftSlotModel {
            drafting_slots: u32,
            most_requests: AtomicU32,
        }
        impl SpeculativeUnifiedModel for DraftSlotModel {
            fn eval_speculative_iter(
                &self,
                batch: &SpeculativeArchInput,
                slots: &mut Vec<LeafMetrics>,
                scratch: &mut Vec<LeafMetrics>,
            ) -> LeafMetrics {
                for group in &batch.groups {
                    let requests = group.request_count();
                    assert!(group.batch_tokens + requests * self.drafting_slots <= BUDGET);
                    self.most_requests.fetch_max(requests, Ordering::Relaxed);
                }
                FakeSpeculativeModel.eval_speculative_iter(batch, slots, scratch)
            }
            fn total_kv_bytes_per_token(&self) -> u64 {
                1024
            }
            fn max_model_len(&self) -> u32 {
                8192
            }
            fn gpus_per_replica(&self) -> u16 {
                1
            }
            fn drafting_slots_per_request(&self) -> u32 {
                self.drafting_slots
            }
        }
        for (drafting_slots, expected_requests) in [(0, 5), (DRAFT_TOKENS, 3)] {
            let rows: Vec<_> = (0..8).map(|id| (id, 1, OUTPUT_TOKENS)).collect();
            let requests = shared_with(&rows);
            for id in 0..8 {
                requests.borrow_mut()[RequestId(id)]
                    .request
                    .definition
                    .decoding = DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::Uniform(0.0),
                };
            }
            let model = Arc::new(DraftSlotModel {
                drafting_slots,
                most_requests: AtomicU32::new(0),
            });
            let mut config = config(KvAdmissionConfig::FullFootprint);
            config.max_batch_tokens = Some(BUDGET);
            let mut worker = build_speculative_worker(
                WorkerId(0),
                "test",
                model.clone(),
                requests.clone(),
                config,
                None,
                PoolId(0),
                "test-gpu",
                test_cluster(),
            );
            for id in 0..8 {
                worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
            }
            for tick in 0..200 {
                worker.tick(Time::from_ms(tick as f64), &mut Vec::new());
            }
            for id in 0..8 {
                assert!(requests.borrow()[RequestId(id)].lifecycle.completed);
            }
            assert_eq!(
                model.most_requests.load(Ordering::Relaxed),
                expected_requests,
                "drafting_slots={drafting_slots}"
            );
        }
    }

    #[test]
    fn short_prefills_cannot_overfill_the_following_verify_batch() {
        struct BoundedBatchModel;
        impl SpeculativeUnifiedModel for BoundedBatchModel {
            fn eval_speculative_iter(
                &self,
                batch: &SpeculativeArchInput,
                slots: &mut Vec<LeafMetrics>,
                scratch: &mut Vec<LeafMetrics>,
            ) -> LeafMetrics {
                assert!(batch.groups.iter().all(|group| group.batch_tokens <= 8));
                FakeSpeculativeModel.eval_speculative_iter(batch, slots, scratch)
            }
            fn total_kv_bytes_per_token(&self) -> u64 {
                1024
            }
            fn max_model_len(&self) -> u32 {
                8192
            }
            fn gpus_per_replica(&self) -> u16 {
                1
            }
        }
        for (batch_policy, prompt_tokens) in [
            (crate::worker::config::BatchPolicy::Mix, 1),
            (crate::worker::config::BatchPolicy::Mix, 9),
            (
                crate::worker::config::BatchPolicy::SeparatePrefillPriority,
                1,
            ),
            (
                crate::worker::config::BatchPolicy::SeparatePrefillPriority,
                9,
            ),
        ] {
            let rows: Vec<_> = (0..8)
                .map(|id| (id, prompt_tokens, if id % 3 == 0 { 1 } else { 8 }))
                .collect();
            let requests = shared_with(&rows);
            for id in 0..8 {
                requests.borrow_mut()[RequestId(id)]
                    .request
                    .definition
                    .decoding = DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::Uniform(0.0),
                };
            }
            let mut config = config(KvAdmissionConfig::FullFootprint);
            config.max_batch_tokens = Some(8);
            config.batch_policy = batch_policy;
            let mut worker = build_speculative_worker(
                WorkerId(0),
                "test",
                Arc::new(BoundedBatchModel),
                requests.clone(),
                config,
                None,
                PoolId(0),
                "test-gpu",
                test_cluster(),
            );
            for id in 0..8 {
                worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
            }
            for tick in 0..100 {
                worker.tick(Time::from_ms(tick as f64), &mut Vec::new());
            }
            for id in 0..8 {
                assert!(requests.borrow()[RequestId(id)].lifecycle.completed);
            }
        }
    }
}
