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
use crate::worker::config::KvAdmissionConfig;
use crate::worker::execution::SpeculativeIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::FullAttnKv;
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{IterBatchWorker, SpeculativeWorker};
use crate::worker::workers::iter_build_essentials::{
    full_attention_token_capacity, prepare_iter_build_essentials,
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
    // Bounded-future admission predicts the next allocation from
    // `current_kv % page_size == 0`, which only holds while a decode advances
    // one token at a time. An accepted chain skips page boundaries, so the
    // prediction would under-allocate silently. Full-footprint reserves the
    // whole remaining output up front and is therefore safe at any step size.
    assert!(
        matches!(config.kv_admission, KvAdmissionConfig::FullFootprint),
        "speculative decode requires full-footprint KV admission: bounded-future \
         predicts page crossings from single-token advance and would under-allocate"
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
    let kv_store = FullAttnKv::with_prefix_cache(
        num_partitions,
        essentials.kv_capacity,
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    let balance = if num_partitions == 1 {
        LoadBalance::Single
    } else {
        LoadBalance::RoundRobin { next: 0 }
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
        SpeculativeIterExecution::new(model, essentials.cost, draft_tokens),
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
    #[should_panic(expected = "speculative decode requires full-footprint KV admission")]
    fn bounded_future_admission_cannot_carry_a_speculating_engine() {
        // The defect this catches: bounded-future predicts the next page
        // crossing from `current_kv % page_size == 0`, a predicate that only
        // holds at one token per step. An accepted chain steps over it and the
        // worker over-admits without ever reporting a capacity failure.
        build(
            config(KvAdmissionConfig::BoundedFuture(
                BoundedFutureKvAdmissionConfig {
                    page_size: 1,
                    max_future_tokens: 4,
                    initial_new_token_ratio: 0.7,
                    minimum_new_token_ratio: 0.098,
                    new_token_ratio_decay_steps: 600,
                    retract_decode_steps: 2,
                    retraction_policy: DecodeRetractionPolicy::Length,
                },
            )),
            shared_with(&[]),
        );
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
