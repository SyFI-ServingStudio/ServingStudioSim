//! How many tokens one resident decode request retires per iteration.
//!
//! This is the one thing speculative decoding changes about a lifecycle: an
//! ordinary engine emits exactly one token per resident request per iteration,
//! a speculating engine verifies a fixed-width window and retires between one
//! and `draft_tokens + 1`. Everything else about admission — placement,
//! chunking, capacity gates, retraction — is identical, so the difference is a
//! sub-axis of the lifecycle rather than a second lifecycle.
//!
//! A policy owns exactly two facts:
//!
//! - [`DecodeCompletion::query_tokens_per_request`], the *fixed* query width the
//!   engine submits per resident request. It is a batch-shape fact the
//!   lifecycle needs before the iteration runs (to size its token budget) and
//!   the execution adapter needs to lower the L4 input. It does not depend on
//!   how many tokens are ultimately accepted.
//! - [`DecodeCompletion::complete_decodes`], the post-iteration transition:
//!   emit the retired tokens, stamp finished requests, and grow KV by exactly
//!   as many positions as were retired.
//!
//! It does not own membership, capacity, or batch composition. The lifecycle
//! decides *whether* a partition runs decode; this decides *how far* it moves.

use crate::common::{AcceptanceProfile, DecodingStrategy, RequestId, Time, UnifiedStage};
use crate::worker::kv::IterWorkerKv;
use crate::worker::shared::advance_scope::{AdvanceScope, PartitionId};
use crate::worker::shared::context::WorkerContext;

/// Per-iteration decode advancement, as a lifecycle sub-axis.
///
/// Parameterized by the store rather than method-generic: the capability every
/// implementation needs — `IterWorkerKv` for decode membership plus its
/// `KvStore` supertrait for `advance` — is nameable at the implementation
/// boundary, so `ChunkedPrefillAdmission` carries `D` unbounded and states
/// `D: DecodeCompletion<K>` on its `IterAdmission<K>` impl.
pub trait DecodeCompletion<K: IterWorkerKv> {
    /// Query rows the engine submits per resident decode request each
    /// iteration. Fixed for the worker's lifetime; the token budget and the
    /// L4 input lowering both read it.
    fn query_tokens_per_request(&self) -> u32;

    /// Retire this iteration's accepted tokens on one partition, appending any
    /// request that finished to `completed`, and advance KV to match.
    fn complete_decodes(
        &mut self,
        kv_store: &mut K,
        partition: PartitionId,
        context: &WorkerContext,
        completed: &mut Vec<RequestId>,
        now: Time,
    );
}

/// The ordinary engine: one query row in, one token out, every resident request
/// moving the same single position.
#[derive(Clone, Copy, Debug, Default)]
pub struct SingleTokenDecodeCompletion;

impl<K: IterWorkerKv> DecodeCompletion<K> for SingleTokenDecodeCompletion {
    fn query_tokens_per_request(&self) -> u32 {
        1
    }

    fn complete_decodes(
        &mut self,
        kv_store: &mut K,
        partition: PartitionId,
        context: &WorkerContext,
        completed: &mut Vec<RequestId>,
        now: Time,
    ) {
        {
            let mut store = context.requests.borrow_mut();
            kv_store.visit_decode_members(partition, |request, _| {
                let record = &mut store[request];
                record.record_token(now, context.log_tokens());
                if record.is_complete() {
                    context.stamp_stage(record, now, UnifiedStage::Done as u16);
                    completed.push(request);
                }
            });
        }
        // Every member moved the same distance, so one whole-partition advance
        // is exact — no need to enumerate the membership a second time.
        kv_store.advance(AdvanceScope::WholePartition(partition), 1);
    }
}

/// The speculating engine: a fixed `draft_tokens + 1` verify window per resident
/// request, retiring the target's own token plus the leading run of accepted
/// drafts.
pub struct SpeculativeDecodeCompletion {
    draft_tokens: u32,
    seed: u64,
    /// Scratch, indexed by retired count: `request_buckets[n]` holds the
    /// requests that retired `n` tokens this iteration. Requests in the same
    /// iteration advance by different distances, so KV takes one
    /// `RequestSubset` advance per distinct distance instead of one call each.
    request_buckets: Vec<Vec<RequestId>>,
}

impl SpeculativeDecodeCompletion {
    pub(crate) fn new(draft_tokens: u32, seed: u64) -> Self {
        assert!(
            draft_tokens > 0,
            "speculative draft_tokens must be positive"
        );
        // Indexed by retired count, which runs 1..=draft_tokens + 1.
        let bucket_count = draft_tokens
            .checked_add(2)
            .and_then(|count| usize::try_from(count).ok())
            .expect("speculative acceptance bucket count must fit usize");
        Self {
            draft_tokens,
            seed,
            request_buckets: vec![Vec::new(); bucket_count],
        }
    }

    /// Tokens this verify retires: the target's own token, plus the leading run
    /// of accepted drafts, truncated by the request's remaining output length.
    fn accepted_length(
        &self,
        request: RequestId,
        output_tokens_emitted: u32,
        accept_rate: &AcceptanceProfile,
        remaining_output_tokens: u32,
    ) -> u32 {
        let mut accepted = 1_u32;
        for draft_position in 0..self.draft_tokens {
            let position_rate = accept_rate.at_position(draft_position, self.draft_tokens);
            if !bernoulli(
                self.seed,
                request,
                output_tokens_emitted,
                draft_position,
                position_rate,
            ) {
                break;
            }
            accepted += 1;
        }
        accepted.min(remaining_output_tokens)
    }
}

impl<K: IterWorkerKv> DecodeCompletion<K> for SpeculativeDecodeCompletion {
    fn query_tokens_per_request(&self) -> u32 {
        self.draft_tokens + 1
    }

    fn complete_decodes(
        &mut self,
        kv_store: &mut K,
        partition: PartitionId,
        context: &WorkerContext,
        completed: &mut Vec<RequestId>,
        now: Time,
    ) {
        for bucket in &mut self.request_buckets {
            bucket.clear();
        }
        {
            let mut store = context.requests.borrow_mut();
            kv_store.visit_decode_members(partition, |request, resident_kv| {
                let record = &mut store[request];
                let accept_rate = match &record.request.definition.decoding {
                    DecodingStrategy::Speculative { accept_rate } => accept_rate,
                    // Not a tolerated mode: the verify width is an engine-level
                    // constant, so a request with no declared acceptance would
                    // pay the wide batch and advance one position, which is not
                    // a thing this engine does. It is a composition error.
                    DecodingStrategy::Standard => {
                        panic!(
                            "speculative completion received standard request {}",
                            request.0
                        )
                    }
                };
                let remaining = record
                    .request
                    .definition
                    .target_output_tokens
                    .saturating_sub(record.progress.output_tokens_emitted);
                assert!(
                    remaining > 0,
                    "completed request {} remained in speculative decode membership",
                    request.0
                );
                let accepted = self.accepted_length(
                    request,
                    record.progress.output_tokens_emitted,
                    accept_rate,
                    remaining,
                );
                let observation = record
                    .telemetry
                    .speculative
                    .get_or_insert_with(Default::default);
                observation.query_width = self.draft_tokens + 1;
                observation.decode_rounds += 1;
                observation.resident_kv_sum += resident_kv as u64;
                observation.emitted_tokens += u64::from(accepted);
                observation.pending_decode = None;
                for _ in 0..accepted {
                    record.record_token(now, context.log_tokens());
                }
                self.request_buckets[accepted as usize].push(request);
                if record.is_complete() {
                    context.stamp_stage(record, now, UnifiedStage::Done as u16);
                    completed.push(request);
                }
            });
        }
        for accepted in 1..=self.draft_tokens + 1 {
            let request_ids = &self.request_buckets[accepted as usize];
            if !request_ids.is_empty() {
                kv_store.advance(
                    AdvanceScope::RequestSubset {
                        partition,
                        request_ids,
                    },
                    accepted,
                );
            }
        }
    }
}

/// One draft position's accept/reject draw.
///
/// Keyed, not sequential: the outcome is a pure function of
/// `(seed, request, output_tokens_emitted, draft_position)`, so a request's
/// acceptance chain does not depend on how it was batched, which partition it
/// landed on, or how many other requests drew before it. That is what makes a
/// scheduler A/B meaningful — acceptance is held fixed, so the difference is
/// attributable to the scheduler — and it is what lets an offline analysis
/// replay this sampler for one request without simulating the whole fleet.
fn bernoulli(
    seed: u64,
    request: RequestId,
    output_tokens_emitted: u32,
    draft_position: u32,
    accept_rate: f32,
) -> bool {
    if accept_rate <= 0.0 {
        return false;
    }
    if accept_rate >= 1.0 {
        return true;
    }
    let key = seed
        ^ u64::from(request.0).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(output_tokens_emitted).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ u64::from(draft_position).wrapping_mul(0x94D0_49BB_1331_11EB);
    let draw = splitmix64(key) >> 11;
    let unit = draw as f64 * (1.0 / ((1_u64 << 53) as f64));
    unit < f64::from(accept_rate)
}

/// splitmix64 finalizer. No transcendental ops and no carried state, so a run
/// is bit-identical across machines and across scheduling orders.
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::common::{PoolId, SharedRequests, WorkerId};
    use crate::test_helpers::shared_with;
    use crate::worker::kv::{FullAttnKv, KvStore};

    const DRAFT_TOKENS: u32 = 3;

    #[test]
    fn multi_round_acceptance_matches_python_diagnostic_fixture() {
        #[derive(serde::Deserialize)]
        struct Chain {
            seed: u64,
            request_id: u32,
            output_len: u32,
            rates: Vec<f32>,
            emitted: Vec<u32>,
        }
        let chains: Vec<Chain> = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/speculative_acceptance_chains.json"
        ))
        .unwrap();
        for chain in chains {
            let policy = SpeculativeDecodeCompletion::new(chain.rates.len() as u32, chain.seed);
            let acceptance = AcceptanceProfile::ByPosition(chain.rates);
            let mut emitted = 1;
            let mut observed = Vec::new();
            while emitted < chain.output_len {
                let count = policy.accepted_length(
                    RequestId(chain.request_id),
                    emitted,
                    &acceptance,
                    chain.output_len - emitted,
                );
                observed.push(count);
                emitted += count;
            }
            assert_eq!(observed, chain.emitted);
        }
    }

    fn uniform(rate: f32) -> DecodingStrategy {
        DecodingStrategy::Speculative {
            accept_rate: AcceptanceProfile::Uniform(rate),
        }
    }

    fn by_position(rates: &[f32]) -> DecodingStrategy {
        DecodingStrategy::Speculative {
            accept_rate: AcceptanceProfile::ByPosition(rates.to_vec()),
        }
    }

    fn context_for(requests: SharedRequests) -> WorkerContext {
        WorkerContext {
            id: WorkerId(0),
            pool: PoolId(0),
            requests,
            log_output_token_times: false,
            log_stage_transitions: false,
        }
    }

    /// `count` resident decode requests that have each emitted their first
    /// token, all carrying `decoding` and `target_output_tokens` in total.
    ///
    /// `kv_decode_budget` is the KV-side remaining-decode count, which a test
    /// can raise above the request's own remaining length to isolate which of
    /// the two limits truncated a chain.
    fn resident_decodes(
        per_request: &[DecodingStrategy],
        target_output_tokens: u32,
        kv_decode_budget: u32,
    ) -> (FullAttnKv, WorkerContext) {
        let rows: Vec<(u32, u32, u32)> = (0..per_request.len() as u32)
            .map(|id| (id, 8, target_output_tokens))
            .collect();
        let requests = shared_with(&rows);
        {
            let mut store = requests.borrow_mut();
            for (id, decoding) in per_request.iter().enumerate() {
                let record = &mut store[RequestId(id as u32)];
                record.request.definition.decoding = decoding.clone();
                record.progress.prefill_tokens_processed = 8;
                record.record_token(Time::ZERO, false);
            }
        }
        let mut kv_store = FullAttnKv::without_prefix_cache(1, 10_000, None);
        for id in 0..per_request.len() as u32 {
            let request = RequestId(id);
            let footprint = kv_store.footprint(request, 8, target_output_tokens);
            KvStore::reserve(&mut kv_store, request, 0, footprint, Time::ZERO);
            kv_store.commit_resident(request, 0, 8, kv_decode_budget);
        }
        (kv_store, context_for(requests))
    }

    fn one_resident_decode(
        target_output_tokens: u32,
        kv_decode_budget: u32,
        decoding: DecodingStrategy,
    ) -> (FullAttnKv, WorkerContext) {
        resident_decodes(&[decoding], target_output_tokens, kv_decode_budget)
    }

    /// Tokens emitted and KV positions grown by one iteration on request 0.
    /// They must agree: KV that outruns emission is exactly the silent
    /// over-allocation this policy exists to avoid.
    fn run_one_iteration(
        policy: &mut impl DecodeCompletion<FullAttnKv>,
        kv_store: &mut FullAttnKv,
        context: &WorkerContext,
    ) -> (u32, u64) {
        let before = context.requests.borrow()[RequestId(0)]
            .progress
            .output_tokens_emitted;
        let mut completed = Vec::new();
        policy.complete_decodes(kv_store, 0, context, &mut completed, Time::from_ms(1.0));
        let after = context.requests.borrow()[RequestId(0)]
            .progress
            .output_tokens_emitted;
        let mut grown = 0;
        kv_store.visit_decode_members(0, |request, current_kv| {
            if request == RequestId(0) {
                grown = current_kv - 8;
            }
        });
        (after - before, grown)
    }

    #[test]
    fn a_verify_window_is_wider_than_an_ordinary_decode_row() {
        assert_eq!(
            DecodeCompletion::<FullAttnKv>::query_tokens_per_request(&SingleTokenDecodeCompletion),
            1
        );
        assert_eq!(
            DecodeCompletion::<FullAttnKv>::query_tokens_per_request(
                &SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 0)
            ),
            DRAFT_TOKENS + 1,
            "the engine verifies every draft plus the target's own position"
        );
    }

    #[test]
    fn a_rejected_draft_still_retires_the_target_token() {
        let (mut kv_store, context) = one_resident_decode(64, 63, uniform(0.0));
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        assert_eq!(
            run_one_iteration(&mut policy, &mut kv_store, &context),
            (1, 1),
            "a verify pass always produces the target's token, so no request can stall"
        );
    }

    #[test]
    fn a_fully_accepted_window_retires_every_drafted_position() {
        let (mut kv_store, context) = one_resident_decode(64, 63, uniform(1.0));
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        assert_eq!(
            run_one_iteration(&mut policy, &mut kv_store, &context),
            (DRAFT_TOKENS + 1, u64::from(DRAFT_TOKENS + 1))
        );
    }

    #[test]
    fn a_per_position_profile_reads_each_positions_own_probability() {
        // The defect this catches: collapsing the vector to one number. A
        // measured acceptance curve is not geometric, and position 1 rejecting
        // must stop the chain even though position 2 would have accepted.
        let (mut kv_store, context) = one_resident_decode(64, 63, by_position(&[1.0, 0.0, 1.0]));
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        assert_eq!(
            run_one_iteration(&mut policy, &mut kv_store, &context),
            (2, 2),
            "target token plus position 0 only; the chain stops at the first reject"
        );
    }

    #[test]
    #[should_panic(expected = "must equal speculative draft_tokens")]
    fn a_profile_shallower_than_the_verify_width_is_refused_before_any_draw() {
        // The trace declares a depth and the worker declares a width. If they
        // disagree the run is not modelling what either of them says.
        let (mut kv_store, context) = one_resident_decode(64, 63, by_position(&[0.5, 0.5]));
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        run_one_iteration(&mut policy, &mut kv_store, &context);
    }

    #[test]
    #[should_panic(expected = "received standard request")]
    fn a_request_with_no_declared_acceptance_is_a_composition_error() {
        // Not a tolerated mode. The verify width is engine-level, so such a
        // request would pay the wide batch and advance one position — which is
        // not something a speculating engine does to any request.
        let (mut kv_store, context) = one_resident_decode(64, 63, DecodingStrategy::Standard);
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        run_one_iteration(&mut policy, &mut kv_store, &context);
    }

    #[test]
    fn the_target_output_length_truncates_an_over_long_acceptance_chain() {
        // Two tokens in total, one already emitted at prefill: the window may
        // accept four, but only one token remains to emit. KV is deliberately
        // left with room for 63 more, so the request's own target is the only
        // thing that can truncate the chain.
        let (mut kv_store, context) = one_resident_decode(2, 63, uniform(1.0));
        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        let mut completed = Vec::new();
        policy.complete_decodes(
            &mut kv_store,
            0,
            &context,
            &mut completed,
            Time::from_ms(1.0),
        );
        assert_eq!(completed, vec![RequestId(0)]);
        let store = context.requests.borrow();
        assert_eq!(store[RequestId(0)].progress.output_tokens_emitted, 2);
        let mut grown = 0;
        kv_store.visit_decode_members(0, |_, current_kv| grown = current_kv - 8);
        assert_eq!(
            grown, 1,
            "KV must not grow past the tokens actually emitted"
        );
    }

    #[test]
    fn requests_that_accept_different_lengths_each_grow_by_their_own_chain() {
        let requests: SharedRequests = shared_with(&[(0, 8, 64), (1, 8, 64)]);
        {
            let mut store = requests.borrow_mut();
            for (id, rate) in [(0u32, 0.0f32), (1, 1.0)] {
                let record = &mut store[RequestId(id)];
                record.request.definition.decoding = uniform(rate);
                record.progress.prefill_tokens_processed = 8;
                record.record_token(Time::ZERO, false);
            }
        }
        let context = context_for(requests);
        let mut kv_store = FullAttnKv::without_prefix_cache(1, 10_000, None);
        for id in 0..2u32 {
            let request = RequestId(id);
            let footprint = kv_store.footprint(request, 8, 64);
            KvStore::reserve(&mut kv_store, request, 0, footprint, Time::ZERO);
            kv_store.commit_resident(request, 0, 8, 63);
        }

        let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 7);
        let mut completed = Vec::new();
        policy.complete_decodes(
            &mut kv_store,
            0,
            &context,
            &mut completed,
            Time::from_ms(1.0),
        );

        let mut grown = Vec::new();
        kv_store.visit_decode_members(0, |request, current_kv| {
            grown.push((request, current_kv - 8))
        });
        assert_eq!(
            grown,
            vec![
                (RequestId(0), 1),
                (RequestId(1), u64::from(DRAFT_TOKENS + 1))
            ],
            "one bucketed advance per distinct chain length, insertion order preserved"
        );
    }

    #[test]
    fn a_requests_chain_does_not_depend_on_what_it_was_batched_with() {
        // The defect this catches: drawing from a sequential stream. The two
        // batches below differ only in how many draws the *earlier* requests
        // consume — all-reject spends one apiece, all-accept spends three — so
        // under a shared stream the last request would land on different draws
        // and its acceptance would shift with batch composition, partition
        // placement, and arrival order. Every scheduler comparison would then
        // be confounded by acceptance moving underneath it.
        let observed = |neighbour_rate: f32| {
            let (mut kv_store, context) = resident_decodes(
                &[
                    uniform(neighbour_rate),
                    uniform(neighbour_rate),
                    uniform(0.5),
                ],
                64,
                63,
            );
            let mut policy = SpeculativeDecodeCompletion::new(DRAFT_TOKENS, 12345);
            let mut completed = Vec::new();
            policy.complete_decodes(
                &mut kv_store,
                0,
                &context,
                &mut completed,
                Time::from_ms(1.0),
            );
            let emitted = context.requests.borrow()[RequestId(2)]
                .progress
                .output_tokens_emitted;
            emitted
        };
        assert_eq!(
            observed(0.0),
            observed(1.0),
            "acceptance is keyed by (seed, request, progress, position), not by draw order"
        );
    }

    #[test]
    fn the_acceptance_draw_is_reproducible_and_seed_dependent() {
        let chain = |seed: u64| {
            let mut accepted = 0;
            for position in 0..DRAFT_TOKENS {
                if !bernoulli(seed, RequestId(3), 5, position, 0.5) {
                    break;
                }
                accepted += 1;
            }
            accepted
        };
        assert_eq!(chain(99), chain(99), "same seed reproduces the same chain");
        let spread: Vec<u32> = (0..64).map(chain).collect();
        assert!(
            spread.iter().any(|&length| length != spread[0]),
            "the seed must actually select a stream, not a constant"
        );
    }

    #[test]
    fn an_ordinary_engine_moves_every_resident_request_one_position() {
        let request = RequestId(0);
        let requests = shared_with(&[(0, 8, 64)]);
        {
            let mut store = requests.borrow_mut();
            store[request].progress.prefill_tokens_processed = 8;
        }
        let context = context_for(requests);
        let mut kv_store = FullAttnKv::without_prefix_cache(1, 10_000, None);
        let footprint = kv_store.footprint(request, 8, 64);
        KvStore::reserve(&mut kv_store, request, 0, footprint, Time::ZERO);
        kv_store.commit_resident(request, 0, 8, 64);

        let mut completed = Vec::new();
        SingleTokenDecodeCompletion.complete_decodes(
            &mut kv_store,
            0,
            &context,
            &mut completed,
            Time::from_ms(1.0),
        );
        assert!(completed.is_empty());
        assert_eq!(
            context.requests.borrow()[request]
                .progress
                .output_tokens_emitted,
            1
        );
        let mut grown = 0;
        kv_store.visit_decode_members(0, |_, current_kv| grown = current_kv - 8);
        assert_eq!(grown, 1);
    }
}
