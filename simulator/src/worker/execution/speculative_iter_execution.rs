//! Whole-iteration input construction and cost evaluation for a speculating model.
//!
//! Structurally the sibling of [`UnifiedIterExecution`](super::UnifiedIterExecution):
//! same borrowed-membership rendering, same reused output buffer, same one fused
//! `iter` section. It differs in exactly one fact — a resident decode request
//! submits a whole verify window instead of one row — and that fact reshapes the
//! decode side of the L4 input:
//!
//! - the ordinary adapter's `decode_kv_lens` becomes `decode_requests`, a
//!   `(kv_len, query_len)` pair per request, because query rows can no longer be
//!   recovered from request count;
//! - `decode_tokens` counts **query rows**, not requests, which is what the L4
//!   contract defines it as and what `batch_tokens` must sum to;
//! - `total_kv_len` stays the pre-verify resident KV — the cache-pressure fact —
//!   while each request's own `kv_len` is the longest context its final verify
//!   row sees.
//!
//! It does not own how far a request advances after the verify: that is the
//! lifecycle's `DecodeCompletion`. This adapter only states the width submitted,
//! which is fixed per worker because it selects a profiled kernel shape.

use std::sync::Arc;

use crate::arch::contract::{
    SpeculativeArchGroupInput, SpeculativeArchInput, SpeculativeDecodeInput,
    SpeculativeUnifiedModel,
};
use crate::common::{DecodingStrategy, SharedRequests, Time};
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::execution::{IterModelExecution, ModelKvLayout};
use crate::worker::kv::{IterWorkerKv, PrefixKv};
use crate::worker::types::IterBatchPlan;

pub struct SpeculativeIterExecution<M: SpeculativeUnifiedModel> {
    model: Arc<M>,
    cost: CostBuffers,
    /// Candidate positions drafted per request per iteration. The verify width
    /// is `draft_tokens + 1`; the model was compiled for that width, so it is a
    /// worker-lifetime constant rather than a per-iteration quantity.
    draft_tokens: u32,
}

impl<M: SpeculativeUnifiedModel> SpeculativeIterExecution<M> {
    pub(crate) fn new(model: Arc<M>, cost: CostBuffers, draft_tokens: u32) -> Self {
        assert!(
            draft_tokens > 0,
            "a speculative execution adapter must draft at least one candidate; \
             zero drafts is the ordinary unified adapter"
        );
        Self {
            model,
            cost,
            draft_tokens,
        }
    }

    /// Query rows one resident decode request submits per iteration.
    fn verify_width(&self) -> u32 {
        self.draft_tokens + 1
    }

    fn build_input<K: IterWorkerKv + PrefixKv>(
        &self,
        kv_store: &K,
        requests: &SharedRequests,
        batch_plan: &IterBatchPlan,
        out: &mut SpeculativeArchInput,
    ) {
        let num_partitions = kv_store.num_partitions();
        out.draft_tokens = self.draft_tokens;
        out.groups
            .resize_with(num_partitions, SpeculativeArchGroupInput::default);
        out.groups.truncate(num_partitions);

        let verify_width = self.verify_width();
        let max_model_len = self.model.max_model_len();
        for partition in 0..num_partitions as u16 {
            let group = &mut out.groups[partition as usize];
            group.clear();
            kv_store.visit_prefill_admits(partition, |request| {
                let resolved_prefill = kv_store.resolved_prefill_context(request);
                let (prefix_tokens, chunk_tokens) = resolved_prefill.active_chunk();
                group
                    .prefill_chunk_pairs
                    .push((prefix_tokens, chunk_tokens));
                group.prefill_tokens += chunk_tokens;
            });
            if batch_plan.partition_runs_decode(partition) {
                let request_store = requests.borrow();
                kv_store.visit_decode_members(partition, |request, current_kv| {
                    let record = &request_store[request];
                    // The batch this adapter bills must be the batch the
                    // lifecycle will advance. Both facts are asserted here so a
                    // mismatched composition fails while lowering the input,
                    // not later inside `complete_decodes`.
                    assert!(
                        matches!(
                            record.request.definition.decoding,
                            DecodingStrategy::Speculative { .. }
                        ),
                        "speculative worker received standard request {}",
                        request.0
                    );
                    assert!(
                        record
                            .request
                            .definition
                            .target_output_tokens
                            .saturating_sub(record.progress.output_tokens_emitted)
                            > 0,
                        "completed request {} remained in speculative decode membership",
                        request.0
                    );
                    let current_kv = current_kv as u32;
                    group.decode_requests.push(SpeculativeDecodeInput {
                        // The last verify row attends over the resident KV plus
                        // every drafted position ahead of it, clamped at the
                        // model limit — the width stays fixed, only the context
                        // stops growing.
                        kv_len: (current_kv + self.draft_tokens).min(max_model_len),
                        query_len: verify_width,
                    });
                    group.total_kv_len += current_kv;
                });
            }
            group.decode_tokens = group.decode_requests.len() as u32 * verify_width;
            group.batch_tokens = group.prefill_tokens + group.decode_tokens;
        }
        if num_partitions > 1 {
            out.tokens_per_source_rank.clear();
            out.tokens_per_source_rank
                .extend(out.groups.iter().map(|group| group.batch_tokens));
        } else {
            out.tokens_per_source_rank.clear();
        }
    }

    pub(crate) fn evaluate_iteration(
        &mut self,
        input: &SpeculativeArchInput,
        iteration: u64,
        now: Time,
    ) -> Time {
        self.cost
            .run_speculative_iter(self.model.as_ref(), input, iteration, now)
    }
}

impl<M, K> IterModelExecution<K> for SpeculativeIterExecution<M>
where
    M: SpeculativeUnifiedModel,
    K: IterWorkerKv + PrefixKv,
{
    type Input = SpeculativeArchInput;

    fn model_kv_layout(&self) -> ModelKvLayout {
        ModelKvLayout {
            total_kv_bytes_per_token: self.model.total_kv_bytes_per_token(),
            num_attn_shards: self.model.num_attn_shards(),
        }
    }

    fn build_iteration_input(
        &self,
        kv_store: &K,
        requests: &SharedRequests,
        batch_plan: &IterBatchPlan,
        out: &mut Self::Input,
    ) {
        self.build_input(kv_store, requests, batch_plan, out);
    }

    fn evaluate_iteration(&mut self, input: &Self::Input, iteration: u64, now: Time) -> Time {
        SpeculativeIterExecution::evaluate_iteration(self, input, iteration, now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::common::{AcceptanceProfile, RequestId, SharedRequests, WorkerId};
    use crate::test_helpers::shared_with;
    use crate::timing::{CostManifest, CostManifestDoc, LeafMetrics};
    use crate::worker::kv::{FullAttnKv, KvStore};

    const DRAFT_TOKENS: u32 = 3;
    const MAX_MODEL_LEN: u32 = 4096;
    const PROMPT_TOKENS: u32 = 8;

    /// Answers only what the adapter reads: layout facts and the model length
    /// that clamps a verify group's final row.
    struct FakeSpeculativeModel;

    impl SpeculativeUnifiedModel for FakeSpeculativeModel {
        fn eval_speculative_iter(
            &self,
            _batch: &SpeculativeArchInput,
            _slots: &mut Vec<LeafMetrics>,
            _scratch: &mut Vec<LeafMetrics>,
        ) -> LeafMetrics {
            LeafMetrics::ZERO
        }

        fn cost_log_manifest(&self) -> CostManifest {
            CostManifest {
                slots: Vec::new(),
                nodes: Vec::new(),
                node_labels: Vec::new(),
            }
        }

        fn total_kv_bytes_per_token(&self) -> u64 {
            1
        }

        fn max_model_len(&self) -> u32 {
            MAX_MODEL_LEN
        }

        fn gpus_per_replica(&self) -> u16 {
            1
        }
    }

    fn execution() -> SpeculativeIterExecution<FakeSpeculativeModel> {
        let cost = CostBuffers::new(
            None,
            "test",
            WorkerId(0),
            &CostManifestDoc::single("iter", FakeSpeculativeModel.cost_log_manifest()),
            1.0,
        );
        SpeculativeIterExecution::new(Arc::new(FakeSpeculativeModel), cost, DRAFT_TOKENS)
    }

    /// `count` resident speculative decode requests, each holding
    /// `context_tokens` of KV, plus the request store the adapter reads.
    fn resident_decodes(count: u32, context_tokens: u32) -> (FullAttnKv, SharedRequests) {
        let rows: Vec<(u32, u32, u32)> = (0..count).map(|id| (id, context_tokens, 64)).collect();
        let requests = shared_with(&rows);
        {
            let mut store = requests.borrow_mut();
            for id in 0..count {
                let record = &mut store[RequestId(id)];
                record.request.definition.decoding = DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::Uniform(0.5),
                };
                record.progress.prefill_tokens_processed = context_tokens;
                record.record_token(Time::ZERO, false);
            }
        }
        let mut kv_store = FullAttnKv::without_prefix_cache(1, 10_000_000, None);
        for id in 0..count {
            let request = RequestId(id);
            let footprint = kv_store.footprint(request, context_tokens, 64);
            KvStore::reserve(&mut kv_store, request, 0, footprint, Time::ZERO);
            kv_store.commit_resident(request, 0, context_tokens as u64, 64);
        }
        (kv_store, requests)
    }

    fn plan_running_decode(runs_decode: bool) -> IterBatchPlan {
        let mut plan = IterBatchPlan::default();
        plan.reset_decode_participation(1, runs_decode);
        plan
    }

    #[test]
    fn a_verify_batch_charges_query_rows_not_requests() {
        // The defect this catches: copying the ordinary adapter's
        // `decode_tokens = decode_requests.len()`, which under-charges the
        // verify pass by the whole draft window and makes `batch_tokens`
        // disagree with the rows the model actually forwards.
        let execution = execution();
        let (kv_store, requests) = resident_decodes(4, PROMPT_TOKENS);
        let mut input = SpeculativeArchInput::default();

        execution.build_input(&kv_store, &requests, &plan_running_decode(true), &mut input);

        let group = &input.groups[0];
        assert_eq!(group.decode_requests.len(), 4);
        assert_eq!(group.decode_tokens, 4 * (DRAFT_TOKENS + 1));
        assert_eq!(
            group.batch_tokens,
            group.prefill_tokens + group.decode_tokens
        );
        assert!(
            group
                .decode_requests
                .iter()
                .all(|d| d.query_len == DRAFT_TOKENS + 1),
            "the width is fixed by the profiled kernel shape, not by the batch"
        );
        assert_eq!(input.draft_tokens, DRAFT_TOKENS);
    }

    #[test]
    fn a_verify_row_sees_the_context_the_drafts_would_add() {
        // `total_kv_len` is pre-verify cache pressure; the per-request `kv_len`
        // is what the final row attends over. Reporting one for the other
        // either misprices attention or misprices KV capacity.
        let execution = execution();
        let (kv_store, requests) = resident_decodes(2, PROMPT_TOKENS);
        let mut input = SpeculativeArchInput::default();

        execution.build_input(&kv_store, &requests, &plan_running_decode(true), &mut input);

        let group = &input.groups[0];
        assert_eq!(group.total_kv_len, 2 * PROMPT_TOKENS);
        assert!(group
            .decode_requests
            .iter()
            .all(|d| d.kv_len == PROMPT_TOKENS + DRAFT_TOKENS));
    }

    #[test]
    fn a_verify_group_at_the_model_limit_clamps_its_context_not_its_width() {
        // Widening past the profiled shape would select a kernel the model was
        // never compiled for; the contract says clamp `kv_len` instead.
        let execution = execution();
        let (kv_store, requests) = resident_decodes(1, MAX_MODEL_LEN - 1);
        let mut input = SpeculativeArchInput::default();

        execution.build_input(&kv_store, &requests, &plan_running_decode(true), &mut input);

        let decode = input.groups[0].decode_requests[0];
        assert_eq!(decode.kv_len, MAX_MODEL_LEN);
        assert_eq!(decode.query_len, DRAFT_TOKENS + 1);
    }

    #[test]
    fn a_prefill_only_iteration_charges_no_verify_rows() {
        // Resident decodes that the plan excluded are not advanced this
        // iteration, so charging their verify window would bill work the
        // simulation never performs.
        let execution = execution();
        let (kv_store, requests) = resident_decodes(3, PROMPT_TOKENS);
        let mut input = SpeculativeArchInput::default();

        execution.build_input(
            &kv_store,
            &requests,
            &plan_running_decode(false),
            &mut input,
        );

        let group = &input.groups[0];
        assert!(group.decode_requests.is_empty());
        assert_eq!(group.decode_tokens, 0);
        assert_eq!(group.total_kv_len, 0);
        assert_eq!(group.batch_tokens, 0);
    }

    #[test]
    fn a_reused_input_buffer_carries_nothing_from_the_previous_iteration() {
        // The buffer is held across iterations; a missed `clear` would keep
        // stale verify rows and silently inflate every later batch.
        let execution = execution();
        let mut input = SpeculativeArchInput::default();

        let (busy, busy_requests) = resident_decodes(5, PROMPT_TOKENS);
        execution.build_input(
            &busy,
            &busy_requests,
            &plan_running_decode(true),
            &mut input,
        );
        let (idle, idle_requests) = resident_decodes(1, PROMPT_TOKENS);
        execution.build_input(
            &idle,
            &idle_requests,
            &plan_running_decode(true),
            &mut input,
        );

        let group = &input.groups[0];
        assert_eq!(group.decode_requests.len(), 1);
        assert_eq!(group.decode_tokens, DRAFT_TOKENS + 1);
        assert_eq!(group.total_kv_len, PROMPT_TOKENS);
    }
}
