//! Runtime-variable speculative execution.
//!
//! This is deliberately a sibling of [`IterModelExecution`], not a flag on
//! [`UnifiedIterExecution`](super::UnifiedIterExecution). A speculative iteration
//! has a different input shape (draft proposals plus target verification) and
//! produces a per-request result that the cadence must retain until compute
//! completes.

use std::sync::Arc;

use crate::common::{RequestId, SharedRequests, Time};

use super::super::kv::IterWorkerKv;
use super::super::shared::advance_scope::PartitionId;
use super::ModelKvLayout;

/// One request's draft/verify shape. `proposal_tokens` excludes the target bonus
/// token; `remaining_output_tokens` includes every token still uncommitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftVerifyRequestInput {
    pub request: RequestId,
    pub partition: PartitionId,
    pub target_kv_len: u64,
    pub remaining_output_tokens: u32,
    pub proposal_tokens: u32,
}

/// Whole speculative iteration input. A real L4 adapter lowers this to K
/// autoregressive draft steps plus one target verification query of width K+1.
#[derive(Default)]
pub struct DraftVerifyInput {
    pub prefills: Vec<DraftVerifyPrefillInput>,
    pub requests: Vec<DraftVerifyRequestInput>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftVerifyPrefillInput {
    pub request: RequestId,
    pub partition: PartitionId,
    pub prefix_tokens: u32,
    pub append_tokens: u32,
}

/// Tentative state created before the target verification completes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculativeProposal {
    pub request: RequestId,
    pub partition: PartitionId,
    pub proposal_tokens: u32,
}

/// Tokens committed by target verification for one request. This count includes
/// the optional target bonus token, avoiding the ambiguous old `accepted_len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftVerifyRequestOutcome {
    pub request: RequestId,
    pub partition: PartitionId,
    pub committed_tokens: u32,
}

/// Result retained by the S6 cadence across the compute window.
pub struct DraftVerifyResult {
    pub duration: Time,
    pub requests: Vec<DraftVerifyRequestOutcome>,
}

/// L4-facing cost surface for a real draft/verify batch shape.
///
/// Implementations may compose separate drafter and target models, but the L5
/// execution adapter does not assume whether they are colocated or fused.
pub trait DraftVerifyModel: Send + Sync + 'static {
    fn model_kv_layout(&self) -> ModelKvLayout;
    fn gpus_per_replica(&self) -> u16;
    fn evaluate_draft_verify(&self, input: &DraftVerifyInput, iter: u64, now: Time) -> Time;
}

/// Runtime acceptance source. It may be trace-backed or distribution-backed;
/// unlike the removed build-time scalar, it is sampled for every request and
/// iteration.
pub trait AcceptanceOracle {
    /// Number of draft tokens accepted, excluding the optional target bonus.
    fn accepted_draft_tokens(
        &mut self,
        request: RequestId,
        proposal_tokens: u32,
        remaining_output_tokens: u32,
    ) -> u32;
}

/// S6 execution face consumed by the draft/verify shell.
///
/// The input remains opaque to the shell. `collect_proposals` is the only
/// lifecycle action needed before compute; request-local completion results
/// return from `evaluate_draft_verify_iteration`.
pub trait DraftVerifyModelExecution<K: IterWorkerKv> {
    type Input: Default;

    fn build_draft_verify_input(
        &self,
        kv_store: &K,
        requests: &SharedRequests,
        out: &mut Self::Input,
    );

    fn collect_proposals(input: &Self::Input, out: &mut Vec<SpeculativeProposal>);

    fn evaluate_draft_verify_iteration(
        &mut self,
        input: &Self::Input,
        iter: u64,
        now: Time,
    ) -> DraftVerifyResult;
}

/// Execution adapter that owns the speculative input builder and acceptance
/// oracle. The shell sees only proposals, duration, and committed outcomes.
pub struct DraftVerifyExecution<M: DraftVerifyModel, O: AcceptanceOracle> {
    model: Arc<M>,
    acceptance_oracle: O,
    proposal_tokens: u32,
}

impl<M: DraftVerifyModel, O: AcceptanceOracle> DraftVerifyExecution<M, O> {
    pub fn new(model: Arc<M>, acceptance_oracle: O, proposal_tokens: u32) -> Self {
        assert!(
            proposal_tokens > 0,
            "proposal_tokens must be greater than zero"
        );
        Self {
            model,
            acceptance_oracle,
            proposal_tokens,
        }
    }

    pub fn model_kv_layout(&self) -> ModelKvLayout {
        self.model.model_kv_layout()
    }

    pub fn gpus_per_replica(&self) -> u16 {
        self.model.gpus_per_replica()
    }

    fn build_input<K: IterWorkerKv>(
        &self,
        kv_store: &K,
        requests: &SharedRequests,
        out: &mut DraftVerifyInput,
    ) {
        let store = requests.borrow();
        out.prefills.clear();
        out.requests.clear();
        for partition in 0..kv_store.num_partitions() as u16 {
            for request in kv_store.prefill_admits(partition) {
                let record = &store[request];
                out.prefills.push(DraftVerifyPrefillInput {
                    request,
                    partition,
                    prefix_tokens: record.prefix_kv,
                    append_tokens: record.active_chunk_len,
                });
            }
            for (request, target_kv_len) in kv_store.decode_members(partition) {
                let record = &store[request];
                let remaining_output_tokens =
                    record.decode_len.saturating_sub(record.tokens_emitted);
                let proposal_tokens = self
                    .proposal_tokens
                    .min(remaining_output_tokens.saturating_sub(1));
                out.requests.push(DraftVerifyRequestInput {
                    request,
                    partition,
                    target_kv_len,
                    remaining_output_tokens,
                    proposal_tokens,
                });
            }
        }
    }

    fn proposals(input: &DraftVerifyInput) -> impl Iterator<Item = SpeculativeProposal> + '_ {
        input
            .requests
            .iter()
            .filter(|request| request.remaining_output_tokens > 0)
            .map(|request| SpeculativeProposal {
                request: request.request,
                partition: request.partition,
                proposal_tokens: request.proposal_tokens,
            })
    }

    fn evaluate(&mut self, input: &DraftVerifyInput, iter: u64, now: Time) -> DraftVerifyResult {
        let duration = self.model.evaluate_draft_verify(input, iter, now);
        let requests = input
            .requests
            .iter()
            .filter(|request| request.remaining_output_tokens > 0)
            .map(|request| {
                let accepted_draft_tokens = self
                    .acceptance_oracle
                    .accepted_draft_tokens(
                        request.request,
                        request.proposal_tokens,
                        request.remaining_output_tokens,
                    )
                    .min(request.proposal_tokens);
                // A target verification always commits at least one token while
                // output remains: either the first rejected target token or the
                // bonus token after accepting the whole draft.
                let committed_tokens = accepted_draft_tokens
                    .saturating_add(1)
                    .min(request.remaining_output_tokens);
                DraftVerifyRequestOutcome {
                    request: request.request,
                    partition: request.partition,
                    committed_tokens,
                }
            })
            .collect();
        DraftVerifyResult { duration, requests }
    }
}

impl<M, O, K> DraftVerifyModelExecution<K> for DraftVerifyExecution<M, O>
where
    M: DraftVerifyModel,
    O: AcceptanceOracle,
    K: IterWorkerKv,
{
    type Input = DraftVerifyInput;

    fn build_draft_verify_input(
        &self,
        kv_store: &K,
        requests: &SharedRequests,
        out: &mut Self::Input,
    ) {
        self.build_input(kv_store, requests, out);
    }

    fn collect_proposals(input: &Self::Input, out: &mut Vec<SpeculativeProposal>) {
        out.clear();
        out.extend(Self::proposals(input));
    }

    fn evaluate_draft_verify_iteration(
        &mut self,
        input: &Self::Input,
        iter: u64,
        now: Time,
    ) -> DraftVerifyResult {
        self.evaluate(input, iter, now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct FixedDurationModel;

    impl DraftVerifyModel for FixedDurationModel {
        fn model_kv_layout(&self) -> ModelKvLayout {
            ModelKvLayout {
                total_kv_bytes_per_token: 1,
                num_attn_shards: 1,
            }
        }

        fn gpus_per_replica(&self) -> u16 {
            1
        }

        fn evaluate_draft_verify(&self, _input: &DraftVerifyInput, _iter: u64, _now: Time) -> Time {
            Time::from_ms(2.5)
        }
    }

    struct ScriptedAcceptance {
        accepted: VecDeque<u32>,
    }

    impl AcceptanceOracle for ScriptedAcceptance {
        fn accepted_draft_tokens(
            &mut self,
            _request: RequestId,
            _proposal_tokens: u32,
            _remaining_output_tokens: u32,
        ) -> u32 {
            self.accepted.pop_front().unwrap()
        }
    }

    #[test]
    fn evaluate_returns_request_local_runtime_acceptance() {
        let mut execution = DraftVerifyExecution::new(
            Arc::new(FixedDurationModel),
            ScriptedAcceptance {
                accepted: VecDeque::from([0, 2, 99]),
            },
            4,
        );
        let input = DraftVerifyInput {
            prefills: Vec::new(),
            requests: vec![
                DraftVerifyRequestInput {
                    request: RequestId(1),
                    partition: 0,
                    target_kv_len: 10,
                    remaining_output_tokens: 8,
                    proposal_tokens: 4,
                },
                DraftVerifyRequestInput {
                    request: RequestId(2),
                    partition: 0,
                    target_kv_len: 20,
                    remaining_output_tokens: 8,
                    proposal_tokens: 4,
                },
                DraftVerifyRequestInput {
                    request: RequestId(3),
                    partition: 1,
                    target_kv_len: 30,
                    remaining_output_tokens: 3,
                    proposal_tokens: 2,
                },
            ],
        };

        let result = execution.evaluate(&input, 7, Time::ZERO);

        assert_eq!(result.duration.as_ms(), 2.5);
        assert_eq!(
            result
                .requests
                .iter()
                .map(|outcome| outcome.committed_tokens)
                .collect::<Vec<_>>(),
            vec![1, 3, 3]
        );
        assert_eq!(result.requests[2].partition, 1);
    }
}
