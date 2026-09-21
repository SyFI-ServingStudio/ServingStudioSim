//! Family-agnostic request containers and the shared worker-facing store.
//!
//! Concrete definitions live one-per-file under `common::request_family`.
//! [`ActiveRequest`] composes one of them with family-specific progress,
//! common lifecycle, and telemetry. The store remains generic over the
//! definition, so a text-only worker cannot receive a media-generation request
//! through this seam.

use std::cell::RefCell;
use std::ops::{Index, IndexMut};
use std::rc::Rc;

use serde::{Deserialize, Serialize};

use super::id::{PoolId, RequestId, WorkerId};
use super::request_family::{RequestDefinition, TextGenerationDefinition};
use super::request_stage::StageEvent;
use super::time::Time;

/// Scheduling facts whose meaning is shared by every request family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulingContract {
    /// Higher values rank ahead when the selected admission policy supports it.
    pub priority: i32,
}

/// Metric-specific service bounds declared by one request.
///
/// These are durations, not absolute timestamps. Keeping them separate from
/// [`SchedulingContract`] prevents a latency obligation from becoming an
/// admission-policy knob merely because both arrived on the same trace row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SloContract {
    pub ttft_slo: Option<Time>,
    pub tpot_slo: Option<Time>,
    pub e2e_slo: Option<Time>,
}

/// Where the trace says this request must run.
///
/// Routing, not ranking: a [`SchedulingContract`] orders a queue, a directive
/// names a machine. Separate from scheduling because they have different
/// consumers — L5 admission reads the priority and would have to ignore this,
/// while L6 routing reads this and never sees a queue.
///
/// `None` is a real third state, not worker 0: the trace declines to place this
/// request and the pool's own placement policy chooses. A pool configured to
/// obey the trace refuses the `None` rather than inventing a worker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlacementDirective {
    /// Pool-relative worker index. Bounds-checked by L6, which owns the replica
    /// list; nothing below L6 knows how many workers exist.
    pub worker: Option<WorkerId>,
}

/// Stable facts carried by every live request.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestCore {
    pub id: RequestId,
    /// Actual time the replay scheduler released the request into L6.
    pub arrival_time: Time,
    pub slo: SloContract,
    pub scheduling: SchedulingContract,
    pub placement: PlacementDirective,
}

/// One concrete request whose definition fixes its family at the type level.
///
/// ```compile_fail
/// use simulator::common::{
///     ImageExtent, ImageGenerationDefinition, PlacementDirective, Request, RequestCore,
///     RequestId, SchedulingContract, SloContract, TextGenerationDefinition, Time,
/// };
/// fn accepts_text(_: Request<TextGenerationDefinition>) {}
/// let image = Request::new(
///     RequestCore {
///         id: RequestId(0),
///         arrival_time: Time::ZERO,
///         slo: SloContract::default(),
///         scheduling: SchedulingContract::default(),
///         placement: PlacementDirective::default(),
///     },
///     ImageGenerationDefinition {
///         text_prompt_tokens: 8,
///         target_generation_steps: 20,
///         extent: ImageExtent { width: 1024, height: 1024 },
///     },
/// );
/// accepts_text(image);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Request<Definition = TextGenerationDefinition> {
    pub core: RequestCore,
    pub definition: Definition,
}

impl<Definition> Request<Definition> {
    /// Compose already-resolved cross-family facts with one typed definition.
    /// Trace parsing and replay pacing stay outside this storage constructor.
    pub const fn new(core: RequestCore, definition: Definition) -> Self {
        Self { core, definition }
    }
}

/// Common lifecycle state. Slot presence in [`RequestStore`] is the arrived
/// fact, so no placeholder `arrived` boolean is needed.
#[derive(Clone, Debug)]
pub struct RequestLifecycle {
    pub admitted: bool,
    pub completed: bool,
    pub current_stage: StageEvent,
    pub stage_log: Vec<StageEvent>,
}

/// One recomputation episode after decode retraction. The request keeps its
/// emitted output and TTFT; these fields describe only the extra prefill work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReprocessedPrefillEpisode {
    pub output_tokens_before: u32,
    pub prefix_cache_hit_tokens: u32,
    pub prefill_tokens_processed: u32,
    pub completed: bool,
}

/// Observations collected without changing request-family semantics.
#[derive(Clone, Debug, Default)]
pub struct RequestTelemetry {
    pub first_output_time: Option<Time>,
    pub last_output_time: Option<Time>,
    pub output_times: Vec<Time>,
    /// Prefix tokens physically found when this request was admitted. `None`
    /// means admission never resolved a prefix context; `Some(0)` is a real
    /// cold/disabled-cache result rather than missing telemetry.
    pub prefix_cache_hit_tokens: Option<u32>,
    /// Number of times active decode KV was released and the request returned
    /// to the waiting queue.
    pub retraction_count: u32,
    /// Extra prefill episodes caused by those retractions.
    pub reprocessed_prefills: Vec<ReprocessedPrefillEpisode>,
    /// Allocate only for speculative requests; ordinary traces keep one pointer.
    pub speculative: Option<Box<SpeculativeProgress>>,
}

/// Completion-side observations, independent of the model input/cost logger.
/// Bounded per request even when per-token timestamp logging is disabled.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SpeculativeProgress {
    pub query_width: u32,
    pub prefill_chunks: u64,
    pub decode_rounds: u64,
    pub resident_kv_sum: u64,
    pub emitted_tokens: u64,
    pub pending_prefill: Option<(u32, u32)>,
    pub pending_decode: Option<u64>,
}

/// One live request: immutable definition + progress + lifecycle + telemetry.
#[derive(Clone, Debug)]
pub struct ActiveRequest<Definition: RequestDefinition = TextGenerationDefinition> {
    pub request: Request<Definition>,
    pub progress: Definition::Progress,
    pub lifecycle: RequestLifecycle,
    pub telemetry: RequestTelemetry,
}

pub type RequestRecord = ActiveRequest<TextGenerationDefinition>;

impl<Definition: RequestDefinition> ActiveRequest<Definition> {
    pub fn from_request(request: Request<Definition>) -> Self {
        let progress = request.definition.initial_progress();
        let arrival_time = request.core.arrival_time;
        Self {
            request,
            progress,
            lifecycle: RequestLifecycle {
                admitted: false,
                completed: false,
                current_stage: StageEvent::unset(arrival_time),
                stage_log: Vec::new(),
            },
            telemetry: RequestTelemetry::default(),
        }
    }

    pub fn record_stage(
        &mut self,
        now: Time,
        code: u16,
        pool: PoolId,
        worker: WorkerId,
        log: bool,
    ) {
        if (
            self.lifecycle.current_stage.code,
            self.lifecycle.current_stage.pool,
            self.lifecycle.current_stage.worker,
        ) == (code, pool, worker)
        {
            return;
        }
        self.lifecycle.current_stage = StageEvent {
            time: now,
            code,
            pool,
            worker,
        };
        if log {
            self.lifecycle.stage_log.push(self.lifecycle.current_stage);
        }
    }

    pub fn is_complete(&self) -> bool {
        self.request.definition.is_complete(&self.progress)
    }
}

impl ActiveRequest<TextGenerationDefinition> {
    /// Persist the one worker-local cache observation made at admission.
    /// Request progress remains compute-only; the immutable declaration stays
    /// in `definition.session` for comparison in request-level logs.
    pub fn record_prefix_cache_hit_tokens(&mut self, prefix_cache_hit_tokens: u32) {
        let declared_prefix_tokens = self.request.definition.session.declared_prefix_tokens();
        assert!(
            prefix_cache_hit_tokens <= declared_prefix_tokens,
            "request {} hit {} prefix tokens but declared only {}",
            self.request.core.id.0,
            prefix_cache_hit_tokens,
            declared_prefix_tokens,
        );
        assert!(
            self.telemetry.prefix_cache_hit_tokens.is_none(),
            "request {} recorded prefix-cache hit tokens more than once",
            self.request.core.id.0,
        );
        self.telemetry.prefix_cache_hit_tokens = Some(prefix_cache_hit_tokens);
    }

    pub fn record_retraction(&mut self) {
        self.telemetry.retraction_count = self
            .telemetry
            .retraction_count
            .checked_add(1)
            .expect("request retraction count overflow");
    }

    pub fn begin_reprocessed_prefill(&mut self, prefix_cache_hit_tokens: u32) {
        assert!(
            self.telemetry
                .reprocessed_prefills
                .last()
                .is_none_or(|episode| episode.completed),
            "request {} began overlapping reprocessed-prefill episodes",
            self.request.core.id.0,
        );
        self.telemetry
            .reprocessed_prefills
            .push(ReprocessedPrefillEpisode {
                output_tokens_before: self.progress.output_tokens_emitted,
                prefix_cache_hit_tokens,
                prefill_tokens_processed: 0,
                completed: false,
            });
    }

    pub fn record_reprocessed_prefill_tokens(&mut self, tokens: u32) {
        let episode = self
            .telemetry
            .reprocessed_prefills
            .last_mut()
            .expect("reprocessed prefill tokens require an active episode");
        assert!(
            !episode.completed,
            "reprocessed prefill episode already completed"
        );
        episode.prefill_tokens_processed = episode
            .prefill_tokens_processed
            .checked_add(tokens)
            .expect("reprocessed prefill token count overflow");
    }

    pub fn complete_reprocessed_prefill(&mut self) {
        let episode = self
            .telemetry
            .reprocessed_prefills
            .last_mut()
            .expect("reprocessed prefill completion requires an active episode");
        assert!(!episode.completed, "reprocessed prefill completed twice");
        episode.completed = true;
    }
}

/// Dense id-indexed slots plus compact lifecycle indexes.
///
/// `records[id]` keeps direct request lookup O(1); `None` means the trace
/// reserved the id but the replay scheduler has not released that request yet.
/// The id vectors avoid scanning every reserved slot for arrived/admitted views.
#[derive(Debug)]
pub struct RequestStore<Definition: RequestDefinition = TextGenerationDefinition> {
    records: Vec<Option<ActiveRequest<Definition>>>,
    /// Request ids in release order. Every present record occurs exactly once.
    arrived_ids: Vec<RequestId>,
    /// Request ids in first-admission order. Every admitted record occurs once.
    admitted_ids: Vec<RequestId>,
}

impl<Definition: RequestDefinition> Default for RequestStore<Definition> {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            arrived_ids: Vec::new(),
            admitted_ids: Vec::new(),
        }
    }
}

impl<Definition: RequestDefinition> RequestStore<Definition> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve dense ids without constructing fake requests.
    pub fn reserve_slots(&mut self, count: usize) {
        assert!(
            self.records.is_empty(),
            "request slots may only be reserved once, before arrivals are inserted"
        );
        self.records.resize_with(count, || None);
    }

    /// Insert one released request. A reserved slot must be empty; without
    /// pre-sizing, ids must still append densely.
    pub fn insert(&mut self, request: Request<Definition>) {
        let request_id = request.core.id;
        let slot = request_id.0 as usize;
        if slot == self.records.len() {
            self.records
                .push(Some(ActiveRequest::from_request(request)));
            self.arrived_ids.push(request_id);
            return;
        }
        let Some(record_slot) = self.records.get_mut(slot) else {
            panic!(
                "RequestStore::insert past the end (id={}, next slot={})",
                slot,
                self.records.len()
            );
        };
        assert!(record_slot.is_none(), "request id {slot} inserted twice");
        *record_slot = Some(ActiveRequest::from_request(request));
        self.arrived_ids.push(request_id);
    }

    pub fn get(&self, id: RequestId) -> Option<&ActiveRequest<Definition>> {
        self.records.get(id.0 as usize).and_then(Option::as_ref)
    }

    pub fn get_mut(&mut self, id: RequestId) -> Option<&mut ActiveRequest<Definition>> {
        self.records.get_mut(id.0 as usize).and_then(Option::as_mut)
    }

    pub fn len(&self) -> usize {
        self.arrived_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arrived_ids.is_empty()
    }

    pub fn iter_arrived(&self) -> impl Iterator<Item = (RequestId, &ActiveRequest<Definition>)> {
        self.arrived_ids.iter().copied().map(|request_id| {
            let record = self
                .get(request_id)
                .expect("arrived request id must reference a present record");
            (request_id, record)
        })
    }

    pub fn mark_admitted(&mut self, id: RequestId) {
        let newly_admitted = {
            let record = self
                .get_mut(id)
                .unwrap_or_else(|| panic!("request {} admitted before arrival", id.0));
            if record.lifecycle.admitted {
                false
            } else {
                record.lifecycle.admitted = true;
                true
            }
        };
        if newly_admitted {
            self.admitted_ids.push(id);
        }
    }

    pub fn num_admitted(&self) -> u64 {
        self.admitted_ids.len() as u64
    }

    pub fn iter_admitted(&self) -> impl Iterator<Item = (RequestId, &ActiveRequest<Definition>)> {
        self.admitted_ids.iter().copied().map(|request_id| {
            let record = self
                .get(request_id)
                .expect("admitted request id must reference a present record");
            debug_assert!(record.lifecycle.admitted);
            (request_id, record)
        })
    }

    pub fn all_complete(&self) -> bool {
        self.iter_arrived()
            .all(|(_, record)| record.lifecycle.completed)
    }

    pub fn in_flight(&self) -> usize {
        self.iter_arrived()
            .filter(|(_, record)| !record.lifecycle.completed)
            .count()
    }
}

impl<Definition: RequestDefinition> Index<RequestId> for RequestStore<Definition> {
    type Output = ActiveRequest<Definition>;

    fn index(&self, id: RequestId) -> &Self::Output {
        self.get(id)
            .unwrap_or_else(|| panic!("request {} has not arrived", id.0))
    }
}

impl<Definition: RequestDefinition> IndexMut<RequestId> for RequestStore<Definition> {
    fn index_mut(&mut self, id: RequestId) -> &mut Self::Output {
        self.get_mut(id)
            .unwrap_or_else(|| panic!("request {} has not arrived", id.0))
    }
}

pub type SharedRequests<Definition = TextGenerationDefinition> =
    Rc<RefCell<RequestStore<Definition>>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::text_request;

    #[test]
    fn first_token_completes_single_token_request() {
        let request = text_request(RequestId(0), 8, 1, Time::ZERO);
        let mut record = ActiveRequest::from_request(request);

        record.record_first_token(Time::from_ms(1.0), false);

        assert_eq!(record.progress.output_tokens_emitted, 1);
        assert!(record.is_complete());
        assert!(record.lifecycle.completed);
    }

    #[test]
    fn reserved_slots_do_not_create_arrived_requests() {
        let mut store = RequestStore::new();
        store.reserve_slots(2);
        store.insert(text_request(RequestId(1), 8, 2, Time::ZERO));

        assert_eq!(store.len(), 1);
        assert_eq!(
            store
                .iter_arrived()
                .map(|(request_id, _)| request_id)
                .collect::<Vec<_>>(),
            vec![RequestId(1)]
        );
        assert!(store.get(RequestId(0)).is_none());
        assert!(store.get(RequestId(1)).is_some());
    }

    #[test]
    fn compact_indexes_follow_release_and_first_admission_order() {
        let mut store = RequestStore::new();
        store.reserve_slots(4);
        store.insert(text_request(RequestId(3), 8, 2, Time::ZERO));
        store.insert(text_request(RequestId(1), 8, 2, Time::ZERO));

        assert_eq!(
            store
                .iter_arrived()
                .map(|(request_id, _)| request_id)
                .collect::<Vec<_>>(),
            vec![RequestId(3), RequestId(1)]
        );

        store.mark_admitted(RequestId(1));
        store.mark_admitted(RequestId(1));
        store.mark_admitted(RequestId(3));

        assert_eq!(store.num_admitted(), 2);
        assert_eq!(
            store
                .iter_admitted()
                .map(|(request_id, _)| request_id)
                .collect::<Vec<_>>(),
            vec![RequestId(1), RequestId(3)]
        );
    }
}
