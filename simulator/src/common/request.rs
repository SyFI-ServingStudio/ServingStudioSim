//! Single-round inference request representation + the worker-facing request
//! store.
//!
//! The current lifecycle is single-round and shared by unified, HP unified, and
//! PD workers. Multi-round fields (`round_idx`, `preserved_prefix_kv`, ...) land
//! alongside future L7 lifecycle work.

use std::cell::RefCell;
use std::ops::{Index, IndexMut};
use std::rc::Rc;

use serde::{Deserialize, Serialize};

use super::id::RequestId;
use super::time::Time;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Request {
    pub id: RequestId,
    pub prompt_len: u32,
    pub decode_len: u32,
    /// KV tokens already cached for this request at arrival (a trace-declared
    /// prefix-cache hit). `prompt_len` counts only the uncached suffix that is
    /// actually prefilled; the prefix still occupies KV and is attended over.
    pub prefix_kv: u32,
    /// Session this request belongs to (trace `session_id` column). Groups the
    /// sequential calls of one agent/user conversation: the prefix-cache model
    /// keys retained KV by session, and `session-sticky` placement routes a
    /// session to one worker. `None` = sessionless (always treated cold).
    pub session: Option<u32>,
    pub arrival_time: Time,
}

impl Request {
    pub const fn new(id: RequestId, prompt_len: u32, decode_len: u32, arrival_time: Time) -> Self {
        Self {
            id,
            prompt_len,
            decode_len,
            prefix_kv: 0,
            session: None,
            arrival_time,
        }
    }

    /// Trace rows may declare an already-cached prefix (frontend `prefix_kv`
    /// column); workers seed their admission prefix from it.
    pub const fn with_prefix_kv(mut self, prefix_kv: u32) -> Self {
        self.prefix_kv = prefix_kv;
        self
    }

    /// Trace rows may declare a session id (frontend `session_id` column).
    pub const fn with_session(mut self, session: Option<u32>) -> Self {
        self.session = session;
        self
    }
}

/// The full lifecycle record for one request: arrival facts (immutable),
/// the worker's FSM working fields, and output-token bookkeeping. This is the
/// `RequestVec` entry of L5 design.md §3.4; it lives in the shared store so a
/// pool's workers and the L7 logger all read one source of truth.
#[derive(Clone, Debug)]
pub struct RequestRecord {
    // ── arrival facts (immutable) ──
    pub prompt_len: u32,
    pub decode_len: u32,
    pub arrival_time: Time,
    /// See [`Request::session`].
    pub session: Option<u32>,

    // ── FSM working fields (mutated by the worker) ──
    /// Prefill tokens already processed; `== prefill_target` once prefill is done.
    pub prefill_processed: u32,
    /// Tokens this request must prefill. Starts at `prompt_len`; a prefix-cache
    /// MISS at admission raises it by the missed prefix (those tokens must be
    /// recomputed) while `prompt_len` stays the immutable arrival fact.
    pub prefill_target: u32,
    /// KV (tokens) already present from prior rounds this request builds on.
    pub prefix_kv: u32,
    /// Tokens this request contributes to the current iter (prefill chunk len
    /// while prefilling, 1 while decoding).
    pub active_chunk_len: u32,

    // ── output bookkeeping ──
    pub first_token_time: Option<Time>,
    pub last_token_time: Option<Time>,
    /// Absolute sim-time of every output token (first token + each decode step).
    /// Feeds the `request_slo` per-token timing column; `len() == tokens_emitted`.
    pub output_token_times: Vec<Time>,
    pub tokens_emitted: u32,
    pub completed: bool,
}

impl RequestRecord {
    pub fn from_request(req: &Request) -> Self {
        Self {
            prompt_len: req.prompt_len,
            decode_len: req.decode_len,
            arrival_time: req.arrival_time,
            session: req.session,
            prefill_processed: 0,
            prefill_target: req.prompt_len,
            prefix_kv: req.prefix_kv,
            active_chunk_len: 0,
            first_token_time: None,
            last_token_time: None,
            output_token_times: Vec::new(),
            tokens_emitted: 0,
            completed: false,
        }
    }

    pub fn is_prefill(&self) -> bool {
        self.prefill_processed < self.prefill_target
    }

    pub fn is_complete(&self) -> bool {
        self.tokens_emitted >= self.decode_len
    }

    /// Prefill resolved → first output token emitted. `log_tokens` mirrors
    /// `io.log_output_token_times`: when off, the per-token array is never
    /// allocated or appended (the scalars below feed `slo-general` regardless).
    pub fn record_first_token(&mut self, now: Time, log_tokens: bool) {
        self.tokens_emitted = 1;
        self.first_token_time = Some(now);
        self.last_token_time = Some(now);
        if self.is_complete() {
            // A decode_len==1 request is done at its first token; mirror
            // `record_token` so the completed flag can't go stale.
            self.completed = true;
        }
        if log_tokens {
            // The full decode length is known up front, so size the per-token
            // buffer exactly once here (only for requests that actually start
            // decoding) — the alternative is ~log2(decode_len) doubling reallocs
            // per request, each memcpy'ing the growing array.
            self.output_token_times
                .reserve_exact(self.decode_len as usize);
            self.output_token_times.push(now);
        }
    }

    /// One decode step produced a token. `log_tokens`: see `record_first_token`.
    pub fn record_token(&mut self, now: Time, log_tokens: bool) {
        self.tokens_emitted += 1;
        self.last_token_time = Some(now);
        if log_tokens {
            self.output_token_times.push(now);
        }
        if self.is_complete() {
            self.completed = true;
        }
    }
}

/// Authoritative slab of all arrived requests, keyed by id. Conceptually an L7
/// object (arrival frontend fills it, the logger reads it); for L5/L6 it is
/// created by the owner and shared via [`SharedRequests`].
#[derive(Default, Debug)]
pub struct RequestStore {
    /// Dense, id-indexed: `records[id.0]` is request `id`. The trace frontend
    /// validates ids as sequential `0..N` and emits them in arrival order, so a
    /// `Vec` replaces a per-tick hash lookup with a plain array index (the tick
    /// loop indexes the store on every active request, every tick).
    records: Vec<RequestRecord>,
    /// Highest id ever admitted to a worker (i.e. that started prefill), or
    /// `None` before the first admission. Because requests arrive id-dense in
    /// arrival order. Workers may admit non-FIFO, but the high-watermark remains
    /// monotonic: `iter_admitted` may transiently include zero-workload gaps, and
    /// never drops a real admitted request. Dense `request_state` snapshots skip
    /// the never-admitted pending tail.
    admitted_hi: Option<u32>,
}

impl RequestStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a freshly arrived request's record (arrival facts; working fields
    /// zeroed). Ids must arrive dense + in order so `records[id.0]` holds.
    pub fn insert(&mut self, req: &Request) {
        debug_assert_eq!(
            req.id.0 as usize,
            self.records.len(),
            "RequestStore expects dense, in-order ids (got id={}, next slot={})",
            req.id.0,
            self.records.len(),
        );
        self.records.push(RequestRecord::from_request(req));
    }

    /// Insert-or-overwrite the record for `req`. Flows call this at admission:
    /// in the default modes the store is empty at start, so this is a plain
    /// dense `insert`; in session-dependent replay the store was pre-filled
    /// (`TraceFrontend::preinsert_all`) because emission is out of id order, and
    /// this overwrites the placeholder with the admission-stamped arrival facts
    /// (the request has not run yet, so resetting working fields is a no-op).
    pub fn upsert(&mut self, req: &Request) {
        let slot = req.id.0 as usize;
        if slot < self.records.len() {
            self.records[slot] = RequestRecord::from_request(req);
        } else {
            self.insert(req);
        }
    }

    pub fn get(&self, id: RequestId) -> Option<&RequestRecord> {
        self.records.get(id.0 as usize)
    }

    pub fn get_mut(&mut self, id: RequestId) -> Option<&mut RequestRecord> {
        self.records.get_mut(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Iterate `(id, record)` over every request seen so far (incl. the unserved
    /// pending tail). Used by the end-of-run summary, which counts all arrivals.
    pub fn iter(&self) -> impl Iterator<Item = (RequestId, &RequestRecord)> {
        self.records
            .iter()
            .enumerate()
            .map(|(i, r)| (RequestId(i as u32), r))
    }

    /// Record that `id` has been admitted to a worker (started prefill),
    /// advancing the admitted-prefix watermark. Worker admission/promise helpers
    /// call this for barebone, HP, and PD prefill paths. `max` keeps the watermark
    /// monotonic under non-FIFO placement.
    pub fn mark_admitted(&mut self, id: RequestId) {
        self.admitted_hi = Some(self.admitted_hi.map_or(id.0, |hi| hi.max(id.0)));
    }

    /// Monotonic, O(1) admission watermark: `admitted_hi + 1`, or `0` before any
    /// admission. Strictly increases whenever a new highest-id request starts
    /// prefill, so the stuck-watchdog can detect "a new request was admitted"
    /// without scanning the store.
    pub fn admitted_watermark(&self) -> u64 {
        self.admitted_hi.map_or(0, |hi| hi as u64 + 1)
    }

    /// Iterate `(id, record)` over admitted requests only — the prefix
    /// `records[0..=admitted_hi]`. Empty until the first admission. Dense
    /// `request_state` snapshots use this instead of `iter` so the never-admitted
    /// pending tail is skipped (see `admitted_hi`).
    pub fn iter_admitted(&self) -> impl Iterator<Item = (RequestId, &RequestRecord)> {
        let upto = self.admitted_hi.map_or(0, |hi| hi as usize + 1);
        self.records[..upto]
            .iter()
            .enumerate()
            .map(|(i, r)| (RequestId(i as u32), r))
    }

    /// Every inserted request has completed. O(n); not on the per-tick path —
    /// `run_sim` tracks arrival/completion counters for termination.
    pub fn all_complete(&self) -> bool {
        self.records.iter().all(|r| r.completed)
    }

    /// Count of inserted-but-not-yet-completed requests. O(n); diagnostic helper.
    pub fn in_flight(&self) -> usize {
        self.records.iter().filter(|r| !r.completed).count()
    }
}

impl Index<RequestId> for RequestStore {
    type Output = RequestRecord;
    fn index(&self, id: RequestId) -> &RequestRecord {
        &self.records[id.0 as usize]
    }
}

impl IndexMut<RequestId> for RequestStore {
    fn index_mut(&mut self, id: RequestId) -> &mut RequestRecord {
        &mut self.records[id.0 as usize]
    }
}

/// Shared handle injected at construction into the flow → pool → workers. The
/// sim thread ticks workers sequentially, so `Rc<RefCell<>>` is enough; writer
/// threads receive row copies and do not access the store.
pub type SharedRequests = Rc<RefCell<RequestStore>>;
