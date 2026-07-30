//! Selection policy sub-axis (interfaces doc §3.2/§3.3, simplified: no Oracle).
//!
//! A policy OWNS the pending queue and hands the lifecycle ONE head at a time. It stays
//! pure over pre-computed candidate facts: it never touches KV and never mutates
//! resources — the lifecycle still does `fits`/`reserve` itself and may decline the head
//! (leaving it queued) when the gate or the budget says no.
//!
//! WHY the policy owns the queue instead of being handed a slice. `form_batch` runs once
//! per simulated iteration, and the earlier
//! `order(&[AdmissionCandidate]) -> Vec<usize>` shape
//! forced every caller to materialize AND rank the whole backlog just to use its head:
//! two heap allocations, N store lookups, and (for `ShortestJobFirst`) an
//! O(N log N) sort per
//! iteration — worst precisely when the worker is saturated (deep queue, KV full,
//! nothing admitted, same work again next tick). Production's
//! `worker/unified.rs` does `pending_prefills.front()` in O(1); owning the
//! structure gets that back. `FifoOrder` is a `VecDeque` (O(1)
//! push/peek/pop) and `ShortestJobFirst` a `BinaryHeap` (O(log N)
//! push/pop, O(1) peek) that is never re-sorted. The store lookup moves to
//! `accept_message`, so it is once per REQUEST rather than once per pending
//! request per iteration.
//!
//! A policy whose rank genuinely varies per iteration (deadline slack against the clock,
//! KV pressure, a prefix-hit rate that moves as the cache fills) can still re-evaluate
//! inside its own `peek`/`push`. The point is that the cost then lands on THAT policy
//! instead of every policy paying for the most expensive conceivable one.

use crate::common::{RequestId, Time};

mod fifo;
mod shortest_job_first;
pub use fifo::FifoOrder;
pub use shortest_job_first::ShortestJobFirst;

/// One admission candidate with the facts a policy may rank on, computed by the
/// lifecycle at ARRIVAL. `Copy` so `peek` can hand out a value without lending the
/// policy's internals to the caller (which would block the `pop` that follows).
///
/// Every field here is immutable for the life of the request (`prompt`/`decode`/
/// `arrival_seq` never change), which is what makes freezing them at arrival sound.
/// `matched_tokens` is the one exception in spirit — see
/// `PrefixPrefillDecodeAdmission`, which snapshots it here for ranking and re-reads
/// the live value for the ONE request it admits.
#[derive(Clone, Copy)]
pub struct AdmissionCandidate {
    pub request: RequestId,
    pub arrival_seq: u64,
    pub prompt: u32,
    pub decode: u32,
    pub deadline: Option<Time>,
    pub matched_tokens: u32,
}

pub trait PendingOrderPolicy {
    type Context;

    /// Queue one arrival into the policy's own structure.
    fn push(&mut self, candidate: AdmissionCandidate, context: &mut Self::Context);

    /// The next candidate to try, WITHOUT dequeuing it — the lifecycle may decline it on
    /// `fits`/budget, in which case it must stay queued at the head.
    fn peek(&self) -> Option<AdmissionCandidate>;

    /// Dequeue the head (the lifecycle took it).
    fn pop(&mut self, context: &mut Self::Context) -> Option<AdmissionCandidate>;

    /// Cancellation: drop `req` wherever it sits. O(n) and heap impls rebuild, but this
    /// is a rare control path, unlike the per-iteration head path.
    fn remove(&mut self, req: RequestId) -> bool;

    /// Queue depth, for `IterAdmission::queued_requests`.
    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
