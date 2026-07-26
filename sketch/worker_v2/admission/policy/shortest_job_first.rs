//! `ShortestJobFirst` — a latency-leaning selection policy (AP-class, non-FIFO). Orders
//! candidates by total work (prompt + decode tokens) ascending, so short requests clear
//! first (classic SJF head-of-line-blocking mitigation). Reads only the `prompt`/`decode`
//! facts the lifecycle pre-computes on every `AdmissionCandidate`, so — like
//! `FifoOrder` — it needs no
//! store access and no change to any lifecycle: a worker swaps throughput-FIFO for
//! latency-SJF by the `P` type param alone.
//!
//! A `BinaryHeap` keeps the queue ordered INCREMENTALLY (O(log n) push/pop, O(1) peek)
//! instead of re-sorting the whole backlog once per iteration. Same order, amortized.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::common::RequestId;

use super::{AdmissionCandidate, PendingOrderPolicy};

/// Heap entry ordered smallest-work-first. `BinaryHeap` is a MAX-heap, so `Ord` is
/// written REVERSED (`other.cmp(self)`); `arrival_seq` breaks ties in arrival order,
/// reproducing the FIFO tie-break the previous stable sort gave.
struct ShortestWorkFirst(AdmissionCandidate);

impl ShortestWorkFirst {
    #[inline]
    fn work(&self) -> u64 {
        u64::from(self.0.prompt) + u64::from(self.0.decode)
    }
}

impl Ord for ShortestWorkFirst {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .work()
            .cmp(&self.work())
            .then_with(|| other.0.arrival_seq.cmp(&self.0.arrival_seq))
    }
}

impl PartialOrd for ShortestWorkFirst {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ShortestWorkFirst {}

impl PartialEq for ShortestWorkFirst {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

#[derive(Default)]
pub struct ShortestJobFirst {
    queue: BinaryHeap<ShortestWorkFirst>,
}

impl ShortestJobFirst {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PendingOrderPolicy for ShortestJobFirst {
    type Context = ();

    #[inline]
    fn push(&mut self, candidate: AdmissionCandidate, _ctx: &mut ()) {
        self.queue.push(ShortestWorkFirst(candidate));
    }

    #[inline]
    fn peek(&self) -> Option<AdmissionCandidate> {
        self.queue.peek().map(|entry| entry.0)
    }

    #[inline]
    fn pop(&mut self, _ctx: &mut ()) -> Option<AdmissionCandidate> {
        self.queue.pop().map(|entry| entry.0)
    }

    /// O(n) rebuild — `BinaryHeap` has no keyed removal. Acceptable because cancellation
    /// is a rare control path; the per-iteration head path stays O(1)/O(log n).
    fn remove(&mut self, req: RequestId) -> bool {
        let before = self.queue.len();
        let kept: Vec<ShortestWorkFirst> = std::mem::take(&mut self.queue)
            .into_vec()
            .into_iter()
            .filter(|entry| entry.0.request != req)
            .collect();
        self.queue = BinaryHeap::from(kept);
        self.queue.len() != before
    }

    #[inline]
    fn len(&self) -> usize {
        self.queue.len()
    }
}
