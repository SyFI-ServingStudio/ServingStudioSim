//! Prefill plan for the next `depth` microbatches of a pipeline head.
//!
//! A pipeline of depth `N` holds at most `N` microbatches in flight, so each
//! of the `N` in-flight positions is a serial lane: microbatch `i` starts only
//! after `i - N` exits. A round of `N` consecutive microbatches therefore takes
//! about `N x` its largest member, and the difference from the members' sum is
//! idle downstream time. The plan keeps the `N` microbatches that will launch
//! next as future slots and bin-packs whole requests into them, so work that
//! arrives together spreads over the round instead of riding in one microbatch.
//!
//! Placement never cuts a request on purpose. A request (or a started request's
//! unplanned remainder) goes into the least-loaded future slot after any slot
//! that already holds it, earliest on ties, among the slots from which the
//! plan still has room for all of it. What does not fit that slot's
//! `max_batch_tokens` fills it and continues in the following slots, one chunk
//! per slot, so a request splits only where ordinary chunked prefill would.
//! When no slot leaves room for the whole request (a backlog), it fills from the
//! earliest slot with room, as greedy would, so every future slot ends full; the
//! remainder past the last slot stays unplanned until a launch opens a new tail
//! slot. Planned tokens never move to a later slot; they move earlier
//! only when the head slot is empty at launch (`launch_head_slot`).
//!
//! The plan owns slot membership and nothing else: the lifecycle admits
//! requests, reserves their KV, and schedules the launched slot's chunks.

use std::collections::VecDeque;

use crate::common::RequestId;

/// One request's planned chunk in one future slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PlannedChunk {
    pub(super) request: RequestId,
    pub(super) tokens: u32,
}

/// One future microbatch's planned prefill, in placement order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct PlannedSlot {
    pub(super) chunks: Vec<PlannedChunk>,
    tokens: u32,
}

impl PlannedSlot {
    #[cfg(test)]
    pub(super) fn tokens(&self) -> u32 {
        self.tokens
    }
}

/// An admitted request's prefill tokens that no future slot holds yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct UnplannedPrefill {
    pub(super) request: RequestId,
    pub(super) tokens: u32,
}

pub(super) struct MicrobatchSlotPlan {
    slot_tokens: u32,
    slots: VecDeque<PlannedSlot>,
    /// Admitted requests with prefill tokens left to place, in admission order.
    pub(super) unplanned: VecDeque<UnplannedPrefill>,
}

impl MicrobatchSlotPlan {
    pub(super) fn new(depth: u16, slot_tokens: u32) -> Self {
        assert!(depth > 0, "pipeline depth must be positive");
        assert!(slot_tokens > 0, "slot capacity must be positive");
        Self {
            slot_tokens,
            slots: (0..depth).map(|_| PlannedSlot::default()).collect(),
            unplanned: VecDeque::new(),
        }
    }

    /// Whether any future slot has capacity left.
    pub(super) fn has_room(&self) -> bool {
        self.slots.iter().any(|slot| slot.tokens < self.slot_tokens)
    }

    /// Place up to `tokens` of `request` and return how many found no slot.
    pub(super) fn place(&mut self, request: RequestId, tokens: u32) -> u32 {
        let first = self
            .slots
            .iter()
            .rposition(|slot| slot.chunks.iter().any(|chunk| chunk.request == request))
            .map_or(0, |last| last + 1);
        // Room in slots `index..`, for the fit test below.
        let mut room_from = vec![0_u64; self.slots.len() + 1];
        for index in (first..self.slots.len()).rev() {
            room_from[index] =
                room_from[index + 1] + u64::from(self.slot_tokens - self.slots[index].tokens);
        }
        let has_room = |index: usize| self.slots[index].tokens < self.slot_tokens;
        let least_loaded_fit = (first..self.slots.len())
            .filter(|&index| has_room(index) && room_from[index] >= u64::from(tokens))
            .min_by_key(|&index| (self.slots[index].tokens, index));
        // Under backlog nothing holds the whole request: fill from the earliest
        // room, as greedy chunking would, so every slot ends full.
        let Some(start) =
            least_loaded_fit.or_else(|| (first..self.slots.len()).find(|&i| has_room(i)))
        else {
            return tokens;
        };
        let mut left = tokens;
        for slot in self.slots.iter_mut().skip(start) {
            if left == 0 {
                break;
            }
            let take = (self.slot_tokens - slot.tokens).min(left);
            if take == 0 {
                continue;
            }
            slot.chunks.push(PlannedChunk {
                request,
                tokens: take,
            });
            slot.tokens += take;
            left -= take;
        }
        left
    }

    /// Remove the head slot for launch and open an empty tail slot. An empty
    /// head with work planned behind it is dropped first, moving that work one
    /// slot earlier, so a launch never waits behind an empty slot.
    pub(super) fn launch_head_slot(&mut self) -> PlannedSlot {
        while self.slots[0].chunks.is_empty() && self.slots.iter().any(|s| !s.chunks.is_empty()) {
            self.slots.rotate_left(1);
        }
        let head = self.slots.pop_front().expect("plan has depth slots");
        self.slots.push_back(PlannedSlot::default());
        head
    }

    /// Remove `request`'s chunks from every future slot and return their tokens.
    pub(super) fn withdraw(&mut self, request: RequestId) -> u32 {
        let mut withdrawn = 0;
        for slot in &mut self.slots {
            slot.chunks.retain(|chunk| {
                if chunk.request == request {
                    withdrawn += chunk.tokens;
                    false
                } else {
                    true
                }
            });
            slot.tokens = slot.chunks.iter().map(|chunk| chunk.tokens).sum();
        }
        withdrawn
    }

    #[cfg(test)]
    fn slot_tokens(&self) -> Vec<u32> {
        self.slots.iter().map(|slot| slot.tokens).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_request_takes_one_slot_and_the_next_takes_the_emptiest() {
        let mut plan = MicrobatchSlotPlan::new(4, 8192);
        assert_eq!(plan.place(RequestId(0), 100), 0);
        assert_eq!(plan.place(RequestId(1), 300), 0);
        assert_eq!(plan.place(RequestId(2), 50), 0);
        assert_eq!(plan.slot_tokens(), vec![100, 300, 50, 0]);
        assert_eq!(plan.place(RequestId(3), 10), 0);
        assert_eq!(plan.slot_tokens(), vec![100, 300, 50, 10]);
        assert_eq!(plan.place(RequestId(4), 10), 0);
        assert_eq!(plan.slot_tokens(), vec![100, 300, 50, 20]);
    }

    #[test]
    fn a_long_request_splits_only_at_the_slot_cap() {
        let mut plan = MicrobatchSlotPlan::new(4, 8192);
        assert_eq!(plan.place(RequestId(0), 20_000), 0);
        assert_eq!(plan.slot_tokens(), vec![8192, 8192, 3616, 0]);
        // The next one cannot fit whole: it fills from the earliest room, and
        // what runs past the last slot stays unplanned.
        assert_eq!(plan.place(RequestId(1), 20_000), 20_000 - 4576 - 8192);
        assert_eq!(plan.slot_tokens(), vec![8192, 8192, 8192, 8192]);
        assert!(!plan.has_room());
    }

    #[test]
    fn a_remainder_continues_after_the_requests_last_slot() {
        let mut plan = MicrobatchSlotPlan::new(3, 100);
        assert_eq!(plan.place(RequestId(0), 150), 0);
        assert_eq!(plan.slot_tokens(), vec![100, 50, 0]);
        // The emptiest slot after request 0's last chunk is the tail.
        assert_eq!(plan.place(RequestId(0), 30), 0);
        assert_eq!(plan.slot_tokens(), vec![100, 50, 30]);
    }

    #[test]
    fn launch_moves_work_earlier_past_an_empty_head() {
        let mut plan = MicrobatchSlotPlan::new(3, 100);
        plan.place(RequestId(0), 100);
        plan.place(RequestId(1), 10);
        assert_eq!(plan.launch_head_slot().tokens(), 100);
        assert_eq!(plan.slot_tokens(), vec![10, 0, 0]);
        let mut plan = MicrobatchSlotPlan::new(3, 100);
        plan.place(RequestId(0), 50);
        plan.place(RequestId(1), 10);
        plan.withdraw(RequestId(0));
        assert_eq!(plan.slot_tokens(), vec![0, 10, 0]);
        let head = plan.launch_head_slot();
        assert_eq!(
            head.chunks,
            vec![PlannedChunk {
                request: RequestId(1),
                tokens: 10
            }]
        );
        assert_eq!(plan.slot_tokens(), vec![0, 0, 0]);
    }
}
