//! `UnifiedArch` — the model + cost execution component paired with this Worker.
//!
//! Holds the model + `CostBuffers`. Consumes a `UnifiedArchInput`, returns the
//! iter's wall time. This is not an independently swappable axis: its input and
//! execution protocol are paired with the concrete Worker. Family-A shape is one
//! fused `eval_iter` per iteration; a different protocol gets another short pair.

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::Time;
use crate::worker::cost_buffers::CostBuffers;

pub struct UnifiedArch<M: IterwiseUnifiedModel> {
    model: Arc<M>,
    cost: CostBuffers,
}

impl<M: IterwiseUnifiedModel> UnifiedArch<M> {
    pub(super) fn new(model: Arc<M>, cost: CostBuffers) -> Self {
        Self { model, cost }
    }

    /// Was `start_iter`'s cost query. `CostBuffers::run_iter` writes the cost_log
    /// row synchronously with `(iter, now)`, so the iter id + timestamp are part
    /// of the row identity — preserve them exactly (see plan's cost-log invariant).
    pub(super) fn eval_iter(&mut self, input: &UnifiedArchInput, iter: u64, now: Time) -> Time {
        self.cost.run_iter(self.model.as_ref(), input, iter, now)
    }
}
