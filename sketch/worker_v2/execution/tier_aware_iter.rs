//! `TierAwareIterExecution<M>` — tier-aware execution — PROVES the Root ② fix: an
//! `IterModelExecution<K>` impl can now escalate to a KV capability view, symmetric
//! with `IterAdmission<K>`.
//!
//! Wraps `UnifiedIterExecution` for the arch-input lowering + base cost (NO duplication of the
//! group-building loop — the §15.3 dedup check), and adds ONE thing: it reads the
//! `TieredKvView` view's fast/slow resident split in `build_iteration_input` and charges an
//! offload surcharge in `evaluate_iteration`. Its impl bound is
//! `K: IterWorkerKv + TieredKvView` — a plain
//! `UnifiedIterExecution` (bound only `IterWorkerKv`) could NOT express this, which is exactly the
//! asymmetry the fix closes: before, KV capabilities reached `IterAdmission` but not the cost
//! model. It composes ONLY with a tiered KV (`TieredMemoryKv`); pairing it with
//! `FullAttnKv` is a compile error at the recipe, NOT a dummy method forced onto
//! `FullAttnKv`.
//!
//! Superset: the offload surcharge (slow-tier tokens × a per-token transfer cost) is a
//! modeled placeholder, not a measured PCIe/NVMe kernel. The point is the SEAM reaching
//! the cost model, not fidelity.

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{SharedRequests, Time};
use crate::worker::cost_buffers::CostBuffers;

use super::super::kv::{IterWorkerKv, TieredKvView};
use super::{IterModelExecution, ModelKvLayout, UnifiedIterExecution};

/// Tier-aware input: the base arch input + the slow-tier resident total captured from the
/// `TieredKvView` view, which `evaluate_iteration` charges an offload surcharge on.
#[derive(Default)]
pub struct TierAwareIterInput {
    arch: UnifiedArchInput,
    slow_tier_tokens: u64,
}

pub struct TierAwareIterExecution<M: IterwiseUnifiedModel> {
    inner: UnifiedIterExecution<M>,
    /// Modeled offload cost per slow-tier token (ns). Real impl: a measured transfer curve.
    offload_ns_per_token: u64,
}

impl<M: IterwiseUnifiedModel> TierAwareIterExecution<M> {
    pub fn new(model: Arc<M>, cost: CostBuffers, offload_ns_per_token: u64) -> Self {
        Self {
            inner: UnifiedIterExecution::new(model, cost),
            offload_ns_per_token,
        }
    }
}

impl<M: IterwiseUnifiedModel, K: IterWorkerKv + TieredKvView> IterModelExecution<K>
    for TierAwareIterExecution<M>
{
    type Input = TierAwareIterInput;

    fn model_kv_layout(&self) -> ModelKvLayout {
        // model_kv_layout does not depend on K; qualify to pick the (arbitrary) K impl.
        <UnifiedIterExecution<M> as IterModelExecution<K>>::model_kv_layout(&self.inner)
    }

    /// Delegate the arch-input lowering to `UnifiedIterExecution` (no group-loop
    /// duplication), then read the capability view — the line a plain
    /// `IterModelExecution` could not write.
    fn build_iteration_input(
        &self,
        kv_store: &K,
        requests: &SharedRequests,
        out: &mut TierAwareIterInput,
    ) {
        self.inner
            .build_iteration_input(kv_store, requests, &mut out.arch);
        out.slow_tier_tokens = (0..kv_store.num_partitions() as u16)
            .map(|partition| kv_store.resident_by_tier(partition).1)
            .sum();
    }

    fn evaluate_iteration(&mut self, input: &TierAwareIterInput, iter: u64, now: Time) -> Time {
        let base = <UnifiedIterExecution<M> as IterModelExecution<K>>::evaluate_iteration(
            &mut self.inner,
            &input.arch,
            iter,
            now,
        );
        // The offload surcharge — the tier split reaching the cost model, the whole point.
        let surcharge = Time(
            input
                .slow_tier_tokens
                .saturating_mul(self.offload_ns_per_token),
        );
        base + surcharge
    }
}
