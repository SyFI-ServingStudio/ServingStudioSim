//! `FullAttnKv` — one implementation of the KV resource axis.
//!
//! Owns `batches` (KvPool + decode set per partition), the `promised` reserve
//! ledger, the `request_to_partition` map, and the occupancy `KvSampler`. Answers
//! the Admission axis exactly one capacity question (`fits`) and never leaks a
//! scalar `remaining()` / `capacity()`. `Footprint` is a plain `u64` (tokens)
//! here; HybridKv / MultiModelKv would make it a per-subpool vector (design §4.8).
//!
//! Relocations that make the split honest (same math, moved home):
//!   - `KvAdmission::try_admit`'s strict capacity/peak/promised test → `fits`.
//!   - `group_promised_kv` / `drain_promises_into_admits` → here (KV ledgers).
//!
//! PRESERVATION INVARIANTS (bit-identity hazards — must not change):
//!   - `promised` is a std `HashMap`; `drain_ready` appends in hash order (which
//!     feeds `prefill_admits` order → worker-built arch-input pair order + decode insertion
//!     order). This is already run-to-run nondeterministic today; preserve the
//!     structure, do not "clean it up" into a different order.
//!   - `release` uses `Batch::release` (order-preserving `Vec::remove`);
//!     `release_external` drops a prefill with `swap_remove`. Keep both exact.
//!   - `sample_submit` fires exactly once per iter, AFTER advance/finalize/release.

use std::collections::HashMap;

use crate::common::{RequestId, Time};
use crate::log::{KvSampler, KvSubmit};
use crate::worker::admission_helpers::Batch;

use super::{Grouping, PartitionId};

pub(crate) struct FullAttnKv {
    /// One `Batch` per KV partition. Barebone builds exactly one (partition 0);
    /// HP/DP build N. Indexed by `PartitionId`.
    batches: Vec<Batch>,
    /// Reserve ledger: admitted-but-not-yet-resident. `(PartitionId, footprint)`.
    /// Was `BareboneWorker.runtime.promised`.
    promised: HashMap<RequestId, (PartitionId, u64)>,
    /// Which KV partition each admitted request lives in. Was
    /// `BareboneWorker.runtime.request_to_group` (renamed: it maps to the KV
    /// resource partition, never a request grouping).
    request_to_partition: HashMap<RequestId, PartitionId>,
    /// Per-worker KV occupancy sampler; `None` when no log dir.
    sampler: Option<KvSampler>,
}

impl FullAttnKv {
    pub(crate) fn new(num_partitions: usize, kv_capacity: u64, sampler: Option<KvSampler>) -> Self {
        Self {
            batches: (0..num_partitions as u16)
                .map(|partition| Batch::new(partition, kv_capacity))
                .collect(),
            promised: HashMap::new(),
            request_to_partition: HashMap::new(),
            sampler,
        }
    }

    // ── the ONE currency the Admission axis speaks: (prompt, decode) tokens → footprint ──

    /// Opaque footprint for `(prompt, decode)` tokens plus a matched prefix.
    /// Barebone always passes `prefix = 0`; F1 (prefix cache) discounts here.
    /// For full-attention the footprint is just the token count. [design §9.1 #1]
    #[inline]
    pub(crate) fn footprint(prompt: u32, decode: u32, prefix: u32) -> u64 {
        (prompt + prefix + decode) as u64
    }

    /// The ONLY capacity question. Reproduces strict `KvAdmission::try_admit`
    /// (admission_helpers.rs) verbatim. `projected_peak` stays internal (never
    /// exposed as a scalar). A future fixed safety margin belongs in this KV
    /// manager's construction; dynamic unavailable KV stays in its explicit ledgers.
    /// Keyed by `PartitionId` — a pure resource question about one KV pool, with no
    /// notion of which request grouping will consume it (design §9.0).
    pub(crate) fn fits(&self, partition: PartitionId, footprint: u64) -> bool {
        let batch = &self.batches[partition as usize];
        let admission_capacity = batch.kv.kv_capacity;
        let demand = self.partition_promised(partition) + footprint;
        // Cheap necessary bound before the O(n log n) peak (see try_admit's note).
        if batch.kv.active_kv + demand > admission_capacity {
            return false;
        }
        batch.projected_peak_kv() + demand <= admission_capacity
    }

    // ── reserve → drain → commit_resident → release lifecycle ──

    /// Was `promise`'s KV half only: record the reserve. The store writes +
    /// Prefill stage that `promise` also did now live in `PrefillDecode::admit`
    /// (an admission transition, per design §4.11).
    pub(crate) fn reserve(&mut self, request: RequestId, partition: PartitionId, footprint: u64) {
        self.promised.insert(request, (partition, footprint));
        self.request_to_partition.insert(request, partition);
    }

    /// Was `drain_promises_into_admits`. Barebone readiness is always true (KV is
    /// local; no remote pull), so every promise drains into its partition's
    /// `prefill_admits` this tick. Hash iteration order preserved (see invariant).
    pub(crate) fn drain_ready(&mut self) {
        let drained: Vec<(PartitionId, RequestId)> = self
            .promised
            .iter()
            .map(|(&request, &(partition, _))| (partition, request))
            .collect();
        for (partition, request) in drained {
            self.promised.remove(&request);
            self.batches[partition as usize]
                .prefill_admits
                .push(request);
        }
    }

    /// Iter-end growth, keyed by request `Grouping` (NOT a bare partition): WHICH
    /// requests grow is a request-axis choice. `Partition(p)` grows the whole
    /// decode set (barebone / DP); `Reqs { .. }` is AFD's partial advance;
    /// `steps = N` is SpecDecode's variable accept length. Barebone calls
    /// `advance(Grouping::Partition(0), 1)`. [design §9.1 #3]
    pub(crate) fn advance(&mut self, grouping: Grouping, steps: u32) {
        debug_assert_eq!(steps, 1, "full-attn barebone grows exactly 1 token/iter");
        match grouping {
            Grouping::Partition(partition) => self.batches[partition as usize].advance_decodes(),
            Grouping::Reqs {
                partition,
                request_ids,
            } => {
                self.debug_assert_group_partition(partition, request_ids);
                self.batches[partition as usize].advance_subset(request_ids);
            }
        }
    }

    /// Was `finalize_to_decode`: a realized prefill becomes resident decode.
    /// `initial_kv` (= prompt + prefix) is the RESIDENT amount — distinct from the
    /// reserved footprint (which also counted the decode budget). This is why
    /// `commit_resident` takes `(initial_kv, remaining)` explicitly rather than
    /// the design §9 thin `commit_resident(req, p)` — a real interface finding.
    pub(crate) fn commit_resident(
        &mut self,
        partition: PartitionId,
        request: RequestId,
        initial_kv: u64,
        remaining: u32,
    ) {
        self.batches[partition as usize].finalize_to_decode(request, initial_kv, remaining);
    }

    pub(crate) fn clear_prefill_admits(&mut self, partition: PartitionId) {
        self.batches[partition as usize].prefill_admits.clear();
    }

    /// Was `complete_iter` step (d)'s release: look up the resident kv, drop from
    /// the decode set, forget the partition mapping. A prefill that resolved straight
    /// to done (single-token) is not resident → `None` kv → a no-op `release`, but
    /// its `request_to_partition` entry is still cleared.
    pub(crate) fn release(&mut self, partition: PartitionId, request: RequestId) {
        let current_kv = self.batches[partition as usize]
            .decode_current_kv(request)
            .unwrap_or(0);
        self.batches[partition as usize].release(request, current_kv);
        self.request_to_partition.remove(&request);
    }

    /// Was `release_request`'s admitted branch (cancellation past the pending
    /// queue). Returns the partition it was released from. Uses `swap_remove` on
    /// `prefill_admits` exactly as today (invariant).
    pub(crate) fn release_external(
        &mut self,
        request: RequestId,
        current_kv: u64,
    ) -> Option<PartitionId> {
        let partition = self.request_to_partition.remove(&request)?;
        let batch = &mut self.batches[partition as usize];
        batch.release(request, current_kv);
        if let Some(position) = batch
            .prefill_admits
            .iter()
            .position(|&admit| admit == request)
        {
            batch.prefill_admits.swap_remove(position);
        }
        self.promised.remove(&request);
        Some(partition)
    }

    /// Was `complete_iter`'s tail KV sample submit (once per iter, post advance /
    /// finalize / release, same active/peak/promised triple as today).
    pub(crate) fn sample_submit(&mut self, partition: PartitionId, now: Time) {
        if self.sampler.is_none() {
            return;
        }
        let submit = KvSubmit {
            active_kv: self.batches[partition as usize].kv.active_kv,
            projected_peak: self.batches[partition as usize].projected_peak_kv(),
            promised_kv: self.partition_promised(partition),
        };
        self.sampler
            .as_mut()
            .unwrap()
            .submit(partition, submit, now);
    }

    // ── read-only facts other components ask for (no scalar capacity leaks) ──

    #[inline]
    fn partition_promised(&self, partition: PartitionId) -> u64 {
        self.promised
            .values()
            .filter(|(reserved_partition, _)| *reserved_partition == partition)
            .map(|(_, footprint)| *footprint)
            .sum()
    }

    /// Explicit request groupings are caller-owned, but every member must still
    /// belong to the resource partition the caller declared. Keeping this check in
    /// Kv catches stale slot membership without making Kv own request→slot state.
    #[inline]
    fn debug_assert_group_partition(&self, partition: PartitionId, request_ids: &[RequestId]) {
        debug_assert!(
            request_ids.iter().all(|request| {
                self.request_to_partition.get(request).copied() == Some(partition)
            }),
            "explicit request grouping contains a request outside partition {partition}"
        );
    }

    #[inline]
    pub(crate) fn batch(&self, partition: PartitionId) -> &Batch {
        &self.batches[partition as usize]
    }

    #[inline]
    pub(crate) fn has_live_decode(&self, partition: PartitionId) -> bool {
        self.batches[partition as usize]
            .iter_decoding()
            .next()
            .is_some()
    }

    #[inline]
    pub(crate) fn live_decode_count(&self, partition: PartitionId) -> u32 {
        self.batches[partition as usize].iter_decoding().count() as u32
    }

    #[inline]
    pub(crate) fn has_prefill_admit(&self, partition: PartitionId) -> bool {
        !self.batches[partition as usize].prefill_admits.is_empty()
    }

    /// Was `status`'s active tally: live decodes + promised + this-iter prefills.
    pub(crate) fn status_active(&self, partition: PartitionId) -> u32 {
        self.live_decode_count(partition)
            + self.promised.len() as u32
            + self.batches[partition as usize].prefill_admits.len() as u32
    }
}
