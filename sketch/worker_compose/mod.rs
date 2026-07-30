//! TARGET-STATE SKETCH — one iter-wise worker after the component split.
//!
//! NOT part of the crate, NOT compiled, NOT wired. A blueprint to read: what the
//! unified worker (`simulator/src/worker/unified.rs`) looks like once split into
//! three components + worker-owned state machine and composition glue,
//! reproducing today's behavior.
//!
//! Reading order (each file carries its own component contract):
//!   1. ctx.rs                     — WorkerCtx (component-agnostic shared state)
//!   2. kv/                        — KV vocabulary + concrete resource models
//!   3. admission/                 — concrete batching/admission policies
//!   4. arch_unified.rs            — UnifiedArch (Arch: model + cost)
//!   5. worker.rs                  — state machine + composition + IterWorker impl
//!
//! In the real migration this becomes `pub(crate) mod compose;` under
//! `worker/`, re-exporting `BareboneWorker` through the existing path so the
//! public surface diff stays zero (review: do NOT add a new `pub mod`).

pub mod ctx;

mod admission;
mod arch_unified;
mod kv;
pub mod worker;

pub use worker::{BareboneWorker, Worker};

// ════════════════════════════════════════════════════════════════════════════
// APPENDIX — M2 : lift the two independent variation axes into traits
// ════════════════════════════════════════════════════════════════════════════
//
// M2 turns KV and Admission into trait impls and the container into
// `Worker<M, K, A>`. `UnifiedArch<M>` stays paired with this Worker template.
// The bodies do not change — only their signatures gain trait bounds. Review rule:
// M1's seams MUST already be the final ones (below), or M2 is not a mechanical
// lift. A compile-only trait skeleton should exist before M1 is declared done.
//
//   // Resource axis keyed by PartitionId; request processing keyed by Grouping —
//   // the two are orthogonal (design §9.0). PartitionId is NOT deployment's PoolId.
//   trait KvManager {
//       type Footprint;                 // FullAttnKv: u64
//       fn footprint(prompt: u32, decode: u32, prefix: u32) -> Self::Footprint;
//       fn fits(&self, partition: PartitionId, footprint: &Self::Footprint) -> bool;
//       fn reserve(&mut self, rid: RequestId, partition: PartitionId, footprint: Self::Footprint);  // KV-only (no ctx)
//       fn advance(&mut self, grouping: Grouping, steps: u32);                        // caller-owned request selection
//       fn commit_resident(&mut self, partition: PartitionId, rid: RequestId, initial_kv: u64, remaining: u32); // full args
//       fn release(&mut self, partition: PartitionId, rid: RequestId);
//       fn release_external(&mut self, rid: RequestId, current_kv: u64) -> Option<PartitionId>; // cancellation
//       fn sample_submit(&mut self, partition: PartitionId, now: Time);
//       // + drain_ready / status_active / concrete KV fact queries used by this
//       //   family template's Worker::build_arch_input
//   }
//
//   trait Admission {
//       type Stage: Copy + Into<u16>;   // PrefillDecode: UnifiedStage  (design §4.11)
//       fn accept(&mut self, msg: WorkerMsgCommon, ctx: &WorkerCtx);
//       fn form_batch<K: KvManager>(&mut self, kv: &mut K, ctx: &WorkerCtx, now: Time) -> bool;
//       fn on_iter_complete<K: KvManager>(&mut self, kv: &mut K, ctx: &WorkerCtx,
//                                         events: &mut Vec<WorkerEventCommon>, now: Time);
//       fn remove_pending(&mut self, rid: RequestId) -> bool;                        // cancellation
//       fn status_queued(&self) -> u32;
//   }
//   // NOTE: a variant needing KvHeld (PrefillHandoff) cannot express `where K:
//   // KvHeld` on a method-generic `form_batch<K>` alone — `K` must be a type
//   // param on the impl (or an associated `type Kv`). Resolve in §9 before M2.
//
//   struct Worker<M, K, A> {
//       ctx: WorkerCtx,
//       kv: K,
//       admission: A,
//       arch: UnifiedArch<M>,
//       iter: IterState,
//   }
//   // The concrete worker owns its state transitions and build_arch_input.
//   // status + release_request also stay on the container.
//
//   pub type BareboneWorker<M> = Worker<M, FullAttnKv, PrefillDecode>;
//   // A worker with a different protocol or ArchInput semantics gets another
//   // short worker instead of widening this one.
//
// Interface deltas this sketch surfaced against design §9 (fold back before M2):
//   - build_arch_input stays on the concrete Worker: it joins request/protocol
//     semantics with KV facts and the selected ArchInput type. Kv never owns Input.
//   - KV resource is keyed by `PartitionId` (fits / reserve / commit / release /
//     sample), request advancement by `Grouping` — two orthogonal axes.
//     `Reqs { partition, request_ids }` is caller-owned (the
//     AFD slot FSM owns membership); explicit partition supports empty slots and
//     prevents first-request inference. PartitionId is NOT deployment's PoolId (§9.0).
//   - commit_resident needs (initial_kv, remaining), not (req, p).
//   - fits takes only the resource partition and actual footprint. Fixed safety
//     margin, if ever needed, is Kv construction state; dynamic unavailable KV is
//     represented by explicit promised / held / suspended ledgers.
//   - cancellation (remove_pending / release_external) must exist on the traits.
//   - status is container-composed, not an Admission trait method.
