//! `PrefillDecode` — one implementation of the Admission policy axis.
//!
//! Owns the pending queue AND its policy fields (`admission`, `max_batch_tokens`
//! — distributed out of `WorkerConfig` at construction, not read off `ctx`).
//! Knows nothing about WHAT kind of KV exists — it speaks tokens (prompt, decode,
//! budget) and asks `kv.fits`. Its `Stage` vocab is `UnifiedStage`
//! (design §4.11).
//!
//! Ownership note (reconciled with design §4.11): the unified worker has no
//! separate FFN Terminal, so the iter-end hook `on_iter_complete` — which records
//! tokens and the Decode/Done stage transitions, finalizes prefills, and emits —
//! is owned by THIS axis. The rule is "whoever owns iter-end records" (finding
//! #8): Admission where an admission axis exists, the concrete protocol owner
//! for an admission-less worker.

use std::collections::VecDeque;

use crate::common::{RequestId, Time, UnifiedStage};
use crate::worker::admission_helpers::{prefill_fits_budget, KvAdmission};
use crate::worker::types::{WorkerEventCommon, WorkerMsgCommon};

use super::super::ctx::WorkerCtx;
use super::super::kv::{FullAttnKv, Grouping, PartitionId};

pub(crate) struct PrefillDecode {
    /// Was `BareboneWorker.runtime.pending_prefills`.
    pending_prefills: VecDeque<RequestId>,
    /// Distributed out of `WorkerConfig.admission` (the strict KV gate).
    admission: KvAdmission,
    /// Distributed out of `WorkerConfig.max_batch_tokens` (soft per-iter budget).
    max_batch_tokens: Option<u32>,
}

impl PrefillDecode {
    pub(crate) fn new(admission: KvAdmission, max_batch_tokens: Option<u32>) -> Self {
        Self {
            pending_prefills: VecDeque::new(),
            admission,
            max_batch_tokens,
        }
    }

    /// Was `enqueue`. Push onto the pending queue and stamp `Pending` at the
    /// request's arrival time (enqueue carries no clock).
    pub(crate) fn accept(&mut self, msg: WorkerMsgCommon, ctx: &WorkerCtx) {
        let WorkerMsgCommon::Request(rid) = msg;
        self.pending_prefills.push_back(rid);
        let mut store = ctx.requests.borrow_mut();
        let arrival = store[rid].arrival_time;
        ctx.stamp_stage(&mut store[rid], arrival, UnifiedStage::Pending as u16);
    }

    /// Was `form_batch` (Phase A admit + Phase B drain). Returns whether this iter
    /// has any work. Does NOT bump the iter counter — the concrete Worker owns
    /// iteration order. [risk #1]
    pub(crate) fn form_batch(&mut self, kv: &mut FullAttnKv, ctx: &WorkerCtx, now: Time) -> bool {
        let had_decode = kv.has_live_decode(0);
        // Phase A: admit fresh prefill(s) from the FIFO head. The KV gate always
        // applies (`kv.fits`); with a token budget, room is reserved for live
        // decodes first, then filled with whole prefills (see prefill_fits_budget).
        match self.max_batch_tokens {
            None => {
                if let Some(&rid) = self.pending_prefills.front() {
                    let (prompt, decode) = read_prompt_decode(ctx, rid);
                    if kv.fits(0, FullAttnKv::footprint(prompt, decode, 0)) {
                        self.pending_prefills.pop_front();
                        self.admit(kv, ctx, rid, 0, prompt, decode, now);
                    }
                }
            }
            Some(budget) => {
                let decode_tokens = kv.live_decode_count(0);
                let mut admitted = 0u32;
                while let Some(&rid) = self.pending_prefills.front() {
                    let (prompt, decode) = read_prompt_decode(ctx, rid);
                    if !prefill_fits_budget(budget, decode_tokens, admitted, prompt) {
                        break; // token budget exhausted this iter
                    }
                    if !kv.fits(0, FullAttnKv::footprint(prompt, decode, 0)) {
                        break; // KV gate blocks the FIFO head; retry next iter
                    }
                    self.pending_prefills.pop_front();
                    self.admit(kv, ctx, rid, 0, prompt, decode, now);
                    admitted += prompt;
                }
            }
        }

        // Phase B: drain ready promises into this iter's prefill_admits.
        kv.drain_ready();
        had_decode || kv.has_prefill_admit(0)
    }

    /// Was `promise`'s store half: reserve KV, then mark admitted, seed the arch
    /// input fields, and stamp the `Prefill` stage. `prefix = 0` for barebone;
    /// F1 will set a matched-prefix length. The reserve footprint is `prompt +
    /// prefix + decode` (matches today's `promise`); the resident kv at commit is
    /// only `prompt + prefix` (see FullAttnKv::commit_resident).
    fn admit(
        &mut self,
        kv: &mut FullAttnKv,
        ctx: &WorkerCtx,
        rid: RequestId,
        partition: PartitionId,
        prompt: u32,
        decode: u32,
        now: Time,
    ) {
        kv.reserve(rid, partition, FullAttnKv::footprint(prompt, decode, 0));
        let mut store = ctx.requests.borrow_mut();
        store.mark_admitted(rid);
        let record = &mut store[rid];
        record.active_chunk_len = prompt;
        record.prefix_kv = 0;
        ctx.stamp_stage(record, now, UnifiedStage::Prefill as u16);
    }

    /// Was `complete_iter` steps (a)–(d) + KV sample submit. Records tokens,
    /// advances decodes, finalizes resolved prefills, emits + releases completed.
    /// `record_first_token` now owns the completion invariant (`completed =
    /// is_complete()`), so the redundant `completed =` assignment is gone — but the
    /// `if record.is_complete()` BRANCH stays: it decides Done+event vs
    /// Decode+finalize (design §4.11 / request.rs:147). [review: do not remove the branch]
    ///
    /// Ordering invariants (bit-identity): completed decodes are appended BEFORE
    /// completed prefills; `advance` runs after (a) and before (c); `sample_submit`
    /// fires last, exactly once.
    pub(crate) fn on_iter_complete(
        &mut self,
        kv: &mut FullAttnKv,
        ctx: &WorkerCtx,
        events: &mut Vec<WorkerEventCommon>,
        now: Time,
    ) {
        let mut completed: Vec<RequestId> = Vec::new();

        // (a) live decodes each produced one token this iter.
        {
            let log_tokens = ctx.log_tokens();
            let mut store = ctx.requests.borrow_mut();
            for (rid, _) in kv.batch(0).iter_decoding() {
                let record = &mut store[rid];
                record.record_token(now, log_tokens);
                if record.is_complete() {
                    ctx.stamp_stage(record, now, UnifiedStage::Done as u16);
                    completed.push(rid);
                }
            }
        }

        // (b) bump KV for every live decode of partition 0.
        kv.advance(Grouping::Partition(0), 1);

        // (c) resolved prefills → first token; complete now or enter decode.
        let mut to_finalize: Vec<(RequestId, u64, u32)> = Vec::new();
        {
            let log_tokens = ctx.log_tokens();
            let mut store = ctx.requests.borrow_mut();
            for &rid in &kv.batch(0).prefill_admits {
                let record = &mut store[rid];
                record.prefill_processed = record.prompt_len;
                record.record_first_token(now, log_tokens);
                if record.is_complete() {
                    ctx.stamp_stage(record, now, UnifiedStage::Done as u16);
                    completed.push(rid);
                } else {
                    let kv_len = (record.prompt_len + record.prefix_kv) as u64;
                    let remaining = record.decode_len.saturating_sub(record.tokens_emitted);
                    ctx.stamp_stage(record, now, UnifiedStage::Decode as u16);
                    to_finalize.push((rid, kv_len, remaining));
                }
            }
        }
        for (rid, kv_len, remaining) in to_finalize {
            kv.commit_resident(0, rid, kv_len, remaining);
        }
        kv.clear_prefill_admits(0);

        // (d) emit + release completed requests.
        for rid in completed {
            kv.release(0, rid);
            events.push(WorkerEventCommon::RequestComplete {
                worker: ctx.id,
                req: rid,
            });
        }

        // Iter-end KV occupancy sample (whoever owns iter-end records). [§10.1 #8]
        kv.sample_submit(0, now);
    }

    #[inline]
    pub(crate) fn status_queued(&self) -> u32 {
        self.pending_prefills.len() as u32
    }

    /// Was `release_request`'s pending branch. Returns true if it was queued (and
    /// removed) here; the caller then need not touch the KV axis.
    pub(crate) fn remove_pending(&mut self, rid: RequestId) -> bool {
        if let Some(position) = self
            .pending_prefills
            .iter()
            .position(|&pending| pending == rid)
        {
            self.pending_prefills.remove(position);
            true
        } else {
            false
        }
    }
}

/// Read `(prompt_len, decode_len)` for the FIFO head.
#[inline]
fn read_prompt_decode(ctx: &WorkerCtx, rid: RequestId) -> (u32, u32) {
    let store = ctx.requests.borrow();
    let record = &store[rid];
    (record.prompt_len, record.decode_len)
}
