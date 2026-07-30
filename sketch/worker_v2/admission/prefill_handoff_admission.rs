//! `PrefillHandoffAdmission` — the PD prefill lifecycle (matrix A
//! "PrefillHandoff", iter family). Prefills a request, emits its first token,
//! then HANDS OFF to a decode
//! pool instead of decoding locally: the request's KV becomes HELD (via
//! `HandoffKv`, ref `KvHeld`)
//! until the decode side pulls it and acks `ReleaseKv`. No local decode set.
//!
//! Reuses `IterBatchWorker` + `FullAttnKv` + `UnifiedIterExecution` — the shell's generic
//! `A::Msg`/`A::Event` let this admission carry the PD-specific `PdPrefillMsg` /
//! `PdPrefillEvent` (with `ReleaseKv` / `PrefillDone`) without touching the shell.
//! Impl bound `K: IterWorkerKv + HandoffKv` (ref `IterBatchKv + KvHeld`; the
//! capability escalation, like
//! `ChunkedPrefillAdmission`'s `ChunkedPrefillKv`). External calls verified:
//! `RequestRecord`,
//! `mark_admitted`, `record_first_token`/`is_complete`, `PdStage`.

use std::collections::VecDeque;

use crate::common::{PdStage, RequestId, Time};
use crate::worker::types::{PdPrefillEvent, PdPrefillMsg};

use super::super::kv::{HandoffKv, IterWorkerKv, KvStore};
use super::super::shared::context::WorkerContext;
use super::IterAdmission;

pub struct PrefillHandoffAdmission {
    pending: VecDeque<RequestId>,
    /// This worker's comm group id (registered at construction). Stamped on each
    /// `PrefillDone` so L6 can build the matching decode-side `Handoff` pull.
    send_group_id: u16,
}

impl PrefillHandoffAdmission {
    pub fn new(send_group_id: u16) -> Self {
        Self {
            pending: VecDeque::new(),
            send_group_id,
        }
    }
}

impl<K: KvStore + IterWorkerKv + HandoffKv> IterAdmission<K> for PrefillHandoffAdmission {
    type Msg = PdPrefillMsg;
    type Event = PdPrefillEvent;

    fn accept_message(&mut self, kv_store: &mut K, msg: PdPrefillMsg, context: &WorkerContext) {
        match msg {
            PdPrefillMsg::Request(rid) => {
                self.pending.push_back(rid);
                let mut store = context.requests.borrow_mut();
                let arrival = store[rid].arrival_time;
                context.stamp_stage(&mut store[rid], arrival, PdStage::PendingPrefill as u16);
            }
            // Decode side finished the pull → drop this request's held KV.
            PdPrefillMsg::ReleaseKv { req } => kv_store.drop_held(req),
        }
    }

    fn form_batch(&mut self, kv_store: &mut K, context: &WorkerContext, now: Time) -> bool {
        // Admit ONE fresh prefill by PROMPT only (no local decode → no decode
        // reservation; held KV already gates via `fits`).
        if let Some(&rid) = self.pending.front() {
            let prompt = context.requests.borrow()[rid].prompt_len;
            let footprint = kv_store.footprint(rid, prompt, 0);
            if kv_store.fits(0, &footprint) {
                kv_store.reserve(rid, 0, footprint);
                self.pending.pop_front();
                let mut store = context.requests.borrow_mut();
                store.mark_admitted(rid);
                let record = &mut store[rid];
                record.active_chunk_len = prompt;
                record.prefix_kv = 0;
                context.stamp_stage(record, now, PdStage::Prefill as u16);
            }
        }
        kv_store.drain_ready();
        kv_store.has_prefill_admit(0)
    }

    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        events: &mut Vec<PdPrefillEvent>,
        now: Time,
    ) {
        // Prefills resolved this iter: first token, then either finish (single-token)
        // or hand off to decode (hold KV + PrefillDone). No decode advance.
        let mut done: Vec<RequestId> = Vec::new();
        let mut to_hold: Vec<(RequestId, u64)> = Vec::new();
        {
            let log_tokens = context.log_tokens();
            let prefills = kv_store.prefill_admits(0);
            let mut store = context.requests.borrow_mut();
            for rid in prefills {
                let record = &mut store[rid];
                record.prefill_processed = record.prompt_len;
                record.record_first_token(now, log_tokens);
                let kv_tokens = u64::from(record.prompt_len + record.prefix_kv);
                let complete = record.is_complete();
                context.stamp_stage(
                    record,
                    now,
                    if complete {
                        PdStage::Done
                    } else {
                        PdStage::PrefillDoneAwaitPull
                    } as u16,
                );
                if complete {
                    done.push(rid);
                } else {
                    to_hold.push((rid, kv_tokens));
                }
            }
        }
        kv_store.clear_prefill_admits(0);

        // Single-token requests finish here; release their reservation (no KV was
        // ever made resident — a prefill worker keeps its local KvPool empty).
        for rid in done {
            kv_store.release(rid, 0);
            events.push(PdPrefillEvent::RequestComplete {
                worker: context.id,
                req: rid,
            });
        }
        // Multi-token requests hand off: hold the KV and signal L6 to route a decode.
        for (rid, kv_tokens) in to_hold {
            kv_store.hold(0, rid, kv_tokens);
            events.push(PdPrefillEvent::PrefillDone {
                worker: context.id,
                req: rid,
                send_gid: self.send_group_id,
                kv_tokens,
            });
        }

        kv_store.sample_submit(0, now);
    }

    #[inline]
    fn queued_requests(&self) -> u32 {
        self.pending.len() as u32
    }

    fn cancel_pending(&mut self, rid: RequestId) -> bool {
        if let Some(position) = self.pending.iter().position(|&pending| pending == rid) {
            self.pending.remove(position);
            true
        } else {
            false
        }
    }
}
