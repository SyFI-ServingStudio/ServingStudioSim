//! `BufferedFfnWorker<E>` — the AFD-ffn double-buffer shell (interfaces doc §5, family =
//! AFD-ffn). Generic over ONLY the execution: the ffn family has NO KV and NO admission
//! (L6 hands it pre-composed `FfnTask`s), so the `<K, A, E>` triple degenerates to
//! `<E>`. This is the sketch's strongest expressiveness check: the axes are
//! optional per family, not a fixed shape every worker must fill.
//!
//! Reproduces `DisaggFfnWorker` (disagg_ffn.rs): a double-buffered compute/transfer
//! pipeline — at most one task pulling (`next`) and one computing (`current`), so a
//! section's attn→ffn input transfer overlaps the prior section's compute. Extra
//! tasks queue in `incoming`. The shell owns: the double-buffer FSM, the comm seam
//! (`submit_gather` input pull), the per-slot `iteration_sequence_by_slot`, the token derivation
//! (store scan for Bootstrap; threaded from the attn side otherwise), and the two
//! events (`SectionReady` / `IterComplete`) including the Terminal completion
//! bookkeeping (token recording over `RequestRecord`'s methods, inlined like the
//! iter family — no shared wrapper — stamped on the request's sticky ATTN owner,
//! NOT this ffn worker id, so it bypasses `WorkerContext::stamp_stage`).

use std::collections::VecDeque;

use crate::common::{AfdStage, RequestId, Time, WorkerId};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{FfnTask, FfnTaskKind, FfnWorkerEvent, FfnWorkerMsg, WorkerStatus};

use super::super::super::execution::FfnTaskExecution;
use super::super::super::shared::context::WorkerContext;

fn earliest_wakeup(current: Option<Time>, candidate: Time) -> Option<Time> {
    Some(current.map_or(candidate, |current| current.min(candidate)))
}

/// A task whose input is transferring (attn→ffn); promoted to `current` at `pull_end`.
struct FfnPullInFlight {
    task: FfnTask,
    pull_end: Time,
}

/// A task whose section is computing; finished (event emitted) at `compute_end`.
struct FfnComputeInFlight {
    task: FfnTask,
    compute_end: Time,
}

pub struct BufferedFfnWorker<E: FfnTaskExecution> {
    context: WorkerContext,
    execution: E,
    /// Scratch arch input, rebuilt per task from its token total.
    input: E::Input,
    cluster: SharedGpuCluster,
    /// This worker's comm group: recv endpoint for the attn→ffn pull AND the send
    /// endpoint the next attn layer pulls QKV from.
    communication_group_id: u16,
    incoming: VecDeque<FfnTask>,
    pulling_task: Option<FfnPullInFlight>,
    computing_task: Option<FfnComputeInFlight>,
    /// Per-slot forward-pass counter (the cost-row `iter_id`), bumped on Bootstrap.
    iteration_sequence_by_slot: Vec<u64>,
}

impl<E: FfnTaskExecution> BufferedFfnWorker<E> {
    pub(super) fn from_components(
        context: WorkerContext,
        execution: E,
        cluster: SharedGpuCluster,
        communication_group_id: u16,
    ) -> Self {
        Self {
            context,
            execution,
            input: E::Input::default(),
            cluster,
            communication_group_id,
            incoming: VecDeque::new(),
            pulling_task: None,
            computing_task: None,
            iteration_sequence_by_slot: Vec::new(),
        }
    }

    fn drive_task_pipeline(&mut self, now: Time, events: &mut Vec<FfnWorkerEvent>) -> Option<Time> {
        loop {
            let mut progressed = false;
            // 1. finish the computing task if its window closed.
            if let Some(rt) = &self.computing_task {
                if now >= rt.compute_end {
                    let rt = self.computing_task.take().unwrap();
                    self.complete_task(rt.task, now, events);
                    progressed = true;
                }
            }
            // 2. promote the pulled task to computing if the compute slot is free.
            if self.computing_task.is_none() {
                if let Some(pt) = &self.pulling_task {
                    if now >= pt.pull_end {
                        let pt = self.pulling_task.take().unwrap();
                        self.computing_task = Some(self.start_task_compute(pt.task, now));
                        progressed = true;
                    }
                }
            }
            // 3. start pulling the next queued task if the pull slot is free.
            if self.pulling_task.is_none() {
                if let Some(task) = self.incoming.pop_front() {
                    self.pulling_task = Some(self.start_task_pull(task, now));
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
        self.next_wakeup(now)
    }

    fn next_wakeup(&self, now: Time) -> Option<Time> {
        let mut next_wakeup: Option<Time> = None;
        if let Some(computing_task) = &self.computing_task {
            if computing_task.compute_end > now {
                next_wakeup = earliest_wakeup(next_wakeup, computing_task.compute_end);
            }
        }
        if let Some(pulling_task) = &self.pulling_task {
            if pulling_task.pull_end > now {
                next_wakeup = earliest_wakeup(next_wakeup, pulling_task.pull_end);
            }
        }
        next_wakeup
    }

    fn start_task_pull(&mut self, task: FfnTask, now: Time) -> FfnPullInFlight {
        // Bootstrap input is local (no pull); otherwise one CONCURRENT gather over the
        // per-attn-worker source chunks (latency once, senders parallel).
        let sources: Vec<(u16, u64)> = task
            .pull_sources
            .iter()
            .filter(|source| source.bytes > 0)
            .map(|source| (source.send_gid, source.bytes))
            .collect();
        let pull_end = if sources.is_empty() {
            now
        } else {
            self.cluster.borrow_mut().submit_gather(
                now,
                &sources,
                self.communication_group_id,
                "afd_ffn_pull",
                "",
            )
        };
        FfnPullInFlight { task, pull_end }
    }

    fn start_task_compute(&mut self, mut task: FfnTask, now: Time) -> FfnComputeInFlight {
        let slot = task.slot as usize;
        if matches!(task.kind, FfnTaskKind::Bootstrap) {
            // A Bootstrap opens a new forward pass for this slot → bump the row id, and
            // derive the token total once from the store (the pool could not pre-sum it
            // before any attn layer-output existed; Bridge/Terminal arrive pre-summed).
            if slot >= self.iteration_sequence_by_slot.len() {
                self.iteration_sequence_by_slot.resize(slot + 1, 0);
            }
            self.iteration_sequence_by_slot[slot] += 1;
            task.tokens = self.task_query_tokens(&task.reqs);
        }
        let iter_id = self
            .iteration_sequence_by_slot
            .get(slot)
            .copied()
            .unwrap_or(0);
        // Empty batches are control-plane no-ops: emit the normal event but keep
        // zero-token shapes off the model/profile lookup path.
        let dt = if task.reqs.is_empty() {
            Time::ZERO
        } else {
            self.execution
                .build_task_input(task.tokens, &mut self.input);
            self.execution
                .evaluate_ffn_task(task.kind, task.slot, iter_id, &self.input, now)
        };
        FfnComputeInFlight {
            compute_end: now + dt,
            task,
        }
    }

    /// Total query tokens the task contributes (prefill = prompt_len, decode = 1),
    /// from the store's `is_prefill()` discriminator (the ffn owns no KV, but reads
    /// the shared store to size the batch — mirrors the attn side's token count).
    fn task_query_tokens(&self, request_ids: &[RequestId]) -> u64 {
        let store = self.context.requests.borrow();
        request_ids
            .iter()
            .map(|&rid| {
                let record = &store[rid];
                if record.is_prefill() {
                    u64::from(record.prompt_len)
                } else {
                    1
                }
            })
            .sum()
    }

    fn complete_task(&mut self, task: FfnTask, now: Time, events: &mut Vec<FfnWorkerEvent>) {
        match task.kind {
            FfnTaskKind::Terminal => {
                let log_tokens = self.context.log_output_token_times;
                let log_stage = self.context.log_stage_transitions;
                let mut completed = Vec::new();
                {
                    let mut store = self.context.requests.borrow_mut();
                    for &rid in &task.reqs {
                        let record = &mut store[rid];
                        // AFD placement is sticky on the ATTN side: the Terminal owns
                        // token emission + the prefill→decode flip, but stamps the
                        // request's existing owner, not this ffn worker. So it reads
                        // `current_stage` and calls `record_stage` directly rather than
                        // going through `WorkerContext::stamp_stage` (which uses self.id).
                        let owner = (record.current_stage.pool, record.current_stage.worker);
                        // Token bookkeeping inlined over the shared `RequestRecord`
                        // methods (the real cross-family layer); the iter family's
                        // complete_iteration does the same. First output token flips
                        // prefill→decode; completion stamps Done.
                        let first_token = record.tokens_emitted == 0;
                        if first_token {
                            record.prefill_processed = record.prompt_len;
                            record.record_first_token(now, log_tokens);
                            record.record_stage(
                                now,
                                AfdStage::Decode as u16,
                                owner.0,
                                owner.1,
                                log_stage,
                            );
                        } else {
                            record.record_token(now, log_tokens);
                        }
                        if record.is_complete() {
                            record.record_stage(
                                now,
                                AfdStage::Done as u16,
                                owner.0,
                                owner.1,
                                log_stage,
                            );
                            completed.push(rid);
                        }
                    }
                }
                events.push(FfnWorkerEvent::IterComplete {
                    worker: self.context.id,
                    slot: task.slot,
                    reqs: task.reqs,
                    completed,
                });
            }
            FfnTaskKind::Bootstrap | FfnTaskKind::Bridge { .. } => {
                // Produces this section's QKV for the downstream attn layer; the attn
                // side pulls `ffn_to_attn_bytes_per_token × tokens`. L6 maps `kind` to
                // the downstream layer.
                let out_bytes = self.execution.ffn_to_attn_bytes_per_token() * task.tokens;
                events.push(FfnWorkerEvent::SectionReady {
                    worker: self.context.id,
                    slot: task.slot,
                    kind: task.kind,
                    reqs: task.reqs,
                    out_send_gid: self.communication_group_id,
                    out_bytes,
                });
            }
        }
    }

    fn on_msg_task(&mut self, task: FfnTask) {
        self.incoming.push_back(task);
    }
}

impl<E: FfnTaskExecution> IterWorker for BufferedFfnWorker<E> {
    type Msg = FfnWorkerMsg;
    type Event = FfnWorkerEvent;

    fn id(&self) -> WorkerId {
        self.context.id
    }

    fn enqueue(&mut self, msg: FfnWorkerMsg) {
        match msg {
            FfnWorkerMsg::Task(task) => self.on_msg_task(task),
        }
    }

    fn tick(&mut self, now: Time, events: &mut Vec<FfnWorkerEvent>) -> Option<Time> {
        self.drive_task_pipeline(now, events)
    }

    fn status(&self) -> WorkerStatus {
        let queued = self.incoming.len() + usize::from(self.pulling_task.is_some());
        WorkerStatus {
            queued_requests: queued as u32,
            active_requests: u32::from(self.computing_task.is_some()),
        }
    }
}

// `AfdFfnWorker` is a blanket impl over `IterWorker<Msg = FfnWorkerMsg, Event =
// FfnWorkerEvent>`, so `BufferedFfnWorker<E>` satisfies it automatically — nothing to write.
