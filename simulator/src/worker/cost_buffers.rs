//! `CostBuffers` — the eval / cost-log scratch + writer bundle that every
//! iter-wise worker carries. Each worker would otherwise hold five fields
//! (`cost_logger` + four reusable buffers) and a ~50-line block at the end of
//! every `start_iter` that calls `model.eval_iter*`, builds the cost-log row,
//! and writes it. Folding that into one struct keeps the four worker files
//! focused on FSM / role logic and gives `cost_log` a single point of edit when
//! its schema or call shape changes.
//!
//! The shape: workers hand over their `&Arc<M>` + the freshly-built (or
//! reused) `UnifiedArchInput` + this iter's metadata; `run_iter` runs the eval
//! (with or without input capture, depending on whether a logger is attached),
//! optionally writes one parquet row, and returns the iter's wall time. The
//! buffers stay owned here so `eval_iter` can refill them in place across iters.

use std::path::PathBuf;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{Time, WorkerId};
use crate::log::{CostLogEntry, CostLogger, GroupInputLog};
use crate::timing::{LeafMetrics, SlotInput};

/// Per-worker eval / cost-log scratch state. Drop one of these into a worker
/// struct in place of `cost_logger` + four `Vec` fields, and call `run_iter`
/// at the end of `start_iter` in place of the inline eval + record block.
///
/// The model generic is `?Sized` so the offline `iter-timing-predict` path can
/// drive a `&dyn IterwiseUnifiedModel`; workers pass a concrete model and stay
/// monomorphized, so no dispatch is added to the cost hot path (L4 §4.1).
pub struct CostBuffers {
    /// Reused per-slot eval output buffer (filled by `model.eval_iter*`).
    pub slots: Vec<LeafMetrics>,
    /// Reused scratch the model may use internally during eval.
    pub scratch: Vec<LeafMetrics>,
    /// Reused per-leaf input capture buffer. Filled only on the
    /// `eval_iter_with_inputs` path (active when a logger is attached);
    /// otherwise stays empty.
    pub slot_inputs: Vec<SlotInput>,
    /// Reused per-group input log buffer — repopulated each iter from
    /// `arch_input.groups` before `logger.record`.
    groups: Vec<GroupInputLog>,
    /// `Some` when cost-log writing is active for this worker. A failure to
    /// open the writer at construction degrades to `None` (with a warn) so
    /// the sim still runs.
    logger: Option<CostLogger>,
}

impl CostBuffers {
    /// Build the buffer set, opening the cost-log writer if a `log_dir` was
    /// configured. An open failure logs a warning and disables logging — never
    /// aborts the sim. `pool_tag` disambiguates writer filenames across pools
    /// (PD's `prefill_<id>.parquet` vs `decode_<id>.parquet`).
    pub fn new<M: IterwiseUnifiedModel + ?Sized>(
        log_dir: Option<PathBuf>,
        pool_tag: &'static str,
        worker_id: WorkerId,
        model: &M,
    ) -> Self {
        let logger = match log_dir {
            Some(dir) => match CostLogger::open(&dir, pool_tag, worker_id, &model.cost_log_manifest()) {
                Ok(logger) => Some(logger),
                Err(e) => {
                    tracing::warn!("cost_log disabled: failed to open writer: {e}");
                    None
                }
            },
            None => None,
        };
        Self {
            slots: Vec::new(),
            scratch: Vec::new(),
            slot_inputs: Vec::new(),
            groups: Vec::new(),
            logger,
        }
    }

    /// Run one iter's eval and (if logging is enabled) record the cost-log row.
    /// Returns the iter's wall time (`agg.m.time_ms`) — the worker adds this to
    /// `now` to get `compute_end`. Picks `eval_iter_with_inputs` vs `eval_iter`
    /// automatically based on whether a logger is attached; the per-leaf input
    /// capture is part of the cost_log row contract, so we only pay for it
    /// when we're actually going to write the row.
    ///
    /// `worker_id` / `iter_id` / `now` are stamped into the row; the model is
    /// borrowed only for the eval (no Arc plumbing — workers Deref their Arc).
    pub fn run_iter<M: IterwiseUnifiedModel + ?Sized>(
        &mut self,
        model: &M,
        arch_input: &UnifiedArchInput,
        worker_id: WorkerId,
        iter_id: u64,
        now: Time,
    ) -> Time {
        let agg = if self.logger.is_some() {
            model.eval_iter_with_inputs(
                arch_input,
                &mut self.slots,
                &mut self.scratch,
                &mut self.slot_inputs,
            )
        } else {
            model.eval_iter(arch_input, &mut self.slots, &mut self.scratch)
        };
        let cost_time = Time::from_ms(agg.m.time_ms as f64);
        if let Some(logger) = self.logger.as_mut() {
            // Per-iteration input_section: log each group's context (prefill
            // kept full as `(prefix, append)` pairs; decode aggregated to
            // count + total KV — the per-decode KV list is dropped). Refilled
            // into `groups` in place; the per-slot time/coverage/input
            // breakdowns are appended into the logger's flat chunk buffers
            // by `record`, so the entry owns no `Vec`.
            self.groups.clear();
            for g in &arch_input.groups {
                self.groups.push(GroupInputLog {
                    batch_tokens: g.batch_tokens,
                    prefill_tokens: g.prefill_tokens,
                    decode_request_count: g.decode_tokens,
                    decode_kv_total: g.total_kv_len,
                    prefill_chunk_pairs: g.prefill_chunk_pairs.clone(),
                });
            }
            let entry = CostLogEntry {
                worker_id: worker_id.0,
                iter_id,
                // One batch per iteration today; AFD/TBO will emit several
                // batches sharing this iter_id with distinct batch_id.
                batch_id: 0,
                wall_start_ms: now.as_ms(),
                total_time_ms: agg.m.time_ms as f64,
                energy_j: agg.m.energy_j as f64,
                group_len: 0,
                slot_len: 0,
                slot_input_len: 0,
            };
            if let Err(e) = logger.record(
                entry,
                &self.slots,
                &mut self.groups,
                &mut self.slot_inputs,
            ) {
                tracing::warn!("cost_log record failed: {e}");
            }
        }
        cost_time
    }
}
