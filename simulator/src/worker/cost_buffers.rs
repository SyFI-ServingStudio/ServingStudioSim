//! `CostBuffers` — the eval / cost-log scratch + writer bundle that every
//! cost-logging worker carries. A worker would otherwise hold the writer plus
//! a few reusable buffers and repeat, at every eval site, the ~25-line block
//! that refills the input log, builds the cost-log row, and writes it. Folding
//! that into one struct keeps the worker files focused on FSM / role logic and
//! gives `cost_log` a single point of edit when its schema or call shape changes.
//!
//! Two entry points over the same buffers + writer:
//!   * [`run_section`](CostBuffers::run_section) — the general one. Evaluate one
//!     building block (handed the reused buffers + an optional capture sink via a
//!     closure, so it threads the heterogeneous attn/ffn signatures), then write
//!     one row tagged `section` / `layer` / `batch_id`. The AFD layer-wise workers
//!     call this once per section per layer.
//!   * [`run_iter`](CostBuffers::run_iter) — a convenience for the iter-wise
//!     workers, whose whole iteration is one fused `eval_iter`. It is `run_section`
//!     specialized to `section = "iter"`, `layer = -1`, `batch_id = 0`.
//!
//! Per-leaf `slot_input` capture (via the model's `*_with_inputs` methods) is part
//! of the row contract, so it runs only when a logger is attached — we don't pay
//! the per-leaf clone when no row will be written.

use std::path::PathBuf;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{Time, WorkerId};
use crate::log::{CostLogEntry, CostLogger, GroupInputLog};
use crate::timing::{CostManifestDoc, LeafMetrics, SlotInput};

/// Per-worker eval / cost-log scratch state + writer. Drop one of these into a
/// worker struct in place of a `cost_logger` + the reusable eval buffers, and
/// call [`run_section`](Self::run_section) (layer-wise) or [`run_iter`](Self::run_iter)
/// (iter-wise) where the worker computes a block's cost.
///
/// Opened with a [`CostManifestDoc`] of named sections, so each row's `section`
/// field selects which sub-manifest names its slots. The iter-wise path uses a
/// degenerate single-`iter` manifest (see [`new_iter`](Self::new_iter)); the AFD
/// layer-wise path uses the model's multi-section manifest. `CostLogger` itself
/// stays model-agnostic.
pub struct CostBuffers {
    /// Reused per-slot eval output buffer (filled by the model's eval method).
    slots: Vec<LeafMetrics>,
    /// Reused CostTree-aggregation scratch (contents not meaningful on return).
    scratch: Vec<LeafMetrics>,
    /// Reused per-leaf input capture buffer. Filled only on the `*_with_inputs`
    /// path (active when a logger is attached); otherwise stays untouched.
    slot_inputs: Vec<SlotInput>,
    /// Reused per-group input log scratch — refilled in place per row before
    /// `logger.record`.
    groups: Vec<GroupInputLog>,
    worker_id: WorkerId,
    /// `Some` when cost-log writing is active for this worker. A failure to open
    /// the writer at construction degrades to `None` (with a warn) so the sim
    /// still runs.
    logger: Option<CostLogger>,
}

impl CostBuffers {
    /// Build the buffer set, opening the cost-log writer if a `log_dir` was
    /// configured. An open failure logs a warning and disables logging — never
    /// aborts the sim. `pool_tag` names the writer stream (e.g. PD's `prefill` /
    /// `decode`, AFD's `attn` / `ffn`, the offline `predict`); the per-row
    /// `section` field — not `pool_tag` — distinguishes building blocks.
    pub fn new(
        log_dir: Option<PathBuf>,
        pool_tag: &'static str,
        worker_id: WorkerId,
        manifest: &CostManifestDoc,
    ) -> Self {
        let logger = match log_dir {
            Some(dir) => match CostLogger::open(&dir, pool_tag, worker_id, manifest) {
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
            worker_id,
            logger,
        }
    }

    /// Iter-wise convenience constructor: the model exposes one fused CostTree, so
    /// its manifest is a single `iter` section. Equivalent to [`new`](Self::new)
    /// with `CostManifestDoc::single("iter", model.cost_log_manifest())`.
    pub fn new_iter<M: IterwiseUnifiedModel + ?Sized>(
        log_dir: Option<PathBuf>,
        pool_tag: &'static str,
        worker_id: WorkerId,
        model: &M,
    ) -> Self {
        Self::new(
            log_dir,
            pool_tag,
            worker_id,
            &CostManifestDoc::single("iter", model.cost_log_manifest()),
        )
    }

    /// Run one building block's eval through `eval` and (if logging) write its
    /// `cost_log` row, returning the section aggregate (the caller advances `now`
    /// by `agg.m.time_ms`). `eval` is handed the reused `slots` + `scratch` buffers
    /// and an optional capture sink: `Some(inputs)` when a logger is attached (call
    /// the model's `*_with_inputs` method), `None` otherwise (call the plain `*_cost`
    /// method). `section` names the block (`attn` / `prologue` / `pre_attn` /
    /// `post_attn` / `post_attn_last` / `epilogue` / `iter`), `layer` the layer index
    /// (`-1` for iteration-level prologue/epilogue/iter), `batch_id` the AFD slot,
    /// and `groups` the per-shard input context (logged as the row's `input_section`)
    /// — the attn/iter sides pass `[ArchGroupInput]`, the ffn side bare `[u32]` token
    /// counts, both via [`GroupLogSource`].
    pub fn run_section<G, F>(
        &mut self,
        section: &'static str,
        layer: i16,
        iter_id: u64,
        batch_id: u64,
        groups: &G,
        now: Time,
        eval: F,
    ) -> LeafMetrics
    where
        G: GroupLogSource,
        F: FnOnce(
            &mut Vec<LeafMetrics>,
            &mut Vec<LeafMetrics>,
            Option<&mut Vec<SlotInput>>,
        ) -> LeafMetrics,
    {
        // Capture per-leaf inputs only on the path that will actually write the row
        // — the `*_with_inputs` clone per leaf is part of the cost_log contract, so
        // we don't pay it when no logger is attached.
        let capture = self.logger.is_some();
        let agg = eval(
            &mut self.slots,
            &mut self.scratch,
            capture.then_some(&mut self.slot_inputs),
        );
        if let Some(logger) = self.logger.as_mut() {
            groups.fill_group_log(&mut self.groups);
            let entry = CostLogEntry {
                worker_id: self.worker_id.0,
                iter_id,
                batch_id,
                wall_start_ms: now.as_ms(),
                total_time_ms: agg.m.time_ms as f64,
                energy_j: agg.m.energy_j as f64,
                section,
                layer,
                // Filled by `logger.record` from the slice lengths.
                group_len: 0,
                slot_len: 0,
                slot_input_len: 0,
            };
            if let Err(e) = logger.record(entry, &self.slots, &mut self.groups, &mut self.slot_inputs)
            {
                tracing::warn!("cost_log record failed: {e}");
            }
        }
        agg
    }

    /// Iter-wise convenience over [`run_section`](Self::run_section): the whole
    /// iteration is one fused `eval_iter`, logged as a single `iter` section (no
    /// per-layer split, one batch → `batch_id = 0`). Returns the iter's wall time
    /// (the worker adds it to `now` for `compute_end`). Picks `eval_iter_with_inputs`
    /// vs `eval_iter` automatically based on whether a logger is attached.
    pub fn run_iter<M: IterwiseUnifiedModel + ?Sized>(
        &mut self,
        model: &M,
        arch_input: &UnifiedArchInput,
        iter_id: u64,
        now: Time,
    ) -> Time {
        // One batch per iteration today; AFD/TBO emit several batches sharing an
        // iter_id with distinct batch_id via `run_section` instead.
        let agg = self.run_section(
            "iter",
            -1,
            iter_id,
            0,
            &arch_input.groups,
            now,
            |slots, scratch, inputs| match inputs {
                Some(i) => model.eval_iter_with_inputs(arch_input, slots, scratch, i),
                None => model.eval_iter(arch_input, slots, scratch),
            },
        );
        Time::from_ms(agg.m.time_ms as f64)
    }
}

/// The per-row `input_section` source for [`CostBuffers::run_section`]. Different
/// arch sides describe their batch differently — the attn/iter sides carry the full
/// attention-shaped [`ArchGroupInput`], the ffn side only per-shard token counts —
/// so each lowers itself to the writer's `[GroupInputLog]` here, keeping the cost
/// path's input types honest about what they actually depend on.
pub trait GroupLogSource {
    /// Refill `dst` (cleared) with one [`GroupInputLog`] per shard.
    fn fill_group_log(&self, dst: &mut Vec<GroupInputLog>);
}

impl GroupLogSource for Vec<ArchGroupInput> {
    /// Full attention-shaped context: prefill kept as `(prefix, append)` chunk pairs;
    /// decode aggregated to a request count + total KV (the per-decode KV list is dropped).
    fn fill_group_log(&self, dst: &mut Vec<GroupInputLog>) {
        dst.clear();
        for g in self {
            dst.push(GroupInputLog {
                batch_tokens: g.batch_tokens,
                prefill_tokens: g.prefill_tokens,
                decode_request_count: g.decode_tokens,
                decode_kv_total: g.total_kv_len,
                prefill_chunk_pairs: g.prefill_chunk_pairs.clone(),
            });
        }
    }
}

impl GroupLogSource for Vec<u32> {
    /// Ffn side: only the per-shard token count is meaningful — the attention-shaped
    /// fields stay zero/empty (the ffn cost never read them).
    fn fill_group_log(&self, dst: &mut Vec<GroupInputLog>) {
        dst.clear();
        for &batch_tokens in self {
            dst.push(GroupInputLog {
                batch_tokens,
                prefill_tokens: 0,
                decode_request_count: 0,
                decode_kv_total: 0,
                prefill_chunk_pairs: Vec::new(),
            });
        }
    }
}
