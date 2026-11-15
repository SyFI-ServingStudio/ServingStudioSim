//! Back-pass batching: a host-side stash of prefill tail rows and the rule
//! that decides when to run the layers past a bounded-replay cut on them.
//!
//! A bounded-replay arch (DeepSeek-V4.1: cut = the last KV source) needs the
//! layers from the cut on only for a few rows of each prompt (the rows whose
//! SWA keys and output token later steps read). Under back batching a step's
//! own prefill rows stop after the cut layer's `write_kv`; their tail rows'
//! layer input waits here, and a later step runs the back layers once over
//! every waiting entry ([`UnifiedArchInput::back_pass`]). A request's first
//! token is emitted when its tail's back pass completes.
//!
//! Which rows an entry holds ([`BackRows`]):
//! - `VllmChunk`: every prefill chunk leaves its last `min(q, window)` rows
//!   (vLLM's per-chunk replay rows); only a prompt's final chunk owns its
//!   request's completion.
//! - `PromptTail`: only a prompt's final chunk leaves an entry, the prompt's
//!   last `min(window, rows computed this episode)` rows (DeepInfer: an
//!   intermediate chunk's back rows are never read).
//!
//! When to flush ([`BackFlush`]), with `S` = everything stashed (plus, where
//! the engine can, this step's own tails) and `t(X)` the back pass's cost on
//! `X` over the whole model, measured by the arch (`probe`) with an empty
//! front minus an empty step:
//! - `EveryStep`: run every non-empty stash at once (no batching).
//! - `Knee`: the back cost is `t = a + b R` in its busiest group's rows `R`.
//!   From `t(S)` and `t(2S)` (every group's entries doubled):
//!   `b R = t(2S) - t(S)`, `a = 2 t(S) - t(2S)`, `R* = a / b`. Flush once
//!   `R >= R*`, where the fixed per-pass cost is at most the per-row cost, so
//!   the batched cost per row is within 2x of its asymptote (equivalently
//!   `2 t(2S) >= 3 t(S)`). No constant is tuned: `a` and `b` are the arch's
//!   own curve at the current stash.
//! - `Fixed(k)`: flush once the busiest group holds `k` entries.
//!
//! In every mode the stash also flushes when the front has nothing left to
//! prefill (waiting longer gains nothing) and when its oldest entry has waited
//! `ceil(R* / window)` steps (`k` steps under `Fixed`): the steps a stream of
//! one full tail per step needs to reach the flush size, so a few finishing
//! requests cannot stall behind a front that never empties.

use std::fmt;
use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::kv_layout::{KvCacheLayout, KvGrowth};
use crate::common::RequestId;

/// The rows a back pass keeps per request: the arch's sliding window (the
/// bounded replay keeps `min(q, window)` rows).
pub fn back_window(layout: &KvCacheLayout) -> u32 {
    layout
        .parts
        .iter()
        .filter_map(|part| match part.growth {
            KvGrowth::SlidingWindow { window_tokens } => Some(window_tokens),
            _ => None,
        })
        .max()
        .expect("back batching needs a sliding-window cache (the bounded-replay window)")
}

/// A probe over a whole-forward model with `groups` attention groups: the
/// cost of a step whose front is empty and whose back pass is `back`.
pub fn model_probe<M: IterwiseUnifiedModel>(model: Arc<M>, groups: usize) -> BackProbe {
    Arc::new(move |back: &[Vec<(u32, u32)>]| {
        let input = UnifiedArchInput {
            groups: vec![ArchGroupInput::default(); groups],
            tokens_per_source_rank: Vec::new(),
            back_pass: Some(back.to_vec()),
        };
        let (mut slots, mut scratch) = (Vec::new(), Vec::new());
        f64::from(model.eval_iter(&input, &mut slots, &mut scratch).m.time_ms)
    })
}

/// The back pass's cost (ms) on one list of `(prefix, rows)` entries per
/// group, front empty: the arch's whole forward, as the worker would run it.
pub type BackProbe = Arc<dyn Fn(&[Vec<(u32, u32)>]) -> f64 + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackFlush {
    EveryStep,
    Knee,
    Fixed(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackRows {
    VllmChunk,
    PromptTail,
}

/// The selector's back-batching choice (`Copy`; the probe is built by the
/// worker recipe from its model).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackBatchingSpec {
    pub flush: BackFlush,
    pub rows: BackRows,
}

/// One stashed tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackEntry {
    pub group: usize,
    pub prefix: u32,
    pub rows: u32,
    /// The request whose first token this entry's back pass emits (a final
    /// chunk's entry); `None` for an intermediate chunk under `VllmChunk`.
    pub request: Option<RequestId>,
    /// The front step that produced it.
    pub step: u64,
}

/// The rows a back pass flushes: per group `(prefix, rows)`, and the requests
/// it completes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BackFlushBatch {
    pub groups: Vec<Vec<(u32, u32)>>,
    pub requests: Vec<RequestId>,
}

pub struct BackStash {
    spec: BackBatchingSpec,
    window: u32,
    groups: usize,
    probe: BackProbe,
    entries: Vec<BackEntry>,
    /// Probe cost of an empty back pass (the front-empty step's own cost).
    base_ms: Option<f64>,
}

impl fmt::Debug for BackStash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackStash")
            .field("spec", &self.spec)
            .field("window", &self.window)
            .field("entries", &self.entries.len())
            .finish()
    }
}

impl BackStash {
    pub fn new(spec: BackBatchingSpec, window: u32, groups: usize, probe: BackProbe) -> Self {
        assert!(window > 0 && groups > 0);
        if let BackFlush::Fixed(k) = spec.flush {
            assert!(k > 0, "a fixed back flush needs k >= 1");
        }
        Self {
            spec,
            window,
            groups,
            probe,
            entries: Vec::new(),
            base_ms: None,
        }
    }

    #[allow(dead_code)]
    pub fn spec(&self) -> BackBatchingSpec {
        self.spec
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The entry a chunk `[prefix, prefix + chunk)` leaves, if any.
    /// `episode_start` is the first position this prefill episode computed
    /// (the resident prefix): rows before it have no layer input.
    pub fn chunk_entry(
        &self,
        prefix: u32,
        chunk: u32,
        episode_start: u32,
        finishes: bool,
    ) -> Option<(u32, u32)> {
        let end = prefix + chunk;
        let rows = match self.spec.rows {
            BackRows::VllmChunk => chunk.min(self.window),
            BackRows::PromptTail if finishes => (end - episode_start).min(self.window),
            BackRows::PromptTail => return None,
        };
        (rows > 0).then_some((end - rows, rows))
    }

    pub fn push(&mut self, entry: BackEntry) {
        debug_assert!(entry.group < self.groups);
        self.entries.push(entry);
    }

    fn grouped<'a>(entries: impl Iterator<Item = &'a BackEntry>, groups: usize) -> Vec<Vec<(u32, u32)>> {
        let mut out = vec![Vec::new(); groups];
        for entry in entries {
            out[entry.group].push((entry.prefix, entry.rows));
        }
        out
    }

    fn cost(&mut self, groups: &[Vec<(u32, u32)>]) -> f64 {
        let base = *self
            .base_ms
            .get_or_insert_with(|| (self.probe)(&vec![Vec::new(); self.groups]));
        ((self.probe)(groups) - base).max(0.0)
    }

    /// Whether to run the back pass now over the stash plus `current` (this
    /// step's own entries, when they can join). `front_pending`: the front
    /// still has prompt tokens to schedule after this step. `step`: this
    /// step's front-step counter.
    pub fn should_flush(&mut self, current: &[BackEntry], front_pending: bool, step: u64) -> bool {
        let all: Vec<BackEntry> = self.entries.iter().chain(current).copied().collect();
        if all.is_empty() {
            return false;
        }
        if !front_pending {
            return true;
        }
        let oldest = all.iter().map(|e| e.step).min().unwrap();
        let waited = step.saturating_sub(oldest);
        let mut per_group = vec![(0_u32, 0_u32); self.groups];
        for e in &all {
            per_group[e.group].0 += 1;
            per_group[e.group].1 += e.rows;
        }
        match self.spec.flush {
            BackFlush::EveryStep => true,
            BackFlush::Fixed(k) => {
                per_group.iter().any(|&(n, _)| n >= k) || waited >= u64::from(k)
            }
            BackFlush::Knee => {
                let groups = Self::grouped(all.iter(), self.groups);
                let doubled: Vec<Vec<(u32, u32)>> = groups
                    .iter()
                    .map(|g| g.iter().chain(g.iter()).copied().collect())
                    .collect();
                let t1 = self.cost(&groups);
                let t2 = self.cost(&doubled);
                // Per-row cost b*R and fixed cost a at the current stash.
                let per_row = t2 - t1;
                let fixed = 2.0 * t1 - t2;
                if per_row <= 0.0 || fixed <= per_row {
                    return true;
                }
                let rows = per_group.iter().map(|&(_, r)| r).max().unwrap_or(0) as f64;
                let knee_rows = rows * fixed / per_row;
                let cap = (knee_rows / f64::from(self.window)).ceil() as u64;
                waited >= cap.max(1)
            }
        }
    }

    /// Take the stash plus `current` as one back pass.
    pub fn take(&mut self, current: &[BackEntry]) -> BackFlushBatch {
        let entries: Vec<BackEntry> = std::mem::take(&mut self.entries)
            .into_iter()
            .chain(current.iter().copied())
            .collect();
        BackFlushBatch {
            groups: Self::grouped(entries.iter(), self.groups),
            requests: entries.iter().filter_map(|e| e.request).collect(),
        }
    }

    /// An empty back pass (deferred mode, nothing flushed this step).
    pub fn empty_pass(&self) -> Vec<Vec<(u32, u32)>> {
        vec![Vec::new(); self.groups]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `t = a + b * busiest rows`, a = 10, b = 0.01: R* = 1000 rows.
    fn linear_probe() -> BackProbe {
        Arc::new(|groups: &[Vec<(u32, u32)>]| {
            let rows = groups
                .iter()
                .map(|g| g.iter().map(|&(_, r)| r).sum::<u32>())
                .max()
                .unwrap_or(0);
            if rows == 0 {
                3.0 // the empty step's own cost, subtracted
            } else {
                3.0 + 10.0 + 0.01 * f64::from(rows)
            }
        })
    }

    fn stash(flush: BackFlush, rows: BackRows) -> BackStash {
        BackStash::new(BackBatchingSpec { flush, rows }, 128, 2, linear_probe())
    }

    fn entry(group: usize, rows: u32, step: u64, request: u64) -> BackEntry {
        BackEntry {
            group,
            prefix: 1000,
            rows,
            request: Some(RequestId(request as _)),
            step,
        }
    }

    #[test]
    fn knee_flushes_at_fixed_over_per_row_cost() {
        let mut s = stash(BackFlush::Knee, BackRows::PromptTail);
        // 7 tails of 128 = 896 rows < 1000 in group 0: hold.
        for i in 0..7 {
            s.push(entry(0, 128, 5, i));
        }
        assert!(!s.should_flush(&[], true, 5));
        // Group 1's rows do not add to the busiest group.
        assert!(!s.should_flush(&[entry(1, 128, 5, 99)], true, 5));
        // The 8th tail in group 0 reaches 1024 >= R* = 1000.
        assert!(s.should_flush(&[entry(0, 128, 5, 7)], true, 5));
        let batch = s.take(&[entry(0, 128, 5, 7)]);
        assert_eq!(batch.groups[0].len(), 8);
        assert_eq!(batch.requests.len(), 8);
        assert!(s.is_empty());
    }

    #[test]
    fn knee_age_cap_is_the_steps_to_reach_the_knee() {
        let mut s = stash(BackFlush::Knee, BackRows::PromptTail);
        s.push(entry(0, 100, 10, 1));
        // R = 100: R* = 100 * a / b R = 1000 -> cap ceil(1000/128) = 8 steps.
        assert!(!s.should_flush(&[], true, 17));
        assert!(s.should_flush(&[], true, 18));
    }

    #[test]
    fn an_empty_front_or_fixed_k_flushes() {
        let mut s = stash(BackFlush::Fixed(3), BackRows::PromptTail);
        assert!(!s.should_flush(&[], true, 0));
        s.push(entry(0, 128, 0, 1));
        s.push(entry(1, 128, 0, 2));
        assert!(!s.should_flush(&[], true, 1));
        assert!(s.should_flush(&[], false, 1));
        assert!(s.should_flush(&[entry(0, 5, 1, 3), entry(0, 5, 1, 4)], true, 1));
        assert!(s.should_flush(&[], true, 3));
    }

    #[test]
    fn rows_rules_pick_the_chunk_or_the_prompt_tail() {
        let tail = stash(BackFlush::EveryStep, BackRows::PromptTail);
        // A 50-row final chunk takes 128 rows reaching into the previous chunk.
        assert_eq!(tail.chunk_entry(8000, 50, 0, true), Some((7922, 128)));
        // ...but not past the episode start (a resident prefix).
        assert_eq!(tail.chunk_entry(8000, 50, 7990, true), Some((7990, 60)));
        assert_eq!(tail.chunk_entry(0, 8192, 0, false), None);
        let chunk = stash(BackFlush::EveryStep, BackRows::VllmChunk);
        assert_eq!(chunk.chunk_entry(0, 8192, 0, false), Some((8064, 128)));
        assert_eq!(chunk.chunk_entry(8192, 50, 0, true), Some((8192, 50)));
    }
}
