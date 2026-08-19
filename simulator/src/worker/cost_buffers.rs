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

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::PathBuf;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{Time, WorkerId};
use crate::log::{CostLogEntry, CostLogger, GroupInputLog};
use crate::timing::{CostManifestDoc, LeafMetrics, SlotInput};

/// Upper bound on distinct cached section results per worker. AFD attn keys churn
/// as KV grows (~a few new keys per iteration), so the table is cleared wholesale
/// once it exceeds this — cheap and rare, and a dropped entry merely recomputes.
/// Sized generously so the ffn side's distinct `tokens_per_group` shapes rarely
/// overflow; each entry is tiny when logging is off (just the `agg`).
const COST_CACHE_CAP: usize = 1024;

/// A memoized section result — everything a cache hit needs to reproduce the miss
/// path exactly: the returned aggregate plus, when a logger is attached, the
/// per-slot breakdown and captured inputs for an identical `cost_log` row. With no
/// logger the slot vecs stay empty (a hit emits no row), so the sim-speed path pays
/// no per-entry allocation.
struct SectionSnapshot {
    agg: LeafMetrics,
    slots: Vec<LeafMetrics>,
    slot_inputs: Vec<SlotInput>,
}

/// FNV-1a over the whole key byte stream — a *combining* hash. Unlike
/// [`IdHasher`](crate::common::id::IdHasher), which keeps only the last integer
/// written (a single Fibonacci multiply) and so is unsuitable for a multi-`u32`
/// slice key, this folds every byte. Correctness never depends on the hash — the
/// `Box<[u32]>` key is compared exactly — this only spreads buckets.
struct FlatHasher(u64);

impl Default for FlatHasher {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325) // FNV-1a 64-bit offset basis
    }
}

impl Hasher for FlatHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // The default `write_u32`/`write_usize` (used when hashing the `[u32]` key +
        // its length prefix) route through here, so folding bytes covers the whole key.
        let mut h = self.0;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a 64-bit prime
        }
        self.0 = h;
    }
}

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
    /// Content-addressed cache of per-section eval results, keyed by the packed
    /// section name + the arch input's [`cost_signature`](GroupLogSource::cost_signature).
    /// AFD is layer-homogeneous (every layer re-evaluates the same section on the
    /// same batch), so all but the first layer of an iteration hit. Bounded by
    /// [`COST_CACHE_CAP`]; the exact `Box<[u32]>` key makes a hit bit-identical to a
    /// fresh eval.
    cache: HashMap<Box<[u32]>, SectionSnapshot, BuildHasherDefault<FlatHasher>>,
    /// Reused key buffer so a cache lookup allocates nothing on a hit.
    key_scratch: Vec<u32>,
    /// GPU wall / kernel time multiplier (≥ 1.0): every segment's returned wall
    /// Time is `kernel_time * gpu_time_multiplier` (inter-kernel overhead). From
    /// the worker's [`WorkerConfig`]; 1.0 = no overhead (predict passes 1.0).
    gpu_time_multiplier: f64,
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
        gpu_time_multiplier: f64,
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
            cache: HashMap::default(),
            key_scratch: Vec::new(),
            gpu_time_multiplier,
        }
    }

    /// Iter-wise convenience constructor: the model exposes one fused `CostTree`, so
    /// its manifest is a single `iter` section. Equivalent to [`new`](Self::new)
    /// with `CostManifestDoc::single("iter", model.cost_log_manifest())`.
    pub fn new_iter<M: IterwiseUnifiedModel + ?Sized>(
        log_dir: Option<PathBuf>,
        pool_tag: &'static str,
        worker_id: WorkerId,
        model: &M,
        gpu_time_multiplier: f64,
    ) -> Self {
        Self::new(
            log_dir,
            pool_tag,
            worker_id,
            &CostManifestDoc::single("iter", model.cost_log_manifest()),
            gpu_time_multiplier,
        )
    }

    /// Run one building block's eval through `eval` and (if logging) write its
    /// `cost_log` row, returning the segment's **wall `Time`** = kernel-folded time
    /// × this worker's `gpu_time_multiplier` (inter-kernel overhead). The `cost_log`
    /// row it writes stays PRE-scale (pure kernel time); only the returned Time
    /// carries the overhead, so the overhead surfaces as a gap between iter/section
    /// slices in the trace, never inside a kernel slice. `eval` is handed the reused
    /// `slots` + `scratch` buffers
    /// and an optional capture sink: `Some(inputs)` when a logger is attached (call
    /// the model's `*_with_inputs` method), `None` otherwise (call the plain `*_cost`
    /// method). `section` names the block (`attn` / `prologue` / `pre_attn` /
    /// `post_attn` / `post_attn_last` / `epilogue` / `iter`), `layer` the layer index
    /// (`-1` for iteration-level prologue/epilogue/iter), `batch_id` the AFD slot,
    /// and `groups` the per-shard input context (logged as the row's `input_section`)
    /// — the attn/iter sides pass `[ArchGroupInput]`, the ffn side bare `[u32]` token
    /// counts, both via [`GroupLogSource`].
    ///
    /// `cache_key` opts this call into result memoization: `Some(k)` caches the section
    /// result under `(section, k)`, returning it on a repeat without re-running `eval`.
    /// `k` must be a *small* identity that changes iff the cost input does — the caller
    /// keeps it cheap: attn passes its `(iter_id, slot)` identity (its real input,
    /// `decode_kv_lens`, is O(batch) and costlier to hash than the eval it would save,
    /// and it never repeats across iterations anyway); ffn passes its small
    /// `tokens_per_group` (which *does* recur across randomly-assigned tasks). `None`
    /// disables caching (iter-wise callers eval once per iteration — nothing repeats).
    #[allow(
        clippy::too_many_arguments,
        reason = "each arg is a distinct per-call identity/context piece (section, layer, iter/batch ids, groups, cache key, clock, eval closure) documented above"
    )]
    pub fn run_section<G, F>(
        &mut self,
        section: &'static str,
        layer: i16,
        iter_id: u64,
        batch_id: u64,
        groups: &G,
        cache_key: Option<&[u32]>,
        now: Time,
        eval: F,
    ) -> Time
    where
        G: GroupLogSource,
        F: FnOnce(
            &mut Vec<LeafMetrics>,
            &mut Vec<LeafMetrics>,
            Option<&mut Vec<SlotInput>>,
        ) -> LeafMetrics,
    {
        // Build the key (packed section name + the caller's small identity) and try a
        // hit. Reuses `key_scratch`, so a hit allocates nothing.
        if let Some(ck) = cache_key {
            self.key_scratch.clear();
            let name = section.as_bytes();
            #[allow(
                clippy::cast_possible_truncation,
                reason = "section is a hardcoded &'static str cost-section name, far shorter than u32::MAX"
            )]
            self.key_scratch.push(name.len() as u32);
            for chunk in name.chunks(4) {
                let mut w = 0u32;
                for (i, &b) in chunk.iter().enumerate() {
                    w |= u32::from(b) << (8 * i);
                }
                self.key_scratch.push(w);
            }
            self.key_scratch.extend_from_slice(ck);

            // Hit: the section already ran on this exact input (a prior layer of this
            // iteration, or a repeated ffn batch). Replay the memoized result. The
            // `cost_log` row still uses THIS call's iter_id / batch_id / wall_start /
            // layer — only the input-determined `agg` / `slots` / `slot_inputs` come
            // from the snapshot — so the row is byte-identical to a fresh eval's.
            if let Some(snap) = self.cache.get(self.key_scratch.as_slice()) {
                let agg = snap.agg;
                if let Some(logger) = self.logger.as_mut() {
                    groups.fill_group_log(&mut self.groups);
                    let entry = CostLogEntry {
                        worker_id: self.worker_id.0,
                        iter_id,
                        batch_id,
                        wall_start_ms: now.as_ms(),
                        total_time_ms: f64::from(agg.m.time_ms),
                        energy_j: f64::from(agg.m.energy_j),
                        section,
                        layer,
                        group_len: 0,
                        slot_len: 0,
                        slot_input_len: 0,
                    };
                    if let Err(e) =
                        logger.record(entry, &snap.slots, &mut self.groups, &snap.slot_inputs)
                    {
                        tracing::warn!("cost_log record failed: {e}");
                    }
                }
                return self.wall_time(&agg);
            }
        }

        // Miss (or un-cached): evaluate, write the row. Capture per-leaf inputs only on
        // the path that will actually write the row — the `*_with_inputs` clone per leaf
        // is part of the cost_log contract, so we don't pay it when no logger is attached.
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
                total_time_ms: f64::from(agg.m.time_ms),
                energy_j: f64::from(agg.m.energy_j),
                section,
                layer,
                // Filled by `logger.record` from the slice lengths.
                group_len: 0,
                slot_len: 0,
                slot_input_len: 0,
            };
            if let Err(e) = logger.record(entry, &self.slots, &mut self.groups, &self.slot_inputs) {
                tracing::warn!("cost_log record failed: {e}");
            }
        }
        // Memoize for future hits (`key_scratch` still holds this call's key, untouched
        // by eval/record). With no logger, hits emit no row, so keep just `agg` (empty
        // slot vecs -> no per-entry allocation on the sim-speed path). Clone (not
        // `take`) preserves the worker's reusable buffer capacity.
        if cache_key.is_some() {
            let snap = SectionSnapshot {
                agg,
                slots: if capture {
                    self.slots.clone()
                } else {
                    Vec::new()
                },
                slot_inputs: if capture {
                    self.slot_inputs.clone()
                } else {
                    Vec::new()
                },
            };
            if self.cache.len() >= COST_CACHE_CAP {
                self.cache.clear();
            }
            self.cache.insert(self.key_scratch.as_slice().into(), snap);
        }
        self.wall_time(&agg)
    }

    /// kernel-folded time → wall `Time`, applying this worker's inter-kernel
    /// `gpu_time_multiplier` (≥ 1.0). The single place the overhead enters the
    /// clock; `cost_log` rows are written pre-scale (pure kernel) by the callers
    /// above, so folding a row's `slot_time_ms` still reproduces its `total_time_ms`.
    fn wall_time(&self, agg: &LeafMetrics) -> Time {
        Time::from_ms(f64::from(agg.m.time_ms) * self.gpu_time_multiplier)
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
        // `run_section` already applies `gpu_time_multiplier` and returns wall Time.
        self.run_section(
            "iter",
            -1,
            iter_id,
            0,
            &arch_input.groups,
            None, // one fused eval per iteration — nothing to memoize
            now,
            |slots, scratch, inputs| match inputs {
                Some(i) => model.eval_iter_with_inputs(arch_input, slots, scratch, i),
                None => model.eval_iter(arch_input, slots, scratch),
            },
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::Cell;
    use std::fs::File;

    use arrow_array::{Float64Array, Int16Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    use crate::timing::{CostManifest, CoverageFlags, FlatCostNode, LeafDesc, Metrics4};

    fn leaf(time_ms: f32) -> LeafMetrics {
        LeafMetrics {
            m: Metrics4 {
                time_ms,
                flops: 0.0,
                bytes: 0.0,
                energy_j: 0.0,
            },
            coverage: CoverageFlags::EMPTY,
            backend_index: LeafMetrics::NO_BACKEND,
        }
    }

    /// One decode-only attention group with the given per-request KV lengths.
    fn grp(kv: &[u32]) -> ArchGroupInput {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "kv is a test fixture with a handful of KV lengths, far under u32::MAX"
        )]
        let num_tokens = kv.len() as u32;
        ArchGroupInput {
            batch_tokens: num_tokens,
            prefill_tokens: 0,
            decode_tokens: num_tokens,
            prefill_chunk_pairs: Vec::new(),
            decode_kv_lens: kv.to_vec(),
            total_kv_len: kv.iter().sum(),
        }
    }

    fn trivial_manifest() -> CostManifestDoc {
        CostManifestDoc::single(
            "attn",
            CostManifest {
                slots: vec![LeafDesc {
                    name: "m.test".to_owned(),
                    kind: "unit".to_owned(),
                    kernel_config: serde_json::json!({"shape": 1, "backends": ["torch"]}),
                }],
                nodes: vec![FlatCostNode::Leaf(0)],
                node_labels: vec![None],
            },
        )
    }

    /// With no logger (the sim-speed path), a repeated `(section, cache_key)` hits the
    /// cache: the eval closure is skipped and the *cached* aggregate is returned. A
    /// distinct key or a distinct section misses; `None` never caches.
    #[test]
    fn cache_hit_skips_eval_and_returns_cached_agg() {
        let mut cost = CostBuffers::new(None, "attn", WorkerId(0), &trivial_manifest(), 1.0);
        let a = vec![grp(&[10, 11])];
        let b = vec![grp(&[10, 12])];
        let (ka, kb) = ([7u32], [8u32]); // cheap caller identities (e.g. iter/slot)
        let now = Time::from_ms(0.0);
        let calls = Cell::new(0u32);

        let m0 = cost.run_section("attn", 0, 0, 0, &a, Some(&ka), now, |slots, _sc, _in| {
            calls.set(calls.get() + 1);
            slots.clear();
            leaf(2.5)
        });
        // Same key, next layer: hit. The passed closure returns 777, but the hit must
        // ignore it and replay the cached 2.5 (and never invoke the closure).
        let m1 = cost.run_section("attn", 1, 0, 0, &a, Some(&ka), now, |slots, _sc, _in| {
            calls.set(calls.get() + 1);
            slots.clear();
            leaf(777.0)
        });
        assert_eq!(calls.get(), 1, "identical (section, cache_key) must hit");
        // run_section returns wall Time (× gpu_time_multiplier = 1.0 here).
        assert_eq!(m0.as_ms(), 2.5);
        assert_eq!(
            m1.as_ms(),
            2.5,
            "hit replays the cached agg, not the closure"
        );

        // Distinct key -> miss.
        cost.run_section("attn", 2, 0, 0, &b, Some(&kb), now, |slots, _sc, _in| {
            calls.set(calls.get() + 1);
            slots.clear();
            leaf(3.0)
        });
        assert_eq!(calls.get(), 2, "a distinct cache_key must miss");

        // Distinct section, same key -> miss (section is part of the key).
        cost.run_section(
            "prologue",
            0,
            0,
            0,
            &a,
            Some(&ka),
            now,
            |slots, _sc, _in| {
                calls.set(calls.get() + 1);
                slots.clear();
                leaf(9.0)
            },
        );
        assert_eq!(
            calls.get(),
            3,
            "a distinct section must miss even with same key"
        );

        // `None` never caches: repeats always re-eval.
        cost.run_section("attn", 3, 0, 0, &a, None, now, |slots, _sc, _in| {
            calls.set(calls.get() + 1);
            slots.clear();
            leaf(1.0)
        });
        cost.run_section("attn", 4, 0, 0, &a, None, now, |slots, _sc, _in| {
            calls.set(calls.get() + 1);
            slots.clear();
            leaf(1.0)
        });
        assert_eq!(calls.get(), 5, "cache_key=None must always eval");
    }

    /// `gpu_time_multiplier > 1` scales the RETURNED wall Time (× mult) while the
    /// eval's kernel-folded time is untouched — the overhead lives only in the Time
    /// the worker advances its clock by, never in the kernel cost itself. The cache
    /// hit replays the same scaled wall Time.
    #[test]
    fn gpu_time_multiplier_scales_returned_wall_time() {
        let mut cost = CostBuffers::new(None, "attn", WorkerId(0), &trivial_manifest(), 2.0);
        let a = vec![grp(&[10, 11])];
        let now = Time::from_ms(0.0);
        // kernel-folded time = 2.5ms; wall = 2.5 × 2.0 = 5.0ms.
        let miss = cost.run_section(
            "attn",
            0,
            0,
            0,
            &a,
            Some(&[7u32]),
            now,
            |slots, _sc, _in| {
                slots.clear();
                leaf(2.5)
            },
        );
        assert_eq!(miss.as_ms(), 5.0);
        // A cache hit (same section+key) replays the same scaled wall Time.
        let hit = cost.run_section(
            "attn",
            1,
            0,
            0,
            &a,
            Some(&[7u32]),
            now,
            |slots, _sc, _in| {
                slots.clear();
                leaf(999.0)
            },
        );
        assert_eq!(hit.as_ms(), 5.0);
    }

    /// With a logger attached, every call still emits one `cost_log` row (a hit does
    /// not suppress logging), and the row carries THIS call's `layer` while sharing the
    /// cached `total_time_ms`.
    #[test]
    fn cache_hit_still_emits_logrow_with_per_call_layer() {
        let dir = tempdir().unwrap();
        let mut cost = CostBuffers::new(
            Some(dir.path().to_path_buf()),
            "attn",
            WorkerId(0),
            &trivial_manifest(),
            1.0,
        );
        let a = vec![grp(&[10, 11])];
        let b = vec![grp(&[10, 12])];
        let (ka, kb) = ([7u32], [8u32]);
        let now = Time::from_ms(0.0);

        // layer 0, key a: miss (total_time_ms 2.5).
        cost.run_section("attn", 0, 0, 0, &a, Some(&ka), now, |slots, _sc, _in| {
            slots.clear();
            slots.push(leaf(2.5));
            leaf(2.5)
        });
        // layer 1, key a: hit. Closure returns 777 but must be skipped -> row logs 2.5.
        cost.run_section("attn", 1, 0, 0, &a, Some(&ka), now, |slots, _sc, _in| {
            slots.clear();
            slots.push(leaf(777.0));
            leaf(777.0)
        });
        // layer 2, key b: miss (3.0).
        cost.run_section("attn", 2, 0, 0, &b, Some(&kb), now, |slots, _sc, _in| {
            slots.clear();
            slots.push(leaf(3.0));
            leaf(3.0)
        });
        drop(cost); // flush + join the writer thread

        let path = dir.path().join("raw/cost_log/worker_attn_0.parquet");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(
            batch.num_rows(),
            3,
            "every call (hit or miss) emits one row"
        );
        // Schema column order: 5 = total_time_ms, 12 = layer (see cost_log_schema).
        let ttm = batch
            .column(5)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let layer = batch
            .column(12)
            .as_any()
            .downcast_ref::<Int16Array>()
            .unwrap();
        assert_eq!((layer.value(0), layer.value(1), layer.value(2)), (0, 1, 2));
        assert!((ttm.value(0) - 2.5).abs() < 1e-9);
        assert!(
            (ttm.value(1) - 2.5).abs() < 1e-9,
            "hit re-logs the cached time"
        );
        assert!((ttm.value(2) - 3.0).abs() < 1e-9);
    }
}
