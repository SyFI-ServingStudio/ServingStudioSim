//! Achieved throughput per **cost-tree location**, from `cost_log`'s per-slot
//! `slot_flops` / `slot_bytes` / `slot_time_ms` joined to the `cost_manifest`.
//!
//! For slot `i` the achieved compute rate is `flops / time` (TFLOP/s) and the
//! achieved memory bandwidth is `bytes / time` (GB/s). A slot whose profile row
//! had no throughput rate logs `0` (the "not measured" sentinel) and is excluded.
//!
//! Grouping is by **location**, i.e. the leaf's identity in the cost tree, keyed
//! by the manifest leaf `name` (e.g. `afd.post_attn.o_proj`). `name` is exactly
//! the right key: the parallel branches of a `Max` node (a `max(gemm, gemm)` over
//! micro-batches / overlapped streams) are structural duplicates that carry the
//! SAME name, so grouping by name pools them into one location — precisely
//! "treat the different max leaves together". Names are section-prefixed and
//! shared across workers, so the 8 attention DP replicas of a location pool too.
//! (kind alone would be too coarse — `o_proj` and `qkv_proj` are both
//! `single_gemm` but distinct locations; the cost_log parquet has neither name
//! nor kind, only the slot index — those live in the `cost_manifest` sidecar.)
//!
//! Scale: a long run holds hundreds of millions of slots, so the rows are sampled
//! 1-in-[`SAMPLE_STRIDE`] on `iter_id` (a temporal stride — a location's achieved
//! rate is near-constant across iterations, so the sample is representative) and
//! the per-location distribution is computed exactly over that sample. Complements
//! the per-instance `analyze trace` annotations and the run-total `throughput`.
//! Tier-1, deployment-agnostic; grain = per cost_log slot, hence `batch` category.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use arrow_array::{Array, Float32Array, ListArray, RecordBatch, StringArray};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::cdf::{clean_nonnegative_sorted, stats};
use crate::io::{read_cost_manifests, SCHEMA_VERSION};
use crate::session::{col, collect, register_cost_log, require_columns, value_f64, COST_LOG_TABLE};

/// cost_log columns this subject depends on (drift guard).
const COST_COLS: &[&str] = &[
    "pool_tag",
    "worker_id",
    "section",
    "iter_id",
    "slot_time_ms",
    "slot_flops",
    "slot_bytes",
];

/// Coarsest iteration stride: on a long run we keep ~1-in-50 iterations (the
/// whole run is far too many slots for an exact distribution, and a location's
/// achieved rate is near-constant across iterations, so a temporal stride is
/// representative and ~50× cheaper). Also bounds cost on very long runs.
const MAX_STRIDE: u64 = 50;

/// Target number of iterations to keep. On a short run the stride is shrunk to
/// hit this, so a 10-iter smoke doesn't collapse to iteration 0 (prefill-only
/// warm-up) and one location — it stays representative regardless of run length.
const TARGET_SAMPLED_ITERS: u64 = 80;

/// One tree location's accumulated achieved rates (exact over the sample).
struct Loc {
    name: String,
    kind: String,
    tflops: Vec<f64>,
    gbps: Vec<f64>,
}

pub async fn run_kernel_throughput(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    if !register_cost_log(ctx, log_dir).await? {
        let reason = "cost_log/ dir not found";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }
    require_columns(ctx, COST_LOG_TABLE, COST_COLS).await?;
    let manifests = match read_cost_manifests(log_dir) {
        Ok(m) => m,
        Err(e) => {
            let reason =
                format!("cost_manifest/ unreadable ({e:#}); needed to name tree locations");
            return Ok((
                unavailable(log_dir, &reason),
                unavailable_payload(log_dir, &reason),
            ));
        }
    };

    // Pre-intern every manifest leaf `name` → a `Loc` slot, and map each
    // `(pool, worker, section)` to its `slot index → loc id` vector, so the hot
    // loop is an array index (no per-slot hashing or string cloning).
    let mut loc_id: HashMap<&str, u32> = HashMap::new();
    let mut locs: Vec<Loc> = Vec::new();
    let mut slot_loc: HashMap<(&str, u16, &str), Vec<u32>> = HashMap::new();
    for ((pool, worker), doc) in &manifests {
        for msec in &doc.sections {
            let ids: Vec<u32> = msec
                .manifest
                .slots
                .iter()
                .map(|leaf| match loc_id.get(leaf.name.as_str()) {
                    Some(&id) => id,
                    None => {
                        let id = locs.len() as u32;
                        loc_id.insert(&leaf.name, id);
                        locs.push(Loc {
                            name: leaf.name.clone(),
                            kind: leaf.kind.clone(),
                            tflops: Vec::new(),
                            gbps: Vec::new(),
                        });
                        id
                    }
                })
                .collect();
            slot_loc.insert((pool.as_str(), *worker, msec.section.as_str()), ids);
        }
    }

    let stride = choose_stride(ctx).await?;
    let sampled_rows = accumulate(ctx, stride, &slot_loc, &mut locs).await?;

    // A location contributes iff it saw ≥1 sampled slot with a rate.
    locs.retain(|l| !l.tflops.is_empty() || !l.gbps.is_empty());
    if locs.is_empty() {
        let reason = "no sampled slots with a profiled throughput rate";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }
    locs.sort_by(|a, b| a.name.cmp(&b.name));

    let locations: Vec<Value> = locs
        .iter()
        .map(|l| {
            json!({
                "name": l.name,
                "kind": l.kind,
                "tflops": stats(&clean_nonnegative_sorted(&l.tflops)),
                "gbps": stats(&clean_nonnegative_sorted(&l.gbps)),
            })
        })
        .collect();

    // Overall = pooled across locations (one concat, then the shared stats).
    let mut all_tf = Vec::new();
    let mut all_gb = Vec::new();
    for l in &locs {
        all_tf.extend_from_slice(&l.tflops);
        all_gb.extend_from_slice(&l.gbps);
    }
    let overall = json!({
        "tflops": stats(&clean_nonnegative_sorted(&all_tf)),
        "gbps": stats(&clean_nonnegative_sorted(&all_gb)),
    });

    let meta = json!({
        "log_dir": log_dir.display().to_string(),
        "sample_stride": stride,
        "sampled_rows": sampled_rows,
        "num_locations": locs.len(),
        "sampled_compute_slots": all_tf.len(),
        "sampled_memory_slots": all_gb.len(),
    });
    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "available": true,
        "overall": overall,
        "locations": locations,
        "definitions": definitions(),
    });
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "available": true,
        "locations": locations,
        "definitions": definitions(),
    });
    Ok((report, payload))
}

/// Pick the iteration sampling stride for this run: `num_iters / TARGET`, clamped
/// to `[1, MAX_STRIDE]`. Long runs land at the [`MAX_STRIDE`] cap (~1/50); short
/// runs shrink toward 1 so the sample stays representative. Cheap: one MAX scan.
async fn choose_stride(ctx: &SessionContext) -> Result<u64> {
    let batches = collect(
        ctx,
        "SELECT CAST(COALESCE(MAX(iter_id), 0) AS BIGINT) AS mx FROM cost_log",
    )
    .await?;
    let mut num_iters = 1u64;
    if let Some(b) = batches.first() {
        if b.num_rows() > 0 {
            let mx = value_f64(col(b, "mx")?, 0)?;
            if mx.is_finite() {
                num_iters = mx as u64 + 1;
            }
        }
    }
    Ok((num_iters / TARGET_SAMPLED_ITERS).clamp(1, MAX_STRIDE))
}

/// Scan the 1-in-`stride` rows, flatten each row's slot-parallel lists, and push
/// every rated slot into its location's bucket. The three `List<f32>` columns
/// share offsets (slot-parallel), and a row's slot index == the manifest slot
/// index for its `(pool, worker, section)`, so `slot_loc[..][slot]` names the
/// location. Returns the number of sampled rows actually attributed.
async fn accumulate(
    ctx: &SessionContext,
    stride: u64,
    slot_loc: &HashMap<(&str, u16, &str), Vec<u32>>,
    locs: &mut [Loc],
) -> Result<u64> {
    // `iter_id % stride = 0` keeps every stride-th iteration (all its workers /
    // sections / slots). DataFusion evaluates the predicate; only the surviving
    // rows' list columns reach this loop.
    let sql = format!(
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, worker_id, \
                CAST(section AS VARCHAR) AS section, \
                slot_time_ms, slot_flops, slot_bytes \
         FROM cost_log WHERE iter_id % {stride} = 0"
    );
    let batches = collect(ctx, &sql).await?;
    let mut sampled_rows = 0u64;
    for batch in &batches {
        let pool = str_col(batch, "pool_tag")?;
        let wid = col(batch, "worker_id")?;
        let sec = str_col(batch, "section")?;
        let (offsets, times) = list_f32(batch, "slot_time_ms")?;
        let (_, flops) = list_f32(batch, "slot_flops")?;
        let (_, bytes) = list_f32(batch, "slot_bytes")?;
        // Cache the resolved slot→loc-id vector across the common run of rows that
        // share one (pool, worker, section) — cost_log is worker/iter ordered.
        type CachedSlotLoc<'a> = Option<((&'a str, u16, &'a str), &'a Vec<u32>)>;
        let mut cached: CachedSlotLoc<'_> = None;
        for row in 0..batch.num_rows() {
            let (p, w, s) = (pool.value(row), value_f64(wid, row)? as u16, sec.value(row));
            let ids = match cached {
                Some((k, ids)) if k == (p, w, s) => ids,
                _ => match slot_loc.get(&(p, w, s)) {
                    Some(ids) => {
                        cached = Some(((p, w, s), ids));
                        ids
                    }
                    None => continue, // a row whose worker/section has no manifest
                },
            };
            sampled_rows += 1;
            let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
            for (leaf_idx, j) in (start..end).enumerate() {
                let Some(&id) = ids.get(leaf_idx) else { break }; // list longer than manifest → drift
                let t_s = times.value(j) as f64 / 1e3;
                if t_s <= 0.0 {
                    continue;
                }
                let loc = &mut locs[id as usize];
                let f = flops.value(j) as f64;
                if f > 0.0 {
                    loc.tflops.push(f / t_s / 1e12);
                }
                let b = bytes.value(j) as f64;
                if b > 0.0 {
                    loc.gbps.push(b / t_s / 1e9);
                }
            }
        }
    }
    Ok(sampled_rows)
}

fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    col(batch, name)?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("`{name}` is not a Utf8 array"))
}

/// Downcast a cost_log `List<f32>` column to `(list_offsets, flat_values)` so the
/// caller walks slots by offset with zero per-row allocation.
fn list_f32<'a>(batch: &'a RecordBatch, name: &str) -> Result<(&'a [i32], &'a Float32Array)> {
    let list = col(batch, name)?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("`{name}` is not a List array"))?;
    let vals = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| anyhow!("`{name}` is not List<f32>"))?;
    Ok((list.value_offsets(), vals))
}

fn definitions() -> Value {
    json!({
        "scope": "cost_log slots sampled 1-in-sample_stride on iter_id, grouped by tree location",
        "location": "the leaf's manifest `name` (e.g. afd.post_attn.o_proj); the parallel branches \
                     of a Max node share a name, so max(gemm, gemm) pools into one location",
        "tflops": "achieved compute = slot_flops / (slot_time_ms/1000) / 1e12 (TFLOP/s)",
        "gbps": "achieved memory bandwidth = slot_bytes / (slot_time_ms/1000) / 1e9 (GB/s)",
        "sentinel": "slots whose profile row had no rate log 0 and are excluded",
        "stats": "exact over the sampled slots (not approximate); overall pools all locations",
    })
}

fn unavailable(log_dir: &Path, reason: &str) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"log_dir": log_dir.display().to_string()},
        "available": false,
        "reason": reason,
        "definitions": definitions(),
    })
}

fn unavailable_payload(log_dir: &Path, reason: &str) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"log_dir": log_dir.display().to_string(), "available": false, "reason": reason},
        "locations": [],
    })
}
