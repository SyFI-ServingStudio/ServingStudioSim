//! Per cost-tree **position**, the distribution of that kernel's inputs in feature
//! space, colored by which backend best-of-N selected there.
//!
//! For each position (a leaf's manifest `name`, pooling `Max` siblings exactly as
//! `kernel-throughput` does) we read every executed slot's `slot_input` JSON and
//! its `slot_backend` index (the position-local candidate the kernel picked, or
//! `255` when the leaf did not run that iteration), dedup identical
//! `(input, backend)` observations with a count, bound the point set per position
//! with a per-backend even stride (so a rarely-picked backend still shows), turn
//! each unique input JSON into a numeric feature vector, and project to 2-D:
//! 1 feature → a value axis, 2 → the raw axes, ≥3 → PCA (standardize + top-2
//! principal components). The payload is one scatter per position; the Python side
//! renders `plots/kernel_input_dist/<position>.png` colored by backend.
//!
//! Availability is strict: a run whose `cost_log` predates the `slot_backend`
//! column, or whose `cost_manifest` leaves carry no structured `backends` list,
//! reports `available: false` — the selected backend is never guessed. Tier-1,
//! deployment-agnostic; the grain (selection over input space) is its own
//! `backend` category.
//!
//! Scale: the SQL scan keeps every stride-th iteration (the same temporal stride
//! `kernel-throughput` uses — a position's input mix is near-stationary across
//! iterations), the heavy projection/filter runs in DataFusion, and the per-input
//! dedup + per-position bounded sample happen in Rust, so a large run stays well
//! under budget.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use anyhow::{anyhow, Result};
use arrow_array::{Array, ListArray, RecordBatch, StringArray, UInt8Array};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{read_cost_manifests, SCHEMA_VERSION};
use crate::pca::pca_project_2d;
use crate::session::{col, collect, register_cost_log, require_columns, value_f64, COST_LOG_TABLE};

/// cost_log columns that always exist when cost logging is on (drift guard — a
/// genuine schema regression here should fail loud, not silently degrade).
const CORE_COLS: &[&str] = &["pool_tag", "worker_id", "section", "iter_id"];

/// The two list columns this subject uniquely needs. Absent on runs that predate
/// them → `available: false` (an expected old-run case, not a drift error), so
/// they are probed softly rather than through `require_columns`.
const OPTIONAL_COLS: &[&str] = &["slot_input", "slot_backend"];

/// Sentinel `slot_backend` value: the leaf was not executed this iteration.
const NO_BACKEND: u8 = u8::MAX;

/// Iteration sampling bounds (shared intent with `kernel-throughput`): keep
/// ~1-in-`MAX_STRIDE` iterations on a long run, shrinking toward every iteration
/// on a short run so a smoke test stays representative.
const MAX_STRIDE: u64 = 50;
const TARGET_SAMPLED_ITERS: u64 = 80;

/// Per-position point budget after dedup: each position emits at most this many
/// scatter points, split across its backends with a floor so a rare backend keeps
/// enough points to be visible. Keeps the payload and the PNG bounded on a run
/// with millions of distinct inputs.
const MAX_POINTS_PER_POSITION: usize = 6000;
const MIN_POINTS_PER_BACKEND: usize = 800;

/// A cost-tree position: the leaf `name` (identity, pooling `Max` siblings), its
/// kernel `kind`, and the ordered candidate `backends` from the manifest — the
/// index space of `slot_backend`, so `backends[i]` names candidate `i`.
struct Position {
    name: String,
    kind: String,
    backends: Vec<String>,
}

/// One deduped observation: how many sampled slots at this position picked
/// `backend` for exactly this `input_json`.
struct Point {
    input_json: String,
    backend: u8,
    count: u64,
}

pub async fn run(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    if !register_cost_log(ctx, log_dir).await? {
        let reason = "cost_log/ dir not found";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }
    require_columns(ctx, COST_LOG_TABLE, CORE_COLS).await?;
    for optional in OPTIONAL_COLS {
        if !has_column(ctx, COST_LOG_TABLE, optional).await? {
            let reason = format!(
                "cost_log has no `{optional}` column — this run predates per-slot backend/input \
                 logging; re-run the sim to analyze backend selection"
            );
            return Ok((
                unavailable(log_dir, &reason),
                unavailable_payload(log_dir, &reason),
            ));
        }
    }
    let manifests = match read_cost_manifests(log_dir) {
        Ok(m) => m,
        Err(e) => {
            let reason = format!("cost_manifest/ unreadable ({e:#}); needed to name positions");
            return Ok((
                unavailable(log_dir, &reason),
                unavailable_payload(log_dir, &reason),
            ));
        }
    };

    // Intern every manifest leaf `name` → a `Position`, and map each
    // `(pool, worker, section)` to its `slot index → position id` vector (the same
    // pre-interned `slot_loc` the throughput subject uses, so the hot loop is an
    // array index). A position's candidate `backends` come from the first leaf that
    // carries a non-empty list (filled in if a later duplicate has it).
    let mut pos_id: HashMap<&str, u32> = HashMap::new();
    let mut positions: Vec<Position> = Vec::new();
    let mut slot_loc: HashMap<(&str, u16, &str), Vec<u32>> = HashMap::new();
    for ((pool, worker), doc) in &manifests {
        for msec in &doc.sections {
            let ids: Vec<u32> = msec
                .manifest
                .slots
                .iter()
                .map(|leaf| {
                    let leaf_backends = leaf.backends();
                    match pos_id.get(leaf.name.as_str()) {
                        Some(&id) => {
                            let p = &mut positions[id as usize];
                            if p.backends.is_empty() && !leaf_backends.is_empty() {
                                p.backends = leaf_backends;
                            }
                            id
                        }
                        None => {
                            #[allow(
                                clippy::cast_possible_truncation,
                                reason = "positions is the set of distinct manifest leaf names in \
                                          one run, realistically far below u32::MAX"
                            )]
                            let id = positions.len() as u32;
                            pos_id.insert(&leaf.name, id);
                            positions.push(Position {
                                name: leaf.name.clone(),
                                kind: leaf.kind.clone(),
                                backends: leaf_backends,
                            });
                            id
                        }
                    }
                })
                .collect();
            slot_loc.insert((pool.as_str(), *worker, msec.section.as_str()), ids);
        }
    }

    // A position is plottable only if the manifest gave it a candidate list; a run
    // whose manifests predate structured `backends` has none, so the subject is
    // unavailable rather than guessing which backend `slot_backend` indexes.
    let plottable: Vec<bool> = positions.iter().map(|p| !p.backends.is_empty()).collect();
    if !plottable.iter().any(|&b| b) {
        let reason =
            "cost_manifest leaves carry no structured `backends` list — this run predates \
                      candidate-backend logging; re-run the sim to analyze backend selection";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }

    let stride = choose_stride(ctx).await?;
    let scan = accumulate(ctx, stride, &slot_loc, &plottable, positions.len()).await?;

    // Build one payload position + one report position per plottable location that
    // saw ≥1 sampled executed slot with an input. Positions are emitted in name
    // order for a stable payload; points within a position are already sorted.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "positions is the set of distinct manifest leaf names in one run, realistically \
                  far below u32::MAX"
    )]
    let mut order: Vec<u32> = (0..positions.len() as u32)
        .filter(|&id| plottable[id as usize] && !scan.points[id as usize].is_empty())
        .collect();
    order.sort_by(|&a, &b| positions[a as usize].name.cmp(&positions[b as usize].name));

    let mut payload_positions = Vec::new();
    let mut report_positions = Vec::new();
    let mut num_multi_backend = 0usize;
    for &id in &order {
        let (payload_pos, report_pos, multi) =
            build_position(&positions[id as usize], &scan.points[id as usize]);
        if multi {
            num_multi_backend += 1;
        }
        payload_positions.push(payload_pos);
        report_positions.push(report_pos);
    }

    if payload_positions.is_empty() {
        let reason = "no sampled slots with a recorded backend selection and input";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }

    // Manifest positions that produced no scatter, surfaced in the report so a
    // reader isn't left wondering where a leaf went (e.g. "why no attn.prefill?"):
    // either it carries no candidate list (an old manifest) or it was never
    // executed in the sampled window — every slot is the 255 not-run sentinel, as
    // for a ragged-prefill leaf in a decode-heavy sample or a placement-zeroed comm
    // leg. Report-only (nothing to plot).
    let mut omitted: Vec<Value> = Vec::new();
    for id in 0..positions.len() {
        if plottable[id] && !scan.points[id].is_empty() {
            continue;
        }
        let reason = if !plottable[id] {
            "no candidate backends in the manifest (run predates structured backend logging)"
        } else {
            "never executed in the sampled iterations (every slot is the 255 not-run sentinel)"
        };
        omitted.push(json!({
            "name": positions[id].name,
            "kind": positions[id].kind,
            "reason": reason,
        }));
    }
    omitted.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));

    let meta = json!({
        "log_dir": log_dir.display().to_string(),
        "sample_stride": stride,
        "sampled_rows": scan.sampled_rows,
        "num_positions_plotted": payload_positions.len(),
        "num_positions_multi_backend": num_multi_backend,
        "num_positions_manifest": positions.len(),
        "num_positions_omitted": omitted.len(),
        "num_positions_without_candidates": plottable.iter().filter(|&&b| !b).count(),
        "skipped_not_executed_slots": scan.skipped_not_executed,
        "skipped_empty_input_slots": scan.skipped_empty_input,
        "max_points_per_position": MAX_POINTS_PER_POSITION,
    });
    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "available": true,
        "positions": report_positions,
        "positions_omitted": omitted,
        "definitions": definitions(),
    });
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "available": true,
        "positions": payload_positions,
        "definitions": definitions(),
    });
    Ok((report, payload))
}

/// Whether a registered table has a column (soft probe for the optional list
/// columns — `require_columns` bails, which we don't want for an expected old run).
async fn has_column(ctx: &SessionContext, table: &str, column: &str) -> Result<bool> {
    let df = ctx.table(table).await?;
    Ok(df.schema().field_with_name(None, column).is_ok())
}

/// Same stride policy as `kernel-throughput`: `num_iters / TARGET`, clamped to
/// `[1, MAX_STRIDE]`.
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
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "mx is MAX(iter_id) from this run's own cost_log (a monotone \
                              nonnegative iteration counter, COALESCEd to 0), realistically far \
                              below u64::MAX; Rust's float-to-int cast also saturates rather than \
                              wrapping"
                )]
                {
                    num_iters = mx as u64 + 1;
                }
            }
        }
    }
    Ok((num_iters / TARGET_SAMPLED_ITERS).clamp(1, MAX_STRIDE))
}

/// Result of the strided scan: per-position deduped `(input, backend) → count`
/// points plus the slots dropped for the two skip reasons.
struct Scan {
    /// `points[pos_id]` = deduped observations at that position.
    points: Vec<Vec<Point>>,
    sampled_rows: u64,
    skipped_not_executed: u64,
    skipped_empty_input: u64,
}

/// Scan the 1-in-`stride` rows, explode the slot-parallel `slot_input` /
/// `slot_backend` lists by offset, and dedup `(position, input, backend)` into a
/// count. The two list columns share the row's slot count (both slot-aligned to
/// `slot_time_ms`); a row whose `slot_input` is empty (input logging off for that
/// row) is skipped. Then bound each position's point set with a per-backend even
/// stride so a rare backend survives and the payload stays small.
async fn accumulate(
    ctx: &SessionContext,
    stride: u64,
    slot_loc: &HashMap<(&str, u16, &str), Vec<u32>>,
    plottable: &[bool],
    num_positions: usize,
) -> Result<Scan> {
    let sql = format!(
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, worker_id, \
                CAST(section AS VARCHAR) AS section, slot_input, slot_backend \
         FROM cost_log WHERE iter_id % {stride} = 0"
    );
    let batches = collect(ctx, &sql).await?;

    // Dedup map per position: (backend, input_json) → count. A BTreeMap keeps the
    // later sort deterministic and cheap (already ordered by backend then input).
    let mut dedup: Vec<BTreeMap<(u8, String), u64>> =
        (0..num_positions).map(|_| BTreeMap::new()).collect();
    let mut sampled_rows = 0u64;
    let mut skipped_not_executed = 0u64;
    let mut skipped_empty_input = 0u64;

    for batch in &batches {
        let pool = str_col(batch, "pool_tag")?;
        let wid = col(batch, "worker_id")?;
        let sec = str_col(batch, "section")?;
        let (in_off, in_vals) = list_str(batch, "slot_input")?;
        let (bk_off, bk_vals) = list_u8(batch, "slot_backend")?;
        // Cache the resolved slot→position vector across the run of rows that share
        // one (pool, worker, section) — cost_log is worker/iter ordered.
        type CachedSlotLoc<'a> = ((&'a str, u16, &'a str), &'a Vec<u32>);
        let mut cached: Option<CachedSlotLoc> = None;
        for row in 0..batch.num_rows() {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "worker_id is this run's own cost_log worker index (a small nonnegative \
                          shard id well below u16::MAX by construction of the simulated cluster \
                          topology); Rust's float-to-int cast also saturates rather than wrapping"
            )]
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
            #[allow(
                clippy::cast_sign_loss,
                reason = "Arrow list offsets are always nonnegative by format invariant, and \
                          usize is at least as wide as the i32 offset type on this target"
            )]
            let (is, ie) = (in_off[row] as usize, in_off[row + 1] as usize);
            #[allow(
                clippy::cast_sign_loss,
                reason = "Arrow list offsets are always nonnegative by format invariant, and \
                          usize is at least as wide as the i32 offset type on this target"
            )]
            let (bs, be) = (bk_off[row] as usize, bk_off[row + 1] as usize);
            // Slot-aligned lists must share length; a mismatch means this row logged
            // no inputs (input logging off) — skip it rather than mis-pair slots.
            if ie - is != be - bs {
                continue;
            }
            sampled_rows += 1;
            for (leaf_idx, (ji, jb)) in (is..ie).zip(bs..be).enumerate() {
                let Some(&id) = ids.get(leaf_idx) else { break }; // list longer than manifest → drift
                if !plottable[id as usize] {
                    continue;
                }
                let backend = bk_vals.value(jb);
                if backend == NO_BACKEND {
                    skipped_not_executed += 1;
                    continue;
                }
                let input = in_vals.value(ji);
                if input.is_empty() || input == "null" {
                    skipped_empty_input += 1;
                    continue;
                }
                *dedup[id as usize]
                    .entry((backend, input.to_string()))
                    .or_insert(0) += 1;
            }
        }
    }

    // Per position, bound the deduped observations with a per-backend even stride.
    let points = dedup.into_iter().map(bound_points).collect();

    Ok(Scan {
        points,
        sampled_rows,
        skipped_not_executed,
        skipped_empty_input,
    })
}

/// Turn one position's `(backend, input) → count` map into a bounded point list.
/// Groups by backend, then even-strides each backend group (deterministic, no RNG)
/// to `max(MIN_POINTS_PER_BACKEND, MAX_POINTS_PER_POSITION / num_backends)` — so a
/// rarely-picked backend keeps its points while a dominant one is thinned. The
/// BTreeMap is already ordered `(backend, input)`, so groups are contiguous.
fn bound_points(per_pos: BTreeMap<(u8, String), u64>) -> Vec<Point> {
    if per_pos.is_empty() {
        return Vec::new();
    }
    // Split into per-backend runs (map is sorted by backend first).
    let mut groups: Vec<Vec<Point>> = Vec::new();
    let mut cur_backend: Option<u8> = None;
    for ((backend, input_json), count) in per_pos {
        if cur_backend != Some(backend) {
            groups.push(Vec::new());
            cur_backend = Some(backend);
        }
        groups.last_mut().unwrap().push(Point {
            input_json,
            backend,
            count,
        });
    }
    let cap = (MAX_POINTS_PER_POSITION / groups.len()).max(MIN_POINTS_PER_BACKEND);
    let mut out = Vec::new();
    for group in groups {
        out.extend(even_stride(group, cap));
    }
    out
}

/// Keep at most `cap` items by even stride over `items` (already deterministically
/// ordered). Preserves the endpoints' spread without an RNG.
fn even_stride<T>(items: Vec<T>, cap: usize) -> Vec<T> {
    if items.len() <= cap || cap == 0 {
        return items;
    }
    let stride = items.len().div_ceil(cap);
    items.into_iter().step_by(stride).collect()
}

/// Build one position's payload + report objects from its bounded points. Returns
/// `(payload, report, is_multi_backend)`.
fn build_position(pos: &Position, points: &[Point]) -> (Value, Value, bool) {
    // Full selection tallies (over the deduped counts, i.e. every sampled slot —
    // the bounded point set is only for plotting, not for the ratios).
    let mut selection: BTreeMap<u8, u64> = BTreeMap::new();
    for pt in points {
        *selection.entry(pt.backend).or_insert(0) += pt.count;
    }
    let total: u64 = selection.values().sum();
    let backend_name = |i: u8| -> String {
        pos.backends
            .get(i as usize)
            .cloned()
            .unwrap_or_else(|| format!("candidate_{i}"))
    };
    #[allow(
        clippy::cast_precision_loss,
        reason = "c and total are per-position sampled-slot selection counts, bounded by the \
                  sampled row count of one run and far below 2^53"
    )]
    let selection_json: Vec<Value> = selection
        .iter()
        .map(|(&b, &c)| {
            json!({
                "backend_index": b,
                "backend_name": backend_name(b),
                "count": c,
                "ratio": if total > 0 { c as f64 / total as f64 } else { 0.0 },
            })
        })
        .collect();
    let is_multi_backend = selection.len() > 1;

    // Feature vectors: flatten each unique input JSON once (memoized), keep the
    // features present in every point (rectangular), drop constant ones.
    let mut flat_cache: HashMap<&str, BTreeMap<String, f64>> = HashMap::new();
    for pt in points {
        flat_cache
            .entry(pt.input_json.as_str())
            .or_insert_with(|| flatten_input(&pt.input_json));
    }
    let features = stable_features(points, &flat_cache);

    // Row matrix aligned to `points`, restricted to the chosen features.
    let rows: Vec<Vec<f64>> = points
        .iter()
        .map(|pt| {
            let flat = &flat_cache[pt.input_json.as_str()];
            features.iter().map(|k| flat[k]).collect()
        })
        .collect();

    let (projection, axis_labels, explained, xy) = project(&features, &rows);

    let point_json: Vec<Value> = points
        .iter()
        .zip(xy.iter())
        .map(|(pt, &[x, y])| {
            json!({
                "x": x,
                "y": y,
                "backend_index": pt.backend,
                "backend_name": backend_name(pt.backend),
                "count": pt.count,
            })
        })
        .collect();

    let payload = json!({
        "name": pos.name,
        "kind": pos.kind,
        "candidate_backends": pos.backends,
        "selection": selection_json,
        "projection": projection,
        "axis_labels": axis_labels,
        "explained_variance": explained,
        "points": point_json,
    });
    let report = json!({
        "name": pos.name,
        "kind": pos.kind,
        "candidate_backends": pos.backends,
        "selection": selection_json,
        "raw_executed_slots": total,
        "sampled_points": points.len(),
        "num_features": features.len(),
        "features_used": features,
        "projection": projection,
        "explained_variance": explained,
    });
    (payload, report, is_multi_backend)
}

/// The feature keys to project: those present in *every* point's flattened input
/// (rectangular guarantee), minus constant ones (they add no separation and would
/// be a zero column). Sorted for a stable axis order.
fn stable_features(
    points: &[Point],
    flat_cache: &HashMap<&str, BTreeMap<String, f64>>,
) -> Vec<String> {
    let mut common: Option<BTreeSet<String>> = None;
    for pt in points {
        let keys: BTreeSet<String> = flat_cache[pt.input_json.as_str()].keys().cloned().collect();
        common = Some(match common {
            None => keys,
            Some(prev) => prev.intersection(&keys).cloned().collect(),
        });
    }
    let common = common.unwrap_or_default();
    // Drop constant features (min == max across all points).
    common
        .into_iter()
        .filter(|k| {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for pt in points {
                let v = flat_cache[pt.input_json.as_str()][k];
                lo = lo.min(v);
                hi = hi.max(v);
            }
            hi - lo > 1e-12
        })
        .collect()
}

/// Choose the projection for a position's feature matrix, returning only the
/// *meaningful* coordinates (no cosmetic jitter — the renderer owns de-overlap):
/// - 0 features → `categorical` (no axis carries data; `points` are placeholders),
/// - 1 → `feature_1d`, the feature value on `x` (a true 1-D strip; `y` is `0`),
/// - 2 → `raw_2d`, the two raw features,
/// - ≥3 → `pca` (top-2 principal components), falling back to `categorical` if the
///   projection is undefined (fewer than 2 rows).
///
/// Returns `(projection tag, axis_labels[2], explained_variance | null, points)`.
/// For `feature_1d`/`categorical` the second axis is not a data dimension, so the
/// Python side hides it and adds only a cosmetic spread — a 1-D position is drawn
/// as a strip, never a fake 2-D scatter.
fn project(
    features: &[String],
    rows: &[Vec<f64>],
) -> (&'static str, [String; 2], Value, Vec<[f64; 2]>) {
    let n = rows.len();
    match features.len() {
        0 => (
            "categorical",
            ["(no numeric features)".to_string(), String::new()],
            Value::Null,
            vec![[0.0, 0.0]; n],
        ),
        1 => (
            "feature_1d",
            [features[0].clone(), String::new()],
            Value::Null,
            rows.iter().map(|r| [r[0], 0.0]).collect(),
        ),
        2 => (
            "raw_2d",
            [features[0].clone(), features[1].clone()],
            Value::Null,
            rows.iter().map(|r| [r[0], r[1]]).collect(),
        ),
        _ => match pca_project_2d(rows) {
            Some(p) => (
                "pca",
                ["PC1".to_string(), "PC2".to_string()],
                json!(p.explained_variance),
                p.points,
            ),
            None => (
                "categorical",
                ["(unprojectable)".to_string(), String::new()],
                Value::Null,
                vec![[0.0, 0.0]; n],
            ),
        },
    }
}

/// Flatten one kernel-input JSON into numeric features:
/// - number → its value; bool → 0/1,
/// - object → recurse with a dotted path (`shape.m`),
/// - numeric array → `count`/`sum`/`mean`/`min`/`max` (variable length collapses
///   to fixed features); non-numeric array → just its `count`,
/// - string / null → kept out (non-numeric; metadata only).
fn flatten_input(json: &str) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    if let Ok(v) = serde_json::from_str::<Value>(json) {
        flatten_value(&v, "", &mut out);
    }
    out
}

fn flatten_value(v: &Value, prefix: &str, out: &mut BTreeMap<String, f64>) {
    let key = |p: &str| {
        if p.is_empty() {
            "value".to_string()
        } else {
            p.to_string()
        }
    };
    match v {
        Value::Null | Value::String(_) => {}
        Value::Bool(b) => {
            out.insert(key(prefix), if *b { 1.0 } else { 0.0 });
        }
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                out.insert(key(prefix), f);
            }
        }
        Value::Array(arr) => {
            #[allow(
                clippy::cast_precision_loss,
                reason = "arr.len() is the length of one JSON array feature within a single \
                          sampled kernel input, realistically tiny and far below 2^53"
            )]
            let count = arr.len() as f64;
            out.insert(format!("{prefix}.count"), count);
            if arr.is_empty() {
                return;
            }
            let nums: Vec<f64> = arr.iter().filter_map(Value::as_f64).collect();
            if nums.len() == arr.len() {
                // Flat numeric array (e.g. per-request kv lens) → aggregates.
                insert_aggregates(out, prefix, &nums);
            } else if let Some(cols) = numeric_columns(arr) {
                // Array of equal-length numeric sub-arrays (e.g. attention's
                // `prefill_chunk_pairs = [[prefix, append], ...]`) → aggregate each
                // column across the list, so `.0` is the prefix series and `.1` the
                // append series. For the common ≤1-request step, mean == the value.
                for (j, col) in cols.iter().enumerate() {
                    insert_aggregates(out, &format!("{prefix}.{j}"), col);
                }
            } else if let Some(fields) = numeric_object_fields(arr) {
                // Array of records → aggregate each shared numeric field.
                for (k, col) in &fields {
                    insert_aggregates(out, &format!("{prefix}.{k}"), col);
                }
            }
        }
        Value::Object(map) => {
            for (k, val) in map {
                let child = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten_value(val, &child, out);
            }
        }
    }
}

/// Emit `sum`/`mean`/`min`/`max` of `nums` under `prefix` (the caller owns
/// `.count`). Skips an empty series (no aggregate is defined).
#[allow(
    clippy::cast_precision_loss,
    reason = "nums.len() is the length of one JSON array feature within a single sampled kernel \
              input, realistically tiny and far below 2^53"
)]
fn insert_aggregates(out: &mut BTreeMap<String, f64>, prefix: &str, nums: &[f64]) {
    if nums.is_empty() {
        return;
    }
    let sum: f64 = nums.iter().sum();
    out.insert(format!("{prefix}.sum"), sum);
    out.insert(format!("{prefix}.mean"), sum / nums.len() as f64);
    out.insert(
        format!("{prefix}.min"),
        nums.iter().copied().fold(f64::INFINITY, f64::min),
    );
    out.insert(
        format!("{prefix}.max"),
        nums.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
}

/// If every element of `arr` is a numeric sub-array of the SAME length `L`, return
/// the `L` columns (column `j` = the `j`-th value of every sub-array); else `None`.
/// Turns a `[[prefix, append], ...]` list into a prefix column and an append column.
fn numeric_columns(arr: &[Value]) -> Option<Vec<Vec<f64>>> {
    let width = arr[0].as_array()?.len();
    if width == 0 {
        return None;
    }
    let mut cols = vec![Vec::with_capacity(arr.len()); width];
    for el in arr {
        let sub = el.as_array()?;
        if sub.len() != width {
            return None;
        }
        for (j, v) in sub.iter().enumerate() {
            cols[j].push(v.as_f64()?);
        }
    }
    Some(cols)
}

/// If `arr` is a list of objects, return each key that is numeric in EVERY element
/// paired with its column of values (keys missing/non-numeric anywhere are
/// dropped). `None` if no element is an object. Sorted for a stable feature order.
fn numeric_object_fields(arr: &[Value]) -> Option<Vec<(String, Vec<f64>)>> {
    let first = arr[0].as_object()?;
    let mut out = Vec::new();
    for key in first.keys() {
        let mut col = Vec::with_capacity(arr.len());
        let complete = arr.iter().all(|el| {
            match el
                .as_object()
                .and_then(|o| o.get(key))
                .and_then(Value::as_f64)
            {
                Some(v) => {
                    col.push(v);
                    true
                }
                None => false,
            }
        });
        if complete {
            out.push((key.clone(), col));
        }
    }
    (!out.is_empty()).then_some(out)
}

fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    col(batch, name)?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("`{name}` is not a Utf8 array"))
}

/// Downcast a cost_log `List<Utf8>` column to `(offsets, flat_values)` for
/// offset-walked, allocation-free slot iteration.
fn list_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<(&'a [i32], &'a StringArray)> {
    let list = col(batch, name)?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("`{name}` is not a List array"))?;
    let vals = list
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("`{name}` is not List<Utf8>"))?;
    Ok((list.value_offsets(), vals))
}

/// Downcast a cost_log `List<UInt8>` column to `(offsets, flat_values)`.
fn list_u8<'a>(batch: &'a RecordBatch, name: &str) -> Result<(&'a [i32], &'a UInt8Array)> {
    let list = col(batch, name)?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("`{name}` is not a List array"))?;
    let vals = list
        .values()
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or_else(|| anyhow!("`{name}` is not List<u8>"))?;
    Ok((list.value_offsets(), vals))
}

fn definitions() -> Value {
    json!({
        "scope": "cost_log slots sampled 1-in-sample_stride on iter_id, grouped by tree position",
        "position": "the leaf's manifest `name` (e.g. m.layers.self_attn.qkv_proj); the parallel \
                     branches of a Max node share a name, so they pool into one position",
        "backend": "position-local index into the manifest `backends` candidate list — the backend \
                    best-of-N selected for that slot; 255 = leaf not executed that iteration (dropped)",
        "point": "one deduped (input, backend) observation; `count` = how many sampled slots matched",
        "features": "input JSON flattened to numbers: scalars direct, bool→0/1, nested path a.b, \
                     numeric array→count/sum/mean/min/max; non-numeric kept out; constant features dropped",
        "projection": "1 feature→value axis, 2→raw axes, ≥3→PCA (standardized, top-2 PCs); \
                       explained_variance is the PC1/PC2 variance ratio (null for non-PCA)",
        "sampling": "per position, per-backend even stride bounds points so a rare backend still shows",
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
        "positions": [],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_covers_scalar_bool_nested_and_array() {
        let f =
            flatten_input(r#"{"m": 128, "causal": true, "shape": {"n": 4096}, "lens": [1, 2, 3]}"#);
        assert_eq!(f["m"], 128.0);
        assert_eq!(f["causal"], 1.0);
        assert_eq!(f["shape.n"], 4096.0);
        assert_eq!(f["lens.count"], 3.0);
        assert_eq!(f["lens.sum"], 6.0);
        assert_eq!(f["lens.mean"], 2.0);
        assert_eq!(f["lens.min"], 1.0);
        assert_eq!(f["lens.max"], 3.0);
        // A string field is non-numeric → absent (metadata only).
        let g = flatten_input(r#"{"dtype": "bf16", "k": 5}"#);
        assert!(!g.contains_key("dtype"));
        assert_eq!(g["k"], 5.0);
    }

    #[test]
    fn flatten_aggregates_columnar_pairs_and_records() {
        // Attention prefill: a list of [prefix, append] pairs → per-column
        // aggregates, so the actual prefill shape becomes real features (not just
        // a request count). For a single fresh prefill, mean == the value.
        let f = flatten_input(r#"{"prefill_chunk_pairs": [[0, 140], [16, 60]]}"#);
        assert_eq!(f["prefill_chunk_pairs.count"], 2.0);
        assert_eq!(f["prefill_chunk_pairs.0.max"], 16.0); // prefix column
        assert_eq!(f["prefill_chunk_pairs.1.sum"], 200.0); // append column: 140+60
        assert_eq!(f["prefill_chunk_pairs.1.min"], 60.0);
        // A list of records aggregates each shared numeric field.
        let r = flatten_input(r#"{"experts": [{"id": 1, "load": 8}, {"id": 2, "load": 4}]}"#);
        assert_eq!(r["experts.count"], 2.0);
        assert_eq!(r["experts.load.mean"], 6.0);
        assert_eq!(r["experts.id.max"], 2.0);
    }

    #[test]
    fn stable_features_drops_constant_and_nonshared() {
        let points = vec![
            Point {
                input_json: r#"{"a": 1, "b": 9, "c": 7}"#.into(),
                backend: 0,
                count: 1,
            },
            Point {
                input_json: r#"{"a": 2, "b": 9}"#.into(),
                backend: 1,
                count: 1,
            },
        ];
        let mut cache: HashMap<&str, BTreeMap<String, f64>> = HashMap::new();
        for p in &points {
            cache.insert(p.input_json.as_str(), flatten_input(&p.input_json));
        }
        let feats = stable_features(&points, &cache);
        // `a` varies and is shared → kept; `b` is constant → dropped; `c` is only in
        // one point → not shared → dropped.
        assert_eq!(feats, vec!["a".to_string()]);
    }

    #[test]
    fn even_stride_bounds_length() {
        let v: Vec<u32> = (0..100).collect();
        let s = even_stride(v, 10);
        assert!(s.len() <= 10);
        assert_eq!(s[0], 0); // endpoints preserved
                             // A set already under the cap is returned intact.
        assert_eq!(even_stride(vec![1, 2, 3], 10), vec![1, 2, 3]);
    }

    #[test]
    fn project_dispatches_on_feature_count() {
        // 1 feature → feature_1d with the value on x.
        let (tag, labels, ev, xy) = project(&["m".into()], &[vec![3.0], vec![7.0]]);
        assert_eq!(tag, "feature_1d");
        assert_eq!(labels[0], "m");
        assert!(ev.is_null());
        assert_eq!(xy[0][0], 3.0);
        assert_eq!(xy[1][0], 7.0);
        // 2 features → raw_2d on both axes.
        let (tag, _, _, xy) = project(&["m".into(), "n".into()], &[vec![1.0, 2.0]]);
        assert_eq!(tag, "raw_2d");
        assert_eq!(xy[0], [1.0, 2.0]);
        // ≥3 features with ≥2 rows → pca with an explained-variance pair.
        let rows: Vec<Vec<f64>> = (0..6)
            .map(|i| vec![i as f64, 2.0 * i as f64, -(i as f64)])
            .collect();
        let (tag, labels, ev, xy) = project(&["a".into(), "b".into(), "c".into()], &rows);
        assert_eq!(tag, "pca");
        assert_eq!(labels, ["PC1".to_string(), "PC2".to_string()]);
        assert!(ev.is_array());
        assert_eq!(xy.len(), 6);
    }
}
