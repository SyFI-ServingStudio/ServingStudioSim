//! `analyze trace` — Perfetto per-kernel timeline export.
//!
//! Reads `cost_log.parquet` (per-iteration leaf timings + captured kernel
//! inputs) and `cost_manifest.json` (the cost-tree structure), then lays each
//! iteration out as a nested slice tree (iter → layers → kernels) via
//! [`place::Placer`] onto a [`TraceWriter`], and writes
//! `traces/<prefix>.pftrace.gz` for ui.perfetto.dev.
//!
//! Sampling (the "easy" of ref's two modes — uniform, not k-means clustering):
//! pick `regions` contiguous windows of `region_ms` each, evenly spaced across
//! the run, and concatenate them on a compressed time axis with small gaps
//! (like ref's `write_cluster_trace` layout). A top-level "Regions" track labels
//! each window with its real wall-clock position. `max_slices` caps the output
//! at iteration granularity (never truncating mid-iteration).

pub mod manifest;
mod place;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use datafusion::prelude::SessionContext;

use crate::io::{resolve_artifact_path, trace_path};
use crate::perfetto::{Annotation, TraceWriter};
use crate::session::{
    col, collect, register_if_exists, require_columns, value_f64, value_f32_list, value_str_list,
};
use place::{slice_pairs_per_iter, Placer};

/// Columns the trace reads from `cost_log` (drift-guarded).
const COLUMNS: &[&str] = &[
    "worker_id",
    "iter_id",
    "batch_id",
    "wall_start_ms",
    "total_time_ms",
    "slot_time_ms",
    "slot_input",
];

/// One iteration's row, materialized from the parquet.
struct IterRow {
    worker_id: i64,
    iter_id: u64,
    batch_id: u64,
    wall_start_ms: f64,
    total_time_ms: f64,
    /// Per-slot leaf duration, pre-rounded to ns (index = manifest slot).
    slot_ns: Vec<i64>,
    /// Per-slot captured kernel input JSON (empty when the run lacked capture).
    slot_input: Vec<String>,
}

pub async fn run(
    ctx: &SessionContext,
    log_dir: &Path,
    regions: usize,
    region_ms: f64,
    max_slices: usize,
) -> Result<()> {
    let manifest = crate::io::read_manifest(log_dir)?;
    let per_iter_pairs = slice_pairs_per_iter(&manifest);

    let path = resolve_artifact_path(log_dir, "cost_log.parquet");
    if !register_if_exists(ctx, "cost_log", path).await? {
        bail!("cost_log.parquet not found under {}", log_dir.display());
    }
    require_columns(ctx, "cost_log", COLUMNS).await?;

    let sql = format!(
        "SELECT {} FROM cost_log ORDER BY worker_id, wall_start_ms",
        COLUMNS.join(", ")
    );
    let batches = collect(ctx, &sql).await?;

    let mut rows: Vec<IterRow> = Vec::new();
    for b in &batches {
        let (wid, iid, bid) = (col(b, "worker_id")?, col(b, "iter_id")?, col(b, "batch_id")?);
        let (ws, tt) = (col(b, "wall_start_ms")?, col(b, "total_time_ms")?);
        let (st, si) = (col(b, "slot_time_ms")?, col(b, "slot_input")?);
        for r in 0..b.num_rows() {
            let slot_ns = value_f32_list(st, r)?
                .iter()
                .map(|ms| (ms * 1e6).round() as i64)
                .collect();
            rows.push(IterRow {
                worker_id: value_f64(wid, r)? as i64,
                iter_id: value_f64(iid, r)? as u64,
                batch_id: value_f64(bid, r)? as u64,
                wall_start_ms: value_f64(ws, r)?,
                total_time_ms: value_f64(tt, r)?,
                slot_ns,
                slot_input: value_str_list(si, r)?,
            });
        }
    }
    if rows.is_empty() {
        bail!("cost_log has no rows");
    }

    // Run span across all workers.
    let t0 = rows.iter().map(|r| r.wall_start_ms).fold(f64::INFINITY, f64::min);
    let t1 = rows
        .iter()
        .map(|r| r.wall_start_ms + r.total_time_ms)
        .fold(f64::NEG_INFINITY, f64::max);
    let span = t1 - t0;

    // Evenly-spaced region anchors. A run shorter than one region collapses to a
    // single window covering the whole thing.
    let regions = regions.max(1);
    let region_ms = region_ms.max(0.0);
    let anchors: Vec<f64> = if span <= region_ms || regions == 1 {
        vec![t0]
    } else {
        (0..regions)
            .map(|i| t0 + (i as f64) * (span - region_ms) / ((regions - 1) as f64))
            .collect()
    };
    let gap_ms = (0.05 * region_ms).clamp(1.0, 10.0);

    let mut w = TraceWriter::new();
    // Top-level label track marking each region's real wall-clock position.
    let region_proc = w.process_track(-1, "Regions");
    let region_track = w.thread_track(region_proc, -1, 0, "regions");

    // Pre-create one process+thread track per worker (avoids a borrow conflict
    // with `w` mid-loop, and fixes a stable track order in the output).
    let mut worker_ids: Vec<i64> = rows.iter().map(|r| r.worker_id).collect();
    worker_ids.sort_unstable();
    worker_ids.dedup();
    let mut worker_tracks: BTreeMap<i64, u64> = BTreeMap::new();
    for &wid in &worker_ids {
        let pid = wid as i32;
        let proc = w.process_track(pid, &format!("worker {wid}"));
        let thr = w.thread_track(proc, pid, 0, &format!("worker {wid}"));
        worker_tracks.insert(wid, thr);
    }

    let mut placed_pairs = 0usize;
    let mut placed_iters = 0usize;
    let mut eligible_iters = 0usize;
    let mut truncated = false;

    'regions: for (i, &pos) in anchors.iter().enumerate() {
        let (lo, hi) = (pos, pos + region_ms);
        let offset_ms = (i as f64) * (region_ms + gap_ms);
        let shift_ms = offset_ms - pos;

        let region_rows: Vec<&IterRow> = rows
            .iter()
            .filter(|r| r.wall_start_ms >= lo && r.wall_start_ms < hi)
            .collect();
        eligible_iters += region_rows.len();

        w.begin(
            region_track,
            (offset_ms * 1e6).round() as i64,
            &format!("region {i} @ {pos:.1}ms"),
            &[
                Annotation::dbl("real_start_ms", lo),
                Annotation::dbl("real_end_ms", hi),
                Annotation::int("iters", region_rows.len() as i64),
            ],
        );

        for row in region_rows {
            if placed_pairs + per_iter_pairs > max_slices {
                truncated = true;
                w.end(region_track, ((offset_ms + region_ms) * 1e6).round() as i64);
                break 'regions;
            }
            let track = worker_tracks[&row.worker_id];
            let base_ns = ((row.wall_start_ms + shift_ms) * 1e6).round() as i64;

            w.begin(
                track,
                base_ns,
                &format!("iter {}", row.iter_id),
                &[
                    Annotation::uint("iter_id", row.iter_id),
                    Annotation::uint("batch_id", row.batch_id),
                    Annotation::dbl("total_ms", row.total_time_ms),
                ],
            );
            let placer = Placer::new(&manifest, &row.slot_ns, &row.slot_input);
            let dur = placer.place_root(&mut w, track, base_ns);
            w.end(track, base_ns + dur);

            // Drift guard: the laid-out tree must reproduce total_time_ms up to
            // per-leaf ns rounding (each of per_iter_pairs rounds at most 1 ns).
            let expected = (row.total_time_ms * 1e6).round() as i64;
            debug_assert!(
                (dur - expected).unsigned_abs() <= per_iter_pairs as u64,
                "placed iter dur {dur}ns vs total_time {expected}ns drifted > {per_iter_pairs}ns"
            );

            placed_pairs += per_iter_pairs;
            placed_iters += 1;
        }

        w.end(region_track, ((offset_ms + region_ms) * 1e6).round() as i64);
    }

    if truncated {
        let omitted = eligible_iters.saturating_sub(placed_iters);
        w.instant(
            region_track,
            0,
            &format!("truncated: {omitted} iterations omitted (--max-slices {max_slices})"),
            &[],
        );
        eprintln!("[trace] hit --max-slices {max_slices}; omitted {omitted} iterations");
    }

    let prefix = log_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("trace");
    let out = trace_path(log_dir, prefix);
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let bytes = w.into_gzip()?;
    fs::write(&out, &bytes).with_context(|| format!("write {}", out.display()))?;
    println!(
        "wrote {} ({} regions, {} iterations, {} slice pairs)",
        out.display(),
        anchors.len(),
        placed_iters,
        placed_pairs
    );
    Ok(())
}

#[cfg(test)]
mod proto_smoke {
    use crate::perfetto::{Annotation, TraceWriter};

    /// Build a tiny nested-slice trace and confirm it gzips to non-empty bytes.
    #[test]
    fn writer_emits_gzipped_trace() {
        let mut w = TraceWriter::new();
        let proc = w.process_track(0, "worker 0");
        let thr = w.thread_track(proc, 0, 0, "nest");
        w.begin(thr, 0, "iter 1", &[Annotation::int("iter_id", 1)]);
        w.begin(thr, 100, "layer 0", &[]);
        w.begin(
            thr,
            100,
            "qkv_proj",
            &[Annotation::str("input", "{\"m\":91}"), Annotation::dbl("dur_ms", 0.02)],
        );
        w.end(thr, 120);
        w.end(thr, 120);
        w.end(thr, 200);
        let bytes = w.into_gzip().expect("gzip");
        assert!(bytes.len() > 20, "expected non-trivial gzip output");
        // gzip magic — decoded against the authoritative perfetto proto during dev.
        assert_eq!(&bytes[..2], &[0x1f, 0x8b]);
    }
}
