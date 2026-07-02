//! `analyze trace` — Perfetto per-kernel timeline export.
//!
//! Reads `cost_log/worker_<pool>_<id>.parquet` (per-iteration leaf timings +
//! captured kernel inputs) and the matching
//! `cost_manifest/worker_<pool>_<id>.json` CostTree structure, then lays each
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

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use datafusion::prelude::SessionContext;

use crate::io::trace_path;
use crate::perfetto::{Annotation, TraceWriter};
use crate::session::{
    col, collect, register_cost_log, register_gpu_cluster, require_columns, value_f32_list,
    value_f64, value_groups, value_str_list, value_string, GroupInput, COST_LOG_TABLE,
    GPU_CLUSTER_TABLE,
};
use place::{critical_pairs_per_iter, slice_pairs_per_iter, Placer};

/// Columns the trace reads from `cost_log` (drift-guarded).
const COLUMNS: &[&str] = &[
    "pool_tag",
    "worker_id",
    "iter_id",
    "batch_id",
    "section",
    "wall_start_ms",
    "total_time_ms",
    "slot_time_ms",
    "slot_input",
    "groups",
];

/// One iteration's row, materialized from the parquet.
struct IterRow {
    pool_tag: String,
    worker_id: u16,
    iter_id: u64,
    batch_id: u64,
    /// Building-block section (`iter` for iter-wise; `attn` / `prologue` /
    /// `pre_attn` / `post_attn` / `post_attn_last` / `epilogue` for AFD). Selects
    /// which sub-manifest interprets `slot_ns`.
    section: String,
    wall_start_ms: f64,
    total_time_ms: f64,
    /// Per-slot leaf duration, pre-rounded to ns (index = manifest slot).
    slot_ns: Vec<i64>,
    /// Per-slot captured kernel input JSON (empty when the run lacked capture).
    slot_input: Vec<String>,
    /// This iter/section's top-level arch input (the `input_section`), one per HP
    /// group — the batch composition fed to the model_arch to cost this row.
    /// Rendered as a JSON annotation on the per-iter slice.
    groups: Vec<GroupInput>,
}

impl IterRow {
    fn manifest_key(&self) -> (String, u16) {
        (self.pool_tag.clone(), self.worker_id)
    }
}

fn worker_label(key: &(String, u16)) -> String {
    format!("{}/{}", key.0, key.1)
}

/// Columns the transfer overlay reads from `gpu_cluster` (drift-guarded).
const NET_COLUMNS: &[&str] = &[
    "net_start_ms",
    "net_end_ms",
    "src_pool_tag",
    "src_worker_id",
    "dst_pool_tag",
    "dst_worker_id",
    "send_gid",
    "recv_gid",
    "bytes",
    "kind",
];

/// One cross-worker transfer, materialized from `gpu_cluster.parquet`. Placed on
/// the **receiving** (`dst_*`) worker's dedicated comm lane; the sender (`src_*`)
/// rides along as a slice annotation.
struct NetRow {
    net_start_ms: f64,
    net_end_ms: f64,
    src_pool_tag: String,
    src_worker_id: u16,
    dst_pool_tag: String,
    dst_worker_id: u16,
    send_gid: u16,
    recv_gid: u16,
    bytes: u64,
    kind: String,
}

impl NetRow {
    /// The receiving worker — the comm lane this transfer is drawn on. Matches the
    /// `cost_log` `(pool_tag, worker_id)` compute-track key.
    fn dst_key(&self) -> (String, u16) {
        (self.dst_pool_tag.clone(), self.dst_worker_id)
    }
}

pub async fn run(
    ctx: &SessionContext,
    log_dir: &Path,
    regions: usize,
    region_ms: f64,
    max_slices: usize,
    expanded: bool,
) -> Result<()> {
    let manifests = crate::io::read_cost_manifests(log_dir)?;

    // Cost log is one parquet per worker under `raw/cost_log/` (a single shared
    // file would race when 32+ decode workers all open it). DataFusion takes
    // the directory and unions every `*.parquet` inside.
    if !register_cost_log(ctx, log_dir).await? {
        bail!("cost_log/ dir not found under {}", log_dir.display());
    }
    require_columns(ctx, COST_LOG_TABLE, COLUMNS).await?;

    // Global time span via a cheap aggregate — the anchors below need it, but
    // scanning every row (tens of millions on AFD) just for min/max is wasteful.
    let span_batches = collect(
        ctx,
        "SELECT MIN(wall_start_ms) AS t0, MAX(wall_start_ms + total_time_ms) AS t1 FROM cost_log",
    )
    .await?;
    let (t0, t1) = match span_batches.first() {
        Some(b) if b.num_rows() > 0 => (value_f64(col(b, "t0")?, 0)?, value_f64(col(b, "t1")?, 0)?),
        _ => bail!("cost_log has no rows"),
    };
    if t0.is_nan() || t1.is_nan() {
        bail!("cost_log has no rows");
    }
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

    // Only rows inside a sample window are ever placed, so push the window union
    // into SQL: keep a row whose `col_name` falls in any `[pos, pos+region_ms)`.
    // This mirrors the in-Rust region filter below (which also keys on the start
    // column), so the placed output is identical — but we materialize thousands of
    // rows, not the full cost_log (dropping peak RSS from tens of GB to MB).
    let window_pred = |col_name: &str| -> String {
        anchors
            .iter()
            .map(|&pos| format!("({col_name} >= {pos} AND {col_name} < {})", pos + region_ms))
            .collect::<Vec<_>>()
            .join(" OR ")
    };

    // Cast the low-cardinality string columns (`pool_tag`, `section`) to VARCHAR in
    // the projection: parquet RLE_DICTIONARY-encodes them (pool_tag is just
    // "prefill" / "decode" repeated millions of times; section likewise), and
    // DataFusion preserves that encoding, surfacing the column as a `DictionaryArray`.
    // `value_string` downcasts to `StringArray`, which fails on dictionaries —
    // flattening at projection sidesteps the per-column dictionary dispatch.
    let other_cols = COLUMNS
        .iter()
        .filter(|c| **c != "pool_tag" && **c != "section")
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, CAST(section AS VARCHAR) AS section, \
         {other_cols} FROM cost_log WHERE {} ORDER BY pool_tag, worker_id, wall_start_ms",
        window_pred("wall_start_ms")
    );
    let batches = collect(ctx, &sql).await?;

    let mut rows: Vec<IterRow> = Vec::new();
    for b in &batches {
        let (pool_tag, wid, iid, bid) = (
            col(b, "pool_tag")?,
            col(b, "worker_id")?,
            col(b, "iter_id")?,
            col(b, "batch_id")?,
        );
        let sec = col(b, "section")?;
        let (ws, tt) = (col(b, "wall_start_ms")?, col(b, "total_time_ms")?);
        let (st, si) = (col(b, "slot_time_ms")?, col(b, "slot_input")?);
        let gr = col(b, "groups")?;
        for r in 0..b.num_rows() {
            let slot_ns = value_f32_list(st, r)?
                .iter()
                .map(|ms| (ms * 1e6).round() as i64)
                .collect();
            rows.push(IterRow {
                pool_tag: value_string(pool_tag, r)?,
                worker_id: value_f64(wid, r)? as u16,
                iter_id: value_f64(iid, r)? as u64,
                batch_id: value_f64(bid, r)? as u64,
                section: value_string(sec, r)?,
                wall_start_ms: value_f64(ws, r)?,
                total_time_ms: value_f64(tt, r)?,
                slot_ns,
                slot_input: value_str_list(si, r)?,
                groups: value_groups(gr, r)?,
            });
        }
    }
    if rows.is_empty() {
        bail!("cost_log has no rows");
    }

    // Optional transfer overlay: read `gpu_cluster` if the run wrote it. Absent on
    // runs that never transfer (unified / single-worker), where the overlay is
    // simply skipped — never an error. The compute span above is left untouched;
    // transfers only get drawn into the windows the compute log already defines.
    let mut net_rows: Vec<NetRow> = Vec::new();
    if register_gpu_cluster(ctx, log_dir).await? {
        require_columns(ctx, GPU_CLUSTER_TABLE, NET_COLUMNS).await?;
        // Flatten the RLE_DICTIONARY string columns (pool tags / kind) to VARCHAR,
        // same reason as the cost_log projection above.
        let sql = format!(
            "SELECT net_start_ms, net_end_ms, \
             CAST(src_pool_tag AS VARCHAR) AS src_pool_tag, src_worker_id, \
             CAST(dst_pool_tag AS VARCHAR) AS dst_pool_tag, dst_worker_id, \
             send_gid, recv_gid, bytes, CAST(kind AS VARCHAR) AS kind \
             FROM gpu_cluster WHERE {} ORDER BY dst_pool_tag, dst_worker_id, net_start_ms",
            window_pred("net_start_ms")
        );
        for b in &collect(ctx, &sql).await? {
            let (ns, ne) = (col(b, "net_start_ms")?, col(b, "net_end_ms")?);
            let (sp, sw) = (col(b, "src_pool_tag")?, col(b, "src_worker_id")?);
            let (dp, dw) = (col(b, "dst_pool_tag")?, col(b, "dst_worker_id")?);
            let (sg, rg) = (col(b, "send_gid")?, col(b, "recv_gid")?);
            let (by, kd) = (col(b, "bytes")?, col(b, "kind")?);
            for r in 0..b.num_rows() {
                net_rows.push(NetRow {
                    net_start_ms: value_f64(ns, r)?,
                    net_end_ms: value_f64(ne, r)?,
                    src_pool_tag: value_string(sp, r)?,
                    src_worker_id: value_f64(sw, r)? as u16,
                    dst_pool_tag: value_string(dp, r)?,
                    dst_worker_id: value_f64(dw, r)? as u16,
                    send_gid: value_f64(sg, r)? as u16,
                    recv_gid: value_f64(rg, r)? as u16,
                    bytes: value_f64(by, r)? as u64,
                    kind: value_string(kd, r)?,
                });
            }
        }
    }

    let mut w = TraceWriter::new();
    // Top-level label track marking each region's real wall-clock position.
    let region_proc = w.process_track(-1, "Regions");
    let region_track = w.thread_track(region_proc, -1, 0, "regions");

    // Pre-create one process+thread track per worker (avoids a borrow conflict
    // with `w` mid-loop, and fixes a stable track order in the output). The roster
    // comes from the cost manifests (every worker writes one), NOT from `rows` — a
    // worker with no row inside any sample window still needs its track so the
    // per-row `worker_tracks[&key]` lookup below never misses.
    let mut worker_keys: Vec<(String, u16)> = manifests.keys().cloned().collect();
    worker_keys.sort_unstable();
    worker_keys.dedup();
    // Which workers RECEIVE a transfer — only these get a comm lane, so a
    // pure-sender (or a run with no transfers) sprouts no empty lane.
    let dst_keys: BTreeSet<(String, u16)> = net_rows.iter().map(NetRow::dst_key).collect();
    let mut worker_tracks: BTreeMap<(String, u16), u64> = BTreeMap::new();
    let mut comm_tracks: BTreeMap<(String, u16), u64> = BTreeMap::new();
    for (idx, key) in worker_keys.iter().enumerate() {
        let pid = idx as i32;
        let label = worker_label(key);
        let proc = w.process_track(pid, &label);
        let thr = w.thread_track(proc, pid, 0, &label);
        worker_tracks.insert(key.clone(), thr);
        // Comm lane: a second thread track (`tid=1`) under the SAME process, so a
        // received transfer renders directly beneath the worker's compute lane —
        // the "same worker row, separate lane" overlay.
        if dst_keys.contains(key) {
            let comm = w.thread_track(proc, pid, 1, &format!("{label} · comm"));
            comm_tracks.insert(key.clone(), comm);
        }
    }

    let mut placed_pairs = 0usize;
    let mut placed_iters = 0usize;
    let mut eligible_iters = 0usize;
    let mut placed_net = 0usize;
    let mut truncated = false;
    // Comm-lane high-water end_ns, mirroring `track_cursor`. Transfers into one
    // recv group serialize (the cluster advances `recv_free`), but a worker with
    // several recv groups can have genuinely overlapping pulls; snap each `begin`
    // forward to the lane's last `end` so slices stay flush siblings rather than
    // nesting (a rare, sub-slice visual nudge).
    let mut net_track_cursor: BTreeMap<u64, i64> = BTreeMap::new();
    // Per-track high-water end_ns. Back-to-back slices (predict's `now += time`
    // layout, or sections within one real task) place each `begin` from the sim
    // clock and each `end` from the placer's leaf-ns sum; the two round
    // independently, so a slice can begin ~1ns before its predecessor's end. That
    // sub-ns overlap makes Perfetto nest the next slice under the previous one (a
    // huge `epilogue`/`lm_head` then appears under a tiny `post_attn_last`). Snap
    // each `begin` to be ≥ the same track's last `end` to keep siblings flush.
    let mut track_cursor: BTreeMap<u64, i64> = BTreeMap::new();

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
            let key = row.manifest_key();
            let doc = manifests
                .get(&key)
                .with_context(|| format!("missing cost manifest for {}", worker_label(&key)))?;
            let manifest = doc.section(&row.section).with_context(|| {
                format!(
                    "cost manifest for {} has no section {:?}",
                    worker_label(&key),
                    row.section
                )
            })?;
            // Cap unit: expanded counts every branch; critical (default) counts
            // only the bottleneck branch each Max collapses to (data-dependent).
            let per_iter_pairs = if expanded {
                slice_pairs_per_iter(manifest)
            } else {
                critical_pairs_per_iter(manifest, &row.slot_ns)
            };
            if placed_pairs + per_iter_pairs > max_slices {
                truncated = true;
                w.end(region_track, ((offset_ms + region_ms) * 1e6).round() as i64);
                break 'regions;
            }
            let track = worker_tracks[&key];
            // Sim-clock begin, snapped forward so it never predates the same track's
            // last end (see `track_cursor`); the shift is ≤1ns rounding noise.
            let base_ns = ((row.wall_start_ms + shift_ms) * 1e6).round() as i64;
            let base_ns = base_ns.max(track_cursor.get(&track).copied().unwrap_or(i64::MIN));

            // Iter-wise rows are one slice per iteration (`iter N`); AFD layer-wise
            // rows are one slice per building block (`iter N · post_attn`).
            let slice_label = if row.section == "iter" {
                format!("iter {}", row.iter_id)
            } else {
                format!("iter {} · {}", row.iter_id, row.section)
            };
            let mut iter_anns = vec![
                Annotation::uint("iter_id", row.iter_id),
                Annotation::uint("batch_id", row.batch_id),
                Annotation::str("section", &row.section),
                Annotation::dbl("total_ms", row.total_time_ms),
            ];
            // Top-level arch input (the `input_section`) as JSON, mirroring how
            // leaf slices carry their per-kernel `input`. Skipped when the run
            // logged no groups (model without a compiled CostTree).
            if !row.groups.is_empty() {
                if let Ok(json) = serde_json::to_string(&row.groups) {
                    iter_anns.push(Annotation::str("groups", &json));
                }
            }
            w.begin(track, base_ns, &slice_label, &iter_anns);
            let placer = Placer::new(manifest, &row.slot_ns, &row.slot_input, expanded);
            let dur = placer.place_root(&mut w, track, base_ns);
            let end_ns = base_ns + dur;
            w.end(track, end_ns);
            track_cursor.insert(track, end_ns);

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

        // Overlay this window's transfers onto their receiving worker's comm lane,
        // using the SAME `shift_ms` as compute so they share the compressed axis.
        // Not capped by `max_slices` (that budgets the compute slice tree);
        // transfers are far fewer, one per handoff.
        for nr in net_rows
            .iter()
            .filter(|n| n.net_start_ms >= lo && n.net_start_ms < hi)
        {
            let Some(&track) = comm_tracks.get(&nr.dst_key()) else {
                // Receiver has no compute track (should not happen — every dst is a
                // cost_log worker). Skip rather than invent a lane.
                continue;
            };
            let begin_ns = ((nr.net_start_ms + shift_ms) * 1e6).round() as i64;
            let begin_ns = begin_ns.max(net_track_cursor.get(&track).copied().unwrap_or(i64::MIN));
            let end_ns = (((nr.net_end_ms + shift_ms) * 1e6).round() as i64).max(begin_ns);
            w.begin(
                track,
                begin_ns,
                &nr.kind,
                &[
                    Annotation::str(
                        "src",
                        worker_label(&(nr.src_pool_tag.clone(), nr.src_worker_id)),
                    ),
                    Annotation::uint("send_gid", nr.send_gid as u64),
                    Annotation::uint("recv_gid", nr.recv_gid as u64),
                    Annotation::uint("bytes", nr.bytes),
                    Annotation::dbl("net_start_ms", nr.net_start_ms),
                    Annotation::dbl("net_end_ms", nr.net_end_ms),
                ],
            );
            w.end(track, end_ns);
            net_track_cursor.insert(track, end_ns);
            placed_net += 1;
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
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let bytes = w.into_gzip()?;
    fs::write(&out, &bytes).with_context(|| format!("write {}", out.display()))?;
    println!(
        "wrote {} ({} regions, {} iterations, {} slice pairs, {} transfers)",
        out.display(),
        anchors.len(),
        placed_iters,
        placed_pairs,
        placed_net
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
            &[
                Annotation::str("input", "{\"m\":91}"),
                Annotation::dbl("dur_ms", 0.02),
            ],
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
