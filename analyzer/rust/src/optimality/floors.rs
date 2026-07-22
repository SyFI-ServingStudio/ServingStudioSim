//! Labeler-derived necessary-work floors — the two global lower bounds that sit
//! BELOW the R5 `hardware_limit` green. R5 is a roofline of the sim's *actual* per
//! -kernel work (weights re-loaded every iteration, activation I/O, sim's attention
//! approximation), so it is not the true floor. Here we aggregate each level's
//! workload (reusing the `workload-conservation` subject's `groups` parser — never
//! the cost tree's `slot_flops`/`slot_bytes`, to stay independent) and hand it to the
//! Python `model.work` labeler, which returns two roofline floors per level in GPU·s:
//!
//! - **necessary** — global roofline `max(ΣFLOPs/peak, Σbytes/bw)` of the run's whole
//!   token workload as one fused mega-forward (weights counted once): the loosest,
//!   truly irreducible floor.
//! - **segmented** — `Σ_seg max(compute, memory)`: per-op serial bound (≥ necessary).
//!
//! The labeler is a subprocess (`uv run python -m model.work.floors`); any failure
//! degrades gracefully to the plain 6-rung ladder (the caller records a caveat).

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use datafusion::prelude::SessionContext;
use serde_json::{json, Map, Value};

use crate::conservation::workload::{collect_workload_by_worker, WorkloadTotals};
use crate::kernel_query::repo_root;

/// One level's two labeler floors, in GPU·seconds. Ordered `necessary ≤ segmented`.
#[derive(Clone, Copy, Default)]
pub(super) struct Floors {
    pub(super) necessary: f64,
    pub(super) segmented: f64,
}

/// Per-level floors keyed exactly like `levels.rs`'s level `key`:
/// `"cluster"` | `<pool_tag>` | `"<pool_tag>/<worker_id>"`.
pub(super) type FloorsByLevel = HashMap<String, Floors>;

/// Aggregate the run's workload per level, hand it to the labeler, and return the two
/// floors per level. Errors (SQL, missing labeler, non-zero exit, bad JSON) propagate
/// so the caller can turn them into a caveat + graceful degrade — they never abort the
/// optimality subject.
pub(super) async fn compute_floors(
    ctx: &SessionContext,
    log_dir: &Path,
) -> Result<FloorsByLevel> {
    let by_worker = collect_workload_by_worker(ctx).await?;
    if by_worker.is_empty() {
        return Err(anyhow!("no cost_log workload rows to aggregate"));
    }
    let levels = rollup_levels(&by_worker);
    run_labeler(log_dir, &levels)
}

/// Roll the per-worker totals up into the three level granularities the waterfall
/// draws. All `WorkloadTotals` fields are additive, so pool = Σ its workers and
/// cluster = Σ all — the roofline (a max) is applied per level by the labeler.
fn rollup_levels(
    by_worker: &HashMap<(String, u16), WorkloadTotals>,
) -> HashMap<String, WorkloadTotals> {
    let mut levels: HashMap<String, WorkloadTotals> = HashMap::new();
    for ((pool_tag, worker_id), totals) in by_worker {
        levels.entry("cluster".to_string()).or_default().add(totals);
        levels.entry(pool_tag.clone()).or_default().add(totals);
        levels
            .entry(format!("{pool_tag}/{worker_id}"))
            .or_default()
            .add(totals);
    }
    levels
}

fn run_labeler(
    log_dir: &Path,
    levels: &HashMap<String, WorkloadTotals>,
) -> Result<FloorsByLevel> {
    let root = repo_root().context("repo root not found for the labeler subprocess")?;
    let request = build_request(levels);
    let log_dir_abs: PathBuf = if log_dir.is_absolute() {
        log_dir.to_path_buf()
    } else {
        root.join(log_dir)
    };

    let mut child = Command::new("uv")
        .args(["run", "python", "-m", "model.work.floors"])
        .arg(&log_dir_abs)
        .current_dir(&root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn `uv run python -m model.work.floors`")?;
    child
        .stdin
        .take()
        .context("labeler subprocess stdin unavailable")?
        .write_all(&serde_json::to_vec(&request)?)?; // dropping the handle sends EOF
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(anyhow!(
            "labeler floors exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    parse_response(&output.stdout)
}

fn build_request(levels: &HashMap<String, WorkloadTotals>) -> Value {
    let mut level_json = Map::new();
    for (key, totals) in levels {
        level_json.insert(
            key.clone(),
            json!({
                "matmul_tokens": totals.matmul_tokens,
                "prefill_tokens": totals.prefill_tokens,
                "decode_passes": totals.decode_passes,
                "prefill_pairs": totals.prefill_pairs,
                "prefill_cached": totals.prefill_cached,
                "decode_kv": totals.decode_kv,
                "prefill_requests": totals.prefill_requests,
            }),
        );
    }
    json!({ "levels": Value::Object(level_json) })
}

fn parse_response(stdout: &[u8]) -> Result<FloorsByLevel> {
    let parsed: Value = serde_json::from_slice(stdout).context("parse labeler stdout as JSON")?;
    let level_map = parsed
        .get("levels")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("labeler output missing `levels` object"))?;
    let mut floors = FloorsByLevel::new();
    for (key, value) in level_map {
        floors.insert(
            key.clone(),
            Floors {
                necessary: value.get("necessary").and_then(Value::as_f64).unwrap_or(0.0),
                segmented: value.get("segmented").and_then(Value::as_f64).unwrap_or(0.0),
            },
        );
    }
    Ok(floors)
}
