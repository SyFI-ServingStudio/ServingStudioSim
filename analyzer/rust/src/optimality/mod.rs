//! `optimality` subject — how far a run is from the optimal use of its GPUs.
//!
//! Optimal = the minimal GPU·seconds to complete the same work with **no idle**,
//! **perfect load balancing**, and **kernels at their best batching / rate**. Not
//! one optimal but a **ladder of increasingly-idealized lower bounds**, so each
//! successive gap is an attributable source of sub-optimality. All values are in
//! unit **GPU·seconds** (= wall time × the worker's physical GPU count), computed
//! by re-folding each worker's per-row CostTree manifest with a different leaf-rate
//! / structure substitution per rung:
//!
//! | Rung | leaf value / structure          | gap vs previous = cause         |
//! |------|---------------------------------|---------------------------------|
//! | R0 Real            | `span × G`         | — (GPU·s actually held)         |
//! | R1 Busy            | `Σ total_time × G` | **idle** (scheduler gaps)       |
//! | R2 Balanced        | real times, Max→mean | **imbalance** (DP/EP straggler)|
//! | R3 per-config best | work / grid-peak rate | **batching** (small-batch loss)|
//! | R4 ignore network  | R3, comm leaves→0  | **communication**               |
//! | R5 hardware limit  | work / spec peak   | **profiled↔hardware** (maturity)|
//!
//! Buckets telescope and sum exactly back to Real, so the output is an additive
//! stacked **waterfall** `[idle | imbalance | batching | communication | hw-gap |
//! hw-optimal]` rendered at five levels (cluster / pool / worker / iteration —
//! idle 0 by construction / per-kernel).
//!
//! Cost model. `G_worker` is read from `run_meta` (`workers[].gpu_ids.len()`),
//! never inferred from tp×dp×ep. R0/R1 are exact SQL sums over every row; R2..R5
//! fold a **stride-sampled** set of rows (a location's rates are near-constant
//! across iterations) and are anchored to the exact R1 by their sampled ratio, so
//! the ladder stays monotone and the buckets stay exact. The mean-mode fold is
//! linear, so each rung factors into a precomputed per-leaf weight `α` times that
//! leaf's value — one dot product per row, and the same `α` gives the additive
//! per-kernel attribution for the kernel-level bars.

mod grid_peaks;
mod spec;

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{anyhow, Result};
use arrow_array::{Array, Float32Array, ListArray, RecordBatch, StringArray};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{read_cost_manifests, read_run_meta, read_worker_gpu_counts, SCHEMA_VERSION};
use crate::kernel_query::repo_root;
use crate::session::{col, collect, register_cost_log, require_columns, value_f64, COST_LOG_TABLE};
use crate::trace::manifest::{fold_mean, ManifestDoc};

use grid_peaks::Peaks;
use spec::GpuSpec;

/// cost_log columns this subject depends on (drift guard).
const COST_COLS: &[&str] = &[
    "pool_tag",
    "worker_id",
    "section",
    "iter_id",
    "wall_start_ms",
    "total_time_ms",
    "slot_time_ms",
    "slot_flops",
    "slot_bytes",
];

/// Iteration sampling bounds for the R2..R5 fold (mirrors `kernel-throughput`).
const MAX_STRIDE: u64 = 50;
const TARGET_SAMPLED_ITERS: u64 = 80;

/// Top-N kernels drawn as individual bars at the kernel level; the rest fold into
/// an `other` bar so the figure stays legible on a many-location deployment.
const TOP_KERNELS: usize = 16;

/// Rung index into the per-unit `[f64; 6]` GPU·ms accumulators (R0..R5).
const R0: usize = 0;
const R1: usize = 1;
const R2: usize = 2;
const R3: usize = 3;
const R4: usize = 4;
const R5: usize = 5;

/// One cost-tree location (leaf identity) pooled across workers, keyed by the
/// manifest `name` — exactly like `kernel-throughput`, so `Max` siblings and DP
/// replicas of a location pool together.
struct Loc {
    name: String,
    kind: String,
    is_comm: bool,
}

/// Per `(pool_tag, worker_id, section)` structural metadata, precomputed once so
/// the hot row loop is array indexing: the mean-fold weight `α` per slot, the
/// slot→location map, and each slot's rate ceilings.
struct SectionMeta {
    /// Mean-mode fold weight per slot (`Σ` of leaf weights if a slot recurs).
    alpha: Vec<f64>,
    loc_id: Vec<u32>,
    is_comm: Vec<bool>,
    /// Grid-peak ceilings for R3 (`0` = no sidecar entry → that leaf's R3 = R2).
    peak_tflops: Vec<f64>,
    peak_gbps: Vec<f64>,
    /// Hardware spec compute peak for R5 by the slot's dtype (`0` = no spec).
    hw_tflops: Vec<f64>,
    /// Whether this leaf's logged `bytes` are physical HBM traffic — false when
    /// its grid `peak_gbps` exceeds the GPU's HBM peak (e.g. grouped_gemm counts
    /// logical operand bytes, not HBM movement). Gates the R5 memory roofline: a
    /// non-physical byte count would otherwise inflate `τ_hw` and erase the leaf's
    /// hardware-gap. R3 is unaffected (its huge peak_gbps makes the term negligible).
    mem_physical: Vec<bool>,
}

/// Per-worker exact totals + sampled fold accumulators.
struct WorkerAgg {
    pool_tag: String,
    worker_id: u16,
    g: f64,
    /// Exact `Σ total_time_ms` and wall span (ms) over ALL rows.
    busy_ms: f64,
    span_ms: f64,
    /// Sampled `Σ total_time_ms` and the sampled mean-fold sums for R2..R5 (ms).
    samp_busy_ms: f64,
    samp: [f64; 4],
}

pub async fn run_optimality(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    if !register_cost_log(ctx, log_dir).await? {
        let reason = "cost_log/ dir not found";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    require_columns(ctx, COST_LOG_TABLE, COST_COLS).await?;
    let manifests = match read_cost_manifests(log_dir) {
        Ok(m) => m,
        Err(e) => {
            let reason = format!("cost_manifest/ unreadable ({e:#}); needed to fold the ladder");
            return Ok((unavailable(log_dir, &reason), unavailable_payload(log_dir, &reason)));
        }
    };

    let mut caveats: Vec<String> = Vec::new();

    // G_worker from run_meta — READ, never inferred. Absent → single-GPU degrade.
    let gpu_counts = read_worker_gpu_counts(log_dir);
    let gpu_counts_available = gpu_counts.is_some();
    if !gpu_counts_available {
        caveats.push(
            "run_meta worker GPU counts unavailable; treating every worker as 1 GPU \
             (worker/pool/cluster GPU·s are not physical)"
                .to_string(),
        );
    }
    let g_of: HashMap<(String, u16), f64> = gpu_counts
        .unwrap_or_default()
        .into_iter()
        .map(|(tag, wid, g)| ((tag, wid), g.max(1) as f64))
        .collect();

    // Hardware roofline (R5) + grid-peak ceilings (R3).
    let (_num_gpus, gpu_name) = read_run_meta(log_dir).unwrap_or((1, String::new()));
    let root = repo_root().ok();
    let gpu_spec = root
        .as_deref()
        .and_then(|r| spec::load_gpu_spec(r, &gpu_name));
    let (gpu_spec_matched, hw): (Option<String>, GpuSpec) = match gpu_spec {
        Some((name, s)) => (Some(name), s),
        None => {
            caveats.push(format!(
                "no gpu/spec.json entry for gpu_name {gpu_name:?}; R5 hardware limit \
                 collapses onto R4 (no hardware-gap bucket) — add an alias"
            ));
            (None, GpuSpec::default())
        }
    };
    let hw_bw = hw.mem_bandwidth_gbps;

    let peaks = grid_peaks::load_or_generate(log_dir, &manifests);
    if peaks.is_empty() {
        caveats.push(format!(
            "grid-peaks sidecar {}: batching headroom not computed (R3 = R2)",
            peaks.source
        ));
    }

    // Intern locations + precompute per-section metadata.
    let (locs, section_meta) = build_section_meta(&manifests, &peaks, &hw);

    // Exact per-worker busy + span (all rows).
    let mut workers = exact_worker_totals(ctx, &g_of).await?;
    if workers.is_empty() {
        let reason = "cost_log has no (pool_tag, worker_id) rows";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    // A cost_log worker with no run_meta GPU count falls back to G=1 (see
    // `exact_worker_totals`), which understates its GPU·s. On a v4 roster this never
    // happens; on a pre-v4 log a non-KV worker (null `pool_tag`) is missing from the
    // tagged roster — surface it rather than silently under-counting.
    if gpu_counts_available {
        let missing = workers
            .iter()
            .filter(|w| !g_of.contains_key(&(w.pool_tag.clone(), w.worker_id)))
            .count();
        if missing > 0 {
            caveats.push(format!(
                "{missing} worker(s) in cost_log are absent from run_meta's tagged roster \
                 (pre-v4 run_meta with a null non-KV pool_tag?); their GPU·s use G=1 and are \
                 understated — re-simulate to v4 for exact GPU counts"
            ));
        }
    }
    let widx: HashMap<(String, u16), usize> = workers
        .iter()
        .enumerate()
        .map(|(i, w)| ((w.pool_tag.clone(), w.worker_id), i))
        .collect();

    // Sampled fold for R2..R5 + per-(location, worker) raw contributions.
    let stride = choose_stride(ctx).await?;
    let mut loc_acc: HashMap<(u32, usize), [f64; 4]> = HashMap::new();
    let sampled_rows = accumulate_fold(
        ctx,
        stride,
        hw_bw,
        &section_meta,
        &widx,
        &mut workers,
        &mut loc_acc,
    )
    .await?;

    // ---- Assemble rungs (GPU·ms) per worker, then pool + cluster. ----
    let mut cluster = [0.0f64; 6];
    let mut pools: HashMap<String, [f64; 6]> = HashMap::new();
    let mut worker_rungs: Vec<(String, u16, [f64; 6])> = Vec::new();
    // Anchor factor per worker (GPU·ms per sampled fold-ms): folds in G and the
    // exact/sampled busy upscale, so per-worker rungs and per-kernel sums agree.
    let mut anchor: Vec<f64> = vec![0.0; workers.len()];
    for (i, w) in workers.iter().enumerate() {
        let mut r = [0.0f64; 6];
        r[R0] = w.span_ms * w.g;
        r[R1] = w.busy_ms * w.g;
        if w.samp_busy_ms > 0.0 {
            let a = w.busy_ms * w.g / w.samp_busy_ms;
            anchor[i] = a;
            r[R2] = a * w.samp[0];
            r[R3] = a * w.samp[1];
            r[R4] = a * w.samp[2];
            r[R5] = a * w.samp[3];
        } else {
            // No sampled rows for this worker: leave R2..R5 = R1 (all lower buckets
            // 0) rather than 0 (which would over-attribute to imbalance).
            r[R2] = r[R1];
            r[R3] = r[R1];
            r[R4] = r[R1];
            r[R5] = r[R1];
        }
        // Enforce monotonicity defensively (sampling noise / clamp interplay).
        for k in 1..6 {
            r[k] = r[k].min(r[k - 1]).max(0.0);
        }
        for k in 0..6 {
            cluster[k] += r[k];
            *pools.entry(w.pool_tag.clone()).or_insert([0.0; 6]).get_mut(k).unwrap() += r[k];
        }
        worker_rungs.push((w.pool_tag.clone(), w.worker_id, r));
    }

    // ---- Per-kernel breakdown (GPU·ms), anchored & summed across workers. ----
    let mut kernel_rk: Vec<[f64; 4]> = vec![[0.0; 4]; locs.len()]; // R2..R5 per location
    for ((loc_id, wi), raw) in &loc_acc {
        let a = anchor[*wi];
        let acc = &mut kernel_rk[*loc_id as usize];
        for k in 0..4 {
            acc[k] += raw[k] * a;
        }
    }

    let headline_ratio = ratio(cluster[R5], cluster[R0]);

    // ---- Build levels (cluster, pools, workers, iteration). ----
    let mut levels: Vec<Value> = Vec::new();
    levels.push(level_json("cluster", "cluster", "Cluster", &cluster, false));
    let mut pool_names: Vec<&String> = pools.keys().collect();
    pool_names.sort();
    for name in &pool_names {
        let r = &pools[*name];
        levels.push(level_json("pool", name, &format!("Pool {name}"), r, false));
    }
    worker_rungs.sort_by(|a, b| (a.0.as_str(), a.1).cmp(&(b.0.as_str(), b.1)));
    for (tag, wid, r) in &worker_rungs {
        let key = format!("{tag}/{wid}");
        levels.push(level_json("worker", &key, &key, r, false));
    }
    // Iteration level: the cluster waterfall with idle forced to 0 (idle is a
    // between-iteration gap, none within an iteration), so the bar tops at Busy.
    levels.push(level_json("iteration", "iteration", "Iteration", &cluster, true));

    let kernels = kernel_json(&locs, &kernel_rk);

    let meta = json!({
        "log_dir": log_dir.display().to_string(),
        "gpu_name": gpu_name,
        "gpu_spec_matched": gpu_spec_matched,
        "peaks_source": peaks.source,
        "sample_stride": stride,
        "sampled_rows": sampled_rows,
        "gpu_counts_available": gpu_counts_available,
        "num_workers": workers.len(),
        "num_locations": locs.len(),
        "caveats": caveats,
    });

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "available": true,
        "meta": meta,
        "optimality_ratio": headline_ratio,
        "unit": "gpu_seconds",
        "cluster": rung_report(&cluster),
        "pools": pool_names
            .iter()
            .map(|n| json!({"pool": n, "rungs": rung_report(&pools[*n])}))
            .collect::<Vec<_>>(),
        "worst_batching_kernels": worst_batching(&locs, &kernel_rk),
        "definitions": definitions(),
    });

    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "available": true,
        "meta": meta,
        "unit": "gpu_seconds",
        "optimality_ratio": headline_ratio,
        "bucket_keys": BUCKET_KEYS,
        "rung_keys": RUNG_KEYS,
        "levels": levels,
        "kernels": kernels,
        "definitions": definitions(),
    });

    Ok((report, payload))
}

/// Waterfall segment order (top of the Real bar → the irreducible floor).
const BUCKET_KEYS: [&str; 6] = [
    "idle",
    "imbalance",
    "batching",
    "communication",
    "hardware_gap",
    "hardware_optimal",
];
const RUNG_KEYS: [&str; 6] = [
    "real",
    "busy",
    "balanced",
    "per_config_best",
    "ignore_network",
    "hardware_limit",
];

/// Intern every manifest leaf into a global location, and precompute each
/// `(pool_tag, worker_id, section)`'s fold weights + per-slot rate ceilings.
fn build_section_meta(
    manifests: &BTreeMap<(String, u16), ManifestDoc>,
    peaks: &Peaks,
    hw: &GpuSpec,
) -> (Vec<Loc>, HashMap<(String, u16, String), SectionMeta>) {
    let mut loc_id: HashMap<String, u32> = HashMap::new();
    let mut locs: Vec<Loc> = Vec::new();
    let mut section_meta = HashMap::new();

    for ((pool, worker), doc) in manifests {
        for msec in &doc.sections {
            let m = &msec.manifest;
            let n = m.slots.len();
            let mut alpha = vec![0.0; n];
            // Root is node 0 (BFS layout: parent precedes children).
            if !m.nodes.is_empty() {
                fold_mean(m, 0, 1.0, &mut |slot, w| {
                    if let Some(a) = alpha.get_mut(slot) {
                        *a += w;
                    }
                });
            }

            let mut ids = Vec::with_capacity(n);
            let mut comm_slots = Vec::with_capacity(n);
            let mut peak_tflops = Vec::with_capacity(n);
            let mut peak_gbps = Vec::with_capacity(n);
            let mut hw_tflops = Vec::with_capacity(n);
            let mut mem_physical = Vec::with_capacity(n);
            for leaf in &m.slots {
                let comm = is_comm(&leaf.kind);
                let id = *loc_id.entry(leaf.name.clone()).or_insert_with(|| {
                    let id = locs.len() as u32;
                    locs.push(Loc {
                        name: leaf.name.clone(),
                        kind: leaf.kind.clone(),
                        is_comm: comm,
                    });
                    id
                });
                ids.push(id);
                comm_slots.push(comm);
                let peak = peaks.get(&leaf.kind, &leaf.kernel_config).unwrap_or_default();
                peak_tflops.push(peak.tflops);
                peak_gbps.push(peak.gbps);
                let dtype = compute_dtype(&leaf.kernel_config);
                hw_tflops.push(if comm { 0.0 } else { hw.peak_tflops(&dtype) });
                // Trust `bytes` as HBM traffic unless the grid says the leaf
                // achieved a bandwidth above the GPU's physical HBM peak.
                let bw = hw.mem_bandwidth_gbps;
                mem_physical.push(bw > 0.0 && (peak.gbps <= 0.0 || peak.gbps <= bw));
            }
            section_meta.insert(
                (pool.clone(), *worker, msec.section.clone()),
                SectionMeta {
                    alpha,
                    loc_id: ids,
                    is_comm: comm_slots,
                    peak_tflops,
                    peak_gbps,
                    hw_tflops,
                    mem_physical,
                },
            );
        }
    }
    (locs, section_meta)
}

/// Exact per-worker `Σ total_time_ms` and wall span from ALL rows (cheap SQL).
async fn exact_worker_totals(
    ctx: &SessionContext,
    g_of: &HashMap<(String, u16), f64>,
) -> Result<Vec<WorkerAgg>> {
    let batches = collect(
        ctx,
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, worker_id, \
                SUM(total_time_ms) AS busy, \
                MIN(wall_start_ms) AS w0, \
                MAX(wall_start_ms + total_time_ms) AS w1 \
         FROM cost_log GROUP BY pool_tag, worker_id",
    )
    .await?;
    let mut out = Vec::new();
    for batch in &batches {
        let pool = str_col(batch, "pool_tag")?;
        let wid = col(batch, "worker_id")?;
        let busy = col(batch, "busy")?;
        let w0 = col(batch, "w0")?;
        let w1 = col(batch, "w1")?;
        for row in 0..batch.num_rows() {
            let pool_tag = pool.value(row).to_string();
            let worker_id = value_f64(wid, row)? as u16;
            let g = g_of
                .get(&(pool_tag.clone(), worker_id))
                .copied()
                .unwrap_or(1.0);
            let busy_ms = value_f64(busy, row)?.max(0.0);
            let span_ms = (value_f64(w1, row)? - value_f64(w0, row)?).max(0.0);
            out.push(WorkerAgg {
                pool_tag,
                worker_id,
                g,
                busy_ms,
                span_ms,
                samp_busy_ms: 0.0,
                samp: [0.0; 4],
            });
        }
    }
    Ok(out)
}

/// Fold the stride-sampled rows: per row `Σ_slot α·value` for R2..R5 into its
/// worker, and the same per-slot contributions into `(location, worker)`.
async fn accumulate_fold(
    ctx: &SessionContext,
    stride: u64,
    hw_bw: f64,
    section_meta: &HashMap<(String, u16, String), SectionMeta>,
    widx: &HashMap<(String, u16), usize>,
    workers: &mut [WorkerAgg],
    loc_acc: &mut HashMap<(u32, usize), [f64; 4]>,
) -> Result<u64> {
    let sql = format!(
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, worker_id, \
                CAST(section AS VARCHAR) AS section, total_time_ms, \
                slot_time_ms, slot_flops, slot_bytes \
         FROM cost_log WHERE iter_id % {stride} = 0"
    );
    let batches = collect(ctx, &sql).await?;
    let mut sampled_rows = 0u64;
    for batch in &batches {
        let pool = str_col(batch, "pool_tag")?;
        let wid = col(batch, "worker_id")?;
        let sec = str_col(batch, "section")?;
        let total = col(batch, "total_time_ms")?;
        let (offsets, times) = list_f32(batch, "slot_time_ms")?;
        let (_, flops) = list_f32(batch, "slot_flops")?;
        let (_, bytes) = list_f32(batch, "slot_bytes")?;
        // Cache the resolved (meta, worker index) across the run of rows sharing
        // one (pool, worker, section) — cost_log is worker/iter ordered.
        let mut cached: Option<((String, u16, String), (&SectionMeta, usize))> = None;
        for row in 0..batch.num_rows() {
            let p = pool.value(row);
            let w = value_f64(wid, row)? as u16;
            let s = sec.value(row);
            let resolved = match &cached {
                Some((k, v)) if k.0 == p && k.1 == w && k.2 == s => Some(*v),
                _ => {
                    let key = (p.to_string(), w, s.to_string());
                    match (section_meta.get(&key), widx.get(&(p.to_string(), w))) {
                        (Some(meta), Some(&wi)) => {
                            cached = Some((key, (meta, wi)));
                            Some((meta, wi))
                        }
                        _ => None,
                    }
                }
            };
            let Some((meta, wi)) = resolved else { continue };
            sampled_rows += 1;
            workers[wi].samp_busy_ms += value_f64(total, row)?.max(0.0);

            let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
            let mut row_sum = [0.0f64; 4];
            for (slot, j) in (start..end).enumerate() {
                let Some(&a) = meta.alpha.get(slot) else { break };
                if a == 0.0 {
                    continue;
                }
                let t = times.value(j) as f64; // ms
                let f = flops.value(j) as f64;
                let b = bytes.value(j) as f64;
                let comm = meta.is_comm[slot];
                let tau_cfg = leaf_optimal_ms(
                    t,
                    f,
                    b,
                    comm,
                    meta.peak_tflops[slot],
                    meta.peak_gbps[slot],
                );
                let tau_hw = if comm {
                    0.0
                } else {
                    // Drop the memory term for a non-physical byte count (grouped_gemm),
                    // leaving the compute roofline — its real hardware gap survives.
                    let hw_bw_eff = if meta.mem_physical[slot] { hw_bw } else { 0.0 };
                    leaf_optimal_ms(tau_cfg, f, b, false, meta.hw_tflops[slot], hw_bw_eff)
                };
                // vR2 real, vR3 per-config-best, vR4 drop-comm, vR5 hardware.
                let v = [
                    t,
                    tau_cfg,
                    if comm { 0.0 } else { tau_cfg },
                    if comm { 0.0 } else { tau_hw },
                ];
                let contrib = [a * v[0], a * v[1], a * v[2], a * v[3]];
                for k in 0..4 {
                    row_sum[k] += contrib[k];
                }
                let cell = loc_acc.entry((meta.loc_id[slot], wi)).or_insert([0.0; 4]);
                for k in 0..4 {
                    cell[k] += contrib[k];
                }
            }
            for k in 0..4 {
                workers[wi].samp[k] += row_sum[k];
            }
        }
    }
    Ok(sampled_rows)
}

/// One leaf's optimal time (ms) = `work / peak_rate`, roofline over compute and
/// bandwidth, clamped to `real` (a peak can't make a leaf slower than observed).
/// A comm leaf uses only the bandwidth term. No measurable work/peak → `real`
/// (that leaf contributes no headroom at this rung).
fn leaf_optimal_ms(real_ms: f64, flops: f64, bytes: f64, comm: bool, peak_tflops: f64, peak_gbps: f64) -> f64 {
    let mut best = 0.0f64;
    let mut any = false;
    if !comm && peak_tflops > 0.0 && flops > 0.0 {
        best = best.max(flops / peak_tflops / 1e9); // flops/(TFLOP/s) → ms
        any = true;
    }
    if peak_gbps > 0.0 && bytes > 0.0 {
        best = best.max(bytes / peak_gbps / 1e6); // bytes/(GB/s) → ms
        any = true;
    }
    if !any {
        return real_ms;
    }
    best.min(real_ms).max(0.0)
}

/// Best-effort compute dtype for a leaf's roofline: the first present of the
/// GEMM/norm `dtype`, then attention `q_dtype`, then an input dtype; else bf16.
fn compute_dtype(config: &Value) -> String {
    for key in ["dtype", "q_dtype", "input_dtype", "kv_dtype"] {
        if let Some(s) = config.get(key).and_then(Value::as_str) {
            return s.to_string();
        }
    }
    "bf16".to_string()
}

/// Collective / point-to-point leaves — dropped at R4 ("ignore network") and R5.
fn is_comm(kind: &str) -> bool {
    matches!(
        kind,
        "all_reduce"
            | "all_gather"
            | "reduce_scatter"
            | "all_to_all"
            | "broadcast"
            | "gather"
            | "scatter"
            | "send"
            | "recv"
    ) || kind.starts_with("p2p")
        || kind.starts_with("nccl")
        || kind.starts_with("comm")
}

/// Pick the iteration stride: `num_iters / TARGET`, clamped `[1, MAX_STRIDE]`.
async fn choose_stride(ctx: &SessionContext) -> Result<u64> {
    let batches =
        collect(ctx, "SELECT CAST(COALESCE(MAX(iter_id), 0) AS BIGINT) AS mx FROM cost_log").await?;
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

/// Convert a rung array (GPU·ms) into one level's payload object: the six waterfall
/// buckets (GPU·s) plus the rung values + optimality ratio. `idle_zero` tops the
/// bar at Busy (the iteration level) instead of Real.
fn level_json(level: &str, key: &str, label: &str, r: &[f64; 6], idle_zero: bool) -> Value {
    let top = if idle_zero { r[R1] } else { r[R0] };
    let idle = if idle_zero { 0.0 } else { r[R0] - r[R1] };
    let buckets = [
        idle,
        r[R1] - r[R2],
        r[R2] - r[R3],
        r[R3] - r[R4],
        r[R4] - r[R5],
        r[R5],
    ];
    let bucket_obj: serde_json::Map<String, Value> = BUCKET_KEYS
        .iter()
        .zip(buckets.iter())
        .map(|(k, v)| (k.to_string(), json!(ms_to_s(v.max(0.0)))))
        .collect();
    let rung_obj: serde_json::Map<String, Value> = RUNG_KEYS
        .iter()
        .zip(r.iter())
        .map(|(k, v)| (k.to_string(), json!(ms_to_s(*v))))
        .collect();
    json!({
        "level": level,
        "key": key,
        "label": label,
        "total": ms_to_s(top),
        "buckets": bucket_obj,
        "rungs": rung_obj,
        "optimality_ratio": ratio(r[R5], top),
    })
}

/// Kernel-level bars: per location the Real (balanced) GPU·s split into batching /
/// communication / hw-gap / hw-optimal. Top-N by Real + an `other` roll-up.
fn kernel_json(locs: &[Loc], kernel_rk: &[[f64; 4]]) -> Vec<Value> {
    let mut idx: Vec<usize> = (0..locs.len()).filter(|&i| kernel_rk[i][0] > 0.0).collect();
    idx.sort_by(|&a, &b| kernel_rk[b][0].total_cmp(&kernel_rk[a][0]));
    let mut out = Vec::new();
    let mut other = [0.0f64; 4];
    for (rank, &i) in idx.iter().enumerate() {
        if rank < TOP_KERNELS {
            out.push(kernel_entry(&locs[i].name, &locs[i].kind, locs[i].is_comm, &kernel_rk[i]));
        } else {
            for k in 0..4 {
                other[k] += kernel_rk[i][k];
            }
        }
    }
    if other[0] > 0.0 {
        out.push(kernel_entry("other", "other", false, &other));
    }
    out
}

fn kernel_entry(name: &str, kind: &str, is_comm: bool, rk: &[f64; 4]) -> Value {
    // rk = [R2 real, R3 per-config-best, R4 ignore-network, R5 hardware].
    json!({
        "name": name,
        "kind": kind,
        "is_comm": is_comm,
        "real": ms_to_s(rk[0]),
        "buckets": {
            "batching": ms_to_s((rk[0] - rk[1]).max(0.0)),
            "communication": ms_to_s((rk[1] - rk[2]).max(0.0)),
            "hardware_gap": ms_to_s((rk[2] - rk[3]).max(0.0)),
            "hardware_optimal": ms_to_s(rk[3].max(0.0)),
        },
    })
}

/// The worst-headroom kernels for the report (by batching GPU·s), a quick "where
/// to look first" list next to the full kernel array in the payload.
fn worst_batching(locs: &[Loc], kernel_rk: &[[f64; 4]]) -> Vec<Value> {
    let mut idx: Vec<usize> = (0..locs.len()).collect();
    idx.sort_by(|&a, &b| {
        (kernel_rk[b][0] - kernel_rk[b][1]).total_cmp(&(kernel_rk[a][0] - kernel_rk[a][1]))
    });
    idx.iter()
        .take(8)
        .filter(|&&i| kernel_rk[i][0] - kernel_rk[i][1] > 0.0)
        .map(|&i| {
            json!({
                "name": locs[i].name,
                "kind": locs[i].kind,
                "batching_gpu_s": ms_to_s(kernel_rk[i][0] - kernel_rk[i][1]),
                "real_gpu_s": ms_to_s(kernel_rk[i][0]),
            })
        })
        .collect()
}

fn rung_report(r: &[f64; 6]) -> Value {
    let top = r[R0].max(1e-9);
    let bucket = |a: f64| json!({"gpu_s": ms_to_s(a.max(0.0)), "frac": (a.max(0.0) / top)});
    json!({
        "real": ms_to_s(r[R0]),
        "busy": ms_to_s(r[R1]),
        "balanced": ms_to_s(r[R2]),
        "per_config_best": ms_to_s(r[R3]),
        "ignore_network": ms_to_s(r[R4]),
        "hardware_limit": ms_to_s(r[R5]),
        "optimality_ratio": ratio(r[R5], r[R0]),
        "buckets": {
            "idle": bucket(r[R0] - r[R1]),
            "imbalance": bucket(r[R1] - r[R2]),
            "batching": bucket(r[R2] - r[R3]),
            "communication": bucket(r[R3] - r[R4]),
            "hardware_gap": bucket(r[R4] - r[R5]),
            "hardware_optimal": bucket(r[R5]),
        },
    })
}

fn ms_to_s(ms: f64) -> f64 {
    ms / 1000.0
}

fn ratio(num: f64, den: f64) -> f64 {
    if den > 0.0 {
        (num / den).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    col(batch, name)?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("`{name}` is not a Utf8 array"))
}

/// Downcast a cost_log `List<f32>` column to `(list_offsets, flat_values)`.
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
        "scope": "per-worker CostTree manifest re-folded with per-rung leaf substitutions; \
                  unit GPU-seconds = wall time × the worker's run_meta gpu_ids count",
        "ladder": {
            "real": "span × G — GPU·s actually held (includes idle)",
            "busy": "Σ total_time_ms × G — no scheduler idle; gap vs real = idle",
            "balanced": "real slot times, Max folded to mean — perfect load balance; gap = imbalance",
            "per_config_best": "work / grid-peak rate for the leaf's fixed config over its batch axis; gap = batching",
            "ignore_network": "per_config_best with comm leaves dropped; gap = communication",
            "hardware_limit": "work / gpu-spec dense peak (roofline); gap = profiled↔hardware; this rung = irreducible",
        },
        "buckets": "idle, imbalance, batching, communication, hardware_gap, hardware_optimal — \
                    telescoping differences of the rungs; sum to Real",
        "sampling": "R0/R1 exact over all rows; R2..R5 folded over 1-in-sample_stride iterations \
                     and anchored to the exact R1 by their sampled ratio",
        "kernel_level": "per manifest location (name); Max siblings + DP replicas pool; \
                         single-leaf so no idle/imbalance — only batching/communication/hw",
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
        "bucket_keys": BUCKET_KEYS,
        "rung_keys": RUNG_KEYS,
        "levels": [],
        "kernels": [],
    })
}
