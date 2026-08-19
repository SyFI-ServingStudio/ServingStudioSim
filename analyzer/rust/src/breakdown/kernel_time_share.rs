//! Run-wide kernel-time composition by semantic CostTree leaf position.
//!
//! The hot `cost_log` stores slot-aligned durations while the matching manifest
//! owns each slot's semantic name and the Sum/Max/Scale aggregation tree. This
//! subject reconstructs an overlap-aware root attribution for every selected
//! row, then rolls it up at three levels: `(pool_tag, worker_id)`, `pool_tag`, and
//! the whole run. A Sum forwards attribution to every child, Scale multiplies its
//! child, and Max forwards only to the row's critical child (equal critical
//! children split the attribution). Therefore every emitted scope's position
//! shares sum to 100% of that scope's CostTree root kernel time.
//!
//! Performance is intentionally bounded. DataFusion first scans only scalar
//! columns to count rows per worker. Small runs replay every row exactly; large
//! runs choose a worker-local regular `iter_id` stride targeting at most
//! `MAX_REPLAY_ROWS` rows in aggregate, then use predicate/projection pushdown to
//! read only selected slot lists. Each manifest section is compiled once into a
//! flat plan, and repeated slot vectors (common across AFD layers) reuse the last
//! computed position shares instead of replaying the tree again.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::{Array, Float32Array, ListArray, RecordBatch, StringArray};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{read_cost_manifests, SCHEMA_VERSION};
use crate::session::{col, collect, register_cost_log, require_columns, value_f64, COST_LOG_TABLE};
use crate::trace::manifest::{FlatCostNode, Manifest};

const COST_COLUMNS: &[&str] = &[
    "pool_tag",
    "worker_id",
    "iter_id",
    "section",
    "total_time_ms",
    "slot_time_ms",
];

/// Soft cap on the number of heavy list rows replayed. The scalar planning pass
/// remains exact and cheap; the per-worker stride can overshoot slightly when a
/// single iteration owns many layer-wise rows.
const MAX_REPLAY_ROWS: u64 = 250_000;
const TIME_EPSILON_MS: f64 = 1e-12;
const TIE_REL_EPSILON: f64 = 1e-9;

#[derive(Clone)]
struct Position {
    name: String,
    kind: String,
}

#[derive(Default)]
struct ScopeTotals {
    kernel_time_ms: f64,
    position_time_ms: Vec<f64>,
}

impl ScopeTotals {
    fn with_positions(num_positions: usize) -> Self {
        Self {
            kernel_time_ms: 0.0,
            position_time_ms: vec![0.0; num_positions],
        }
    }

    fn add(&mut self, other: &Self) {
        self.kernel_time_ms += other.kernel_time_ms;
        for (dst, src) in self
            .position_time_ms
            .iter_mut()
            .zip(&other.position_time_ms)
        {
            *dst += src;
        }
    }
}

struct WorkerTotals {
    pool_tag: String,
    worker_id: u16,
    raw_rows: u64,
    sampled_rows: u64,
    stride: u64,
    exact_kernel_time_ms: f64,
    totals: ScopeTotals,
}

#[derive(Clone, Copy)]
struct WorkerScan {
    raw_rows: u64,
    stride: u64,
    exact_kernel_time_ms: f64,
}

/// A manifest section compiled into flat scratch arrays. `last_slot_bits` and
/// `last_shares` are a one-entry hot cache: layer-wise AFD repeatedly logs the
/// same cost vector for adjacent homogeneous layers, so most rows avoid another
/// bottom-up/top-down tree replay.
struct SectionPlan {
    nodes: Vec<FlatCostNode>,
    slot_position_ids: Vec<usize>,
    node_times: Vec<f64>,
    node_weights: Vec<f64>,
    last_slot_bits: Vec<u32>,
    last_shares: Vec<(usize, f64)>,
    cache_hits: u64,
    cache_misses: u64,
}

impl SectionPlan {
    fn new(manifest: &Manifest, slot_position_ids: Vec<usize>) -> Result<Self> {
        validate_tree(manifest)?;
        Ok(Self {
            nodes: manifest.nodes.clone(),
            slot_position_ids,
            node_times: vec![0.0; manifest.nodes.len()],
            node_weights: vec![0.0; manifest.nodes.len()],
            last_slot_bits: Vec::new(),
            last_shares: Vec::new(),
            cache_hits: 0,
            cache_misses: 0,
        })
    }

    /// Return sparse `(position_id, root_share)` pairs. Shares sum to 1 for a
    /// positive root. The caller multiplies by the logged root time and sampling
    /// weight, keeping the row's authoritative `total_time_ms` as denominator.
    fn position_shares(&mut self, slot_times: &[f32]) -> Result<&[(usize, f64)]> {
        if slot_times.len() != self.slot_position_ids.len() {
            bail!(
                "slot_time_ms length {} != manifest slot count {}",
                slot_times.len(),
                self.slot_position_ids.len()
            );
        }
        if same_slot_bits(slot_times, &self.last_slot_bits) {
            self.cache_hits += 1;
            return Ok(&self.last_shares);
        }
        self.cache_misses += 1;

        for idx in (0..self.nodes.len()).rev() {
            self.node_times[idx] = match &self.nodes[idx] {
                FlatCostNode::Leaf(slot) => {
                    let value = f64::from(slot_times[*slot]);
                    if !value.is_finite() || value < 0.0 {
                        bail!("slot {slot} has invalid time {value}");
                    }
                    value
                }
                FlatCostNode::Sum { children } => {
                    children.clone().map(|child| self.node_times[child]).sum()
                }
                FlatCostNode::Max { overlap, children } => {
                    let max_child = children
                        .clone()
                        .map(|child| self.node_times[child])
                        .fold(0.0_f64, f64::max);
                    max_child / f64::from(*overlap).max(TIME_EPSILON_MS)
                }
                FlatCostNode::Scale { n, children } => {
                    f64::from(*n) * self.node_times[children.start]
                }
            };
        }

        let root_time = self.node_times[0];
        self.node_weights.fill(0.0);
        self.last_shares.clear();
        if root_time > TIME_EPSILON_MS {
            self.node_weights[0] = 1.0 / root_time;
            for idx in 0..self.nodes.len() {
                let weight = self.node_weights[idx];
                if weight == 0.0 {
                    continue;
                }
                match &self.nodes[idx] {
                    FlatCostNode::Leaf(slot) => {
                        let share = f64::from(slot_times[*slot]) * weight;
                        if share > 0.0 {
                            add_sparse_share(
                                &mut self.last_shares,
                                self.slot_position_ids[*slot],
                                share,
                            );
                        }
                    }
                    FlatCostNode::Sum { children } => {
                        for child in children.clone() {
                            self.node_weights[child] += weight;
                        }
                    }
                    FlatCostNode::Scale { n, children } => {
                        self.node_weights[children.start] += weight * f64::from(*n);
                    }
                    FlatCostNode::Max { overlap, children } => {
                        let max_child = children
                            .clone()
                            .map(|child| self.node_times[child])
                            .fold(0.0_f64, f64::max);
                        let tolerance = TIE_REL_EPSILON * max_child.abs().max(1.0);
                        let critical: Vec<usize> = children
                            .clone()
                            .filter(|&child| {
                                (self.node_times[child] - max_child).abs() <= tolerance
                            })
                            .collect();
                        if !critical.is_empty() {
                            let child_weight = weight
                                / f64::from(*overlap).max(TIME_EPSILON_MS)
                                / critical.len() as f64;
                            for child in critical {
                                self.node_weights[child] += child_weight;
                            }
                        }
                    }
                }
            }
            self.last_shares
                .sort_by_key(|(position_id, _)| *position_id);
            let share_sum: f64 = self.last_shares.iter().map(|(_, share)| share).sum();
            if share_sum > TIME_EPSILON_MS {
                // Remove only floating-point fold drift; the semantic attribution
                // above already determined which branches own the root.
                for (_, share) in &mut self.last_shares {
                    *share /= share_sum;
                }
            }
        }
        self.last_slot_bits.clear();
        self.last_slot_bits
            .extend(slot_times.iter().map(|value| value.to_bits()));
        Ok(&self.last_shares)
    }
}

pub async fn run(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    if !register_cost_log(ctx, log_dir).await? {
        let reason = "cost_log/ dir not found";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }
    require_columns(ctx, COST_LOG_TABLE, COST_COLUMNS).await?;
    let manifests = match read_cost_manifests(log_dir) {
        Ok(manifests) => manifests,
        Err(error) => {
            let reason = format!("cost_manifest/ unreadable ({error:#})");
            return Ok((
                unavailable(log_dir, &reason),
                unavailable_payload(log_dir, &reason),
            ));
        }
    };

    let worker_scans = plan_worker_scans(ctx).await?;
    if worker_scans.is_empty() {
        let reason = "cost_log has no rows";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }

    let (positions, mut plans, plan_ids, worker_ids, mut workers) =
        compile_plans(&manifests, &worker_scans)?;
    let filter = sampling_filter(&worker_scans);
    let sql = format!(
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, worker_id, \
                CAST(section AS VARCHAR) AS section, total_time_ms, slot_time_ms \
         FROM cost_log{filter}"
    );
    let batches = collect(ctx, &sql).await?;
    for batch in &batches {
        accumulate_batch(batch, &mut plans, &plan_ids, &worker_ids, &mut workers)?;
    }
    normalize_workers_to_exact_roots(&mut workers)?;

    let sampled_rows: u64 = workers.iter().map(|worker| worker.sampled_rows).sum();
    let raw_rows: u64 = workers.iter().map(|worker| worker.raw_rows).sum();
    let mut pools: BTreeMap<String, ScopeTotals> = BTreeMap::new();
    let mut overall = ScopeTotals::with_positions(positions.len());
    for worker in &workers {
        pools
            .entry(worker.pool_tag.clone())
            .or_insert_with(|| ScopeTotals::with_positions(positions.len()))
            .add(&worker.totals);
        overall.add(&worker.totals);
    }
    if overall.kernel_time_ms <= TIME_EPSILON_MS {
        let reason = "selected cost_log rows have zero CostTree root kernel time";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }

    let mut position_order: Vec<usize> = (0..positions.len()).collect();
    position_order.sort_by(|&a, &b| {
        overall.position_time_ms[b]
            .total_cmp(&overall.position_time_ms[a])
            .then_with(|| positions[a].name.cmp(&positions[b].name))
    });
    let overall_json = composition_json(&overall, &positions, &position_order);
    let pool_json: Vec<Value> = pools
        .iter()
        .map(|(pool_tag, totals)| {
            let mut value = composition_json(totals, &positions, &position_order);
            value["pool_tag"] = json!(pool_tag);
            value["num_workers"] =
                json!(workers.iter().filter(|w| w.pool_tag == *pool_tag).count());
            value
        })
        .collect();
    let worker_json: Vec<Value> = workers
        .iter()
        .map(|worker| {
            let mut value = composition_json(&worker.totals, &positions, &position_order);
            value["pool_tag"] = json!(worker.pool_tag);
            value["worker_id"] = json!(worker.worker_id);
            value["raw_rows"] = json!(worker.raw_rows);
            value["sampled_rows"] = json!(worker.sampled_rows);
            value["sample_stride"] = json!(worker.stride);
            value
        })
        .collect();
    let position_json: Vec<Value> = position_order
        .iter()
        .map(|&position_id| {
            json!({
                "name": positions[position_id].name,
                "kind": positions[position_id].kind,
                "overall_share_pct": 100.0 * overall.position_time_ms[position_id]
                    / overall.kernel_time_ms,
            })
        })
        .collect();
    let cache_hits: u64 = plans.iter().map(|plan| plan.cache_hits).sum();
    let cache_misses: u64 = plans.iter().map(|plan| plan.cache_misses).sum();
    let exact = worker_scans.values().all(|scan| scan.stride == 1);
    let meta = json!({
        "log_dir": log_dir.display().to_string(),
        "exact": exact,
        "sampling_method": if exact { "all rows" } else { "worker-local regular iter_id stride" },
        "max_replay_rows_target": MAX_REPLAY_ROWS,
        "raw_rows": raw_rows,
        "sampled_rows": sampled_rows,
        "num_positions": positions.len(),
        "num_pools": pools.len(),
        "num_workers": workers.len(),
        "tree_cache_hits": cache_hits,
        "tree_cache_misses": cache_misses,
        "kernel_time_totals_exact": true,
    });
    let definitions = definitions();
    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "available": true,
        "totals": {
            "overall": overall_json,
            "pools": pool_json,
            "workers": worker_json,
        },
        "positions": position_json,
        "definitions": definitions,
    });
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "available": true,
        "overall": overall_json,
        "pools": pool_json,
        "workers": worker_json,
        "positions": position_json,
        "definitions": definitions,
    });
    Ok((report, payload))
}

async fn plan_worker_scans(ctx: &SessionContext) -> Result<BTreeMap<(String, u16), WorkerScan>> {
    let batches = collect(
        ctx,
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, worker_id, COUNT(*) AS raw_rows, \
                SUM(total_time_ms) AS kernel_time_ms \
         FROM cost_log GROUP BY pool_tag, worker_id ORDER BY pool_tag, worker_id",
    )
    .await?;
    let mut counts = Vec::new();
    for batch in &batches {
        let pools = string_column(batch, "pool_tag")?;
        let worker_ids = col(batch, "worker_id")?;
        let raw_rows = col(batch, "raw_rows")?;
        let kernel_time_ms = col(batch, "kernel_time_ms")?;
        for row in 0..batch.num_rows() {
            counts.push((
                pools.value(row).to_owned(),
                value_f64(worker_ids, row)? as u16,
                value_f64(raw_rows, row)? as u64,
                value_f64(kernel_time_ms, row)?,
            ));
        }
    }
    if counts.is_empty() {
        return Ok(BTreeMap::new());
    }
    let target_per_worker = (MAX_REPLAY_ROWS / counts.len() as u64).max(1);
    Ok(counts
        .into_iter()
        .map(|(pool_tag, worker_id, raw_rows, exact_kernel_time_ms)| {
            let stride = raw_rows.div_ceil(target_per_worker).max(1);
            (
                (pool_tag, worker_id),
                WorkerScan {
                    raw_rows,
                    stride,
                    exact_kernel_time_ms,
                },
            )
        })
        .collect())
}

type PlanIds = HashMap<String, HashMap<u16, HashMap<String, usize>>>;
type WorkerIds = HashMap<String, HashMap<u16, usize>>;

/// Return type of [`compile_plans`]: positions, section plans, the two id
/// lookup tables, and per-worker totals.
type CompiledPlans = (
    Vec<Position>,
    Vec<SectionPlan>,
    PlanIds,
    WorkerIds,
    Vec<WorkerTotals>,
);

fn compile_plans(
    manifests: &BTreeMap<(String, u16), crate::trace::manifest::ManifestDoc>,
    worker_scans: &BTreeMap<(String, u16), WorkerScan>,
) -> Result<CompiledPlans> {
    let mut positions: Vec<Position> = Vec::new();
    let mut position_ids: HashMap<String, usize> = HashMap::new();
    let mut plans = Vec::new();
    let mut plan_ids: PlanIds = HashMap::new();
    let mut worker_ids: WorkerIds = HashMap::new();
    let mut worker_specs = Vec::new();

    for ((pool_tag, worker_id), doc) in manifests {
        let Some(scan) = worker_scans.get(&(pool_tag.clone(), *worker_id)) else {
            continue;
        };
        let worker_index = worker_specs.len();
        worker_specs.push((pool_tag.clone(), *worker_id, *scan));
        worker_ids
            .entry(pool_tag.clone())
            .or_default()
            .insert(*worker_id, worker_index);
        for section in &doc.sections {
            let slot_position_ids = section
                .manifest
                .slots
                .iter()
                .map(|leaf| {
                    if let Some(&position_id) = position_ids.get(&leaf.name) {
                        if positions[position_id].kind != leaf.kind {
                            bail!(
                                "leaf position {:?} has conflicting kinds {:?} and {:?}",
                                leaf.name,
                                positions[position_id].kind,
                                leaf.kind
                            );
                        }
                        Ok(position_id)
                    } else {
                        let position_id = positions.len();
                        position_ids.insert(leaf.name.clone(), position_id);
                        positions.push(Position {
                            name: leaf.name.clone(),
                            kind: leaf.kind.clone(),
                        });
                        Ok(position_id)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let plan_index = plans.len();
            plans.push(SectionPlan::new(&section.manifest, slot_position_ids)?);
            let old = plan_ids
                .entry(pool_tag.clone())
                .or_default()
                .entry(*worker_id)
                .or_default()
                .insert(section.section.clone(), plan_index);
            if old.is_some() {
                bail!(
                    "duplicate manifest section {:?} for {pool_tag}/{worker_id}",
                    section.section
                );
            }
        }
    }
    for key in worker_scans.keys() {
        if !manifests.contains_key(key) {
            bail!(
                "cost_log worker {}/{} has no matching manifest",
                key.0,
                key.1
            );
        }
    }
    let workers = worker_specs
        .into_iter()
        .map(|(pool_tag, worker_id, scan)| WorkerTotals {
            pool_tag,
            worker_id,
            raw_rows: scan.raw_rows,
            sampled_rows: 0,
            stride: scan.stride,
            exact_kernel_time_ms: scan.exact_kernel_time_ms,
            totals: ScopeTotals::with_positions(positions.len()),
        })
        .collect();
    Ok((positions, plans, plan_ids, worker_ids, workers))
}

fn sampling_filter(worker_scans: &BTreeMap<(String, u16), WorkerScan>) -> String {
    if worker_scans.values().all(|scan| scan.stride == 1) {
        return String::new();
    }
    let predicates = worker_scans
        .iter()
        .map(|((pool_tag, worker_id), scan)| {
            let pool_tag = pool_tag.replace('\'', "''");
            format!(
                "(pool_tag = '{pool_tag}' AND worker_id = {worker_id} \
                 AND iter_id % {} = 0)",
                scan.stride
            )
        })
        .collect::<Vec<_>>()
        .join(" OR ");
    format!(" WHERE {predicates}")
}

fn accumulate_batch(
    batch: &RecordBatch,
    plans: &mut [SectionPlan],
    plan_ids: &PlanIds,
    worker_ids: &WorkerIds,
    workers: &mut [WorkerTotals],
) -> Result<()> {
    let pools = string_column(batch, "pool_tag")?;
    let worker_column = col(batch, "worker_id")?;
    let sections = string_column(batch, "section")?;
    let total_times = col(batch, "total_time_ms")?;
    let (offsets, slot_values) = list_f32(batch, "slot_time_ms")?;
    for row in 0..batch.num_rows() {
        let pool_tag = pools.value(row);
        let worker_id = value_f64(worker_column, row)? as u16;
        let section = sections.value(row);
        let worker_index = worker_ids
            .get(pool_tag)
            .and_then(|ids| ids.get(&worker_id))
            .copied()
            .with_context(|| format!("unknown cost_log worker {pool_tag}/{worker_id}"))?;
        let plan_index = plan_ids
            .get(pool_tag)
            .and_then(|workers| workers.get(&worker_id))
            .and_then(|sections| sections.get(section))
            .copied()
            .with_context(|| {
                format!("manifest for {pool_tag}/{worker_id} has no section {section:?}")
            })?;
        let total_time_ms = value_f64(total_times, row)?;
        if !total_time_ms.is_finite() || total_time_ms < 0.0 {
            bail!(
                "cost_log {pool_tag}/{worker_id} section {section:?} has invalid root time {total_time_ms}"
            );
        }
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        let slot_times = &slot_values.values()[start..end];
        let worker = &mut workers[worker_index];
        worker.sampled_rows += 1;
        let weighted_root_ms = total_time_ms * worker.stride as f64;
        worker.totals.kernel_time_ms += weighted_root_ms;
        for &(position_id, share) in plans[plan_index].position_shares(slot_times)? {
            worker.totals.position_time_ms[position_id] += weighted_root_ms * share;
        }
    }
    Ok(())
}

/// The heavy scan estimates only each worker's position mix. Its scalar root
/// total is known exactly from the DataFusion planning pass, so rescale the
/// sampled position totals to that exact denominator. This removes stride-tail
/// bias from pool/overall weighting without reading another list value.
fn normalize_workers_to_exact_roots(workers: &mut [WorkerTotals]) -> Result<()> {
    for worker in workers {
        let sampled_kernel_time_ms = worker.totals.kernel_time_ms;
        if worker.exact_kernel_time_ms > TIME_EPSILON_MS {
            if sampled_kernel_time_ms <= TIME_EPSILON_MS {
                bail!(
                    "sampling selected no positive kernel time for {}/{} (stride {})",
                    worker.pool_tag,
                    worker.worker_id,
                    worker.stride
                );
            }
            let scale = worker.exact_kernel_time_ms / sampled_kernel_time_ms;
            for position_time_ms in &mut worker.totals.position_time_ms {
                *position_time_ms *= scale;
            }
        }
        worker.totals.kernel_time_ms = worker.exact_kernel_time_ms;
    }
    Ok(())
}

fn composition_json(
    totals: &ScopeTotals,
    positions: &[Position],
    position_order: &[usize],
) -> Value {
    let segments = position_order
        .iter()
        .filter_map(|&position_id| {
            let time_ms = totals.position_time_ms[position_id];
            (time_ms > TIME_EPSILON_MS).then(|| {
                json!({
                    "position": positions[position_id].name,
                    "kind": positions[position_id].kind,
                    "kernel_time_ms": time_ms,
                    "share_pct": 100.0 * time_ms / totals.kernel_time_ms,
                })
            })
        })
        .collect::<Vec<_>>();
    json!({
        "kernel_time_ms": totals.kernel_time_ms,
        "segments": segments,
    })
}

fn validate_tree(manifest: &Manifest) -> Result<()> {
    if manifest.nodes.is_empty() {
        bail!("cost manifest section has no root node");
    }
    for (idx, node) in manifest.nodes.iter().enumerate() {
        let children = match node {
            FlatCostNode::Leaf(slot) => {
                if *slot >= manifest.slots.len() {
                    bail!("leaf node {idx} references missing slot {slot}");
                }
                continue;
            }
            FlatCostNode::Sum { children } | FlatCostNode::Max { children, .. } => children,
            FlatCostNode::Scale { children, .. } => {
                if children.end != children.start + 1 {
                    bail!("Scale node {idx} must own exactly one child");
                }
                children
            }
        };
        if children.is_empty() || children.start <= idx || children.end > manifest.nodes.len() {
            bail!("node {idx} has invalid BFS child range {children:?}");
        }
    }
    Ok(())
}

fn same_slot_bits(slot_times: &[f32], cached_bits: &[u32]) -> bool {
    slot_times.len() == cached_bits.len()
        && slot_times
            .iter()
            .zip(cached_bits)
            .all(|(value, bits)| value.to_bits() == *bits)
}

fn add_sparse_share(shares: &mut Vec<(usize, f64)>, position_id: usize, share: f64) {
    if let Some((_, accumulated)) = shares.iter_mut().find(|(id, _)| *id == position_id) {
        *accumulated += share;
    } else {
        shares.push((position_id, share));
    }
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    col(batch, name)?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("`{name}` is not a Utf8 array"))
}

fn list_f32<'a>(batch: &'a RecordBatch, name: &str) -> Result<(&'a [i32], &'a Float32Array)> {
    let list = col(batch, name)?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("`{name}` is not a List array"))?;
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| anyhow!("`{name}` is not List<Float32>"))?;
    Ok((list.value_offsets(), values))
}

fn definitions() -> Value {
    json!({
        "scope": "all cost_log workers; exact for small runs, worker-stratified regular iter_id sampling for large runs",
        "position": "the manifest leaf's full semantic name; identical names across Max siblings and worker replicas are pooled",
        "kernel_time_ms": "exact DataFusion SUM(cost_log.total_time_ms) for the scope; sampled position mixtures are normalized to each worker's exact root total",
        "share_pct": "position-attributed kernel_time_ms / scope kernel_time_ms × 100; segments sum to 100%",
        "tree_attribution": "Sum forwards to all children; Scale multiplies its child; Max forwards to the critical child and divides by overlap; exactly tied critical children split evenly",
        "levels": "workers are keyed by (pool_tag, worker_id); pools sum their workers; overall sums all pools",
        "sampling": "DataFusion first counts scalar rows per worker, then predicate/projection-pushes a worker-local regular iter_id stride into the heavy slot_time_ms scan; meta.exact=false marks estimates",
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
        "meta": {"log_dir": log_dir.display().to_string(), "reason": reason},
        "available": false,
        "overall": {},
        "pools": [],
        "workers": [],
        "positions": [],
        "definitions": definitions(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::manifest::LeafDesc;

    fn leaf(name: &str) -> LeafDesc {
        LeafDesc {
            name: name.into(),
            kind: "k".into(),
            kernel_config: json!({"backends": []}),
        }
    }

    fn mixed_manifest() -> Manifest {
        Manifest {
            slots: vec![leaf("m.a"), leaf("m.b"), leaf("m.c")],
            // Sum(Leaf a, Scale{2}(Max{overlap=2}(Leaf b, Leaf c)))
            nodes: vec![
                FlatCostNode::Sum { children: 1..3 },
                FlatCostNode::Leaf(0),
                FlatCostNode::Scale {
                    n: 2,
                    children: 3..4,
                },
                FlatCostNode::Max {
                    overlap: 2.0,
                    children: 4..6,
                },
                FlatCostNode::Leaf(1),
                FlatCostNode::Leaf(2),
            ],
            node_labels: vec![None; 6],
        }
    }

    #[test]
    fn attributes_scale_and_max_to_critical_leaf() {
        let mut plan = SectionPlan::new(&mixed_manifest(), vec![0, 1, 2]).unwrap();
        let shares = plan.position_shares(&[4.0, 6.0, 10.0]).unwrap();
        // root = 4 + 2 × (10 / 2) = 14; a owns 4, c owns 10.
        assert_eq!(shares.len(), 2);
        assert!((shares.iter().find(|(id, _)| *id == 0).unwrap().1 - 4.0 / 14.0).abs() < 1e-12);
        assert!((shares.iter().find(|(id, _)| *id == 2).unwrap().1 - 10.0 / 14.0).abs() < 1e-12);
        assert!((shares.iter().map(|(_, share)| share).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn splits_exact_max_tie_and_reuses_cached_vector() {
        let manifest = Manifest {
            slots: vec![leaf("m.a"), leaf("m.b")],
            nodes: vec![
                FlatCostNode::Max {
                    overlap: 1.0,
                    children: 1..3,
                },
                FlatCostNode::Leaf(0),
                FlatCostNode::Leaf(1),
            ],
            node_labels: vec![None; 3],
        };
        let mut plan = SectionPlan::new(&manifest, vec![0, 1]).unwrap();
        let first = plan.position_shares(&[5.0, 5.0]).unwrap().to_vec();
        let second = plan.position_shares(&[5.0, 5.0]).unwrap().to_vec();
        assert_eq!(first, vec![(0, 0.5), (1, 0.5)]);
        assert_eq!(second, first);
        assert_eq!(plan.cache_hits, 1);
        assert_eq!(plan.cache_misses, 1);
    }

    #[test]
    fn sampling_filter_is_worker_local() {
        let scans = BTreeMap::from([
            (
                ("attn".to_string(), 0),
                WorkerScan {
                    raw_rows: 10,
                    stride: 1,
                    exact_kernel_time_ms: 1.0,
                },
            ),
            (
                ("ffn".to_string(), 2),
                WorkerScan {
                    raw_rows: 30,
                    stride: 3,
                    exact_kernel_time_ms: 1.0,
                },
            ),
        ]);
        let filter = sampling_filter(&scans);
        assert!(filter.contains("pool_tag = 'attn'"));
        assert!(filter.contains("worker_id = 2"));
        assert!(filter.contains("iter_id % 3 = 0"));
    }
}
