//! Lazy, worker-scoped operation index and exact CostTree resources.
//!
//! Selection is operation-only: the public window is a globally ordered slice
//! of raw `worker_cost` rows, while an exact CostTree reopens one row by its
//! `(iter_id, batch_id, operation_id)` identity.

use std::collections::HashMap;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::{
    Array, Int16Array, Int64Array, ListArray, StringArray, UInt32Array, UInt64Array, UInt8Array,
};
use datafusion::prelude::{SessionConfig, SessionContext};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::Mutex as AsyncMutex;

use crate::io::{read_cost_manifests, resolve_artifact_path};
use crate::session::{
    build_session, col, collect, column_f64, register_if_exists, require_columns, value_f32_list,
    value_f64, value_groups, value_str_list, value_string,
};
use crate::trace::manifest::{FlatCostNode, Manifest, ManifestDoc};

use super::discovery::{regular_file, DiscoveredRun};

const TABLE: &str = "worker_cost";
const INDEX_COLUMNS: &[&str] = &[
    "iter_id",
    "batch_id",
    "section",
    "layer",
    "wall_start_ms",
    "total_time_ms",
];
const TREE_COLUMNS: &[&str] = &[
    "iter_id",
    "batch_id",
    "section",
    "layer",
    "wall_start_ms",
    "total_time_ms",
    "groups",
    "slot_time_ms",
];

#[derive(Clone, Copy)]
struct OptionalColumns {
    slot_input: bool,
    slot_flops: bool,
    slot_bytes: bool,
    slot_backend: bool,
}

const MAX_RANGE_LIMIT: usize = 384;
const SEEK_VIEWPORT_LIMIT: usize = 64;
const SEEK_BUFFER_LIMIT: usize = SEEK_VIEWPORT_LIMIT * 3;
const INDEX_CACHE_BUDGET_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(super) struct OperationRange {
    pub(super) offset: usize,
    pub(super) limit: usize,
}

/// One matched cost-log/manifest pair. UI resource families resolve this
/// internal source independently, so exact CostTree replay does not depend on
/// simulation-run topology.
#[derive(Clone, Debug)]
pub(super) struct CostLogSource {
    artifact_root: PathBuf,
    pool_tag: String,
    worker_id: u16,
}

impl CostLogSource {
    pub(super) fn simulation_worker(run: &DiscoveredRun, pool_tag: &str, worker_id: u16) -> Self {
        Self {
            artifact_root: run.path.clone(),
            pool_tag: pool_tag.to_owned(),
            worker_id,
        }
    }

    pub(super) fn unique_prediction_source(artifact_root: &Path) -> Result<Self> {
        let manifests = read_cost_manifests(artifact_root)?;
        if manifests.len() != 1 {
            bail!(
                "timing prediction requires exactly one cost-log source, found {}",
                manifests.len()
            );
        }
        let ((pool_tag, worker_id), _) = manifests
            .into_iter()
            .next()
            .context("timing prediction cost manifest is empty")?;
        let source = Self {
            artifact_root: artifact_root.to_path_buf(),
            pool_tag,
            worker_id,
        };
        if !regular_file(&source.cost_path()) {
            bail!("timing prediction cost log is missing");
        }
        Ok(source)
    }

    pub(super) fn pool_tag(&self) -> &str {
        &self.pool_tag
    }

    pub(super) fn worker_id(&self) -> u16 {
        self.worker_id
    }

    pub(super) fn artifact_root(&self) -> &std::path::Path {
        &self.artifact_root
    }

    fn cost_path(&self) -> PathBuf {
        resolve_artifact_path(&self.artifact_root, "cost_log").join(format!(
            "worker_{}_{}.parquet",
            self.pool_tag, self.worker_id
        ))
    }
}

#[derive(Default)]
pub(super) struct OperationIndexCache {
    indexes: RwLock<OperationIndexStore>,
    build_lock: AsyncMutex<()>,
    access_clock: AtomicU64,
}

#[derive(Default)]
struct OperationIndexStore {
    entries: HashMap<PathBuf, CachedOperationIndex>,
    bytes: usize,
}

struct CachedOperationIndex {
    index: Arc<WorkerOperationIndex>,
    last_access: u64,
}

struct WorkerOperationIndex {
    worker_kind: &'static str,
    batch_role: &'static str,
    span_start_ms: f64,
    span_end_ms: f64,
    section_names: Vec<String>,
    operations: Vec<IndexedOperation>,
    estimated_bytes: usize,
}

struct IndexedOperation {
    iter_id: u64,
    batch_id: u64,
    operation_id: u32,
    section_id: u16,
    layer: i16,
    start_ms: f64,
    end_ms: f64,
}

impl OperationRange {
    pub(super) fn bounded(offset: usize, limit: usize) -> Self {
        Self {
            offset,
            limit: limit.clamp(1, MAX_RANGE_LIMIT),
        }
    }
}

struct ExactRow {
    section: String,
    layer: i16,
    wall_start_ms: f64,
    total_time_ms: f64,
    groups: Value,
    slot_time_ms: Vec<f64>,
    slot_input: Vec<String>,
    slot_flops: Vec<f64>,
    slot_bytes: Vec<f64>,
    slot_backend: Vec<u8>,
}

pub(super) fn details_descriptor(run: &DiscoveredRun) -> Option<Value> {
    let cost_log = resolve_artifact_path(&run.path, "cost_log");
    let manifests = resolve_artifact_path(&run.path, "cost_manifest");
    (cost_log.is_dir() && manifests.is_dir()).then(|| {
        json!({
            "status": "ready",
            "schema_version": 1,
            "views": ["payload"],
        })
    })
}

pub(super) async fn operation_range(
    cache: &OperationIndexCache,
    run: &DiscoveredRun,
    pool_tag: &str,
    worker_id: u16,
    range: OperationRange,
    request_id: u64,
) -> Result<Value> {
    let source = CostLogSource::simulation_worker(run, pool_tag, worker_id);
    let started = Instant::now();
    let index = cached_operation_index(cache, &source, request_id).await?;
    let value = index.range_json(pool_tag, worker_id, range);
    prof(request_id, "operation_range event=complete", started);
    Ok(value)
}

pub(super) async fn operation_seek(
    cache: &OperationIndexCache,
    run: &DiscoveredRun,
    pool_tag: &str,
    worker_id: u16,
    at_ms: f64,
    request_id: u64,
) -> Result<Value> {
    let source = CostLogSource::simulation_worker(run, pool_tag, worker_id);
    let started = Instant::now();
    if !at_ms.is_finite() {
        bail!("operation seek time must be finite");
    }
    let index = cached_operation_index(cache, &source, request_id).await?;
    let compute_started = Instant::now();
    let hit_ordinals = index.matching_ordinals(at_ms);
    let (anchor_ordinal, anchor_kind) = if let Some(ordinal) = hit_ordinals.first().copied() {
        (ordinal, "hit")
    } else {
        (index.nearest_ordinal(at_ms), "nearest")
    };
    let viewport_offset =
        centered_offset(anchor_ordinal, SEEK_VIEWPORT_LIMIT, index.operations.len());
    let buffer_offset = viewport_offset
        .saturating_sub(SEEK_VIEWPORT_LIMIT)
        .min(index.operations.len().saturating_sub(SEEK_BUFFER_LIMIT));
    let hits = hit_ordinals
        .iter()
        .map(|ordinal| index.operation_json(*ordinal))
        .collect::<Vec<_>>();
    let buffer = index.range_body(OperationRange::bounded(buffer_offset, SEEK_BUFFER_LIMIT));
    eprintln!(
        "[worker-prof] request_id={request_id} operation_seek event=compute hits={} anchor={} buffer_offset={} elapsed_ms={:.3}",
        hits.len(), anchor_ordinal, buffer_offset, compute_started.elapsed().as_secs_f64() * 1000.0
    );
    let value = json!({
        "schema_version": 1,
        "worker": {"pool_tag": pool_tag, "worker_id": worker_id},
        "worker_kind": index.worker_kind,
        "batch_role": index.batch_role,
        "at_ms": at_ms,
        "total_operations": index.operations.len(),
        "span": {"start_ms": index.span_start_ms, "end_ms": index.span_end_ms},
        "hits": hits,
        "anchor": {"ordinal": anchor_ordinal, "kind": anchor_kind},
        "suggested_viewport": {"offset": viewport_offset, "limit": SEEK_VIEWPORT_LIMIT},
        "buffer": buffer,
    });
    prof(request_id, "operation_seek event=complete", started);
    Ok(value)
}

/// Project one predictor case from the shared operation index without exposing
/// the predictor's internal cost-log worker key.
pub(super) async fn prediction_case_summary(
    cache: &OperationIndexCache,
    source: &CostLogSource,
    case_id: u64,
    request_id: u64,
) -> Result<Value> {
    let index = cached_operation_index(cache, source, request_id).await?;
    let matching = index
        .operations
        .iter()
        .filter(|operation| operation.iter_id == case_id)
        .collect::<Vec<_>>();
    if matching.is_empty() {
        bail!("prediction case {case_id} has no cost-log operations");
    }
    let total_time_ms = matching
        .iter()
        .map(|operation| operation.end_ms - operation.start_ms)
        .sum::<f64>();
    let operations = matching
        .into_iter()
        .enumerate()
        .map(|(public_operation_id, operation)| {
            json!({
                "operation_id": public_operation_id.to_string(),
                "section": index.section_names[usize::from(operation.section_id)],
                "layer": operation.layer,
                "time_ms": operation.end_ms - operation.start_ms,
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "case_id": case_id.to_string(),
        "total_time_ms": total_time_ms,
        "operations": operations,
    }))
}

pub(super) async fn prediction_operation_cost_tree(
    cache: &OperationIndexCache,
    source: &CostLogSource,
    prediction_id: &str,
    case_id: u64,
    public_operation_id: usize,
    request_id: u64,
) -> Result<Value> {
    let index = cached_operation_index(cache, source, request_id).await?;
    let operation = index
        .operations
        .iter()
        .filter(|operation| operation.iter_id == case_id)
        .nth(public_operation_id)
        .with_context(|| {
            format!("prediction operation {public_operation_id} is absent from case {case_id}")
        })?;
    exact_source_cost_tree(
        source,
        operation.iter_id,
        operation.batch_id,
        u64::from(operation.operation_id),
        json!({
            "prediction_id": prediction_id,
            "case_id": case_id.to_string(),
            "operation_id": public_operation_id.to_string(),
        }),
        request_id,
    )
    .await
}

fn centered_offset(anchor: usize, limit: usize, total: usize) -> usize {
    anchor
        .saturating_sub(limit / 2)
        .min(total.saturating_sub(limit))
}

impl WorkerOperationIndex {
    fn range_json(&self, pool_tag: &str, worker_id: u16, range: OperationRange) -> Value {
        json!({
            "schema_version": 1,
            "worker": {"pool_tag": pool_tag, "worker_id": worker_id},
            "worker_kind": self.worker_kind,
            "batch_role": self.batch_role,
            "total_operations": self.operations.len(),
            "span": {"start_ms": self.span_start_ms, "end_ms": self.span_end_ms},
            "range": {"offset": range.offset, "limit": range.limit, "returned": self.range_len(range)},
            "operations": self.range_operations(range),
        })
    }

    fn range_body(&self, range: OperationRange) -> Value {
        json!({
            "offset": range.offset,
            "limit": range.limit,
            "returned": self.range_len(range),
            "operations": self.range_operations(range),
        })
    }

    fn range_len(&self, range: OperationRange) -> usize {
        range
            .offset
            .saturating_add(range.limit)
            .min(self.operations.len())
            .saturating_sub(range.offset.min(self.operations.len()))
    }

    fn range_operations(&self, range: OperationRange) -> Vec<Value> {
        let end = range
            .offset
            .saturating_add(range.limit)
            .min(self.operations.len());
        (range.offset.min(end)..end)
            .map(|ordinal| self.operation_json(ordinal))
            .collect()
    }

    fn operation_json(&self, ordinal: usize) -> Value {
        let operation = &self.operations[ordinal];
        json!({
            "ordinal": ordinal,
            "iter_id": operation.iter_id.to_string(),
            "batch_id": operation.batch_id.to_string(),
            "operation_id": operation.operation_id.to_string(),
            "section": self.section_names[usize::from(operation.section_id)],
            "layer": operation.layer,
            "start_ms": operation.start_ms,
            "end_ms": operation.end_ms,
        })
    }

    fn matching_ordinals(&self, at_ms: f64) -> Vec<usize> {
        let upper = self
            .operations
            .partition_point(|operation| operation.start_ms <= at_ms);
        let lower = self.operations[..upper].partition_point(|operation| operation.end_ms <= at_ms);
        (lower..upper)
            .filter(|ordinal| at_ms < self.operations[*ordinal].end_ms)
            .collect()
    }

    fn nearest_ordinal(&self, at_ms: f64) -> usize {
        let upper = self
            .operations
            .partition_point(|operation| operation.start_ms <= at_ms);
        let next = (upper < self.operations.len()).then_some(upper);
        let previous = upper.checked_sub(1);
        match (previous, next) {
            (Some(previous), Some(next)) => {
                let previous_distance = at_ms - self.operations[previous].end_ms;
                let next_distance = self.operations[next].start_ms - at_ms;
                if previous_distance <= next_distance {
                    previous
                } else {
                    next
                }
            }
            (Some(previous), None) => previous,
            (None, Some(next)) => next,
            (None, None) => 0,
        }
    }
}

async fn cached_operation_index(
    cache: &OperationIndexCache,
    source: &CostLogSource,
    request_id: u64,
) -> Result<Arc<WorkerOperationIndex>> {
    let path = source.cost_path();
    if let Some(index) = cache.get(&path, request_id)? {
        eprintln!("[worker-prof] request_id={request_id} operation_index cache=hit");
        return Ok(index);
    }
    let wait_started = Instant::now();
    let _build_guard = cache.build_lock.lock().await;
    prof(
        request_id,
        "operation_index build_lock=acquired",
        wait_started,
    );
    if let Some(index) = cache.get(&path, request_id)? {
        eprintln!("[worker-prof] request_id={request_id} operation_index cache=hit_after_wait");
        return Ok(index);
    }
    eprintln!("[worker-prof] request_id={request_id} operation_index cache=miss event=build_start");
    let started = Instant::now();
    let built = Arc::new(build_operation_index(source, request_id).await?);
    prof(request_id, "operation_index event=build_end", started);
    cache.insert(path, built, request_id)
}

impl OperationIndexCache {
    fn get(&self, path: &PathBuf, request_id: u64) -> Result<Option<Arc<WorkerOperationIndex>>> {
        let wait_started = Instant::now();
        let mut store = self
            .indexes
            .write()
            .map_err(|_| anyhow!("worker operation index cache lock is poisoned"))?;
        prof(request_id, "operation_index cache_lock=write", wait_started);
        let access = self.access_clock.fetch_add(1, Ordering::Relaxed);
        Ok(store.entries.get_mut(path).map(|entry| {
            entry.last_access = access;
            Arc::clone(&entry.index)
        }))
    }

    fn insert(
        &self,
        path: PathBuf,
        built: Arc<WorkerOperationIndex>,
        request_id: u64,
    ) -> Result<Arc<WorkerOperationIndex>> {
        let wait_started = Instant::now();
        let mut store = self
            .indexes
            .write()
            .map_err(|_| anyhow!("worker operation index cache lock is poisoned"))?;
        prof(request_id, "operation_index cache_lock=write", wait_started);
        while !store.entries.is_empty()
            && store.bytes.saturating_add(built.estimated_bytes) > INDEX_CACHE_BUDGET_BYTES
        {
            let oldest_path = store
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(path, _)| path.clone())
                .expect("non-empty operation index cache");
            if let Some(evicted) = store.entries.remove(&oldest_path) {
                store.bytes = store.bytes.saturating_sub(evicted.index.estimated_bytes);
                eprintln!(
                    "[worker-prof] request_id={request_id} operation_index cache_evict path={} bytes={}",
                    oldest_path.display(), evicted.index.estimated_bytes
                );
            }
        }
        let access = self.access_clock.fetch_add(1, Ordering::Relaxed);
        store.bytes = store.bytes.saturating_add(built.estimated_bytes);
        store.entries.insert(
            path,
            CachedOperationIndex {
                index: Arc::clone(&built),
                last_access: access,
            },
        );
        eprintln!(
            "[worker-prof] request_id={request_id} operation_index cache_insert bytes={} cache_bytes={} budget_bytes={INDEX_CACHE_BUDGET_BYTES}",
            built.estimated_bytes, store.bytes
        );
        Ok(built)
    }
}

async fn build_operation_index(
    source: &CostLogSource,
    request_id: u64,
) -> Result<WorkerOperationIndex> {
    let started = Instant::now();
    let serial_config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false);
    let serial_ctx = SessionContext::new_with_config(serial_config);
    let (ctx, manifest) =
        open_source_in_session(serial_ctx, source, request_id, "operation_index").await?;
    require_columns(&ctx, TABLE, INDEX_COLUMNS).await?;
    let sql = "SELECT iter_id, batch_id, CAST(section AS VARCHAR) AS section, layer, \
                      wall_start_ms, total_time_ms FROM worker_cost";
    let dataframe = ctx
        .sql(sql)
        .await
        .with_context(|| format!("sql planning failed: {sql}"))?;
    let mut stream = dataframe
        .execute_stream()
        .await
        .with_context(|| format!("sql execution failed: {sql}"))?;
    let scan_started = Instant::now();
    let mut operations = Vec::<IndexedOperation>::new();
    let mut section_ids = HashMap::<String, u16>::new();
    let mut section_names = Vec::<String>::new();
    let mut batch_count = 0_usize;
    while let Some(batch) = stream.next().await {
        let batch = batch.context("worker operation index stream failed")?;
        batch_count += 1;
        let iter_ids = column_u64(col(&batch, "iter_id")?)?;
        let batch_ids = column_u64(col(&batch, "batch_id")?)?;
        let starts_ms = column_f64(col(&batch, "wall_start_ms")?)?;
        let durations_ms = column_f64(col(&batch, "total_time_ms")?)?;
        let sections = col(&batch, "section")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("worker operation section must be Utf8")?;
        let layers = col(&batch, "layer")?
            .as_any()
            .downcast_ref::<Int16Array>()
            .context("worker operation layer must be Int16")?;
        for row in 0..batch.num_rows() {
            if sections.is_null(row) || layers.is_null(row) {
                bail!("worker operation section and layer must be non-null");
            }
            checked_operation_end_ms(starts_ms[row], durations_ms[row])?;
            let section = sections.value(row);
            let section_id = if let Some(section_id) = section_ids.get(section) {
                *section_id
            } else {
                let section_id = u16::try_from(section_names.len())
                    .context("worker has more than u16::MAX distinct sections")?;
                section_names.push(section.to_owned());
                section_ids.insert(section.to_owned(), section_id);
                section_id
            };
            operations.push(IndexedOperation {
                iter_id: iter_ids[row],
                batch_id: batch_ids[row],
                operation_id: 0,
                section_id,
                layer: layers.value(row),
                start_ms: starts_ms[row],
                // Temporarily hold duration; the final pass replaces it with end.
                end_ms: durations_ms[row],
            });
        }
    }
    if operations.is_empty() {
        bail!("worker operation index is empty");
    }
    eprintln!(
        "[worker-prof] request_id={request_id} operation_index event=stream_decode batches={batch_count} rows={} elapsed_ms={:.3}",
        operations.len(), scan_started.elapsed().as_secs_f64() * 1000.0
    );
    let sort_started = Instant::now();
    let stream_was_sorted = operations
        .windows(2)
        .all(|pair| operation_build_order(&pair[0], &pair[1], &section_names).is_le());
    if !stream_was_sorted {
        operations
            .sort_unstable_by(|left, right| operation_build_order(left, right, &section_names));
    }
    for pair in operations.windows(2) {
        if pair[0].start_ms == pair[1].start_ms
            && pair[0].iter_id == pair[1].iter_id
            && pair[0].batch_id == pair[1].batch_id
            && pair[0].section_id == pair[1].section_id
            && pair[0].layer == pair[1].layer
            && pair[0].end_ms == pair[1].end_ms
        {
            bail!(
                "worker has duplicate operation ordering keys for iter {} batch {}",
                pair[0].iter_id,
                pair[0].batch_id
            );
        }
    }
    eprintln!(
        "[worker-prof] request_id={request_id} operation_index event=sort already_sorted={stream_was_sorted} elapsed_ms={:.3}",
        sort_started.elapsed().as_secs_f64() * 1000.0
    );
    let finalize_started = Instant::now();
    let mut next_operation_ids = HashMap::<(u64, u64), u32>::new();
    for operation in &mut operations {
        let next_id = next_operation_ids
            .entry((operation.iter_id, operation.batch_id))
            .or_default();
        operation.operation_id = *next_id;
        *next_id = next_id
            .checked_add(1)
            .context("one iter/batch has more than u32::MAX operations")?;
        operation.end_ms = checked_operation_end_ms(operation.start_ms, operation.end_ms)?;
    }
    validate_monotonic_operation_ends(&operations)?;
    operations.shrink_to_fit();
    let span_start_ms = operations[0].start_ms;
    let span_end_ms = operations.last().expect("non-empty operation index").end_ms;
    let (worker_kind, batch_role) = if is_ffn(&manifest) {
        ("afd_ffn", "slot")
    } else if has_section(&manifest, "attn") {
        ("afd_attn", "slot")
    } else {
        ("iterwise", "batch")
    };
    let estimated_bytes = operations.capacity() * size_of::<IndexedOperation>()
        + section_names
            .iter()
            .map(|section| section.capacity())
            .sum::<usize>();
    prof(
        request_id,
        "operation_index event=finalize",
        finalize_started,
    );
    eprintln!(
        "[worker-prof] request_id={request_id} operation_index event=constructed operations={} estimated_bytes={} elapsed_ms={:.3}",
        operations.len(), estimated_bytes, started.elapsed().as_secs_f64() * 1000.0
    );
    Ok(WorkerOperationIndex {
        worker_kind,
        batch_role,
        span_start_ms,
        span_end_ms,
        section_names,
        operations,
        estimated_bytes,
    })
}

fn operation_build_order(
    left: &IndexedOperation,
    right: &IndexedOperation,
    section_names: &[String],
) -> std::cmp::Ordering {
    left.start_ms
        .total_cmp(&right.start_ms)
        .then_with(|| left.iter_id.cmp(&right.iter_id))
        .then_with(|| left.batch_id.cmp(&right.batch_id))
        .then_with(|| {
            section_names[usize::from(left.section_id)]
                .cmp(&section_names[usize::from(right.section_id)])
        })
        .then_with(|| left.layer.cmp(&right.layer))
        .then_with(|| left.end_ms.total_cmp(&right.end_ms))
}

fn validate_monotonic_operation_ends(operations: &[IndexedOperation]) -> Result<()> {
    if let Some(pair) = operations
        .windows(2)
        .find(|pair| pair[1].end_ms < pair[0].end_ms)
    {
        let operation = &pair[1];
        bail!(
            "worker operation ends are not monotonic at iter {} batch {} operation {}",
            operation.iter_id,
            operation.batch_id,
            operation.operation_id
        );
    }
    Ok(())
}

fn checked_operation_end_ms(start_ms: f64, total_time_ms: f64) -> Result<f64> {
    if !start_ms.is_finite() {
        bail!("worker operation start must be finite, got {start_ms}");
    }
    if !total_time_ms.is_finite() || total_time_ms < 0.0 {
        bail!("worker operation duration must be finite and non-negative, got {total_time_ms}");
    }
    let end_ms = start_ms + total_time_ms;
    if !end_ms.is_finite() || end_ms < start_ms {
        bail!("worker operation end must be finite and not precede its start");
    }
    Ok(end_ms)
}

pub(super) async fn exact_operation_cost_tree(
    run: &DiscoveredRun,
    pool_tag: &str,
    worker_id: u16,
    iter_id: u64,
    batch_id: u64,
    operation_id: u64,
    request_id: u64,
) -> Result<Value> {
    let source = CostLogSource::simulation_worker(run, pool_tag, worker_id);
    exact_source_cost_tree(
        &source,
        iter_id,
        batch_id,
        operation_id,
        json!({
            "pool_tag": pool_tag,
            "worker_id": worker_id,
            "iter_id": iter_id.to_string(),
            "batch_id": batch_id.to_string(),
            "operation_id": operation_id.to_string(),
        }),
        request_id,
    )
    .await
}

async fn exact_source_cost_tree(
    source: &CostLogSource,
    iter_id: u64,
    batch_id: u64,
    operation_id: u64,
    mut public_identity: Value,
    request_id: u64,
) -> Result<Value> {
    let started = Instant::now();
    let (ctx, manifest_doc) = open_source(source, request_id, "cost_tree").await?;
    let require_started = Instant::now();
    require_columns(&ctx, TABLE, TREE_COLUMNS).await?;
    let optional = optional_columns(&ctx).await?;
    prof(request_id, "cost_tree event=schema", require_started);
    let mut projection = vec![
        "CAST(section AS VARCHAR) AS section",
        "layer",
        "wall_start_ms",
        "total_time_ms",
        "groups",
        "slot_time_ms",
    ];
    if optional.slot_input {
        projection.push("slot_input");
    }
    if optional.slot_flops {
        projection.push("slot_flops");
    }
    if optional.slot_bytes {
        projection.push("slot_bytes");
    }
    if optional.slot_backend {
        projection.push("slot_backend");
    }
    let sql = format!(
        "SELECT {} \
         FROM worker_cost WHERE iter_id = {iter_id} AND batch_id = {batch_id} \
         ORDER BY wall_start_ms, section, layer",
        projection.join(", ")
    );
    let query_started = Instant::now();
    let mut rows = read_exact_rows(&ctx, &sql, optional).await?;
    eprintln!(
        "[worker-prof] request_id={request_id} cost_tree event=datafusion_collect_decode rows={} elapsed_ms={:.3}",
        rows.len(), query_started.elapsed().as_secs_f64() * 1000.0
    );
    if rows.is_empty() {
        bail!("no operations for iter {iter_id}, batch {batch_id}");
    }
    let tree_started = Instant::now();
    sort_exact_operations(&mut rows, iter_id, batch_id)?;
    let operation_index = usize::try_from(operation_id).context("operation id exceeds usize")?;
    let row = rows.get(operation_index).with_context(|| {
        format!("operation {operation_id} is absent from iter {iter_id}, batch {batch_id}")
    })?;
    let end_ms = checked_operation_end_ms(row.wall_start_ms, row.total_time_ms)?;
    let manifest = manifest_doc
        .section(&row.section)
        .with_context(|| format!("worker manifest has no section {:?}", row.section))?;
    let tree = tree_json(manifest, row, 0)?;
    let identity = public_identity
        .as_object_mut()
        .context("cost-tree public identity must be an object")?;
    identity.insert("section".to_owned(), json!(row.section));
    identity.insert("layer".to_owned(), json!(row.layer));
    let value = json!({
        "schema_version": 1,
        "identity": public_identity,
        "interval": {"start_ms": row.wall_start_ms, "end_ms": end_ms},
        "inputs": [{
            "section": row.section,
            "layer": row.layer,
            "groups": row.groups,
        }],
        "tree": tree,
    });
    prof(request_id, "cost_tree event=tree_and_json", tree_started);
    prof(request_id, "cost_tree event=complete", started);
    Ok(value)
}

fn sort_exact_operations(rows: &mut [ExactRow], iter_id: u64, batch_id: u64) -> Result<()> {
    for row in rows.iter() {
        checked_operation_end_ms(row.wall_start_ms, row.total_time_ms)?;
    }
    rows.sort_by(|left, right| {
        left.wall_start_ms
            .total_cmp(&right.wall_start_ms)
            .then_with(|| left.section.cmp(&right.section))
            .then_with(|| left.layer.cmp(&right.layer))
            .then_with(|| left.total_time_ms.total_cmp(&right.total_time_ms))
    });
    if rows.windows(2).any(|pair| {
        pair[0].wall_start_ms == pair[1].wall_start_ms
            && pair[0].section == pair[1].section
            && pair[0].layer == pair[1].layer
            && pair[0].total_time_ms == pair[1].total_time_ms
    }) {
        bail!("iter {iter_id} batch {batch_id} has duplicate operation ordering keys");
    }
    Ok(())
}

async fn open_source(
    source: &CostLogSource,
    request_id: u64,
    purpose: &str,
) -> Result<(SessionContext, ManifestDoc)> {
    open_source_in_session(build_session(), source, request_id, purpose).await
}

async fn open_source_in_session(
    ctx: SessionContext,
    source: &CostLogSource,
    request_id: u64,
    purpose: &str,
) -> Result<(SessionContext, ManifestDoc)> {
    let started = Instant::now();
    let manifest_started = Instant::now();
    let manifests = read_cost_manifests(source.artifact_root())?;
    let manifest = manifests
        .get(&(source.pool_tag().to_owned(), source.worker_id()))
        .cloned()
        .with_context(|| {
            format!(
                "unknown cost-log source {}/{}",
                source.pool_tag(),
                source.worker_id()
            )
        })?;
    eprintln!(
        "[worker-prof] request_id={request_id} open_worker purpose={purpose} event=manifest_read manifests={} elapsed_ms={:.3}",
        manifests.len(), manifest_started.elapsed().as_secs_f64() * 1000.0
    );
    let path = source.cost_path();
    if !regular_file(&path) {
        bail!(
            "cost log missing for source {}/{}",
            source.pool_tag(),
            source.worker_id()
        );
    }
    let register_started = Instant::now();
    register_if_exists(&ctx, TABLE, path).await?;
    prof(
        request_id,
        &format!("open_worker purpose={purpose} event=register"),
        register_started,
    );
    prof(
        request_id,
        &format!("open_worker purpose={purpose} event=complete"),
        started,
    );
    Ok((ctx, manifest))
}

fn prof(request_id: u64, event: &str, started: Instant) {
    eprintln!(
        "[worker-prof] request_id={request_id} {event} elapsed_ms={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
}

fn has_section(manifest: &ManifestDoc, section: &str) -> bool {
    manifest
        .sections
        .iter()
        .any(|candidate| candidate.section == section)
}

fn is_ffn(manifest: &ManifestDoc) -> bool {
    has_section(manifest, "prologue") && has_section(manifest, "post_attn")
}

async fn optional_columns(ctx: &SessionContext) -> Result<OptionalColumns> {
    let table = ctx.table(TABLE).await?;
    let has = |name: &str| table.schema().field_with_name(None, name).is_ok();
    Ok(OptionalColumns {
        slot_input: has("slot_input"),
        slot_flops: has("slot_flops"),
        slot_bytes: has("slot_bytes"),
        slot_backend: has("slot_backend"),
    })
}

async fn read_exact_rows(
    ctx: &SessionContext,
    sql: &str,
    optional: OptionalColumns,
) -> Result<Vec<ExactRow>> {
    let batches = collect(ctx, sql).await?;
    let mut rows = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            rows.push(ExactRow {
                section: value_string(col(batch, "section")?, row)?,
                layer: value_f64(col(batch, "layer")?, row)? as i16,
                wall_start_ms: value_f64(col(batch, "wall_start_ms")?, row)?,
                total_time_ms: value_f64(col(batch, "total_time_ms")?, row)?,
                groups: json!(value_groups(col(batch, "groups")?, row)?),
                slot_time_ms: value_f32_list(col(batch, "slot_time_ms")?, row)?,
                slot_input: if optional.slot_input {
                    value_str_list(col(batch, "slot_input")?, row)?
                } else {
                    Vec::new()
                },
                slot_flops: if optional.slot_flops {
                    value_f32_list(col(batch, "slot_flops")?, row)?
                } else {
                    Vec::new()
                },
                slot_bytes: if optional.slot_bytes {
                    value_f32_list(col(batch, "slot_bytes")?, row)?
                } else {
                    Vec::new()
                },
                slot_backend: if optional.slot_backend {
                    value_u8_list(col(batch, "slot_backend")?, row)?
                } else {
                    Vec::new()
                },
            });
        }
    }
    Ok(rows)
}

fn tree_json(manifest: &Manifest, row: &ExactRow, node_index: usize) -> Result<Value> {
    let node = manifest
        .nodes
        .get(node_index)
        .with_context(|| format!("manifest node index {node_index} is out of range"))?;
    let label = manifest
        .node_labels
        .get(node_index)
        .and_then(|label| label.clone());
    match node {
        FlatCostNode::Leaf(slot_index) => {
            let slot = manifest
                .slots
                .get(*slot_index)
                .with_context(|| format!("manifest slot index {slot_index} is out of range"))?;
            let time_ms = value_at(&row.slot_time_ms, *slot_index, "slot_time_ms")?;
            let flops = row
                .slot_flops
                .get(*slot_index)
                .copied()
                .filter(|value| *value > 0.0);
            let bytes = row
                .slot_bytes
                .get(*slot_index)
                .copied()
                .filter(|value| *value > 0.0);
            let backend = row
                .slot_backend
                .get(*slot_index)
                .and_then(|index| (*index != u8::MAX).then_some(*index as usize))
                .and_then(|index| slot.backends().get(index).cloned());
            let input = row
                .slot_input
                .get(*slot_index)
                .filter(|input| !input.is_empty())
                .map(|input| {
                    serde_json::from_str(input).unwrap_or_else(|_| Value::String(input.clone()))
                })
                .unwrap_or(Value::Null);
            Ok(json!({
                "kind": "leaf",
                "slot": {
                    "name": slot.name,
                    "kind": slot.kind,
                    "kernel_config": slot.kernel_config,
                    "backend": backend,
                },
                "base": time_ms,
                "stats": {
                    "input": input,
                    "flops": flops,
                    "bytes": bytes,
                    "tflops": flops.filter(|_| time_ms > 0.0).map(|value| value / time_ms / 1.0e9),
                    "gbps": bytes.filter(|_| time_ms > 0.0).map(|value| value / time_ms / 1.0e6),
                },
            }))
        }
        FlatCostNode::Sum { children } => container_json(
            "sum",
            label,
            tree_children(manifest, row, children.clone())?,
            None,
        ),
        FlatCostNode::Max { overlap, children } => container_json(
            "max",
            label,
            tree_children(manifest, row, children.clone())?,
            Some(("overlap", json!(overlap))),
        ),
        FlatCostNode::Scale { n, children } => container_json(
            "scale",
            label,
            tree_children(manifest, row, children.clone())?,
            Some(("n", json!(n))),
        ),
    }
}

fn container_json(
    kind: &str,
    label: Option<String>,
    children: Vec<Value>,
    extra: Option<(&str, Value)>,
) -> Result<Value> {
    let mut node = json!({"kind": kind, "children": children});
    let object = node
        .as_object_mut()
        .expect("container JSON is always an object");
    if let Some(label) = label {
        object.insert("label".to_owned(), Value::String(label));
    }
    if let Some((key, value)) = extra {
        object.insert(key.to_owned(), value);
    }
    Ok(node)
}

fn tree_children(
    manifest: &Manifest,
    row: &ExactRow,
    children: std::ops::Range<usize>,
) -> Result<Vec<Value>> {
    children
        .map(|child| tree_json(manifest, row, child))
        .collect()
}

fn value_at(values: &[f64], index: usize, field: &str) -> Result<f64> {
    values
        .get(index)
        .copied()
        .ok_or_else(|| anyhow!("{field} has no value at manifest slot {index}"))
}

fn value_u8_list(array: &arrow_array::ArrayRef, row: usize) -> Result<Vec<u8>> {
    let list = array
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("expected List array"))?;
    if list.is_null(row) {
        return Ok(Vec::new());
    }
    let values = list.value(row);
    let values = values
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or_else(|| anyhow!("expected List<UInt8> values"))?;
    Ok((0..values.len()).map(|index| values.value(index)).collect())
}

fn column_u64(array: &arrow_array::ArrayRef) -> Result<Vec<u64>> {
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return (0..values.len())
            .map(|row| {
                (!values.is_null(row))
                    .then(|| values.value(row))
                    .context("expected non-null integer identity")
            })
            .collect();
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return (0..values.len())
            .map(|row| {
                (!values.is_null(row))
                    .then(|| values.value(row) as u64)
                    .context("expected non-null integer identity")
            })
            .collect();
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return (0..values.len())
            .map(|row| {
                if values.is_null(row) {
                    bail!("expected non-null integer identity");
                }
                u64::try_from(values.value(row)).context("integer identity is negative")
            })
            .collect();
    }
    bail!("expected UInt64, UInt32, or non-negative Int64 identity")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::manifest::LeafDesc;

    fn row(section: &str, layer: i16) -> ExactRow {
        ExactRow {
            section: section.to_owned(),
            layer,
            wall_start_ms: 1.0,
            total_time_ms: 2.0,
            groups: json!([]),
            slot_time_ms: vec![2.0, 1.0],
            slot_input: vec![r#"{"m":8}"#.to_owned(), String::new()],
            slot_flops: vec![4.0e9, 0.0],
            slot_bytes: vec![0.0, 2.0e6],
            slot_backend: vec![1, u8::MAX],
        }
    }

    #[test]
    fn exact_operations_sort_without_collapsing_sections() {
        let mut rows = vec![row("pre_attn", 0), row("prologue", -1)];

        sort_exact_operations(&mut rows, 7, 1).unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].section, "pre_attn");
        assert_eq!(rows[1].section, "prologue");
    }

    #[test]
    fn ambiguous_operation_ordinals_and_invalid_intervals_fail_loud() {
        let mut duplicate_rows = vec![row("pre_attn", 0), row("pre_attn", 0)];
        assert!(sort_exact_operations(&mut duplicate_rows, 7, 1).is_err());
        assert!(checked_operation_end_ms(f64::NAN, 1.0).is_err());
        assert!(checked_operation_end_ms(1.0, -1.0).is_err());
        assert_eq!(checked_operation_end_ms(1.0, 0.0).unwrap(), 1.0);
    }

    #[test]
    fn exact_tree_preserves_overlap_backend_input_and_rates() {
        let manifest = Manifest {
            slots: vec![
                LeafDesc {
                    name: "gemm".to_owned(),
                    kind: "single_gemm".to_owned(),
                    kernel_config: json!({
                        "n": {"value": 8, "expression": null, "bindings": {}},
                        "backends": ["torch", "triton"],
                    }),
                },
                LeafDesc {
                    name: "copy".to_owned(),
                    kind: "elementwise".to_owned(),
                    kernel_config: json!({"backends": []}),
                },
            ],
            nodes: vec![
                FlatCostNode::Max {
                    overlap: 2.0,
                    children: 1..3,
                },
                FlatCostNode::Leaf(0),
                FlatCostNode::Leaf(1),
            ],
            node_labels: vec![Some("parallel".to_owned()), None, None],
        };

        let tree = tree_json(&manifest, &row("post_attn", 1), 0).unwrap();

        assert_eq!(tree["kind"], "max");
        assert_eq!(tree["overlap"], 2.0);
        assert_eq!(tree["children"][0]["slot"]["backend"], "triton");
        assert_eq!(tree["children"][0]["stats"]["input"]["m"], 8);
        assert_eq!(tree["children"][0]["stats"]["tflops"], 2.0);
        assert!(tree["children"][1]["stats"]["flops"].is_null());
        assert_eq!(tree["children"][1]["stats"]["gbps"], 2.0);
    }

    #[test]
    fn integer_identity_reader_does_not_round_large_ids() {
        let expected = (1_u64 << 53) + 17;
        let values: arrow_array::ArrayRef = Arc::new(UInt64Array::from(vec![expected]));

        assert_eq!(column_u64(&values).unwrap(), vec![expected]);
    }

    fn indexed_operation(
        iter_id: u64,
        batch_id: u64,
        operation_id: u32,
        start_ms: f64,
        end_ms: f64,
    ) -> IndexedOperation {
        IndexedOperation {
            iter_id,
            batch_id,
            operation_id,
            section_id: 0,
            layer: operation_id as i16,
            start_ms,
            end_ms,
        }
    }

    fn operation_index(operations: Vec<IndexedOperation>) -> WorkerOperationIndex {
        WorkerOperationIndex {
            worker_kind: "afd_ffn",
            batch_role: "slot",
            span_start_ms: operations
                .first()
                .map_or(0.0, |operation| operation.start_ms),
            span_end_ms: operations.last().map_or(0.0, |operation| operation.end_ms),
            section_names: vec!["post_attn".to_owned()],
            operations,
            estimated_bytes: 0,
        }
    }

    #[test]
    fn operation_seek_returns_overlaps_with_half_open_boundaries() {
        let index = operation_index(vec![
            indexed_operation(7, 0, 0, 10.0, 20.0),
            indexed_operation(8, 1, 0, 15.0, 25.0),
            indexed_operation(9, 0, 0, 20.0, 30.0),
        ]);

        let overlap_ordinals = index.matching_ordinals(17.0);
        assert_eq!(overlap_ordinals, vec![0, 1]);

        let boundary_ordinals = index.matching_ordinals(20.0);
        assert_eq!(boundary_ordinals, vec![1, 2]);
        assert_eq!(index.nearest_ordinal(0.0), 0);
        assert_eq!(index.nearest_ordinal(100.0), 2);
    }

    #[test]
    fn seek_index_requires_monotonic_ends_but_allows_real_overlap() {
        let overlapping = vec![
            indexed_operation(7, 0, 0, 10.0, 20.0),
            indexed_operation(8, 1, 0, 15.0, 25.0),
        ];
        assert!(validate_monotonic_operation_ends(&overlapping).is_ok());

        let decreasing = vec![
            indexed_operation(7, 0, 0, 10.0, 30.0),
            indexed_operation(8, 1, 0, 15.0, 25.0),
        ];
        assert!(validate_monotonic_operation_ends(&decreasing).is_err());
    }

    #[test]
    fn operation_range_is_globally_ordinal_and_bounded() {
        let index = operation_index(vec![
            indexed_operation(7, 2, 0, 10.0, 11.0),
            indexed_operation(7, 2, 1, 11.0, 12.0),
            indexed_operation(8, 1, 0, 12.0, 13.0),
        ]);

        let range = index.range_json("ffn", 1, OperationRange::bounded(1, 128));

        assert_eq!(range["worker_kind"], "afd_ffn");
        assert_eq!(range["batch_role"], "slot");
        assert_eq!(range["range"]["offset"], 1);
        assert_eq!(range["range"]["returned"], 2);
        assert_eq!(range["operations"][0]["ordinal"], 1);
        assert_eq!(range["operations"][0]["operation_id"], "1");
        assert_eq!(range["operations"][1]["iter_id"], "8");
        assert!(index
            .range_operations(OperationRange::bounded(99, 128))
            .is_empty());
        assert_eq!(SEEK_BUFFER_LIMIT, 192);
        assert_eq!(centered_offset(200, SEEK_VIEWPORT_LIMIT, 1_000), 168);
        assert_eq!(centered_offset(999, SEEK_VIEWPORT_LIMIT, 1_000), 936);
    }
}
