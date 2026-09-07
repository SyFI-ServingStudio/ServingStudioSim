//! DataFusion session + parquet registration + Arrow→Vec extraction helpers.
//! Ported from ref `analyze-rust/src/session.rs`, trimmed to what the analyzer
//! currently needs. The drift guard ([`require_columns`]) is the one addition:
//! the analyzer reads sim parquet by column name, so it asserts the expected
//! columns exist and fails loud (not silently NaN) when the log schema moves.

use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    ListArray, RecordBatch, StringArray, StructArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};

pub fn build_session() -> SessionContext {
    let config = SessionConfig::new()
        .with_repartition_file_scans(true)
        .with_repartition_aggregations(true);
    SessionContext::new_with_config(config)
}

pub async fn register_if_exists(ctx: &SessionContext, name: &str, path: PathBuf) -> Result<bool> {
    // Subjects share one `SessionContext` and each declares the streams it needs,
    // so the same logical table (e.g. `state` = request_state.parquet) is often
    // requested by several subjects in one `analyze run`. Register it once: a
    // later request for an already-registered table just reuses it.
    if ctx.table_exist(name)? {
        return Ok(true);
    }
    if !path.exists() {
        return Ok(false);
    }
    match ctx
        .register_parquet(name, path.to_str().unwrap(), ParquetReadOptions::default())
        .await
    {
        Ok(()) => Ok(true),
        // Subjects now run concurrently over one shared ctx, so a peer may have
        // registered this same shared table between the `table_exist` check above and
        // here (DataFusion's `register_parquet` errors on a duplicate rather than
        // replacing). If the table is present now, the race is benign — treat it as
        // registered; otherwise the error is real.
        Err(_) if ctx.table_exist(name)? => Ok(true),
        Err(e) => Err(e).with_context(|| format!("register_parquet({name})")),
    }
}

/// Canonical table name for the per-worker `cost_log/` directory union.
pub const COST_LOG_TABLE: &str = "cost_log";

/// Register the per-worker `cost_log/` DIRECTORY (`raw/cost_log/worker_*.parquet`)
/// as the [`COST_LOG_TABLE`] table, so DataFusion unions every worker's parquet —
/// PD has 1 prefill + N decode workers; unified has one. Returns `false` when the
/// dir is absent (caller emits `unavailable` / `bail!`).
///
/// This is the ONE correct way to read cost_log. Do NOT register the legacy
/// single-file `cost_log.parquet`: it predates the per-worker split (commit that
/// added `cost_log/<worker>.parquet`), so on any multi-worker run it is either
/// missing (→ false negative) or a stale leftover (→ silently wrong numbers).
pub async fn register_cost_log(ctx: &SessionContext, log_dir: &std::path::Path) -> Result<bool> {
    let path = crate::io::resolve_artifact_path(log_dir, "cost_log");
    register_if_exists(ctx, COST_LOG_TABLE, path).await
}

/// Canonical table name for the per-worker `kv_snapshot/` directory union.
pub const KV_SNAPSHOT_TABLE: &str = "kv_snapshot";

/// Register the per-worker `kv_snapshot/` DIRECTORY
/// (`raw/kv_snapshot/worker_*.parquet`) as the [`KV_SNAPSHOT_TABLE`] table — one
/// file per KV-bearing worker (unified: one; PD: prefill + decode; AFD: every attn
/// shard), unioned by DataFusion exactly like [`register_cost_log`]. Returns
/// `false` when the dir is absent (a run with KV logging off, or a deployment with
/// no KV pool), so the `kv-occupancy` subject emits `unavailable` rather than
/// `bail!`.
pub async fn register_kv_snapshot(ctx: &SessionContext, log_dir: &std::path::Path) -> Result<bool> {
    let path = crate::io::resolve_artifact_path(log_dir, "kv_snapshot");
    register_if_exists(ctx, KV_SNAPSHOT_TABLE, path).await
}

/// Canonical table name for the single-file `gpu_cluster` stream.
pub const GPU_CLUSTER_TABLE: &str = "gpu_cluster";

/// Register the run's single `raw/gpu_cluster.parquet` (the cross-worker transfer
/// log) as [`GPU_CLUSTER_TABLE`]. Unlike `cost_log`, this is ONE file (the whole
/// run shares one `GpuCluster` writer), so no directory union. Returns `false`
/// when absent — the transfer overlay is optional, so callers skip it rather than
/// `bail!` (e.g. a unified/single-worker run that never transfers).
pub async fn register_gpu_cluster(ctx: &SessionContext, log_dir: &std::path::Path) -> Result<bool> {
    let path = crate::io::resolve_artifact_path(log_dir, "gpu_cluster.parquet");
    register_if_exists(ctx, GPU_CLUSTER_TABLE, path).await
}

/// Drift guard: fail with a clear message if any expected column is absent from
/// a registered table, instead of letting a later extraction read NaN/null.
pub async fn require_columns(ctx: &SessionContext, table: &str, columns: &[&str]) -> Result<()> {
    let df = ctx.table(table).await?;
    let missing: Vec<&str> = columns
        .iter()
        .copied()
        .filter(|c| df.schema().field_with_name(None, c).is_err())
        .collect();
    if !missing.is_empty() {
        bail!(
            "table `{table}` is missing expected columns [{}] — sim log schema drifted; \
             update the analyzer's column contract",
            missing.join(", ")
        );
    }
    Ok(())
}

pub async fn collect(ctx: &SessionContext, sql: &str) -> Result<Vec<RecordBatch>> {
    ctx.sql(sql)
        .await
        .with_context(|| format!("sql planning failed: {sql}"))?
        .collect()
        .await
        .with_context(|| format!("sql execution failed: {sql}"))
}

/// Column index by name within a batch (batches in one `collect` share a schema).
pub fn col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef> {
    let idx = batch
        .schema()
        .index_of(name)
        .with_context(|| format!("column `{name}` not in batch"))?;
    Ok(batch.column(idx))
}

pub fn value_f64(array: &ArrayRef, row: usize) -> Result<f64> {
    if array.is_null(row) {
        return Ok(f64::NAN);
    }
    if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(a.value(row));
    }
    if let Some(a) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok(a.value(row) as f64);
    }
    if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(a.value(row) as f64);
    }
    if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(a.value(row) as f64);
    }
    if let Some(a) = array.as_any().downcast_ref::<Int16Array>() {
        return Ok(a.value(row) as f64);
    }
    if let Some(a) = array.as_any().downcast_ref::<Int8Array>() {
        return Ok(a.value(row) as f64);
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(a.value(row) as f64);
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(a.value(row) as f64);
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt16Array>() {
        return Ok(a.value(row) as f64);
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt8Array>() {
        return Ok(a.value(row) as f64);
    }
    bail!("unsupported numeric array type")
}

/// Extract an entire numeric column as `Vec<f64>` in one type dispatch (null →
/// NaN), instead of [`value_f64`]'s per-element 11-branch downcast. The hot
/// cost_log paths read tens of millions of rows, so paying the downcast once per
/// column rather than once per element is the difference between seconds and
/// minutes. Same numeric-type coverage as `value_f64`.
#[allow(clippy::unnecessary_cast)] // uniform `as f64` across the int/float arms
pub fn column_f64(array: &ArrayRef) -> Result<Vec<f64>> {
    macro_rules! collect_as {
        ($ty:ty) => {{
            let a = array.as_any().downcast_ref::<$ty>().unwrap();
            return Ok((0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        f64::NAN
                    } else {
                        a.value(i) as f64
                    }
                })
                .collect());
        }};
    }
    if array.as_any().is::<Float64Array>() {
        collect_as!(Float64Array);
    } else if array.as_any().is::<Float32Array>() {
        collect_as!(Float32Array);
    } else if array.as_any().is::<Int64Array>() {
        collect_as!(Int64Array);
    } else if array.as_any().is::<Int32Array>() {
        collect_as!(Int32Array);
    } else if array.as_any().is::<Int16Array>() {
        collect_as!(Int16Array);
    } else if array.as_any().is::<Int8Array>() {
        collect_as!(Int8Array);
    } else if array.as_any().is::<UInt64Array>() {
        collect_as!(UInt64Array);
    } else if array.as_any().is::<UInt32Array>() {
        collect_as!(UInt32Array);
    } else if array.as_any().is::<UInt16Array>() {
        collect_as!(UInt16Array);
    } else if array.as_any().is::<UInt8Array>() {
        collect_as!(UInt8Array);
    }
    bail!("unsupported numeric array type for column_f64")
}

pub fn value_string(array: &ArrayRef, row: usize) -> Result<String> {
    if array.is_null(row) {
        return Ok(String::new());
    }
    let strs = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("expected Utf8 array"))?;
    Ok(strs.value(row).to_string())
}

/// One row of a `List<Float32>` column as an owned `Vec<f64>` (e.g.
/// `output_token_times`). Null row → empty vec.
pub fn value_f32_list(array: &ArrayRef, row: usize) -> Result<Vec<f64>> {
    let list = array
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("expected List array"))?;
    if list.is_null(row) {
        return Ok(Vec::new());
    }
    let values = list.value(row);
    let floats = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| anyhow!("expected List<Float32> values"))?;
    Ok((0..floats.len()).map(|i| floats.value(i) as f64).collect())
}

/// One iteration's per-HP-group arch input, read from the `groups` `List<Struct>`
/// column (analyzer-side mirror of the sim's `GroupInputLog`). Only the fields the
/// breakdown header renders are kept.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct GroupInput {
    pub batch_tokens: u32,
    pub prefill_tokens: u32,
    pub decode_request_count: u32,
    pub decode_kv_total: u32,
    /// Per prefill request `(prefix_len, append_len)`, re-zipped from the two
    /// parallel `prefill_*_lens` sub-lists the sim writer splits them into.
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
}

/// One row of the `groups` `List<Struct>` column → that iteration's per-group
/// inputs. Null row → empty vec. Struct field order is fixed by the sim writer
/// (`simulator/src/log/rows.rs::build_groups_column`): 0 batch_tokens, 1
/// prefill_tokens, 2 decode_request_count, 3 decode_kv_total, 4 prefill_prefix_lens
/// (`List<UInt32>`), 5 prefill_append_lens (`List<UInt32>`), 6 decode_query_rows.
/// These indices are positional, so the writer appends new columns after the last
/// one and never inserts. Field 6 is written but not read here yet.
pub fn value_groups(array: &ArrayRef, row: usize) -> Result<Vec<GroupInput>> {
    let list = array
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("expected List array for `groups`"))?;
    if list.is_null(row) {
        return Ok(Vec::new());
    }
    let structs_ref = list.value(row);
    let structs = structs_ref
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| anyhow!("expected List<Struct> for `groups`"))?;
    let u32_col = |idx: usize| -> Result<&UInt32Array> {
        structs
            .column(idx)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| anyhow!("groups struct field {idx} is not UInt32"))
    };
    let (bt, pt, dc, dk) = (u32_col(0)?, u32_col(1)?, u32_col(2)?, u32_col(3)?);
    let list_col = |idx: usize| -> Result<&ListArray> {
        structs
            .column(idx)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("groups struct field {idx} is not List"))
    };
    let (prefix_lens, append_lens) = (list_col(4)?, list_col(5)?);
    let u32_items = |la: &ListArray, g: usize| -> Result<Vec<u32>> {
        let vals = la.value(g);
        let arr = vals
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| anyhow!("groups prefill list values are not UInt32"))?;
        Ok((0..arr.len()).map(|i| arr.value(i)).collect())
    };
    let mut out = Vec::with_capacity(structs.len());
    for g in 0..structs.len() {
        let prefix = u32_items(prefix_lens, g)?;
        let append = u32_items(append_lens, g)?;
        out.push(GroupInput {
            batch_tokens: bt.value(g),
            prefill_tokens: pt.value(g),
            decode_request_count: dc.value(g),
            decode_kv_total: dk.value(g),
            prefill_chunk_pairs: prefix.into_iter().zip(append).collect(),
        });
    }
    Ok(out)
}

/// One row of a `List<Utf8>` column as owned `Vec<String>` (e.g. `slot_input`).
/// Null row → empty vec. A null *element* within the list → empty string, so the
/// returned vec stays index-aligned to its sibling lists (e.g. `slot_time_ms`).
pub fn value_str_list(array: &ArrayRef, row: usize) -> Result<Vec<String>> {
    let list = array
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("expected List array"))?;
    if list.is_null(row) {
        return Ok(Vec::new());
    }
    let values = list.value(row);
    let strs = values
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("expected List<Utf8> values"))?;
    Ok((0..strs.len())
        .map(|i| {
            if strs.is_null(i) {
                String::new()
            } else {
                strs.value(i).to_string()
            }
        })
        .collect())
}
