//! DataFusion session + parquet registration + Arrow→Vec extraction helpers.
//! Ported from ref `analyze-rust/src/session.rs`, trimmed to what the analyzer
//! currently needs. The drift guard ([`require_columns`]) is the one addition:
//! the analyzer reads sim parquet by column name, so it asserts the expected
//! columns exist and fails loud (not silently NaN) when the log schema moves.

use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    ListArray, RecordBatch, StringArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
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
    ctx.register_parquet(name, path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .with_context(|| format!("register_parquet({name})"))?;
    Ok(true)
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
