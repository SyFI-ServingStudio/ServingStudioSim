//! Speculation conserves executed queries, including rejected candidates.
//! Cost-log inputs and admission/completion observations are independent sources.

use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Geometry {
    draft_tokens: u64,
    max_model_len: u64,
    prefill: Vec<(u64, u64)>,
    decode: Vec<(u64, u64)>,
}

#[derive(Default, Deserialize)]
struct Progress {
    query_width: u64,
    prefill_chunks: u64,
    decode_rounds: u64,
    resident_kv_sum: u64,
    emitted_tokens: u64,
    pending_prefill: Option<(u64, u64)>,
    pending_decode: Option<u64>,
}

#[derive(Default)]
struct Work {
    prefill_chunks: f64,
    decode_rounds: f64,
    query_rows: f64,
    resident_kv: f64,
    verify_pairs: f64,
    recurrent_rows: f64,
}

pub(super) async fn checks(
    ctx: &SessionContext,
    actual: &Actual,
    expected: &mut Expected,
) -> Result<Vec<Value>> {
    require_columns(ctx, "slo", &["speculative_progress"]).await?;
    let mut observed = Work::default();
    let mut raw_totals = WorkloadTotals::default();
    for totals in collect_workload_by_worker(ctx).await?.values() {
        ensure!(
            !totals.speculative_geometry.is_empty(),
            "missing per-stage cost-log geometry"
        );
        raw_totals.add(totals);
        for (encoded, count) in &totals.speculative_geometry {
            let geometry: Geometry = serde_json::from_str(encoded)?;
            ensure!(
                geometry.draft_tokens > 0 && geometry.draft_tokens < geometry.max_model_len,
                "invalid logged draft depth"
            );
            let width = geometry.draft_tokens + 1;
            observed.prefill_chunks += geometry.prefill.len() as f64 * count;
            observed.recurrent_rows += (geometry.prefill.len() + geometry.decode.len()) as f64
                * (geometry.draft_tokens - 1) as f64
                * count;
            for (final_context, query) in geometry.decode {
                ensure!(
                    query == width
                        && final_context >= query
                        && final_context <= geometry.max_model_len,
                    "invalid logged verify geometry"
                );
                let resident = (final_context - query) as f64;
                let q = query as f64;
                observed.decode_rounds += count;
                observed.query_rows += q * count;
                observed.resident_kv += resident * count;
                observed.verify_pairs += (q * resident + q * (q + 1.0) / 2.0) * count;
            }
        }
    }
    let mut committed = Work::default();
    let mut output_actual = 0.0;
    let mut output_expected = 0.0;
    let mut violations = 0.0;
    // Parquet may preserve dictionary encoding; normalize strings at projection
    // just as the shared cost-log readers do for pool/section names.
    let batches = collect(ctx, "SELECT CAST(speculative_progress AS VARCHAR) AS speculative_progress, num_output_tokens, completed, prefill_processed, reprocessed_prefill_completed FROM slo").await?;
    for batch in batches {
        let progress_column = col(&batch, "speculative_progress")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("speculative_progress must be Utf8"))?;
        let episodes = col(&batch, "reprocessed_prefill_completed")?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("reprocessed episodes must be a list"))?;
        for row in 0..batch.num_rows() {
            let output = value_f64(col(&batch, "num_output_tokens")?, row)?;
            let processed = value_f64(col(&batch, "prefill_processed")?, row)?;
            if progress_column.is_null(row) {
                ensure!(
                    output == 0.0 && processed == 0.0,
                    "admitted request lacks speculative progress"
                );
                continue;
            }
            let progress: Progress = serde_json::from_str(progress_column.value(row))?;
            ensure!(progress.query_width > 1, "invalid request verify width");
            let q = progress.query_width as f64;
            let rounds =
                progress.decode_rounds as f64 + f64::from(progress.pending_decode.is_some());
            let resident =
                progress.resident_kv_sum as f64 + progress.pending_decode.unwrap_or(0) as f64;
            let chunks =
                progress.prefill_chunks as f64 + f64::from(progress.pending_prefill.is_some());
            committed.prefill_chunks += chunks;
            committed.decode_rounds += rounds;
            committed.query_rows += rounds * q;
            committed.resident_kv += resident;
            committed.verify_pairs += q * resident + rounds * q * (q + 1.0) / 2.0;
            committed.recurrent_rows += (chunks + rounds) * (q - 2.0);
            if let Some((prefix, append)) = progress.pending_prefill {
                let (p, a) = (prefix as f64, append as f64);
                let pairs = a * p + a * (a + 1.0) / 2.0;
                expected.prefill_tokens += a;
                expected.causal += pairs;
                expected.cold_causal += pairs;
            }
            let completed = col(&batch, "completed")?
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| anyhow!("completed must be Boolean"))?
                .value(row);
            if completed
                && (progress.pending_decode.is_some() || progress.pending_prefill.is_some())
            {
                violations += 1.0;
            }
            if progress.emitted_tokens < progress.decode_rounds
                || progress.emitted_tokens > progress.decode_rounds * progress.query_width
            {
                violations += 1.0;
            }
            let episode_values = episodes.value(row);
            let episode_values = episode_values
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| anyhow!("reprocessed completion must be Boolean"))?;
            let completed_episodes = episode_values.iter().filter(|v| *v == Some(true)).count();
            output_actual += output;
            output_expected += progress.emitted_tokens as f64
                + f64::from(output > 0.0)
                + completed_episodes as f64;
        }
    }
    let excluded = [
        "decode_passes",
        "ffn_token_pass",
        "decode_kv_sum",
        "cost_log_batch_self_consistency",
    ];
    let mut checks: Vec<Value> = checks_for_mode(WorkloadMode::Iterwise, actual, expected)
        .into_iter()
        .filter(|check| !excluded.contains(&check["name"].as_str().unwrap_or("")))
        .collect();
    for (name, description, a, e) in [
        (
            "decode_passes",
            "Executed verify rounds versus completed and pending request rounds",
            observed.decode_rounds,
            committed.decode_rounds,
        ),
        (
            "verify_query_rows",
            "All verify queries, including rejected candidates",
            observed.query_rows,
            committed.query_rows,
        ),
        (
            "verify_causal_pairs",
            "Per-query causal context versus request-side resident KV",
            observed.verify_pairs,
            committed.verify_pairs,
        ),
        (
            "decode_kv_sum",
            "Resident KV before verify; pending query excluded",
            observed.resident_kv,
            committed.resident_kv,
        ),
        (
            "prefill_chunks",
            "Executed chunks versus completed and pending request chunks",
            observed.prefill_chunks,
            committed.prefill_chunks,
        ),
        (
            "mtp_recurrent_rows",
            "One row per endpoint per recurrent draft position",
            observed.recurrent_rows,
            committed.recurrent_rows,
        ),
        (
            "ffn_token_pass",
            "Target and first MTP pass forward prefill plus all verify rows",
            actual.batch_tokens,
            expected.prefill_tokens + committed.query_rows,
        ),
        (
            "cost_log_batch_self_consistency",
            "Batch tokens equal prefill plus verify queries",
            actual.batch_tokens,
            actual.prefill_tokens + observed.query_rows,
        ),
        (
            "cost_log_resident_consistency",
            "Raw geometry reconstructs the logged resident KV total",
            raw_totals.decode_kv,
            observed.resident_kv,
        ),
        (
            "cost_log_request_consistency",
            "Raw geometry reconstructs the logged decode request count",
            raw_totals.decode_passes,
            observed.decode_rounds,
        ),
        (
            "speculative_output_tokens",
            "Committed outputs equal prefill outputs plus accepted verify output",
            output_actual,
            output_expected,
        ),
        (
            "speculative_progress_contract",
            "Completed requests have no pending work; each committed round emits 1..q tokens",
            violations,
            0.0,
        ),
    ] {
        checks.push(check(name, description, a, e, 0.0));
    }
    Ok(checks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::arrow::json::ReaderBuilder;
    use std::sync::Arc;

    fn register(ctx: &SessionContext, name: &str, schema: Schema, row: Value) {
        let encoded = row.to_string();
        let batch = ReaderBuilder::new(Arc::new(schema))
            .build(encoded.as_bytes())
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        ctx.register_batch(name, batch).unwrap();
    }

    fn fixture(
        pending: &str,
        final_context: u64,
        output: u64,
    ) -> (SessionContext, Actual, Expected) {
        let ctx = SessionContext::new();
        let u32_list = || DataType::List(Arc::new(Field::new("item", DataType::UInt32, false)));
        let group_fields = vec![
            Field::new("batch_tokens", DataType::UInt32, false),
            Field::new("prefill_tokens", DataType::UInt32, false),
            Field::new("decode_request_count", DataType::UInt32, false),
            Field::new("decode_kv_total", DataType::UInt32, false),
            Field::new("prefill_prefix_lens", u32_list(), false),
            Field::new("prefill_append_lens", u32_list(), false),
            Field::new("speculative_geometry", DataType::Utf8, true),
        ];
        let decode = pending != "prefill";
        let geometry = json!({"draft_tokens": 5, "max_model_len": 8192,
            "prefill": [[0,8]], "decode": if decode { vec![(final_context,6)] } else { vec![] }});
        register(
            &ctx,
            "cost_log",
            Schema::new(vec![
                Field::new("pool_tag", DataType::Utf8, false),
                Field::new("worker_id", DataType::UInt16, false),
                Field::new(
                    "groups",
                    DataType::List(Arc::new(Field::new(
                        "item",
                        DataType::Struct(group_fields.into()),
                        false,
                    ))),
                    false,
                ),
            ]),
            json!({"pool_tag":"main", "worker_id":0, "groups":[{
                "batch_tokens": if decode {14} else {8}, "prefill_tokens":8,
                "decode_request_count":u32::from(decode), "decode_kv_total":if decode {8} else {0},
                "prefill_prefix_lens":[0],"prefill_append_lens":[8],"speculative_geometry":geometry.to_string()
            }]}),
        );
        let progress = json!({"query_width":6,"prefill_chunks":u32::from(pending != "prefill"),
            "decode_rounds":u32::from(pending.is_empty()),
            "resident_kv_sum":if pending.is_empty() {8} else {0},
            "emitted_tokens":if pending.is_empty() {6} else {0},
            "pending_prefill":if pending == "prefill" {Some((0,8))} else {None},
            "pending_decode":if pending == "decode" {Some(8)} else {None}});
        register(
            &ctx,
            "slo",
            Schema::new(vec![
                Field::new("speculative_progress", DataType::Utf8, true),
                Field::new("num_output_tokens", DataType::UInt32, false),
                Field::new("completed", DataType::Boolean, false),
                Field::new("prefill_processed", DataType::UInt32, false),
                Field::new(
                    "reprocessed_prefill_completed",
                    DataType::List(Arc::new(Field::new("item", DataType::Boolean, false))),
                    false,
                ),
            ]),
            json!({"speculative_progress":progress.to_string(), "num_output_tokens":output,
            "completed":pending.is_empty(), "prefill_processed":if pending == "prefill" {0} else {8},
            "reprocessed_prefill_completed":[]}),
        );
        let actual = Actual {
            prefill_tokens: 8.0,
            causal: 36.0,
            batch_tokens: if decode { 14.0 } else { 8.0 },
            ..Actual::default()
        };
        let expected = Expected {
            prefill_tokens: if pending == "prefill" { 0.0 } else { 8.0 },
            causal: if pending == "prefill" { 0.0 } else { 36.0 },
            cold_causal: if pending == "prefill" { 0.0 } else { 36.0 },
            ..Expected::default()
        };
        (ctx, actual, expected)
    }

    #[tokio::test]
    async fn completed_and_pending_work_conserve_without_boundary_allowances() {
        for (pending, output) in [("", 7), ("decode", 1), ("prefill", 0)] {
            let (ctx, actual, mut expected) = fixture(pending, 14, output);
            let checks = checks(&ctx, &actual, &mut expected).await.unwrap();
            assert!(
                checks.iter().all(|row| row["status"] == "OK"),
                "{pending}: {checks:#?}"
            );
            assert!(checks
                .iter()
                .all(|row| row.get("positive_boundary_allowance").is_none()));
        }
    }

    #[tokio::test]
    async fn dictionary_encoded_request_progress_conserves() {
        let (ctx, actual, mut expected) = fixture("decode", 14, 1);
        let batches = collect(&ctx, "SELECT * FROM slo").await.unwrap();
        let batch = &batches[0];
        let mut columns = batch.columns().to_vec();
        let encoded_type =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        columns[0] = datafusion::arrow::compute::cast(&columns[0], &encoded_type).unwrap();
        let mut fields: Vec<Field> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields[0] = fields[0].clone().with_data_type(encoded_type);
        let encoded =
            arrow_array::RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        ctx.deregister_table("slo").unwrap();
        ctx.register_batch("slo", encoded).unwrap();
        let checks = checks(&ctx, &actual, &mut expected).await.unwrap();
        assert!(
            checks.iter().all(|row| row["status"] == "OK"),
            "{checks:#?}"
        );
    }

    #[tokio::test]
    async fn context_and_output_corruption_are_detected_independently() {
        for (context, output, failed) in [
            (15, 7, "verify_causal_pairs"),
            (14, 8, "speculative_output_tokens"),
        ] {
            let (ctx, actual, mut expected) = fixture("", context, output);
            let checks = checks(&ctx, &actual, &mut expected).await.unwrap();
            assert_eq!(
                checks.iter().find(|row| row["name"] == failed).unwrap()["status"],
                "FAIL"
            );
        }
    }
}
