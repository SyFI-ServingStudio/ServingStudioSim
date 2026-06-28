//! `workload-conservation` — run-wide work-accounting invariants.
//!
//! Two independent computation paths must agree:
//!   - **actual**: summed from `cost_log`'s per-iteration `groups` (what the cost
//!     model was actually asked to compute each iteration).
//!   - **expected**: closed forms over each request's terminal `(p, d)` in
//!     `request_slo` (`p = prefill_processed`, `d = num_output_tokens`).
//!
//! `request_slo` covers exactly the admitted set (completed rows at their
//! completion tick + sim-end-flush partial rows for in-flight reqs), which is
//! exactly the set `cost_log` did work for, so the two sides are comparable on
//! any run — not just fully drained ones.
//!
//! Checks (single-round today, so preserved-prefix `pre = 0`):
//!   1. `prefill_tokens`        Σ prefill_tokens            vs Σ p
//!   2. `decode_passes`         Σ decode_request_count      vs Σ max(d-1, 0)
//!      (the first output token is produced by the prefill pass, not a decode
//!      pass, so a request incurs `d-1` decode forward passes)
//!   3. `ffn_token_pass`        Σ batch_tokens              vs Σ [p + max(d-1,0)]
//!   4. `prefill_causal_attn_work`
//!         Σ_chunks [a·prefix + a(a+1)/2]                   vs Σ p(p+1)/2
//!      The causal per-chunk work telescopes to the single-shot value
//!      `p·pre + p(p+1)/2` regardless of how prefill is chunked (the `Σ aᵢ²`
//!      terms cancel) — so this is exact even with `ChunkedPrefill`. NOTE: this
//!      is the *causal* count (≈ p²/2), NOT the dense `a·kv_len` (= p²) the ref
//!      moesim validator uses; ref gets away with dense only because it never
//!      sub-chunks a prefill.
//!   5. `decode_kv_sum`         Σ decode_kv_total
//!         vs Σ [m·p + m(m-1)/2], m = max(d-1, 0)
//!      (decode step reading context p, p+1, …, p+m-1 — a single query over the
//!      full KV each step, so no causal /2 here.)
//!   6. `cost_log_batch_self_consistency`  Σ batch_tokens vs Σ(prefill_tokens +
//!      decode_request_count) — a cost_log-internal invariant (no request side).
//!
//! When EP/HP multi-group lands, the per-group reduction needs the same revisit
//! as `batch::composition` (sum for partition-style, pick-one for replicate-style
//! HP) — one group today (unified dense asserts a single HP group).

use std::path::Path;

use anyhow::{anyhow, Result};
use arrow_array::{Array, ListArray, StructArray, UInt32Array};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{resolve_artifact_path, SCHEMA_VERSION};
use crate::session::{
    col, collect, register_cost_log, register_if_exists, require_columns, value_f64, COST_LOG_TABLE,
};

/// cost_log columns this subject depends on (drift guard).
const COST_COLS: &[&str] = &["groups"];
/// request_slo columns this subject depends on (drift guard). `prefill_processed`
/// exists only on runs logged after it was added; an older run fails the guard
/// loudly rather than silently mis-computing the expected side.
const SLO_COLS: &[&str] = &["prefill_processed", "num_output_tokens"];

/// |Δ| ≤ this fraction of expected ⇒ OK; ≤ [`WARN_PCT`] ⇒ WARN; else FAIL. Every
/// quantity is integer-exact when the sim is correct, so OK is effectively an
/// exact match and any real accounting bug lands in WARN/FAIL.
const TOLERANCE_PCT: f64 = 0.01;
const WARN_PCT: f64 = 5.0;

/// Run-wide actuals summed from `cost_log` groups. f64 is exact for these integer
/// sums on realistic runs (all well under 2^53).
#[derive(Default)]
struct Actual {
    prefill_tokens: f64,
    decode_passes: f64,
    batch_tokens: f64,
    decode_kv: f64,
    causal: f64,
    iters: usize,
}

/// Run-wide expecteds from per-request `(p, d)` in `request_slo`.
#[derive(Default)]
struct Expected {
    prefill_tokens: f64,
    decode_passes: f64,
    batch_tokens: f64,
    decode_kv: f64,
    causal: f64,
    requests: usize,
}

pub async fn run_workload(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    if !register_cost_log(ctx, log_dir).await? {
        let reason = "cost_log/ dir not found";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    require_columns(ctx, COST_LOG_TABLE, COST_COLS).await?;

    let slo_path = resolve_artifact_path(log_dir, "request_slo.parquet");
    if !register_if_exists(ctx, "slo", slo_path).await? {
        let reason = "request_slo.parquet not found";
        return Ok((unavailable(log_dir, reason), unavailable_payload(log_dir, reason)));
    }
    require_columns(ctx, "slo", SLO_COLS).await?;

    let actual = collect_actual(ctx).await?;
    let expected = collect_expected(ctx).await?;

    // (name, description, actual, expected)
    let specs: [(&str, &str, f64, f64); 6] = [
        (
            "prefill_tokens",
            "tokens prefilled: Σ cost_log prefill_tokens vs Σ request_slo prefill_processed",
            actual.prefill_tokens,
            expected.prefill_tokens,
        ),
        (
            "decode_passes",
            "decode forward passes: Σ cost_log decode_request_count vs Σ max(d-1,0)",
            actual.decode_passes,
            expected.decode_passes,
        ),
        (
            "ffn_token_pass",
            "tokens through FFN: Σ cost_log batch_tokens vs Σ [p + max(d-1,0)]",
            actual.batch_tokens,
            expected.batch_tokens,
        ),
        (
            "prefill_causal_attn_work",
            "causal prefill attn work: Σ [a·prefix + a(a+1)/2] vs Σ p(p+1)/2",
            actual.causal,
            expected.causal,
        ),
        (
            "decode_kv_sum",
            "decode KV read: Σ cost_log decode_kv_total vs Σ [m·p + m(m-1)/2], m=max(d-1,0)",
            actual.decode_kv,
            expected.decode_kv,
        ),
        (
            "cost_log_batch_self_consistency",
            "cost_log internal: Σ batch_tokens vs Σ(prefill_tokens + decode_request_count)",
            actual.batch_tokens,
            actual.prefill_tokens + actual.decode_passes,
        ),
    ];

    let checks: Vec<Value> = specs.iter().map(|&(n, d, a, e)| check(n, d, a, e)).collect();
    let all_ok = checks
        .iter()
        .all(|c| c["status"] == "OK");

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "num_iterations": actual.iters,
            "num_requests": expected.requests,
        },
        "available": true,
        "tolerance_pct": TOLERANCE_PCT,
        "warn_pct": WARN_PCT,
        "all_ok": all_ok,
        "checks": checks,
        "definitions": definitions(),
    });

    // Payload mirrors the checks as parallel arrays for the Δ% bar chart.
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "available": true,
            "tolerance_pct": TOLERANCE_PCT,
            "warn_pct": WARN_PCT,
            "all_ok": all_ok,
        },
        "checks": checks,
        "definitions": definitions(),
    });

    Ok((report, payload))
}

/// One check as JSON: Δ = actual − expected, Δ% relative to expected (null when
/// expected is 0 and there is a nonzero delta, i.e. undefined), status by tol.
fn check(name: &str, desc: &str, actual: f64, expected: f64) -> Value {
    let delta = actual - expected;
    let pct = if expected.abs() > 0.0 {
        delta / expected * 100.0
    } else if delta == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let status = if pct.abs() <= TOLERANCE_PCT {
        "OK"
    } else if pct.abs() <= WARN_PCT {
        "WARN"
    } else {
        "FAIL"
    };
    json!({
        "name": name,
        "description": desc,
        "actual": actual,
        "expected": expected,
        "delta": delta,
        // serde_json renders non-finite f64 as null; make that explicit.
        "delta_pct": if pct.is_finite() { json!(pct) } else { Value::Null },
        "status": status,
    })
}

/// Sum the actuals across every iteration's `groups` list.
async fn collect_actual(ctx: &SessionContext) -> Result<Actual> {
    let batches = collect(ctx, "SELECT groups FROM cost_log").await?;
    let mut a = Actual::default();
    for batch in &batches {
        let groups = col(batch, "groups")?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("`groups` is not a List array"))?;
        for row in 0..batch.num_rows() {
            a.iters += 1;
            if groups.is_null(row) {
                continue;
            }
            let g = groups.value(row);
            let gs = g
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| anyhow!("`groups` elements are not Structs"))?;
            a.prefill_tokens += sum_scalar(gs, "prefill_tokens")?;
            a.decode_passes += sum_scalar(gs, "decode_request_count")?;
            a.batch_tokens += sum_scalar(gs, "batch_tokens")?;
            a.decode_kv += sum_scalar(gs, "decode_kv_total")?;
            a.causal += causal_work(gs)?;
        }
    }
    Ok(a)
}

/// Sum one scalar struct field across all groups of an iteration (one group today).
fn sum_scalar(gs: &StructArray, field: &str) -> Result<f64> {
    let arr = gs
        .column_by_name(field)
        .ok_or_else(|| anyhow!("`groups` struct missing field `{field}`"))?;
    let mut total = 0.0;
    for i in 0..arr.len() {
        total += value_f64(arr, i)?;
    }
    Ok(total)
}

/// Causal prefill attention work for one iteration's groups: over each group's
/// parallel `(prefill_prefix_lens, prefill_append_lens)` chunk lists, accumulate
/// `a·prefix + a(a+1)/2`.
fn causal_work(gs: &StructArray) -> Result<f64> {
    let prefix = list_field(gs, "prefill_prefix_lens")?;
    let append = list_field(gs, "prefill_append_lens")?;
    let mut work = 0.0;
    for gi in 0..gs.len() {
        if prefix.is_null(gi) || append.is_null(gi) {
            continue;
        }
        let pv = prefix.value(gi);
        let av = append.value(gi);
        let pv = u32_values(&pv, "prefill_prefix_lens")?;
        let av = u32_values(&av, "prefill_append_lens")?;
        for k in 0..pv.len().min(av.len()) {
            let prefix_len = pv.value(k) as f64;
            let a = av.value(k) as f64;
            work += a * prefix_len + a * (a + 1.0) / 2.0;
        }
    }
    Ok(work)
}

fn list_field<'a>(gs: &'a StructArray, field: &str) -> Result<&'a ListArray> {
    gs.column_by_name(field)
        .ok_or_else(|| anyhow!("`groups` struct missing list field `{field}`"))?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("`{field}` is not a List array"))
}

fn u32_values<'a>(arr: &'a arrow_array::ArrayRef, field: &str) -> Result<&'a UInt32Array> {
    arr.as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| anyhow!("`{field}` items are not UInt32"))
}

/// Per-request expected work from `request_slo` `(prefill_processed, num_output_tokens)`.
async fn collect_expected(ctx: &SessionContext) -> Result<Expected> {
    let batches = collect(ctx, "SELECT prefill_processed, num_output_tokens FROM slo").await?;
    let mut e = Expected::default();
    for batch in &batches {
        let p_arr = col(batch, "prefill_processed")?;
        let d_arr = col(batch, "num_output_tokens")?;
        for row in 0..batch.num_rows() {
            e.requests += 1;
            let p = value_f64(p_arr, row)?;
            let d = value_f64(d_arr, row)?;
            let m = (d - 1.0).max(0.0); // decode forward passes (first token from prefill)
            e.prefill_tokens += p;
            e.decode_passes += m;
            e.decode_kv += m * p + m * (m - 1.0) / 2.0;
            e.causal += p * (p + 1.0) / 2.0;
        }
    }
    e.batch_tokens = e.prefill_tokens + e.decode_passes;
    Ok(e)
}

fn definitions() -> Value {
    json!({
        "scope": "whole run — cost_log actuals vs request_slo per-request expected",
        "p": "request_slo.prefill_processed (terminal prefill length)",
        "d": "request_slo.num_output_tokens (terminal decode length)",
        "preserved_prefix": "assumed 0 (single-round today); add when multi-round lands",
        "status": "OK |Δ%| <= tolerance_pct; WARN <= warn_pct; FAIL otherwise",
        "causal_note": "prefill work is the causal count (~p^2/2), chunk-invariant; \
                        NOT the dense a*kv_len (=p^2)",
        "decode_boundary_note": "a tiny nonzero Δ on decode_passes / decode_kv_sum (well \
                                 under tolerance) is the sim-end truncation boundary: an \
                                 in-flight request's final decode iteration is counted in \
                                 cost_log but its token postdates the terminal request_slo \
                                 snapshot, so actual exceeds expected by ~1 per such request",
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
        "checks": [],
    })
}
