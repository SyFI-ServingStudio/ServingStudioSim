//! `workload-conservation` — run-wide work-accounting invariants.
//!
//! Two independent computation paths must agree:
//!   - **actual**: summed from `cost_log` `groups` (what the cost model was
//!     actually asked to compute).
//!   - **expected**: closed forms over each request's terminal `(p, d)` in
//!     `request_slo` (`p = prefill_processed`, `d = num_output_tokens`).
//!
//! `request_slo` covers exactly the admitted set (completed rows at their
//! completion tick + sim-end-flush partial rows for in-flight reqs), which is
//! exactly the set `cost_log` did work for, so the two sides are comparable on
//! any run — not just fully drained ones.
//!
//! Checks for iter-wise deployments (single-round today, so preserved-prefix
//! `pre = 0`):
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
//! AFD logs attention once per layer and FFN once per section. For
//! `deployment=afd`, attention checks therefore use only
//! `(pool_tag=attn, section=attn)` and multiply request-side expected work by
//! the observed layer count; FFN checks use `pool_tag=ffn` and expect
//! `Σ[p + max(d-1,0)] × (layers + 3)` section-token passes.
//!
//! When EP/HP multi-group lands, the per-group reduction needs the same revisit
//! as `batch::composition` (sum for partition-style, pick-one for replicate-style
//! HP) — one group today (unified dense asserts a single HP group).

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{anyhow, Result};
use arrow_array::{Array, BooleanArray, ListArray, StructArray, UInt32Array};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{read_deployment, resolve_artifact_path, SCHEMA_VERSION};
use crate::session::{
    col, collect, column_f64, register_cost_log, register_if_exists, require_columns, value_f64,
    COST_LOG_TABLE,
};

/// cost_log columns this subject depends on (drift guard).
const COST_COLS: &[&str] = &["pool_tag", "section", "layer", "groups"];
/// request_slo columns this subject depends on (drift guard). `prefill_processed`
/// exists only on runs logged after it was added; an older run fails the guard
/// loudly rather than silently mis-computing the expected side.
const SLO_COLS: &[&str] = &["completed", "prefill_processed", "num_output_tokens"];

/// |unexplained Δ| ≤ this fraction of expected ⇒ OK; ≤ [`WARN_PCT`] ⇒ WARN; else
/// FAIL. Every quantity is integer-exact when the sim is correct, so OK is
/// effectively an exact match except for the explicit positive-only boundary
/// allowance on DurationReached pipeline tails.
const TOLERANCE_PCT: f64 = 0.01;
const WARN_PCT: f64 = 5.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkloadMode {
    Iterwise,
    Afd,
}

impl WorkloadMode {
    fn from_deployment(deployment: Option<&str>) -> Self {
        match deployment {
            Some("afd") => Self::Afd,
            _ => Self::Iterwise,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Iterwise => "iterwise",
            Self::Afd => "afd-layered",
        }
    }
}

/// Run-wide actuals summed from `cost_log` groups. f64 is exact for these integer
/// sums on realistic runs (all well under 2^53).
#[derive(Default)]
struct Actual {
    prefill_tokens: f64,
    decode_passes: f64,
    attn_batch_tokens: f64,
    batch_tokens: f64,
    decode_kv: f64,
    causal: f64,
    iters: usize,
    num_layers: usize,
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
    incomplete_requests: usize,
    max_context_len: f64,
}

pub async fn run_workload(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    if !register_cost_log(ctx, log_dir).await? {
        let reason = "cost_log/ dir not found";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }
    require_columns(ctx, COST_LOG_TABLE, COST_COLS).await?;

    let slo_path = resolve_artifact_path(log_dir, "request_slo.parquet");
    if !register_if_exists(ctx, "slo", slo_path).await? {
        let reason = "request_slo.parquet not found";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }
    require_columns(ctx, "slo", SLO_COLS).await?;

    let deployment = read_deployment(log_dir);
    let mode = WorkloadMode::from_deployment(deployment.as_deref());
    let actual = collect_actual(ctx, mode).await?;
    let expected = collect_expected(ctx, mode, actual.num_layers).await?;

    let checks = checks_for_mode(mode, &actual, &expected);
    let all_ok = checks.iter().all(|c| c["status"] == "OK");

    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "log_dir": log_dir.display().to_string(),
            "deployment": deployment.as_deref().unwrap_or("unknown"),
            "mode": mode.label(),
            "num_iterations": actual.iters,
            "num_layers": actual.num_layers,
            "num_requests": expected.requests,
            "num_incomplete_requests": expected.incomplete_requests,
            "max_context_len": expected.max_context_len,
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
            "deployment": deployment.as_deref().unwrap_or("unknown"),
            "mode": mode.label(),
            "tolerance_pct": TOLERANCE_PCT,
            "warn_pct": WARN_PCT,
            "all_ok": all_ok,
        },
        "checks": checks,
        "definitions": definitions(),
    });

    Ok((report, payload))
}

fn checks_for_mode(mode: WorkloadMode, actual: &Actual, expected: &Expected) -> Vec<Value> {
    let layer_multiplier = match mode {
        WorkloadMode::Iterwise => 1.0,
        WorkloadMode::Afd => actual.num_layers as f64,
    };
    let ffn_multiplier = match mode {
        WorkloadMode::Iterwise => 1.0,
        WorkloadMode::Afd => actual.num_layers as f64 + 3.0,
    };
    let incomplete = expected.incomplete_requests as f64;
    let decode_boundary_allowance = incomplete * layer_multiplier;
    let ffn_boundary_allowance = incomplete * ffn_multiplier;
    let decode_kv_boundary_allowance =
        (actual.decode_passes - expected.decode_passes).max(0.0) * expected.max_context_len;

    let mut specs: Vec<(&str, &str, f64, f64, f64)> = vec![
        (
            "prefill_tokens",
            match mode {
                WorkloadMode::Iterwise => {
                    "tokens prefilled: Σ cost_log prefill_tokens vs Σ request_slo prefill_processed"
                }
                WorkloadMode::Afd => {
                    "AFD attention-layer prefill work: Σ attn cost_log prefill_tokens vs Σ p × layers"
                }
            },
            actual.prefill_tokens,
            expected.prefill_tokens,
            0.0,
        ),
        (
            "decode_passes",
            match mode {
                WorkloadMode::Iterwise => {
                    "decode forward passes: Σ cost_log decode_request_count vs Σ max(d-1,0)"
                }
                WorkloadMode::Afd => {
                    "AFD attention-layer decode passes: Σ attn cost_log decode_request_count vs Σ max(d-1,0) × layers"
                }
            },
            actual.decode_passes,
            expected.decode_passes,
            decode_boundary_allowance,
        ),
        (
            "ffn_token_pass",
            match mode {
                WorkloadMode::Iterwise => {
                    "tokens through FFN: Σ cost_log batch_tokens vs Σ [p + max(d-1,0)]"
                }
                WorkloadMode::Afd => {
                    "AFD FFN section token pass: Σ ffn cost_log batch_tokens vs Σ[p + max(d-1,0)] × (layers+3)"
                }
            },
            actual.batch_tokens,
            expected.batch_tokens,
            ffn_boundary_allowance,
        ),
        (
            "prefill_causal_attn_work",
            match mode {
                WorkloadMode::Iterwise => {
                    "causal prefill attn work: Σ [a·prefix + a(a+1)/2] vs Σ p(p+1)/2"
                }
                WorkloadMode::Afd => {
                    "AFD attention-layer causal prefill work: Σ [a·prefix + a(a+1)/2] vs Σ p(p+1)/2 × layers"
                }
            },
            actual.causal,
            expected.causal,
            0.0,
        ),
        (
            "decode_kv_sum",
            match mode {
                WorkloadMode::Iterwise => {
                    "decode KV read: Σ cost_log decode_kv_total vs Σ [m·p + m(m-1)/2], m=max(d-1,0)"
                }
                WorkloadMode::Afd => {
                    "AFD attention-layer decode KV read: Σ attn cost_log decode_kv_total vs Σ [m·p + m(m-1)/2] × layers"
                }
            },
            actual.decode_kv,
            expected.decode_kv,
            decode_kv_boundary_allowance,
        ),
    ];

    let self_actual = match mode {
        WorkloadMode::Iterwise => actual.batch_tokens,
        WorkloadMode::Afd => actual.attn_batch_tokens,
    };
    specs.push((
        "cost_log_batch_self_consistency",
        match mode {
            WorkloadMode::Iterwise => {
                "cost_log internal: Σ batch_tokens vs Σ(prefill_tokens + decode_request_count)"
            }
            WorkloadMode::Afd => {
                "AFD attn cost_log internal: Σ attn batch_tokens vs Σ(attn prefill_tokens + attn decode_request_count)"
            }
        },
        self_actual,
        actual.prefill_tokens + actual.decode_passes,
        0.0,
    ));

    specs
        .iter()
        .map(|&(n, d, a, e, allowance)| check(n, d, a, e, allowance))
        .collect()
}

/// One check as JSON: Δ = actual − expected. For DurationReached-style partial
/// pipeline tails, callers may pass a positive-only allowance; negative deltas
/// still indicate missing work and are never absorbed by that allowance.
fn check(name: &str, desc: &str, actual: f64, expected: f64, positive_allowance: f64) -> Value {
    let delta = actual - expected;
    let unexplained_delta = if delta > 0.0 {
        (delta - positive_allowance).max(0.0)
    } else {
        delta
    };
    let raw_pct = pct_of(delta, expected);
    let unexplained_pct = pct_of(unexplained_delta, expected);
    let status = if unexplained_pct.abs() <= TOLERANCE_PCT {
        "OK"
    } else if unexplained_pct.abs() <= WARN_PCT {
        "WARN"
    } else {
        "FAIL"
    };
    let mut out = json!({
        "name": name,
        "description": desc,
        "actual": actual,
        "expected": expected,
        "delta": delta,
        "delta_pct": if raw_pct.is_finite() { json!(raw_pct) } else { Value::Null },
        "status": status,
    });
    if positive_allowance > 0.0 {
        out["positive_boundary_allowance"] = json!(positive_allowance);
        out["unexplained_delta"] = json!(unexplained_delta);
        out["unexplained_delta_pct"] = if unexplained_pct.is_finite() {
            json!(unexplained_pct)
        } else {
            Value::Null
        };
    }
    out
}

fn pct_of(delta: f64, expected: f64) -> f64 {
    if expected.abs() > 0.0 {
        delta / expected * 100.0
    } else if delta == 0.0 {
        0.0
    } else {
        delta.signum() * f64::INFINITY
    }
}

/// Sum the actuals from `cost_log`'s `groups`. AFD logs attention per layer and FFN
/// per section, so its formulas use different row subsets — split at the SQL layer
/// (a `WHERE` per side) instead of a per-row pool_tag/section string compare across
/// tens of millions of rows. Each side then sums over the flattened `groups` child
/// in one typed pass per column (see [`sum_field`] / [`column_f64`]).
async fn collect_actual(ctx: &SessionContext, mode: WorkloadMode) -> Result<Actual> {
    let mut a = Actual::default();
    // `num_iterations` in the report stays the raw cost_log row count (every worker
    // × layer × section row) for continuity — a cheap COUNT(*), not a full scan.
    a.iters = count_rows(ctx, "SELECT COUNT(*) AS c FROM cost_log").await?;

    match mode {
        WorkloadMode::Iterwise => {
            // One row = one iteration; sum every field over the flattened groups.
            let batches = collect(ctx, "SELECT groups FROM cost_log").await?;
            for batch in &batches {
                let gs = groups_struct(groups_list(batch)?)?;
                a.prefill_tokens += sum_field(gs, "prefill_tokens")?;
                a.decode_passes += sum_field(gs, "decode_request_count")?;
                a.batch_tokens += sum_field(gs, "batch_tokens")?;
                a.decode_kv += sum_field(gs, "decode_kv_total")?;
                a.causal += causal_work(gs)?;
            }
            a.attn_batch_tokens = a.batch_tokens;
        }
        WorkloadMode::Afd => {
            // Attention side: sum over `(pool_tag=attn, section=attn)` rows and count
            // the distinct layers (the expected side multiplies request work by it).
            let attn = collect(
                ctx,
                "SELECT layer, groups FROM cost_log \
                 WHERE pool_tag = 'attn' AND section = 'attn'",
            )
            .await?;
            let mut layers = BTreeSet::new();
            for batch in &attn {
                for l in column_f64(col(batch, "layer")?)? {
                    if l.is_finite() && l >= 0.0 {
                        layers.insert(l as i16);
                    }
                }
                let gs = groups_struct(groups_list(batch)?)?;
                a.prefill_tokens += sum_field(gs, "prefill_tokens")?;
                a.decode_passes += sum_field(gs, "decode_request_count")?;
                a.attn_batch_tokens += sum_field(gs, "batch_tokens")?;
                a.decode_kv += sum_field(gs, "decode_kv_total")?;
                a.causal += causal_work(gs)?;
            }
            a.num_layers = layers.len();
            if a.num_layers == 0 {
                return Err(anyhow!(
                    "AFD workload conservation found no attention layer rows in cost_log"
                ));
            }
            // FFN section-token pass: sum batch_tokens over the ffn pool's rows.
            let ffn = collect(ctx, "SELECT groups FROM cost_log WHERE pool_tag = 'ffn'").await?;
            for batch in &ffn {
                let gs = groups_struct(groups_list(batch)?)?;
                a.batch_tokens += sum_field(gs, "batch_tokens")?;
            }
        }
    }
    Ok(a)
}

/// The `groups` column of a batch as a `ListArray`.
fn groups_list(batch: &arrow_array::RecordBatch) -> Result<&ListArray> {
    col(batch, "groups")?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("`groups` is not a List array"))
}

/// The flattened struct child of a `groups` list — one entry per (row, group).
/// Summing/looping over it equals the per-row per-group nesting but pays each column
/// downcast once, not once per element. Null list rows contribute no child entries,
/// so the flattened sum matches the old per-row loop that skipped null `groups`.
fn groups_struct(list: &ListArray) -> Result<&StructArray> {
    list.values()
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| anyhow!("`groups` elements are not Structs"))
}

/// Sum a scalar struct field over a whole batch's flattened `groups` child in one
/// typed pass. Fields are `not null`, so the plain sum matches the old accumulate.
fn sum_field(gs: &StructArray, field: &str) -> Result<f64> {
    let arr = gs
        .column_by_name(field)
        .ok_or_else(|| anyhow!("`groups` struct missing field `{field}`"))?;
    Ok(column_f64(arr)?.iter().sum())
}

/// Read a single-row `COUNT(*)` result as `usize`.
async fn count_rows(ctx: &SessionContext, sql: &str) -> Result<usize> {
    let batches = collect(ctx, sql).await?;
    match batches.first() {
        Some(b) if b.num_rows() > 0 => Ok(value_f64(col(b, "c")?, 0)? as usize),
        _ => Ok(0),
    }
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
async fn collect_expected(
    ctx: &SessionContext,
    mode: WorkloadMode,
    num_layers: usize,
) -> Result<Expected> {
    let batches = collect(
        ctx,
        "SELECT completed, prefill_processed, num_output_tokens FROM slo",
    )
    .await?;
    let mut e = Expected::default();
    let layer_multiplier = match mode {
        WorkloadMode::Iterwise => 1.0,
        WorkloadMode::Afd => num_layers as f64,
    };
    let ffn_multiplier = match mode {
        WorkloadMode::Iterwise => 1.0,
        WorkloadMode::Afd => num_layers as f64 + 3.0,
    };
    for batch in &batches {
        let completed = col(batch, "completed")?
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| anyhow!("`completed` is not a Boolean array"))?;
        let p_arr = col(batch, "prefill_processed")?;
        let d_arr = col(batch, "num_output_tokens")?;
        for row in 0..batch.num_rows() {
            e.requests += 1;
            let p = value_f64(p_arr, row)?;
            let d = value_f64(d_arr, row)?;
            if !completed.value(row) {
                e.incomplete_requests += 1;
            }
            e.max_context_len = e.max_context_len.max(p + d);
            let m = (d - 1.0).max(0.0); // decode forward passes (first token from prefill)
            e.prefill_tokens += p * layer_multiplier;
            e.decode_passes += m * layer_multiplier;
            e.batch_tokens += (p + m) * ffn_multiplier;
            e.decode_kv += (m * p + m * (m - 1.0) / 2.0) * layer_multiplier;
            e.causal += p * (p + 1.0) / 2.0 * layer_multiplier;
        }
    }
    Ok(e)
}

fn definitions() -> Value {
    json!({
        "scope": "whole run — cost_log actuals vs request_slo per-request expected",
        "p": "request_slo.prefill_processed (terminal prefill length)",
        "d": "request_slo.num_output_tokens (terminal decode length)",
        "preserved_prefix": "assumed 0 (single-round today); add when multi-round lands",
        "status": "OK |unexplained Δ%| <= tolerance_pct; WARN <= warn_pct; FAIL otherwise",
        "boundary_allowance_note": "positive-only allowance for DurationReached pipeline tails: \
                                   an incomplete request may have at most one un-emitted decode \
                                   token partially through the layer/section pipeline; negative \
                                   deltas are never absorbed",
        "afd_note": "for deployment=afd, attention actuals are summed from pool_tag=attn, \
                     section=attn and multiplied by observed layer count; FFN actuals are \
                     summed from pool_tag=ffn section rows and expected as Σ[p+m]×(layers+3)",
        "causal_note": "prefill work is the causal count (~p^2/2), chunk-invariant; \
                        NOT the dense a*kv_len (=p^2)",
        "decode_boundary_note": "a positive Δ on decode_passes / decode_kv_sum can be the \
                                 sim-end truncation boundary: an in-flight request's final \
                                 decode iteration is counted in cost_log but its token \
                                 postdates the terminal request_slo snapshot; the positive \
                                 boundary allowance makes only that explained tail OK",
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
