//! `workload-conservation` — run-wide work-accounting invariants.
//!
//! Two independent computation paths must agree:
//!   - **actual**: summed from `cost_log` `groups` (what the cost model was
//!     actually asked to compute).
//!   - **expected**: closed forms over each request's immutable fresh/declared
//!     prefill contract plus its terminal computed/hit/decode observations in
//!     `request_slo`.
//!
//! `request_slo` covers every arrived request (completed rows at their completion
//! tick + sim-end-flush partial rows for incomplete reqs). Never-admitted rows
//! contribute zero expected work. They are excluded from the positive boundary
//! allowance, which applies only after a request has emitted its first token and
//! may therefore have one uncommitted decode pass at sim-end.
//!
//! For one context-ready request, let `fresh` be its new suffix, `declared` its
//! reusable-prefix requirement, `hit` the resident prefix found at admission,
//! `p` the prefill tokens actually computed, and `d` the emitted output tokens.
//! Prefix-aware requests must conserve `hit + p = fresh + declared`; the common
//! logical context after prefill is therefore `context = hit + p` regardless of
//! the dynamic cache result.
//!
//! Checks for iter-wise deployments:
//!   1. `prefill_tokens`        Σ prefill_tokens            vs Σ p
//!   2. prefix telemetry presence, `hit <= declared`, and both aggregate and
//!      per-request forms of `hit + p = fresh + declared`
//!   3. `decode_passes`         Σ decode_request_count      vs Σ max(d-1, 0)
//!      (the first output token is produced by the prefill pass, not a decode
//!      pass, so a request incurs `d-1` decode forward passes)
//!   4. `ffn_token_pass`        Σ batch_tokens              vs Σ [p + max(d-1,0)]
//!   5. `prefill_causal_attn_work`
//!         Σ_chunks [a·prefix + a(a+1)/2]
//!             vs Σ [p·hit + p(p+1)/2]
//!      The causal per-chunk work telescopes to the single-shot value
//!      `p·hit + p(p+1)/2` regardless of how prefill is chunked (the `Σ aᵢ²`
//!      terms cancel) — so this is exact even with `ChunkedPrefill`. NOTE: this
//!      is the *causal* count (≈ p²/2), NOT the dense `a·kv_len` (= p²) the ref
//!      moesim validator uses; ref gets away with dense only because it never
//!      sub-chunks a prefill.
//!   6. `prefill_cold_equivalent_work` adds the causally skipped prefix triangle
//!      `hit(hit+1)/2` back to actual work and compares against the cold
//!      `context(context+1)/2` baseline. `context = fresh + declared` for a
//!      context-ready request; a sim-end partial prefill uses only its observed
//!      `hit + p` context so unfinished future work is not invented.
//!   7. `decode_kv_sum`         Σ decode_kv_total
//!         vs Σ [m·context + m(m-1)/2], m = max(d-1, 0)
//!      (decode reads the full post-prefill context, independent of how much of
//!      that context was a cache hit.)
//!   8. `cost_log_batch_self_consistency`  Σ batch_tokens vs Σ(prefill_tokens +
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

use std::collections::{BTreeSet, HashMap};
use std::ops::Range;
use std::path::Path;

use anyhow::{anyhow, Result};
use arrow_array::{
    Array, BooleanArray, ListArray, StringArray, StructArray, UInt16Array, UInt32Array,
};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{read_deployment, resolve_artifact_path, SCHEMA_VERSION};
use crate::session::{
    col, collect, column_f64, register_cost_log, register_if_exists, require_columns, value_f64,
    COST_LOG_TABLE,
};

/// cost_log columns this subject depends on (drift guard).
const COST_COLS: &[&str] = &["pool_tag", "section", "layer", "groups"];
/// request_slo columns this subject depends on (drift guard). Older logs that
/// predate the independent prefix/fresh observations fail the guard loudly
/// rather than silently reverting to a no-prefix expected-work formula.
const SLO_COLS: &[&str] = &[
    "completed",
    "fresh_prompt_tokens",
    "declared_prefix_tokens",
    "prefix_cache_hit_tokens",
    "prefill_processed",
    "num_output_tokens",
];

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

/// Run-wide expecteds and request-contract violations from `request_slo`.
#[derive(Default)]
struct Expected {
    prefill_tokens: f64,
    decode_passes: f64,
    batch_tokens: f64,
    decode_kv: f64,
    causal: f64,
    saved_causal: f64,
    cold_causal: f64,
    prefix_token_balance_actual: f64,
    prefix_token_balance_expected: f64,
    missing_prefix_resolution_requests: usize,
    prefix_hit_bound_violations: usize,
    prefix_token_balance_violations: usize,
    requests: usize,
    context_ready_requests: usize,
    incomplete_requests: usize,
    boundary_decode_requests: usize,
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
            "num_context_ready_requests": expected.context_ready_requests,
            "num_incomplete_requests": expected.incomplete_requests,
            "num_boundary_decode_requests": expected.boundary_decode_requests,
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
    let boundary_decode_requests = expected.boundary_decode_requests as f64;
    let decode_boundary_allowance = boundary_decode_requests * layer_multiplier;
    let ffn_boundary_allowance = boundary_decode_requests * ffn_multiplier;
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
            "prefix_resolution_presence",
            "context-ready requests missing an admission-time prefix-cache observation vs 0",
            expected.missing_prefix_resolution_requests as f64,
            0.0,
            0.0,
        ),
        (
            "prefix_hit_bounds",
            "requests whose prefix_cache_hit_tokens exceeds declared_prefix_tokens vs 0",
            expected.prefix_hit_bound_violations as f64,
            0.0,
            0.0,
        ),
        (
            "prefix_token_balance",
            "context-ready token balance: Σ(hit + computed) vs Σ(fresh + declared)",
            expected.prefix_token_balance_actual,
            expected.prefix_token_balance_expected,
            0.0,
        ),
        (
            "prefix_token_balance_violations",
            "context-ready requests violating hit + computed = fresh + declared vs 0",
            expected.prefix_token_balance_violations as f64,
            0.0,
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
                    "causal prefill attn work: Σ [a·prefix + a(a+1)/2] vs Σ [p·hit + p(p+1)/2]"
                }
                WorkloadMode::Afd => {
                    "AFD attention-layer causal prefill work: Σ [a·prefix + a(a+1)/2] vs Σ [p·hit + p(p+1)/2] × layers"
                }
            },
            actual.causal,
            expected.causal,
            0.0,
        ),
        (
            "prefill_cold_equivalent_work",
            match mode {
                WorkloadMode::Iterwise => {
                    "cold-equivalent causal prefill work: actual + Σ hit(hit+1)/2 vs Σ (fresh+declared)(fresh+declared+1)/2"
                }
                WorkloadMode::Afd => {
                    "AFD cold-equivalent causal prefill work: actual + saved prefix triangle vs immutable cold baseline × layers"
                }
            },
            actual.causal + expected.saved_causal,
            expected.cold_causal,
            0.0,
        ),
        (
            "decode_kv_sum",
            match mode {
                WorkloadMode::Iterwise => {
                    "decode KV read: Σ cost_log decode_kv_total vs Σ [m·(hit+p) + m(m-1)/2], m=max(d-1,0)"
                }
                WorkloadMode::Afd => {
                    "AFD attention-layer decode KV read: Σ attn cost_log decode_kv_total vs Σ [m·(hit+p) + m(m-1)/2] × layers"
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

/// Per-(pool_tag, worker_id) workload aggregate for the optimality floors. The same
/// `groups` quantities [`collect_actual`] sums run-wide, plus the two extra fields the
/// labeler roofline needs (prefill KV read + request count). All fields are additive,
/// so pool / cluster levels are plain rollups of this map. A rollup of every worker
/// reproduces the run-wide conservation `actual`, which is the cross-check.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WorkloadTotals {
    pub(crate) matmul_tokens: f64,    // Σ batch_tokens
    pub(crate) prefill_tokens: f64,   // Σ prefill_tokens
    pub(crate) decode_passes: f64,    // Σ decode_request_count
    pub(crate) prefill_pairs: f64, // Σ_chunk [a·prefix + a(a+1)/2]  (causal, = labeler prefill pairs)
    pub(crate) prefill_cached: f64, // Σ prefix                        (prefill KV read)
    pub(crate) decode_kv: f64, // Σ decode_kv_total               (= labeler decode pairs / cached)
    pub(crate) prefill_requests: f64, // Σ prefill chunk count           (lm_head sampled positions)
}

/// One exact fixed-batch workload and the number of iterations with that shape.
/// The labeler evaluates each distinct shape once; callers multiply its roofline
/// result by `occurrences`, preserving batch boundaries without one subprocess row
/// per iteration.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WeightedWorkload {
    pub(crate) totals: WorkloadTotals,
    pub(crate) occurrences: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct WorkloadShape {
    matmul_tokens: u64,
    prefill_tokens: u64,
    decode_passes: u64,
    prefill_pairs: u64,
    prefill_cached: u64,
    decode_kv: u64,
    prefill_requests: u64,
}

impl WorkloadShape {
    fn from_totals(totals: WorkloadTotals) -> Result<Self> {
        Ok(Self {
            matmul_tokens: exact_count(totals.matmul_tokens, "matmul_tokens")?,
            prefill_tokens: exact_count(totals.prefill_tokens, "prefill_tokens")?,
            decode_passes: exact_count(totals.decode_passes, "decode_passes")?,
            prefill_pairs: exact_count(totals.prefill_pairs, "prefill_pairs")?,
            prefill_cached: exact_count(totals.prefill_cached, "prefill_cached")?,
            decode_kv: exact_count(totals.decode_kv, "decode_kv")?,
            prefill_requests: exact_count(totals.prefill_requests, "prefill_requests")?,
        })
    }

    fn totals(self) -> WorkloadTotals {
        WorkloadTotals {
            matmul_tokens: self.matmul_tokens as f64,
            prefill_tokens: self.prefill_tokens as f64,
            decode_passes: self.decode_passes as f64,
            prefill_pairs: self.prefill_pairs as f64,
            prefill_cached: self.prefill_cached as f64,
            decode_kv: self.decode_kv as f64,
            prefill_requests: self.prefill_requests as f64,
        }
    }
}

fn exact_count(value: f64, field_name: &str) -> Result<u64> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > u64::MAX as f64 {
        return Err(anyhow!(
            "workload field {field_name:?} is not a non-negative exact u64: {value}"
        ));
    }
    Ok(value as u64)
}

impl WorkloadTotals {
    pub(crate) fn add(&mut self, other: &WorkloadTotals) {
        self.matmul_tokens += other.matmul_tokens;
        self.prefill_tokens += other.prefill_tokens;
        self.decode_passes += other.decode_passes;
        self.prefill_pairs += other.prefill_pairs;
        self.prefill_cached += other.prefill_cached;
        self.decode_kv += other.decode_kv;
        self.prefill_requests += other.prefill_requests;
    }

    /// Replicate independent copies of one workload without changing sequence
    /// geometry. Every stored quantity is additive across batch entries, so this
    /// models rebatching; it must never be interpreted as lengthening context.
    pub(crate) fn scale(&mut self, factor: f64) {
        self.matmul_tokens *= factor;
        self.prefill_tokens *= factor;
        self.decode_passes *= factor;
        self.prefill_pairs *= factor;
        self.prefill_cached *= factor;
        self.decode_kv *= factor;
        self.prefill_requests *= factor;
    }
}

/// Aggregate `cost_log` `groups` per (pool_tag, worker_id) in one scan. Walks the list
/// offsets so each flattened group maps back to its row's worker, accumulating into a
/// per-worker running total that is flushed to the map whenever the worker changes (so
/// a string key is allocated ~once per worker, not per row). Reuses the exact same
/// `groups` parsing as the run-wide path — the two cannot drift.
pub(crate) async fn collect_workload_by_worker(
    ctx: &SessionContext,
) -> Result<HashMap<(String, u16), WorkloadTotals>> {
    let batches = collect(
        ctx,
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, worker_id, groups FROM cost_log",
    )
    .await?;
    let mut by_worker: HashMap<(String, u16), WorkloadTotals> = HashMap::new();
    for batch in &batches {
        let pool_tags = col(batch, "pool_tag")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("`pool_tag` is not a String array"))?;
        let worker_ids = col(batch, "worker_id")?
            .as_any()
            .downcast_ref::<UInt16Array>()
            .ok_or_else(|| anyhow!("`worker_id` is not a UInt16 array"))?;
        let groups = groups_list(batch)?;
        let gs = groups_struct(groups)?;
        let offsets = groups.value_offsets();
        let workload_columns = WorkloadGroupColumns::new(gs)?;

        // Running total for the current worker; flushed on change / at batch end. Rows
        // of one worker are contiguous (one cost_log file per worker), but a worker may
        // straddle a batch boundary — the flush merges into any existing map entry.
        let mut current: Option<((String, u16), WorkloadTotals)> = None;
        for row in 0..batch.num_rows() {
            let pool = pool_tags.value(row);
            let worker_id = worker_ids.value(row);
            let same = current
                .as_ref()
                .is_some_and(|((p, w), _)| p.as_str() == pool && *w == worker_id);
            if !same {
                if let Some((key, totals)) = current.take() {
                    by_worker.entry(key).or_default().add(&totals);
                }
                current = Some(((pool.to_string(), worker_id), WorkloadTotals::default()));
            }
            let totals = &mut current.as_mut().expect("current set above").1;
            workload_columns
                .add_range((offsets[row] as usize)..(offsets[row + 1] as usize), totals)?;
        }
        if let Some((key, totals)) = current.take() {
            by_worker.entry(key).or_default().add(&totals);
        }
    }
    Ok(by_worker)
}

/// Exact fixed-batch workload for one worker iteration. Unlike
/// [`collect_workload_by_worker`], this does not combine separate iterations, so
/// `model.work` reads each weight matrix once for this observed batch rather than
/// once for a run-wide mega-batch.
pub(crate) async fn collect_iteration_workload(
    ctx: &SessionContext,
    pool_tag: &str,
    worker_id: u16,
    iter_id: u64,
) -> Result<WorkloadTotals> {
    let escaped_pool_tag = pool_tag.replace('\'', "''");
    let sql = format!(
        "SELECT groups FROM cost_log \
         WHERE CAST(pool_tag AS VARCHAR) = '{escaped_pool_tag}' \
         AND worker_id = {worker_id} AND iter_id = {iter_id}"
    );
    let batches = collect(ctx, &sql).await?;
    let mut totals = WorkloadTotals::default();
    for batch in &batches {
        let groups = groups_list(batch)?;
        let workload_columns = WorkloadGroupColumns::new(groups_struct(groups)?)?;
        workload_columns.add_range(0..groups.values().len(), &mut totals)?;
    }
    Ok(totals)
}

/// Collect fixed-batch workloads for a whole run in one scan and deduplicate equal
/// iteration shapes per worker. Rows sharing an `iter_id` are summed before shape
/// comparison, which keeps AFD/layered logs correct as well as one-row iterwise logs.
pub(crate) async fn collect_workload_shapes_by_worker(
    ctx: &SessionContext,
) -> Result<HashMap<(String, u16), Vec<WeightedWorkload>>> {
    let batches = collect(
        ctx,
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, worker_id, iter_id, groups FROM cost_log",
    )
    .await?;
    let mut totals_by_iteration: HashMap<(String, u16, u64), WorkloadTotals> = HashMap::new();
    for batch in &batches {
        let pool_tags = col(batch, "pool_tag")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("`pool_tag` is not a String array"))?;
        let worker_ids = col(batch, "worker_id")?
            .as_any()
            .downcast_ref::<UInt16Array>()
            .ok_or_else(|| anyhow!("`worker_id` is not a UInt16 array"))?;
        let iteration_ids = col(batch, "iter_id")?;
        let groups = groups_list(batch)?;
        let offsets = groups.value_offsets();
        let workload_columns = WorkloadGroupColumns::new(groups_struct(groups)?)?;
        for row_index in 0..batch.num_rows() {
            let iteration_id = exact_count(value_f64(iteration_ids, row_index)?, "iter_id")?;
            let totals = totals_by_iteration
                .entry((
                    pool_tags.value(row_index).to_string(),
                    worker_ids.value(row_index),
                    iteration_id,
                ))
                .or_default();
            workload_columns.add_range(
                (offsets[row_index] as usize)..(offsets[row_index + 1] as usize),
                totals,
            )?;
        }
    }

    let mut occurrence_by_shape: HashMap<((String, u16), WorkloadShape), u64> = HashMap::new();
    for ((pool_tag, worker_id, _iteration_id), totals) in totals_by_iteration {
        let shape = WorkloadShape::from_totals(totals)?;
        *occurrence_by_shape
            .entry(((pool_tag, worker_id), shape))
            .or_default() += 1;
    }
    let mut shapes_by_worker: HashMap<(String, u16), Vec<WeightedWorkload>> = HashMap::new();
    for ((worker_key, shape), occurrences) in occurrence_by_shape {
        shapes_by_worker
            .entry(worker_key)
            .or_default()
            .push(WeightedWorkload {
                totals: shape.totals(),
                occurrences,
            });
    }
    for shapes in shapes_by_worker.values_mut() {
        shapes.sort_by_key(|shape| {
            let totals = shape.totals;
            (
                totals.matmul_tokens as u64,
                totals.prefill_tokens as u64,
                totals.decode_passes as u64,
                totals.decode_kv as u64,
                totals.prefill_pairs as u64,
                totals.prefill_cached as u64,
                totals.prefill_requests as u64,
            )
        });
    }
    Ok(shapes_by_worker)
}

/// Typed, once-per-record-batch view of the fields needed by `model.work`.
/// Keeping range accumulation here makes the run aggregate and exact-iteration
/// paths share one workload formula without paying Arrow downcasts per row.
struct WorkloadGroupColumns<'a> {
    batch_tokens: Vec<f64>,
    prefill_tokens: Vec<f64>,
    decode_requests: Vec<f64>,
    decode_kv: Vec<f64>,
    prefill_prefix_lists: &'a ListArray,
    prefill_append_lists: &'a ListArray,
}

impl<'a> WorkloadGroupColumns<'a> {
    fn new(groups: &'a StructArray) -> Result<Self> {
        Ok(Self {
            batch_tokens: column_f64(field(groups, "batch_tokens")?)?,
            prefill_tokens: column_f64(field(groups, "prefill_tokens")?)?,
            decode_requests: column_f64(field(groups, "decode_request_count")?)?,
            decode_kv: column_f64(field(groups, "decode_kv_total")?)?,
            prefill_prefix_lists: list_field(groups, "prefill_prefix_lens")?,
            prefill_append_lists: list_field(groups, "prefill_append_lens")?,
        })
    }

    fn add_range(&self, elements: Range<usize>, totals: &mut WorkloadTotals) -> Result<()> {
        for element in elements {
            totals.matmul_tokens += self.batch_tokens[element];
            totals.prefill_tokens += self.prefill_tokens[element];
            totals.decode_passes += self.decode_requests[element];
            totals.decode_kv += self.decode_kv[element];
            if self.prefill_prefix_lists.is_null(element)
                || self.prefill_append_lists.is_null(element)
            {
                continue;
            }
            let prefix_values = self.prefill_prefix_lists.value(element);
            let append_values = self.prefill_append_lists.value(element);
            let prefix_values = u32_values(&prefix_values, "prefill_prefix_lens")?;
            let append_values = u32_values(&append_values, "prefill_append_lens")?;
            for index in 0..prefix_values.len().min(append_values.len()) {
                let prefix_length = prefix_values.value(index) as f64;
                let append_length = append_values.value(index) as f64;
                totals.prefill_pairs +=
                    append_length * prefix_length + append_length * (append_length + 1.0) / 2.0;
                totals.prefill_cached += prefix_length;
                totals.prefill_requests += 1.0;
            }
        }
        Ok(())
    }
}

/// A scalar struct field of the flattened `groups` child by name.
fn field<'a>(gs: &'a StructArray, name: &str) -> Result<&'a arrow_array::ArrayRef> {
    gs.column_by_name(name)
        .ok_or_else(|| anyhow!("`groups` struct missing field `{name}`"))
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

#[derive(Clone, Copy, Debug)]
struct RequestSloWorkInput {
    completed: bool,
    fresh_prompt_tokens: f64,
    declared_prefix_tokens: f64,
    prefix_cache_hit_tokens: Option<f64>,
    prefill_tokens_processed: f64,
    num_output_tokens: f64,
}

impl Expected {
    fn add_request(&mut self, mode: WorkloadMode, num_layers: usize, input: RequestSloWorkInput) {
        self.requests += 1;
        if !input.completed {
            self.incomplete_requests += 1;
        }
        if contributes_boundary_allowance(input.completed, input.num_output_tokens) {
            self.boundary_decode_requests += 1;
        }

        if input
            .prefix_cache_hit_tokens
            .is_some_and(|hit_tokens| hit_tokens > input.declared_prefix_tokens)
        {
            self.prefix_hit_bound_violations += 1;
        }

        let context_ready = input.num_output_tokens > 0.0;
        let prefix_cache_hit_tokens = input.prefix_cache_hit_tokens.unwrap_or(0.0);
        let logical_context_tokens = prefix_cache_hit_tokens + input.prefill_tokens_processed;
        if context_ready {
            self.context_ready_requests += 1;
            if input.prefix_cache_hit_tokens.is_none() {
                self.missing_prefix_resolution_requests += 1;
            }
            let requested_context_tokens = input.fresh_prompt_tokens + input.declared_prefix_tokens;
            self.prefix_token_balance_actual += logical_context_tokens;
            self.prefix_token_balance_expected += requested_context_tokens;
            if logical_context_tokens != requested_context_tokens {
                self.prefix_token_balance_violations += 1;
            }
        }

        let layer_multiplier = match mode {
            WorkloadMode::Iterwise => 1.0,
            WorkloadMode::Afd => num_layers as f64,
        };
        let ffn_multiplier = match mode {
            WorkloadMode::Iterwise => 1.0,
            WorkloadMode::Afd => num_layers as f64 + 3.0,
        };
        let decode_forward_passes = (input.num_output_tokens - 1.0).max(0.0);
        self.max_context_len = self
            .max_context_len
            .max(logical_context_tokens + input.num_output_tokens);
        self.prefill_tokens += input.prefill_tokens_processed * layer_multiplier;
        self.decode_passes += decode_forward_passes * layer_multiplier;
        self.batch_tokens +=
            (input.prefill_tokens_processed + decode_forward_passes) * ffn_multiplier;
        self.decode_kv += (decode_forward_passes * logical_context_tokens
            + decode_forward_passes * (decode_forward_passes - 1.0) / 2.0)
            * layer_multiplier;
        self.causal += (input.prefill_tokens_processed * prefix_cache_hit_tokens
            + triangular(input.prefill_tokens_processed))
            * layer_multiplier;

        self.saved_causal += triangular(prefix_cache_hit_tokens) * layer_multiplier;
        let cold_equivalent_context_tokens = if context_ready {
            input.fresh_prompt_tokens + input.declared_prefix_tokens
        } else {
            logical_context_tokens
        };
        self.cold_causal += triangular(cold_equivalent_context_tokens) * layer_multiplier;
    }
}

fn triangular(tokens: f64) -> f64 {
    tokens * (tokens + 1.0) / 2.0
}

/// Per-request expected work from immutable request facts plus runtime observations.
async fn collect_expected(
    ctx: &SessionContext,
    mode: WorkloadMode,
    num_layers: usize,
) -> Result<Expected> {
    let batches = collect(
        ctx,
        "SELECT completed, fresh_prompt_tokens, declared_prefix_tokens, \
         prefix_cache_hit_tokens, prefill_processed, num_output_tokens FROM slo",
    )
    .await?;
    let mut expected = Expected::default();
    for batch in &batches {
        let completed = col(batch, "completed")?
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| anyhow!("`completed` is not a Boolean array"))?;
        let fresh_prompt_tokens =
            u32_values(col(batch, "fresh_prompt_tokens")?, "fresh_prompt_tokens")?;
        let declared_prefix_tokens = u32_values(
            col(batch, "declared_prefix_tokens")?,
            "declared_prefix_tokens",
        )?;
        let prefix_cache_hit_tokens = u32_values(
            col(batch, "prefix_cache_hit_tokens")?,
            "prefix_cache_hit_tokens",
        )?;
        let prefill_tokens_processed = col(batch, "prefill_processed")?;
        let num_output_tokens = col(batch, "num_output_tokens")?;
        for row in 0..batch.num_rows() {
            expected.add_request(
                mode,
                num_layers,
                RequestSloWorkInput {
                    completed: completed.value(row),
                    fresh_prompt_tokens: fresh_prompt_tokens.value(row) as f64,
                    declared_prefix_tokens: declared_prefix_tokens.value(row) as f64,
                    prefix_cache_hit_tokens: (!prefix_cache_hit_tokens.is_null(row))
                        .then(|| prefix_cache_hit_tokens.value(row) as f64),
                    prefill_tokens_processed: value_f64(prefill_tokens_processed, row)?,
                    num_output_tokens: value_f64(num_output_tokens, row)?,
                },
            );
        }
    }
    Ok(expected)
}

fn contributes_boundary_allowance(completed: bool, num_output_tokens: f64) -> bool {
    !completed && num_output_tokens > 0.0
}

fn definitions() -> Value {
    json!({
        "scope": "whole run — cost_log actuals vs request_slo per-request expected",
        "fresh": "request_slo.fresh_prompt_tokens (immutable new suffix)",
        "declared": "request_slo.declared_prefix_tokens (immutable reusable-prefix requirement)",
        "hit": "request_slo.prefix_cache_hit_tokens (nullable admission-time resident-prefix observation)",
        "p": "request_slo.prefill_processed (tokens actually computed by prefill)",
        "d": "request_slo.num_output_tokens (terminal decode length)",
        "context": "hit + p = fresh + declared after prefill completes",
        "status": "OK |unexplained Δ%| <= tolerance_pct; WARN <= warn_pct; FAIL otherwise",
        "prefix_scope_note": "a request with d > 0 has completed prefill and must have a non-null hit observation plus exact hit+p=fresh+declared balance; never-admitted and sim-end prefill-only rows are partial observations and are excluded from that full-context equation",
        "boundary_allowance_note": "positive-only allowance for DurationReached pipeline tails: \
                                   an incomplete request that has emitted at least one output \
                                   token may have at most one additional decode token partially \
                                   through the layer/section pipeline; never-admitted and prefill-only \
                                   requests do not enlarge the allowance; negative deltas are never absorbed",
        "afd_note": "for deployment=afd, attention actuals are summed from pool_tag=attn, \
                     section=attn and multiplied by observed layer count; FFN actuals are \
                     summed from pool_tag=ffn section rows and expected as Σ[p+m]×(layers+3)",
        "causal_note": "cache-aware prefill work is p*hit+p(p+1)/2 and remains chunk-invariant; adding the saved hit(hit+1)/2 triangle recovers the cold baseline. For context-ready rows that baseline is the immutable (fresh+declared)(fresh+declared+1)/2; sim-end partial-prefill rows use only their observed hit+p context so unfinished work is not invented. This is causal work, not dense a*kv_len",
        "decode_kv_note": "each decode pass reads the full logical post-prefill context hit+p, so m passes consume m(hit+p)+m(m-1)/2 KV-token reads where m=max(d-1,0)",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_allowance_excludes_pending_and_completed_requests() {
        assert!(!contributes_boundary_allowance(false, 0.0));
        assert!(contributes_boundary_allowance(false, 1.0));
        assert!(!contributes_boundary_allowance(true, 1.0));
    }

    #[test]
    fn boundary_allowance_uses_eligible_count_not_all_incomplete_requests() {
        let expected = Expected {
            requests: 4,
            incomplete_requests: 3,
            boundary_decode_requests: 1,
            ..Expected::default()
        };

        let checks = checks_for_mode(WorkloadMode::Iterwise, &Actual::default(), &expected);
        let decode = checks
            .iter()
            .find(|check| check["name"] == "decode_passes")
            .unwrap();
        let ffn = checks
            .iter()
            .find(|check| check["name"] == "ffn_token_pass")
            .unwrap();

        assert_eq!(decode["positive_boundary_allowance"], 1.0);
        assert_eq!(ffn["positive_boundary_allowance"], 1.0);
    }

    #[test]
    fn prefix_hit_conserves_tokens_and_attention_work() {
        let mut expected = Expected::default();
        expected.add_request(
            WorkloadMode::Iterwise,
            0,
            RequestSloWorkInput {
                completed: true,
                fresh_prompt_tokens: 32.0,
                declared_prefix_tokens: 64.0,
                prefix_cache_hit_tokens: Some(64.0),
                prefill_tokens_processed: 32.0,
                num_output_tokens: 4.0,
            },
        );

        assert_eq!(expected.context_ready_requests, 1);
        assert_eq!(expected.prefix_token_balance_actual, 96.0);
        assert_eq!(expected.prefix_token_balance_expected, 96.0);
        assert_eq!(expected.missing_prefix_resolution_requests, 0);
        assert_eq!(expected.prefix_hit_bound_violations, 0);
        assert_eq!(expected.prefix_token_balance_violations, 0);
        assert_eq!(expected.prefill_tokens, 32.0);
        assert_eq!(expected.decode_passes, 3.0);
        assert_eq!(expected.batch_tokens, 35.0);
        assert_eq!(expected.causal, 2_576.0);
        assert_eq!(expected.saved_causal, 2_080.0);
        assert_eq!(expected.cold_causal, 4_656.0);
        assert_eq!(
            expected.causal + expected.saved_causal,
            expected.cold_causal
        );

        assert_eq!(expected.decode_kv, 291.0);

        let actual = Actual {
            prefill_tokens: 32.0,
            decode_passes: 3.0,
            attn_batch_tokens: 35.0,
            batch_tokens: 35.0,
            decode_kv: 291.0,
            causal: 2_576.0,
            ..Actual::default()
        };
        let checks = checks_for_mode(WorkloadMode::Iterwise, &actual, &expected);
        assert_eq!(checks.len(), 11);
        assert!(checks.iter().all(|check| check["status"] == "OK"));
    }

    #[test]
    fn prefix_contract_violations_are_counted_per_request() {
        let mut expected = Expected::default();
        expected.add_request(
            WorkloadMode::Iterwise,
            0,
            RequestSloWorkInput {
                completed: true,
                fresh_prompt_tokens: 32.0,
                declared_prefix_tokens: 64.0,
                prefix_cache_hit_tokens: Some(65.0),
                prefill_tokens_processed: 32.0,
                num_output_tokens: 1.0,
            },
        );
        expected.add_request(
            WorkloadMode::Iterwise,
            0,
            RequestSloWorkInput {
                completed: true,
                fresh_prompt_tokens: 16.0,
                declared_prefix_tokens: 0.0,
                prefix_cache_hit_tokens: None,
                prefill_tokens_processed: 16.0,
                num_output_tokens: 1.0,
            },
        );

        assert_eq!(expected.context_ready_requests, 2);
        assert_eq!(expected.prefix_hit_bound_violations, 1);
        assert_eq!(expected.missing_prefix_resolution_requests, 1);
        assert_eq!(expected.prefix_token_balance_violations, 1);
    }

    #[test]
    fn prefill_only_partial_request_does_not_claim_full_context_balance() {
        let mut expected = Expected::default();
        expected.add_request(
            WorkloadMode::Iterwise,
            0,
            RequestSloWorkInput {
                completed: false,
                fresh_prompt_tokens: 32.0,
                declared_prefix_tokens: 64.0,
                prefix_cache_hit_tokens: Some(64.0),
                prefill_tokens_processed: 8.0,
                num_output_tokens: 0.0,
            },
        );

        assert_eq!(expected.requests, 1);
        assert_eq!(expected.incomplete_requests, 1);
        assert_eq!(expected.context_ready_requests, 0);
        assert_eq!(expected.missing_prefix_resolution_requests, 0);
        assert_eq!(expected.prefix_token_balance_actual, 0.0);
        assert_eq!(expected.prefix_token_balance_expected, 0.0);
        assert_eq!(expected.prefix_token_balance_violations, 0);
        assert_eq!(expected.prefill_tokens, 8.0);
        assert_eq!(expected.causal, 548.0);
        assert_eq!(expected.saved_causal, 2_080.0);
        assert_eq!(expected.cold_causal, 2_628.0);
        assert_eq!(
            expected.causal + expected.saved_causal,
            expected.cold_causal
        );

        let actual = Actual {
            prefill_tokens: 8.0,
            attn_batch_tokens: 8.0,
            batch_tokens: 8.0,
            causal: 548.0,
            ..Actual::default()
        };
        let checks = checks_for_mode(WorkloadMode::Iterwise, &actual, &expected);
        assert!(checks.iter().all(|check| check["status"] == "OK"));
    }

    #[test]
    fn workload_replication_scales_additive_geometry_without_recomputing_it() {
        let mut totals = WorkloadTotals {
            matmul_tokens: 3.0,
            prefill_tokens: 2.0,
            decode_passes: 1.0,
            prefill_pairs: 7.0,
            prefill_cached: 4.0,
            decode_kv: 16.0,
            prefill_requests: 1.0,
        };
        totals.scale(1_000.0);
        assert_eq!(totals.matmul_tokens, 3_000.0);
        assert_eq!(totals.prefill_pairs, 7_000.0);
        assert_eq!(totals.decode_kv, 16_000.0);
    }
}
