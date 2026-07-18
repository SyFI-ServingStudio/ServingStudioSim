//! `kernel-query` — cost-model cache introspection for the cache-fidelity harness.
//!
//! This is *introspection*, not simulation — the same family as `list-params` /
//! `dry-run` / `build-cache-only` (build/inspect the cost model without running a
//! sim). One subcommand, three interfaces selected by the request's `op`, each
//! describing **one kernel by its own config** (`kind` + the kernel's
//! `KernelConfig` fields):
//!
//!   - **`grid`** → the fitted `grid_axes` + resolved config, straight from
//!     `sweep_grid`. Pure metadata: no bridge, no profiling, no GPU. The driver
//!     calls this first to place off-grid probes.
//!   - **`eval`** → best-of-N interpolated metrics at a batch of `query_points`
//!     (each the kernel's own `Input` fields). Builds the kernel (profiles
//!     missing grid rows via JIT), so it needs the perf_api bridge.
//!   - **`peak`** → the fitted grid's peak achieved compute/BW rates (the
//!     per-config "best batching" ceiling the optimality analyzer divides work
//!     by). Builds the kernel like `eval`, then reads the peak straight off the
//!     cache cells — no query points, no coords remap. Needs the bridge.
//!
//! Division of labor: Python (`tools/cache_fidelity.py`) owns the one config and
//! feeds it to **both** the Rust interpolation (here) and the perf_api ground
//! truth, so they can't describe different kernels. Rust owns only the
//! authoritative interpolation + grid metadata; it never profiles ground truth
//! and never reimplements bilinear. To test a different cache (grid/axes/type)
//! you change the Rust `KernelSpec`, not this file.

use std::io::Read;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::timing::kernels::engine::KernelQueryEntry;
use crate::timing::PerfApiBridge;

/// `grid`/`eval` carry one kernel `kind` + its `KernelConfig` fields (`eval` adds
/// the points); `peak` carries a batch of `{kind, config}` items. Tagged by `op`
/// so the one subcommand serves all three interfaces.
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum KernelQueryRequest {
    /// Report the fitted grid + resolved config. No bridge/GPU.
    Grid { kind: String, config: Value },
    /// Interpolate the cache at `query_points` (best-of-N). Builds the kernel.
    Eval {
        kind: String,
        config: Value,
        /// Each a JSON object of the kernel's own `Input` fields
        /// (e.g. `{"prefix_len":0,"append_len":192}`).
        query_points: Vec<Value>,
    },
    /// Report each config's fitted-grid peak achieved compute/BW rates (the
    /// per-config batching ceiling). A **batch** so the optimality sidecar amortizes
    /// the one-time PyO3/bridge import across every unique run config in a single
    /// subprocess. Builds each kernel like `eval`, then reads the cache cells
    /// directly — no query points, no coords remap.
    Peak { requests: Vec<PeakRequestItem> },
}

/// One entry of a batched `peak` request: a kernel `kind` + its `KernelConfig`.
#[derive(Deserialize)]
struct PeakRequestItem {
    kind: String,
    config: Value,
}

#[derive(Serialize)]
struct GridResponse {
    kind: String,
    /// Structured resolved config; rich Dim objects retain formula provenance.
    describe_config: Value,
    /// The Input field names a query point must carry, in `grid_axes` order:
    /// `input_fields[i]` labels `grid_axes[i]` (e.g. `["prefix_len","append_len"]`).
    input_fields: &'static [&'static str],
    /// The fitted grid, in coords space, one ascending axis per dim.
    grid_axes: Vec<Vec<f64>>,
}

#[derive(Serialize)]
struct PointResult {
    /// Echo of the input object queried.
    input: Value,
    /// Best-of-N interpolated metrics (what the sim's `Kernel::eval` uses).
    time_ms: f32,
    flops: f32,
    bytes: f32,
    energy_j: f32,
    /// `CoverageFlags` bits (EXTRAPOLATED=1, JIT=2, NO_COVERAGE=4).
    coverage: u8,
}

#[derive(Serialize)]
struct EvalResponse {
    kind: String,
    results: Vec<PointResult>,
}

/// One config's peak result within a batched `peak` response.
#[derive(Serialize)]
struct PeakItemResult {
    kind: String,
    /// Max achieved TFLOP/s over the fitted grid (compute ceiling); `0` for a
    /// comm kernel (no flops).
    peak_tflops: f64,
    /// Max achieved GB/s over the fitted grid (bandwidth ceiling).
    peak_gbps: f64,
    /// Set (with the rates left `0`) when this one config failed to build — e.g.
    /// a profile.db row is missing. A per-item error keeps one bad config from
    /// sinking the whole sidecar; the caller degrades that leaf to its observed peak.
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// `peak` response: one result per requested config, in request order.
#[derive(Serialize)]
struct PeakResponse {
    results: Vec<PeakItemResult>,
}

/// Find a kernel's registry entry by `kind` (each kernel self-registers via
/// `register_kernel!`). No central match — adding a kernel never touches this
/// file; the "have:" list stays correct automatically.
fn lookup(kind: &str) -> anyhow::Result<&'static KernelQueryEntry> {
    inventory::iter::<KernelQueryEntry>
        .into_iter()
        .find(|e| e.kind == kind)
        .ok_or_else(|| {
            let have: Vec<&str> = inventory::iter::<KernelQueryEntry>
                .into_iter()
                .map(|e| e.kind)
                .collect();
            anyhow::anyhow!(
                "kernel-query: unknown kernel kind '{kind}' (have: {})",
                have.join(", ")
            )
        })
}

/// Entry point for `simulator kernel-query` (stdin JSON → stdout JSON).
pub fn run_kernel_query() -> anyhow::Result<()> {
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .context("reading kernel-query request from stdin")?;
    let req: KernelQueryRequest =
        serde_json::from_str(&buf).context("parsing kernel-query JSON request")?;

    let out = match req {
        // grid: pure metadata from `sweep_grid` — no bridge, no profiling.
        KernelQueryRequest::Grid { kind, config } => {
            let (describe_config, grid_axes, input_fields) = (lookup(&kind)?.describe)(config)
                .with_context(|| format!("describing '{kind}' grid"))?;
            serde_json::to_string_pretty(&GridResponse {
                kind,
                describe_config,
                input_fields,
                grid_axes,
            })?
        }
        // eval: build the kernel (JIT-profiles missing grid rows), interpolate.
        KernelQueryRequest::Eval {
            kind,
            config,
            query_points,
        } => {
            let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
            bridge
                .enable_jit_profiling()
                .context("enabling JIT profiling for the fidelity grid build")?;
            let probe = (lookup(&kind)?.build)(config, &bridge).with_context(|| {
                format!("building '{kind}' kernel (often a missing profile.db row)")
            })?;

            let mut results = Vec::with_capacity(query_points.len());
            for point in &query_points {
                let lm = probe.eval_json(point)?;
                results.push(PointResult {
                    input: point.clone(),
                    time_ms: lm.m.time_ms,
                    flops: lm.m.flops,
                    bytes: lm.m.bytes,
                    energy_j: lm.m.energy_j,
                    coverage: lm.coverage.bits(),
                });
            }
            serde_json::to_string_pretty(&EvalResponse {
                kind: probe.kind().to_string(),
                results,
            })?
        }
        // peak: one bridge for the whole batch; build each kernel (like eval), then
        // read its fitted grid's peak achieved rates straight off the cache cells
        // (best over backends). No query points, no coords remap — correct for
        // re-axis kernels. A per-item build failure is reported, not fatal.
        KernelQueryRequest::Peak { requests } => {
            let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
            bridge
                .enable_jit_profiling()
                .context("enabling JIT profiling for the peak grid build")?;
            let mut results = Vec::with_capacity(requests.len());
            for item in requests {
                let entry = match lookup(&item.kind) {
                    Ok(entry) => entry,
                    Err(e) => {
                        results.push(PeakItemResult {
                            kind: item.kind,
                            peak_tflops: 0.0,
                            peak_gbps: 0.0,
                            error: Some(format!("{e:#}")),
                        });
                        continue;
                    }
                };
                match (entry.build)(item.config, &bridge) {
                    Ok(probe) => {
                        let peak = probe.peak_rates();
                        results.push(PeakItemResult {
                            kind: probe.kind().to_string(),
                            peak_tflops: peak.tflops,
                            peak_gbps: peak.gbps,
                            error: None,
                        });
                    }
                    Err(e) => results.push(PeakItemResult {
                        kind: item.kind,
                        peak_tflops: 0.0,
                        peak_gbps: 0.0,
                        error: Some(format!("build failed (often a missing profile.db row): {e:#}")),
                    }),
                }
            }
            serde_json::to_string_pretty(&PeakResponse { results })?
        }
    };

    println!("{out}");
    Ok(())
}
