//! `kernel-query` — cost-model cache introspection for the cache-fidelity harness.
//!
//! This is *introspection*, not simulation — the same family as `list-params` /
//! `dry-run` / `build-cache-only` (build/inspect the cost model without running a
//! sim). One subcommand, two interfaces selected by the request's `op`, each
//! describing **one kernel by its own config** (`kind` + the kernel's
//! `KernelConfig` fields):
//!
//!   - **`grid`** → the fitted `grid_axes` + resolved config, straight from
//!     `sweep_grid`. Pure metadata: no bridge, no profiling, no GPU. The driver
//!     calls this first to place off-grid probes.
//!   - **`eval`** → best-of-N interpolated metrics at a batch of `query_points`
//!     (each the kernel's own `Input` fields). Builds the kernel (profiles
//!     missing grid rows via JIT), so it needs the perf_api bridge.
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

/// Both ops carry the kernel `kind` + its `KernelConfig` fields; `eval` adds the
/// points. Tagged by `op` so the one subcommand serves two interfaces.
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
}

#[derive(Serialize)]
struct GridResponse {
    kind: String,
    /// One-line `field=value` of the resolved config.
    describe_config: String,
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
            anyhow::anyhow!("kernel-query: unknown kernel kind '{kind}' (have: {})", have.join(", "))
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
        KernelQueryRequest::Eval { kind, config, query_points } => {
            let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
            bridge
                .enable_jit_profiling()
                .context("enabling JIT profiling for the fidelity grid build")?;
            let probe = (lookup(&kind)?.build)(config, &bridge)
                .with_context(|| format!("building '{kind}' kernel (often a missing profile.db row)"))?;

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
            serde_json::to_string_pretty(&EvalResponse { kind: probe.kind().to_string(), results })?
        }
    };

    println!("{out}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::KernelQueryEntry;

    #[test]
    fn registry_covers_every_kernel_kind() {
        // Each kernel self-registers via `register_kernel!`; this guards that the
        // link-time set is exactly the known kinds, so a dropped registration (or
        // a new kernel that forgot the macro) fails here instead of at runtime.
        let mut kinds: Vec<&str> = inventory::iter::<KernelQueryEntry>
            .into_iter()
            .map(|e| e.kind)
            .collect();
        kinds.sort_unstable();
        assert_eq!(
            kinds,
            [
                "all_reduce",
                "elementwise",
                "flashinfer_attn_decode",
                "flashinfer_attn_prefill",
                "flashinfer_attn_rect",
                "rms_norm",
                "single_gemm",
            ]
        );
    }
}
