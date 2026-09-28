//! `kernel-query` — cost-model cache introspection for the cache-fidelity harness.
//! `kernel-list` (at the end) prints every registered kind's config, input and
//! cache-coordinate fields for the Kernel Library.
//!
//! This is *introspection*, not simulation — the same family as `list-params` /
//! `dry-run` / `build-cache-only` (build/inspect the cost model without running a
//! sim). One subcommand, several interfaces selected by the request's `op`, each
//! describing **one kernel by its own config** (`kind` + the kernel's
//! `KernelConfig` fields):
//!
//!   - **`grid`** → the fitted `grid_axes` + resolved config, straight from
//!     `sweep_grid`. Pure metadata: no bridge, no profiling, no GPU. The driver
//!     calls this first to place off-grid probes.
//!   - **`rows`** → the profile.db rows the config measures (from `enumerate`:
//!     which columns it sweeps, which it fixes) and, for each physical `Input`,
//!     its cache coordinates and the grid rows around it. No bridge/GPU/DB.
//!   - **`eval`** → best-of-N interpolated metrics at physical `Input` values.
//!   - **`eval_coords`** → the same cache evaluated directly in coordinate
//!     space. Analyzer uses this for declared-grid inspection because ragged or
//!     re-axis kernels do not have one scalar Input field per cache axis.
//!   - **`peak`** → the fitted grid's peak achieved compute/BW rates (the
//!     per-config "best batching" ceiling the optimality analyzer divides work
//!     by). Builds the kernel like `eval`, then reads the peak straight off the
//!     cache cells — no query points, no coords remap. Needs the bridge.
//!
//! Division of labor: Python (`tools/cache-fidelity-analyzer/cache_fidelity.py`)
//! owns the one config and
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
    /// Interpolate directly in the fitted cache's coordinate space.
    EvalCoords {
        kind: String,
        config: Value,
        query_points: Vec<Vec<f64>>,
    },
    /// Report each config's fitted-grid peak achieved compute/BW rates (the
    /// per-config batching ceiling). A **batch** so the optimality sidecar amortizes
    /// the one-time PyO3/bridge import across every unique run config in a single
    /// subprocess. Builds each kernel like `eval`, then reads the cache cells
    /// directly — no query points, no coords remap.
    Peak { requests: Vec<PeakRequestItem> },
    /// The profile.db rows one config measures (which columns it sweeps and
    /// which it fixes), and the rows each runtime input reads. Uses the
    /// kernel's own `enumerate` and `cache_coords`; no bridge, DB or GPU.
    Rows {
        kind: String,
        config: Value,
        /// Default: the config's first backend.
        #[serde(default)]
        backend: Option<String>,
        /// Each a JSON object of the kernel's own `Input` fields.
        #[serde(default)]
        inputs: Vec<Value>,
    },
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
    /// Coordinate labels in `grid_axes` order. For a direct-axis scalar kernel
    /// these are also Input field names; ragged/re-axis kernels expose derived
    /// work labels instead.
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
    /// Highest algorithmic intensity among fitted cells. The analyzer compares
    /// this with the GPU-spec ridge point to choose R3's one throughput basis.
    max_arithmetic_intensity_flops_per_byte: f64,
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
        // rows: the kernel's enumerate + cache_coords — no bridge, no profiling.
        KernelQueryRequest::Rows {
            kind,
            config,
            backend,
            inputs,
        } => {
            let report = (lookup(&kind)?.rows)(config, backend.as_deref(), &inputs)
                .with_context(|| format!("listing '{kind}' rows"))?;
            serde_json::to_string_pretty(&report)?
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
        KernelQueryRequest::EvalCoords {
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
            for coordinates in &query_points {
                let metrics = probe.eval_coords(coordinates)?;
                results.push(PointResult {
                    input: serde_json::json!(coordinates),
                    time_ms: metrics.m.time_ms,
                    flops: metrics.m.flops,
                    bytes: metrics.m.bytes,
                    energy_j: metrics.m.energy_j,
                    coverage: metrics.coverage.bits(),
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
                            max_arithmetic_intensity_flops_per_byte: 0.0,
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
                            max_arithmetic_intensity_flops_per_byte: peak
                                .max_arithmetic_intensity_flops_per_byte,
                            error: None,
                        });
                    }
                    Err(e) => results.push(PeakItemResult {
                        kind: item.kind,
                        peak_tflops: 0.0,
                        peak_gbps: 0.0,
                        max_arithmetic_intensity_flops_per_byte: 0.0,
                        error: Some(format!(
                            "build failed (often a missing profile.db row): {e:#}"
                        )),
                    }),
                }
            }
            serde_json::to_string_pretty(&PeakResponse { results })?
        }
    };

    println!("{out}");
    Ok(())
}

/// One kernel kind in `simulator kernel-list`.
#[derive(Debug, Serialize)]
pub struct KernelListEntry {
    pub kind: &'static str,
    /// The DB table the kernel reads; differs from `kind` for a variant that
    /// profiles through a base kind.
    pub profile_kind: &'static str,
    /// What fixes one kernel instance (`KernelConfig` fields, without the
    /// `backends` / `gpu_name` every config carries).
    pub config: Vec<&'static str>,
    /// The physical query (`Input` fields): what the profile.db rows sweep.
    pub input: &'static [&'static str],
    /// The coordinates the cache interpolates on. Equal to `input` for most
    /// kinds; a ragged kind summarizes its per-request lists here (e.g.
    /// `swa_valid_counts` → `batch_size`, `mean_valid_rows`).
    pub cache_coords: &'static [&'static str],
    /// The config field holding the compute dtype (`#[compute_dtype]`).
    pub compute_dtype: Option<&'static str>,
    /// The config field holding the KV-cache dtype (`#[kv_dtype]`).
    pub kv_dtype: Option<&'static str>,
}

/// Every registered kernel kind, sorted by `kind`. Reads only the
/// `register_kernel!` inventory: no bridge, no `profile.db`, no GPU.
pub fn kernel_list() -> Vec<KernelListEntry> {
    let mut out: Vec<KernelListEntry> = inventory::iter::<KernelQueryEntry>
        .into_iter()
        .map(|e| KernelListEntry {
            kind: e.kind,
            profile_kind: (e.profile_kind)(),
            config: (e.config_fields)()
                .iter()
                .copied()
                .filter(|f| !matches!(*f, "backends" | "gpu_name"))
                .collect(),
            input: (e.input_fields)(),
            cache_coords: (e.coord_fields)(),
            compute_dtype: e.compute_dtype_field,
            kv_dtype: e.kv_dtype_field,
        })
        .collect();
    out.sort_by_key(|e| e.kind);
    out
}

/// Entry point for `simulator kernel-list` (JSON on stdout).
pub fn run_kernel_list() -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(&kernel_list())?);
    Ok(())
}

#[cfg(test)]
mod kernel_list_tests {
    use super::{kernel_list, KernelQueryEntry};

    #[test]
    fn single_gemm_lists_its_static_key_sweep_and_dtype_field() {
        let list = kernel_list();
        let gemm = list.iter().find(|e| e.kind == "single_gemm").unwrap();
        assert_eq!(gemm.profile_kind, "single_gemm");
        assert_eq!(gemm.config, ["n", "k", "dtype"]);
        assert_eq!(gemm.input, ["m"]);
        assert_eq!(gemm.cache_coords, ["m"]);
        assert_eq!(gemm.compute_dtype, Some("dtype"));
        assert_eq!(gemm.kv_dtype, None);
    }

    #[test]
    fn kinds_are_unique_and_every_dtype_field_is_a_config_field() {
        let list = kernel_list();
        let mut kinds: Vec<_> = list.iter().map(|e| e.kind).collect();
        kinds.dedup();
        assert_eq!(kinds.len(), list.len(), "a kind is registered twice");
        // Every config carries backends + gpu_name, so a read that failed (a
        // config that does not deserialize as a struct) shows as their absence.
        // Some configs carry nothing else: a kernel with fixed shapes.
        for entry in inventory::iter::<KernelQueryEntry> {
            let raw = (entry.config_fields)();
            assert!(
                raw.contains(&"backends") && raw.contains(&"gpu_name"),
                "{}: config fields not read ({raw:?})",
                entry.kind
            );
        }
        for e in &list {
            assert!(!e.input.is_empty(), "{}: no input fields read", e.kind);
            for field in [e.compute_dtype, e.kv_dtype].into_iter().flatten() {
                assert!(
                    e.config.contains(&field),
                    "{}: {field} not in config",
                    e.kind
                );
            }
        }
    }
}

#[cfg(test)]
mod rows_tests {
    use serde_json::{json, Value};

    use super::lookup;

    fn rows(kind: &str, config: Value, inputs: &[Value]) -> Value {
        (lookup(kind).unwrap().rows)(config, None, inputs).unwrap()
    }

    #[test]
    fn a_gemm_sweeps_m_fixes_its_shape_and_reads_the_rows_around_an_input() {
        let config = json!({"backends": ["torch"], "gpu_name": "NVIDIA H200",
                            "n": 4096, "k": 4096, "dtype": "bf16"});
        let report = rows(
            "single_gemm",
            config,
            &[json!({"m": 64}), json!({"m": 100}), json!({"m": 1_000_000})],
        );
        assert_eq!(report["swept"], json!(["m"]));
        assert_eq!(
            report["fixed"],
            json!({"backend": "torch", "n": 4096, "k": 4096, "dtype": "bf16"})
        );
        let m_of = |i: &Value| report["rows"][i.as_u64().unwrap() as usize]["args"]["m"].clone();
        let on_grid = &report["inputs"][0];
        assert_eq!(
            on_grid["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(m_of)
                .collect::<Vec<_>>(),
            [64]
        );
        let between = &report["inputs"][1];
        let around: Vec<Value> = between["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(m_of)
            .collect();
        assert_eq!(around.len(), 2);
        assert!(around[0].as_u64().unwrap() < 100 && around[1].as_u64().unwrap() > 100);
        assert_eq!(between["extrapolated"], false);
        assert_eq!(report["inputs"][2]["extrapolated"], true);
    }

    /// The config holds a routing shard; the DB holds per-expert token counts.
    /// `rows` reports the DB side, which is what the library plots.
    #[test]
    fn a_grouped_gemm_reports_db_columns_not_its_routing_shard() {
        let config = json!({"backends": ["torch"], "gpu_name": "NVIDIA H200", "n": 4096,
                            "k": 8192, "dtype": "bf16", "local_ppm": [300000, 200000]});
        let report = rows(
            "grouped_gemm",
            config,
            &[json!({"global_expert_selections": 64})],
        );
        assert_eq!(report["swept"], json!(["per_group_batches"]));
        assert_eq!(report["fixed"]["num_local_experts"], 2);
        assert!(report["fixed"].get("local_ppm").is_none());
        let row = report["inputs"][0]["rows"][0].as_u64().unwrap() as usize;
        assert_eq!(
            report["rows"][row]["args"]["per_group_batches"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }
}
