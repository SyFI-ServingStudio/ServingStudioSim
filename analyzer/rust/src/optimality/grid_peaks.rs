//! Per-config grid-peaks sidecar — the R3 "batching ceiling" data path.
//!
//! For every unique `(kind, kernel_config)` in a run's manifests, the optimality
//! subject needs the kernel's **best achievable rate for that fixed structural
//! config over its batch/free axis** (a GEMM's peak TFLOP/s at its `n,k,dtype`
//! swept across `m`, etc.). That ceiling is what the sim's `kernel-query peak` op
//! reads straight off the fitted cache grid — see
//! `simulator/src/introspect/mod.rs`. This module enumerates the run's unique
//! configs, asks the simulator for all their peaks in **one** batched subprocess
//! (amortizing the PyO3/torch import), and caches the answer as
//! `raw/kernel_grid_peaks.json` so re-analysis is a pure file read.
//!
//! The subject reads the peaks with serde — no `simulator` link, no new analyzer
//! dep. When neither a cached sidecar nor a live generation is available (no built
//! simulator / venv / profile.db), the subject degrades to run-observed peaks and
//! logs the caveat; nothing here ever aborts an `analyze run`.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::io::{resolve_artifact_path, write_json, SCHEMA_VERSION};
use crate::kernel_query::{repo_root, run_kernel_query, simulator_binary};
use crate::trace::manifest::ManifestDoc;

/// Sidecar filename under `raw/` (co-located with the parquet it derives from).
const SIDECAR: &str = "kernel_grid_peaks.json";

/// One config's fitted-grid ceiling: best achieved compute (TFLOP/s) and memory /
/// collective bandwidth (GB/s) over its batch axis. A comm kernel has `tflops == 0`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GridPeakRates {
    pub tflops: f64,
    pub gbps: f64,
}

/// The run's per-config peaks plus where they came from (for the report caveat).
pub(crate) struct GridPeakCatalog {
    rates_by_config_key: HashMap<String, GridPeakRates>,
    /// `"sidecar"` (cached file), `"generated"` (fresh query, now cached), or
    /// `"unavailable: <why>"` (subject falls back to run-observed peaks).
    pub source: String,
}

impl GridPeakCatalog {
    /// The grid peak for a leaf's `(kind, config)`, or `None` when this config was
    /// never queried (degraded run) or failed to build — caller uses the observed peak.
    pub fn get(&self, kind: &str, config: &Value) -> Option<GridPeakRates> {
        self.rates_by_config_key
            .get(&config_key(kind, config))
            .copied()
    }

    pub fn is_empty(&self) -> bool {
        self.rates_by_config_key.is_empty()
    }
}

/// Stable lookup key for a `(kind, config)` pair. `Value`'s `Display` is compact
/// canonical JSON; both the generator and the subject key off the *same* manifest
/// `LeafDesc.kernel_config` Value, so the strings match byte-for-byte regardless
/// of serde's key-order feature.
fn config_key(kind: &str, config: &Value) -> String {
    format!("{kind}\u{1}{config}")
}

/// Load a cached sidecar, else generate one, else degrade — never errors.
///
/// `manifests` are the already-parsed per-worker cost trees the subject holds, so
/// enumeration re-reads nothing. Generation needs a built simulator + `.venv` +
/// a populated `profile.db` (all present when analyzing from the repo); a run's
/// own configs are already in `profile.db`, so the peak queries are cache reads.
pub(crate) fn load_or_generate(
    log_dir: &Path,
    manifests: &BTreeMap<(String, u16), ManifestDoc>,
) -> GridPeakCatalog {
    if let Some(peaks) = load(log_dir) {
        return peaks;
    }
    match generate(log_dir, manifests) {
        Ok(peaks) => peaks,
        Err(e) => GridPeakCatalog {
            rates_by_config_key: HashMap::new(),
            source: format!("unavailable: {e:#}"),
        },
    }
}

/// Read-only service path: reuse the sidecar produced by `analyze run` without
/// launching kernel-query or mutating the run directory during an HTTP GET.
pub(crate) fn load_cached(log_dir: &Path) -> GridPeakCatalog {
    load(log_dir).unwrap_or_else(|| GridPeakCatalog {
        rates_by_config_key: HashMap::new(),
        source: "unavailable: cached sidecar missing or unreadable".to_string(),
    })
}

/// Read a previously-written `raw/kernel_grid_peaks.json` into a lookup map.
/// `None` if absent/unparseable (the caller then tries generation).
fn load(log_dir: &Path) -> Option<GridPeakCatalog> {
    let path = resolve_artifact_path(log_dir, SIDECAR);
    let text = std::fs::read_to_string(&path).ok()?;
    let doc: SidecarDoc = serde_json::from_str(&text).ok()?;
    let mut rates_by_config_key = HashMap::new();
    for entry in doc.configs {
        // A per-config build failure is recorded with a null peak; skip it so the
        // leaf falls back to its observed peak rather than a bogus 0-rate ceiling.
        if entry.error.is_some() {
            continue;
        }
        rates_by_config_key.insert(
            config_key(&entry.kind, &entry.config),
            GridPeakRates {
                tflops: entry.peak_tflops,
                gbps: entry.peak_gbps,
            },
        );
    }
    Some(GridPeakCatalog {
        rates_by_config_key,
        source: "sidecar".to_string(),
    })
}

/// Enumerate unique configs, query all their peaks in one batched subprocess,
/// write the sidecar, and return the lookup map.
fn generate(
    log_dir: &Path,
    manifests: &BTreeMap<(String, u16), ManifestDoc>,
) -> Result<GridPeakCatalog> {
    let unique = enumerate_unique(manifests);
    if unique.is_empty() {
        anyhow::bail!("no kernel configs found in manifests");
    }
    let root = repo_root()?;
    let simulator = simulator_binary(&root)?;
    let requests: Vec<Value> = unique
        .iter()
        .map(|(kind, config)| json!({"kind": kind, "config": config}))
        .collect();
    let response = run_kernel_query(
        &root,
        &simulator,
        json!({"op": "peak", "requests": requests}),
    )
    .context("kernel-query peak batch")?;
    let results = response
        .get("results")
        .and_then(Value::as_array)
        .context("peak response has no results array")?;
    if results.len() != unique.len() {
        anyhow::bail!(
            "peak returned {} results for {} configs",
            results.len(),
            unique.len()
        );
    }

    let mut rates_by_config_key = HashMap::new();
    let mut sidecar_entries = Vec::with_capacity(unique.len());
    for ((kind, config), result) in unique.iter().zip(results) {
        let error = result
            .get("error")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let peak_tflops = result
            .get("peak_tflops")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let peak_gbps = result
            .get("peak_gbps")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        if error.is_none() {
            rates_by_config_key.insert(
                config_key(kind, config),
                GridPeakRates {
                    tflops: peak_tflops,
                    gbps: peak_gbps,
                },
            );
        }
        sidecar_entries.push(json!({
            "kind": kind,
            "config": config,
            "peak_tflops": peak_tflops,
            "peak_gbps": peak_gbps,
            "error": error,
        }));
    }

    // Cache for the next `analyze run`. A write failure (read-only tree) is
    // non-fatal — we still return the freshly-queried peaks for this run.
    let doc = json!({
        "schema_version": SCHEMA_VERSION,
        "source": "generated",
        "num_configs": sidecar_entries.len(),
        "configs": sidecar_entries,
    });
    let out = resolve_artifact_path(log_dir, SIDECAR);
    if let Err(e) = write_json(&out, &doc) {
        eprintln!("[optimality] could not cache {SIDECAR}: {e:#}");
    }

    Ok(GridPeakCatalog {
        rates_by_config_key,
        source: "generated".to_string(),
    })
}

/// Every distinct `(kind, kernel_config)` across all workers' sections, in a
/// deterministic order. Two leaves with byte-identical config dedup to one query.
fn enumerate_unique(manifests: &BTreeMap<(String, u16), ManifestDoc>) -> Vec<(String, Value)> {
    let mut seen: BTreeMap<String, (String, Value)> = BTreeMap::new();
    for doc in manifests.values() {
        for section in &doc.sections {
            for leaf in &section.manifest.slots {
                seen.entry(config_key(&leaf.kind, &leaf.kernel_config))
                    .or_insert_with(|| (leaf.kind.clone(), leaf.kernel_config.clone()));
            }
        }
    }
    seen.into_values().collect()
}

/// Deserialize view of the cached sidecar.
#[derive(Deserialize)]
struct SidecarDoc {
    configs: Vec<SidecarEntry>,
}

#[derive(Deserialize)]
struct SidecarEntry {
    kind: String,
    config: Value,
    #[serde(default)]
    peak_tflops: f64,
    #[serde(default)]
    peak_gbps: f64,
    #[serde(default)]
    error: Option<String>,
}
