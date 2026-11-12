//! Achievable-peaks sidecar — the "achievable limit" rung between R4 and R5.
//!
//! R5 divides each leaf's work by the spec-sheet peak, which no kernel reaches.
//! The achievable rung divides it by the best rate a real kernel of the same
//! kind of work reaches on this GPU, measured in `profile.db`:
//!
//! - a compute-classified leaf by the fastest `single_gemm` of its dtype class
//!   (every backend and shape);
//! - a bandwidth-classified leaf by the fastest large elementwise copy (input of
//!   at least 1 GiB, so L2 write-back does not inflate the rate).
//!
//! `profiling/db/achievable.py` owns that query; this module runs it once per
//! analysis (`uv run python -m profiling.db.achievable`) in the checkout that
//! owns the run, and caches the answer as `raw/achievable_peaks.json`. A missing
//! `profile.db`, GPU, or dtype class leaves the affected leaves at R4, so the
//! kernel gap reads zero and R5 is unchanged. Nothing here aborts an analysis.

use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::io::{resolve_artifact_path, write_json, SCHEMA_VERSION};
use crate::kernel_query::owning_repo_root;

use super::spec::DtypeClass;

/// Sidecar filename under `raw/`.
const SIDECAR: &str = "achievable_peaks.json";

/// The measured ceilings for one GPU. Rates are `0.0` where nothing was measured.
#[derive(Clone, Debug)]
pub(crate) struct AchievablePeaks {
    gemm_tflops_by_class: Vec<(DtypeClass, f64)>,
    copy_gbps: f64,
    /// The profiling module's own JSON, kept for the report's provenance.
    document: Value,
    /// `"sidecar"`, `"generated"`, or `"unavailable: <why>"`.
    pub source: String,
}

impl AchievablePeaks {
    fn unavailable(reason: impl std::fmt::Display) -> Self {
        Self {
            gemm_tflops_by_class: Vec::new(),
            copy_gbps: 0.0,
            document: Value::Null,
            source: format!("unavailable: {reason}"),
        }
    }

    fn from_document(document: Value, source: &str) -> Self {
        let mut gemm_tflops_by_class: Vec<(DtypeClass, f64)> = Vec::new();
        if let Some(by_dtype) = document.get("gemm_by_dtype").and_then(Value::as_object) {
            for (dtype, entry) in by_dtype {
                let tflops = entry.get("tflops").and_then(Value::as_f64).unwrap_or(0.0);
                let class = DtypeClass::of(dtype);
                match gemm_tflops_by_class.iter_mut().find(|(c, _)| *c == class) {
                    Some((_, best)) => *best = best.max(tflops),
                    None => gemm_tflops_by_class.push((class, tflops)),
                }
            }
        }
        let copy_gbps = document
            .pointer("/copy/gbps")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        Self {
            gemm_tflops_by_class,
            copy_gbps,
            document,
            source: source.to_string(),
        }
    }

    /// Best measured GEMM TFLOP/s of `dtype`'s class; `0.0` = none measured.
    pub(crate) fn peak_tflops(&self, dtype: &str) -> f64 {
        let class = DtypeClass::of(dtype);
        self.gemm_tflops_by_class
            .iter()
            .find(|(c, _)| *c == class)
            .map_or(0.0, |(_, tflops)| *tflops)
    }

    /// Best measured large-copy GB/s; `0.0` = none measured.
    pub(crate) fn copy_gbps(&self) -> f64 {
        self.copy_gbps
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.copy_gbps <= 0.0 && self.gemm_tflops_by_class.is_empty()
    }

    /// The report's `meta.achievable_peaks`: where the rates came from and which
    /// row set each one.
    pub(crate) fn meta_json(&self) -> Value {
        json!({
            "source": self.source,
            "gemm_by_dtype": self.document.get("gemm_by_dtype"),
            "copy": self.document.get("copy"),
            "min_copy_input_bytes": self.document.get("min_copy_input_bytes"),
        })
    }

    #[cfg(test)]
    pub(super) fn for_test(gemm_tflops_by_dtype: &[(&str, f64)], copy_gbps: f64) -> Self {
        let gemm: serde_json::Map<String, Value> = gemm_tflops_by_dtype
            .iter()
            .map(|(dtype, tflops)| (dtype.to_string(), json!({ "tflops": tflops })))
            .collect();
        Self::from_document(
            json!({ "gemm_by_dtype": gemm, "copy": { "gbps": copy_gbps } }),
            "test",
        )
    }
}

/// Load the cached sidecar for `gpu_name`, else query `profile.db` and cache it.
pub(crate) fn load_or_generate(log_dir: &Path, gpu_name: &str) -> AchievablePeaks {
    if let Some(peaks) = load(log_dir, gpu_name) {
        return peaks;
    }
    match generate(log_dir, gpu_name) {
        Ok(peaks) => peaks,
        Err(error) => AchievablePeaks::unavailable(format!("{error:#}")),
    }
}

/// Read-only service path: reuse the sidecar `analyze run` wrote, without
/// launching a subprocess or writing the run directory during an HTTP GET.
pub(crate) fn load_cached(log_dir: &Path, gpu_name: &str) -> AchievablePeaks {
    load(log_dir, gpu_name).unwrap_or_else(|| {
        AchievablePeaks::unavailable(format!(
            "cached {SIDECAR} missing, unreadable, or for another GPU than {gpu_name:?}"
        ))
    })
}

/// The cached sidecar, if it exists and was generated for `gpu_name` (a timing
/// prediction can override the GPU an earlier analysis used).
fn load(log_dir: &Path, gpu_name: &str) -> Option<AchievablePeaks> {
    let text = std::fs::read_to_string(resolve_artifact_path(log_dir, SIDECAR)).ok()?;
    let document: Value = serde_json::from_str(&text).ok()?;
    (document.get("gpu_name").and_then(Value::as_str) == Some(gpu_name))
        .then(|| AchievablePeaks::from_document(document, "sidecar"))
}

fn generate(log_dir: &Path, gpu_name: &str) -> Result<AchievablePeaks> {
    if gpu_name.trim().is_empty() {
        bail!("run names no GPU");
    }
    let root = owning_repo_root(log_dir)?;
    let output = Command::new("uv")
        .args([
            "run",
            "python",
            "-m",
            "profiling.db.achievable",
            "--gpu-name",
        ])
        .arg(gpu_name)
        .current_dir(&root)
        .output()
        .context("spawn `uv run python -m profiling.db.achievable`")?;
    if !output.status.success() {
        bail!(
            "profiling.db.achievable exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let mut document: Value = serde_json::from_slice(&output.stdout)
        .context("parse profiling.db.achievable stdout as JSON")?;
    document["schema_version"] = json!(SCHEMA_VERSION);
    // A write failure (read-only tree) is non-fatal: this analysis still uses
    // the fresh rates.
    if let Err(error) = write_json(&resolve_artifact_path(log_dir, SIDECAR), &document) {
        eprintln!("[optimality] could not cache {SIDECAR}: {error:#}");
    }
    Ok(AchievablePeaks::from_document(document, "generated"))
}

#[cfg(test)]
mod tests {
    use super::AchievablePeaks;

    #[test]
    fn dtype_strings_of_one_class_share_their_best_gemm() {
        let peaks = AchievablePeaks::for_test(
            &[
                ("fp8_e4m3", 3200.0),
                ("mxfp8_e4m3", 2900.0),
                ("bf16", 1700.0),
            ],
            7000.0,
        );
        assert_eq!(peaks.peak_tflops("mxfp8_e4m3"), 3200.0);
        assert_eq!(peaks.peak_tflops("bfloat16"), 1700.0);
        assert_eq!(peaks.peak_tflops("nvfp4_e2m1"), 0.0);
        assert_eq!(peaks.copy_gbps(), 7000.0);
    }
}
