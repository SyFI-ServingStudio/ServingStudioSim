//! Shared alignment artifact envelope.
//!
//! Alignment categories consume the same launcher-written manifest but use
//! disjoint sources beneath it. Keeping path/schema parsing here avoids each
//! subject inventing its own interpretation; metric formulas remain in their
//! category modules.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AlignmentManifest {
    pub schema_version: u32,
    pub profile_log_dir: PathBuf,
    pub simulation_log_dir: PathBuf,
    pub analysis_log_dir: PathBuf,
    pub parsed_nsys: PathBuf,
    pub replay_result: PathBuf,
    pub request_timings_result: Option<PathBuf>,
    pub predict_log_dir: PathBuf,
    pub timing_predict_case_map: PathBuf,
    pub labeled_kernel_sequences: Option<PathBuf>,
    pub iteration: IterationInput,
    #[serde(default)]
    pub workload: WorkloadInput,
    pub e2e: E2eInput,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IterationInput {
    pub enabled: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkloadInput {
    pub enabled: bool,
}

impl Default for WorkloadInput {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct E2eInput {
    pub enabled: bool,
    pub throughput_bins: usize,
}

pub fn read(log_dir: &Path) -> Result<AlignmentManifest> {
    let path = log_dir.join("alignment_manifest.json");
    let text = fs::read_to_string(&path)
        .with_context(|| format!("read alignment manifest at {}", path.display()))?;
    let manifest: AlignmentManifest = serde_json::from_str(&text)
        .with_context(|| format!("parse alignment manifest at {}", path.display()))?;
    ensure!(
        manifest.schema_version == 4,
        "unsupported alignment manifest schema_version {}",
        manifest.schema_version
    );
    ensure!(
        manifest.analysis_log_dir == log_dir
            || manifest.analysis_log_dir.canonicalize().ok().as_deref()
                == log_dir.canonicalize().ok().as_deref(),
        "alignment manifest analysis_log_dir {} does not identify CLI directory {}",
        manifest.analysis_log_dir.display(),
        log_dir.display()
    );
    Ok(manifest)
}
