//! Alignment artifact envelopes — one manifest type per analysis phase.
//!
//! Alignment runs in two phases with disjoint inputs, so each phase gets its own
//! strongly-typed manifest rather than one struct carrying phase-optional fields:
//!
//! - **kernel-align** ([`KernelAlignManifest`], subject `alignment-iteration`)
//!   compares measured kernels against timing-predict totals. It runs before any
//!   simulation, so it names no simulation or replay artifact.
//! - **e2e-align** ([`E2eAlignManifest`], subjects `alignment-workload` and
//!   `alignment-e2e`) compares the measured run against a completed DES
//!   simulation, so the simulation and client replay are mandatory.
//!
//! The two share the bounded NSYS anchor (`profile_log_dir`, `parsed_nsys`) and
//! the envelope bookkeeping. E2E additionally names the full-run workload
//! profile so bounded CUPTI capture cannot truncate scheduler/client metrics.
//! Path/schema parsing lives here;
//! metric formulas stay in the category modules.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::de::DeserializeOwned;
use serde::Deserialize;

/// Manifest schema the launcher writes and every subject expects.
const SCHEMA_VERSION: u32 = 8;

/// kernel-align inputs: measured kernels vs timing-predict totals. No
/// simulation is involved, so every field is mandatory once the phase runs.
#[derive(Debug, Clone, Deserialize)]
pub struct KernelAlignManifest {
    pub schema_version: u32,
    pub analysis_log_dir: PathBuf,
    pub profile_log_dir: PathBuf,
    pub parsed_nsys: PathBuf,
    pub predict_log_dir: PathBuf,
    pub timing_predict_case_map: PathBuf,
    pub labeled_kernel_sequences: PathBuf,
}

/// e2e-align inputs: the measured run vs a completed DES simulation.
#[derive(Debug, Clone, Deserialize)]
pub struct E2eAlignManifest {
    pub schema_version: u32,
    pub analysis_log_dir: PathBuf,
    pub profile_log_dir: PathBuf,
    pub workload_profile_log_dir: PathBuf,
    pub parsed_nsys: PathBuf,
    /// Full-run structured EngineCore iteration records. Unlike parsed NSYS,
    /// this stream continues after the bounded CUPTI window closes.
    pub metrics_jsonl: PathBuf,
    pub simulation_log_dir: PathBuf,
    pub replay_result: PathBuf,
    /// vLLM engine-core per-request timing JSONL. Absent on captures taken
    /// before that instrumentation existed, in which case the server-side
    /// latency overlays are skipped and the client-side ones still render. This
    /// is the one optional field, and it is optional for a genuine
    /// data-availability reason — not to paper over a phase distinction.
    #[serde(default)]
    pub request_timings_result: Option<PathBuf>,
    pub throughput_bins: usize,
}

/// Read a kernel-align manifest from `<log_dir>/alignment_manifest.json`.
pub fn read_kernel_align(log_dir: &Path) -> Result<KernelAlignManifest> {
    let manifest: KernelAlignManifest = read_envelope(log_dir)?;
    check_envelope(manifest.schema_version, &manifest.analysis_log_dir, log_dir)?;
    Ok(manifest)
}

/// Read an e2e-align manifest from `<log_dir>/alignment_manifest.json`.
pub fn read_e2e_align(log_dir: &Path) -> Result<E2eAlignManifest> {
    let manifest: E2eAlignManifest = read_envelope(log_dir)?;
    check_envelope(manifest.schema_version, &manifest.analysis_log_dir, log_dir)?;
    Ok(manifest)
}

fn read_envelope<M: DeserializeOwned>(log_dir: &Path) -> Result<M> {
    let path = log_dir.join("alignment_manifest.json");
    let text = fs::read_to_string(&path)
        .with_context(|| format!("read alignment manifest at {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("parse alignment manifest at {}", path.display()))
}

fn check_envelope(schema_version: u32, analysis_log_dir: &Path, log_dir: &Path) -> Result<()> {
    ensure!(
        schema_version == SCHEMA_VERSION,
        "unsupported alignment manifest schema_version {schema_version} (expected {SCHEMA_VERSION})"
    );
    ensure!(
        analysis_log_dir == log_dir
            || analysis_log_dir.canonicalize().ok().as_deref()
                == log_dir.canonicalize().ok().as_deref(),
        "alignment manifest analysis_log_dir {} does not identify CLI directory {}",
        analysis_log_dir.display(),
        log_dir.display()
    );
    Ok(())
}
