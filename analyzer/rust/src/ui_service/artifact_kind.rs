//! Explicit first-class artifact type marker shared by every discovery catalog.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use super::artifact::read_json;
use super::discovery::regular_file;

pub(super) const ARTIFACT_METADATA_FILE: &str = "artifact.meta.json";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum ArtifactKind {
    SimulationRun,
    SimulationSweep,
    TimingPrediction,
    AlignmentBundle,
    KernelProfile,
    KernelMeasurement,
}

impl ArtifactKind {
    pub(super) fn can_contain_resources(self) -> bool {
        matches!(self, Self::SimulationSweep | Self::AlignmentBundle)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactMetadata {
    schema_version: u32,
    artifact_kind: ArtifactKind,
}

pub(super) fn read_artifact_kind(directory: &Path) -> Result<Option<ArtifactKind>> {
    let metadata_path = directory.join(ARTIFACT_METADATA_FILE);
    if !regular_file(&metadata_path) {
        return Ok(None);
    }
    let metadata: ArtifactMetadata = serde_json::from_value(read_json(&metadata_path)?)
        .with_context(|| format!("decode artifact marker {}", metadata_path.display()))?;
    if metadata.schema_version != 1 {
        bail!(
            "unsupported artifact marker schema_version {} under {}",
            metadata.schema_version,
            directory.display()
        );
    }
    Ok(Some(metadata.artifact_kind))
}
