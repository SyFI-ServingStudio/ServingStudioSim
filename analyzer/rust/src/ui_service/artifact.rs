//! Shared readers for fixed artifacts owned by a discovered run or repository.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use super::discovery::regular_file;
use super::ArtifactNotFound;

pub(super) fn read_run_json(run: &Path, relative: &str) -> Result<Value> {
    read_json(&run.join(relative))
}

pub(super) fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&read_bytes(path)?).with_context(|| format!("parse {}", path.display()))
}

pub(super) fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    if !regular_file(path) {
        return Err(ArtifactNotFound.into());
    }
    fs::read(path).with_context(|| format!("read {}", path.display()))
}
