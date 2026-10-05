//! Each kernel kind's title and category, as the kind's Python DOC declares it
//! (`profiling.db.doc.kind_vocabulary`).
//!
//! The UI names and groups a CostTree leaf, a kernel-time segment or a measured
//! kernel's family by its kind; these DOCs are where that is written down. The
//! registry is fixed for the process's checkout, so the first successful read
//! is kept; a failed read is retried on the next request.

use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use serde_json::Value;

const PROGRAM: &str = "import json; from profiling.db.doc import kind_vocabulary; \
                       print(json.dumps(kind_vocabulary()))";

static VOCABULARY: OnceLock<Value> = OnceLock::new();

pub(super) fn kernel_kinds(repo_root: &Path) -> Result<Value> {
    if let Some(vocabulary) = VOCABULARY.get() {
        return Ok(vocabulary.clone());
    }
    let output = Command::new("uv")
        .args(["run", "python", "-c", PROGRAM])
        .current_dir(repo_root)
        .output()
        .context("start profiling.db.doc.kind_vocabulary via uv run")?;
    if !output.status.success() {
        bail!(
            "profiling.db.doc.kind_vocabulary failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let vocabulary: Value =
        serde_json::from_slice(&output.stdout).context("kind_vocabulary printed no JSON")?;
    Ok(VOCABULARY.get_or_init(|| vocabulary).clone())
}
