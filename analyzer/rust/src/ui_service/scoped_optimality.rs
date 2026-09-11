//! On-demand optimality for one CostTree subtree.
//!
//! The computation belongs to `optimality::scoped`; this module only adapts it
//! to the Analyzer artifact contract and advertises the subject when both raw
//! inputs needed by the computation exist.

use std::path::Path;

use anyhow::Result;
use serde_json::{json, Value};

use crate::io::resolve_artifact_path;
use crate::optimality::compute_scoped;
use crate::session::build_session;

fn scoped_optimality_available(artifact_root: &Path) -> bool {
    let has_cost_log = resolve_artifact_path(artifact_root, "cost_log").is_dir();
    let has_cost_manifest = resolve_artifact_path(artifact_root, "cost_manifest").is_dir();
    has_cost_log && has_cost_manifest
}

pub(super) fn scoped_optimality_subject_descriptor(artifact_root: &Path) -> Option<Value> {
    scoped_optimality_available(artifact_root).then(|| {
        json!({
            "status": "ready",
            "views": ["report"],
            "schema_version": 1,
        })
    })
}

pub(super) fn scoped_optimality_capability(artifact_root: &Path) -> Option<Value> {
    scoped_optimality_available(artifact_root).then(|| {
        json!({
            "views": ["report"],
        })
    })
}

pub(super) async fn scoped_optimality_report(
    artifact_root: &Path,
    path: Option<&str>,
    label: Option<&str>,
) -> Result<Value> {
    let context = build_session();
    let (report, _) = compute_scoped(&context, artifact_root, path, label).await?;
    Ok(serde_json::to_value(report)?)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::{scoped_optimality_capability, scoped_optimality_subject_descriptor};

    #[test]
    fn advertisement_requires_both_raw_inputs() {
        let temporary = TempDir::new().expect("temporary artifact root");
        fs::create_dir_all(temporary.path().join("raw/cost_log")).expect("cost log directory");
        assert!(scoped_optimality_subject_descriptor(temporary.path()).is_none());

        fs::create_dir_all(temporary.path().join("raw/cost_manifest"))
            .expect("cost manifest directory");
        assert_eq!(
            scoped_optimality_subject_descriptor(temporary.path()).expect("run subject")["status"],
            "ready"
        );
        assert_eq!(
            scoped_optimality_capability(temporary.path()).expect("prediction resource")["views"],
            serde_json::json!(["report"])
        );
    }
}
