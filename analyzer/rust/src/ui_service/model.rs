//! Model resource resolved from the config path recorded in run parameters.

use std::collections::BTreeSet;
use std::path::{Component, Path};
use std::process::Command;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use super::artifact::{read_json, read_run_json};
use super::discovery::{regular_file, DiscoveredRun};
use super::ArtifactNotFound;

pub(super) fn read_model(run: &DiscoveredRun, repo_root: &Path) -> Result<Value> {
    let params = read_run_json(&run.path, "raw/params.json")?;
    let source_path =
        model_config_path(&params)?.context("raw/params.json has no model_config path")?;
    let relative = Path::new(&source_path);
    if relative.is_absolute()
        || !relative.starts_with("model/config")
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        anyhow::bail!("model_config must be a relative path below model/config");
    }

    let model_root = repo_root.join("model/config");
    let model_path = repo_root.join(relative);
    if !regular_file(&model_path) {
        return Err(ArtifactNotFound.into());
    }
    let canonical_root = model_root
        .canonicalize()
        .context("canonicalize model/config root")?;
    let canonical_model = model_path
        .canonicalize()
        .with_context(|| format!("canonicalize model config {}", model_path.display()))?;
    if !canonical_model.starts_with(&canonical_root) {
        anyhow::bail!("model_config resolves outside model/config");
    }
    let parameter_counts = model_work_parameter_counts(repo_root, &canonical_model);
    Ok(json!({
        "schema_version": 2,
        "source_path": source_path,
        "config": read_json(&canonical_model)?,
        "parameter_counts": parameter_counts,
    }))
}

/// Keep the overview independent of HF config conventions: `model.work` owns
/// architecture-aware total/active counting. Unsupported models degrade only
/// this optional enrichment, never the raw model resource.
fn model_work_parameter_counts(repo_root: &Path, model_path: &Path) -> Option<Value> {
    let output = Command::new("uv")
        .args(["run", "python", "-m", "model.work.parameter_counts"])
        .arg(model_path)
        .current_dir(repo_root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    serde_json::from_slice(&output.stdout).ok()
}

pub(super) fn model_config_path(params: &Value) -> Result<Option<String>> {
    let Some(pools) = params.get("pools").and_then(Value::as_object) else {
        return Ok(None);
    };
    let paths = pools
        .values()
        .filter_map(|pool| pool.get("groups")?.as_array())
        .flatten()
        .filter_map(|group| group.get("arch")?.get("model_config")?.as_str())
        .collect::<BTreeSet<_>>();
    if paths.len() > 1 {
        anyhow::bail!("run groups reference more than one model_config");
    }
    Ok(paths.into_iter().next().map(str::to_owned))
}
