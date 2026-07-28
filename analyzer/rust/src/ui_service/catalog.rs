//! Protocol-v1 catalog projection over discovered runs.

use std::time::SystemTime;

use anyhow::Result;
use serde::Serialize;

use super::discovery::{discover_runs, timestamp, ConfiguredRoot, Lifecycle};
use super::PROTOCOL_VERSION;

#[derive(Debug, Serialize)]
pub(super) struct RunCatalog {
    protocol_version: u32,
    generated_at: String,
    pub(super) runs: Vec<RunCatalogEntry>,
}

#[derive(Debug, Serialize)]
pub(super) struct RunCatalogEntry {
    workspace_id: String,
    pub(super) run_id: String,
    kind: &'static str,
    pub(super) display_name: String,
    pub(super) descriptor_href: String,
    lifecycle: Lifecycle,
    updated_at: String,
}

pub(super) fn build_catalog(roots: &[ConfiguredRoot]) -> Result<RunCatalog> {
    let mut runs = discover_runs(roots)?;
    runs.sort_by(|left, right| {
        right
            .updated_time
            .cmp(&left.updated_time)
            .then_with(|| left.run_id.cmp(&right.run_id))
    });
    let runs = runs
        .into_iter()
        .map(|run| RunCatalogEntry {
            descriptor_href: format!("runs/{}/descriptor", run.run_id),
            workspace_id: run.workspace_id,
            run_id: run.run_id,
            kind: "simulation",
            display_name: run.display_name,
            lifecycle: run.lifecycle,
            updated_at: timestamp(run.updated_time),
        })
        .collect();
    Ok(RunCatalog {
        protocol_version: PROTOCOL_VERSION,
        generated_at: timestamp(SystemTime::now()),
        runs,
    })
}
