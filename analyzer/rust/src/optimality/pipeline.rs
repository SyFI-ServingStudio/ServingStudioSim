//! Pipeline-parallel stage identity, read from the manifests a PP run writes.
//!
//! A PP pool runs one worker per stage, and every stage forwards the same
//! microbatches. Each stage owns only part of the model: its contiguous decoder
//! layer range, plus the token embedding on the first stage and the output head on
//! the last. Necessary work must follow that ownership, or every stage would carry
//! the whole model and a pool would count it `pp_size` times.
//!
//! The layer range is the one the simulator's arch actually built, which may be a
//! custom `layer_partition` rather than vLLM's default split, so it is read from
//! the stage's manifest and never re-derived from `pp_size`. Every PP stage arch
//! labels its root node `"<name> pipeline stage <i> of <N> (...) [layers
//! <start>..<end>..."`; `tools/pp-layer-balance` reads the same label.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::io::{read_cost_manifests, resolve_artifact_path};
use crate::trace::manifest::ManifestDoc;

/// One worker's place in its pool's pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PipelineStage {
    pub(super) index: u16,
    pub(super) num_stages: u16,
    /// `[start, end)` decoder layers, in checkpoint layer numbering.
    pub(super) layers: (u32, u32),
}

/// `(pool_tag, worker_id)` -> stage, for every worker of every pipelined pool.
pub(super) type PipelineStages = BTreeMap<(String, u16), PipelineStage>;

/// The pipeline stages of the run in `log_dir`. A run without cost manifests
/// (some unit fixtures) has no stages to report.
pub(super) fn read_pipeline_stages(log_dir: &Path) -> Result<PipelineStages> {
    if !resolve_artifact_path(log_dir, "cost_manifest").is_dir() {
        return Ok(PipelineStages::new());
    }
    pipeline_stages(&read_cost_manifests(log_dir)?)
}

/// Every pipelined worker's stage, after checking that each pipelined pool is
/// exactly stages `0..N` whose layer ranges tile one contiguous range from layer 0.
/// A pool is pipelined when any of its workers is a stage; then all must be.
pub(super) fn pipeline_stages(
    manifests_by_worker: &BTreeMap<(String, u16), ManifestDoc>,
) -> Result<PipelineStages> {
    let mut stages = PipelineStages::new();
    let mut plain_workers_by_pool: BTreeMap<&str, usize> = BTreeMap::new();
    for (worker_key, manifest_doc) in manifests_by_worker {
        match manifest_stage(manifest_doc)
            .with_context(|| format!("worker {}/{}", worker_key.0, worker_key.1))?
        {
            Some(stage) => {
                stages.insert(worker_key.clone(), stage);
            }
            None => {
                *plain_workers_by_pool
                    .entry(worker_key.0.as_str())
                    .or_default() += 1
            }
        }
    }
    let mut stages_by_pool: BTreeMap<&str, Vec<PipelineStage>> = BTreeMap::new();
    for ((pool_tag, _), stage) in &stages {
        stages_by_pool.entry(pool_tag).or_default().push(*stage);
    }
    for (pool_tag, mut pool_stages) in stages_by_pool {
        if let Some(plain) = plain_workers_by_pool.get(pool_tag) {
            bail!("pool {pool_tag:?} mixes pipeline stages with {plain} non-stage worker(s)");
        }
        pool_stages.sort_by_key(|stage| stage.index);
        let num_stages = pool_stages[0].num_stages;
        let mut next_layer = 0;
        for (position, stage) in pool_stages.iter().enumerate() {
            if usize::from(stage.index) != position || stage.num_stages != num_stages {
                bail!(
                    "pool {pool_tag:?} pipeline stages are not exactly 0..{num_stages}: {pool_stages:?}"
                );
            }
            if stage.layers.0 != next_layer || stage.layers.1 <= stage.layers.0 {
                bail!("pool {pool_tag:?} stage layer ranges do not tile: {pool_stages:?}");
            }
            next_layer = stage.layers.1;
        }
        if pool_stages.len() != usize::from(num_stages) {
            bail!(
                "pool {pool_tag:?} has {} of its {num_stages} pipeline stages",
                pool_stages.len()
            );
        }
    }
    Ok(stages)
}

/// The stage a worker's manifest declares, from its sections' root labels.
fn manifest_stage(manifest_doc: &ManifestDoc) -> Result<Option<PipelineStage>> {
    let mut found = None;
    for section in &manifest_doc.sections {
        let Some(Some(root_label)) = section.manifest.node_labels.first() else {
            continue;
        };
        let Some(stage) = parse_stage_label(root_label)? else {
            continue;
        };
        if found.is_some_and(|previous| previous != stage) {
            bail!("manifest sections disagree on their pipeline stage");
        }
        found = Some(stage);
    }
    Ok(found)
}

/// `Some` for a root label of the form `"... pipeline stage <i> of <N> ... [layers
/// <start>..<end>..."`; a label naming a stage without a parseable layer range is
/// an error rather than a silently unpipelined worker.
fn parse_stage_label(label: &str) -> Result<Option<PipelineStage>> {
    let Some((_, stage_text)) = label.split_once("pipeline stage ") else {
        return Ok(None);
    };
    let malformed = || format!("malformed pipeline stage label {label:?}");
    let (index, rest) = stage_text.split_once(" of ").with_context(malformed)?;
    let num_stages = rest
        .split(|character: char| !character.is_ascii_digit())
        .next()
        .with_context(malformed)?;
    let (_, layers_text) = rest.split_once("[layers ").with_context(malformed)?;
    let (start, rest) = layers_text.split_once("..").with_context(malformed)?;
    let end = rest
        .split(|character: char| !character.is_ascii_digit())
        .next()
        .with_context(malformed)?;
    let stage = PipelineStage {
        index: index.parse().with_context(malformed)?,
        num_stages: num_stages.parse().with_context(malformed)?,
        layers: (
            start.parse().with_context(malformed)?,
            end.parse().with_context(malformed)?,
        ),
    };
    if stage.index >= stage.num_stages {
        bail!("{}", malformed());
    }
    Ok(Some(stage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::manifest::{Manifest, ManifestSection};

    fn manifest(root_label: &str) -> ManifestDoc {
        ManifestDoc {
            sections: vec![ManifestSection {
                section: "iter".to_string(),
                manifest: Manifest {
                    slots: Vec::new(),
                    nodes: Vec::new(),
                    node_labels: vec![Some(root_label.to_string())],
                },
            }],
        }
    }

    fn stage_label(index: u16, num_stages: u16, start: u32, end: u32) -> String {
        format!(
            "pp pipeline stage {index} of {num_stages} (Glm53FlashVllmFp8PpStageModel) \
             [layers {start}..{end}: 4 KDA + 1 DSA; TP1/EP1 on one GPU, no collectives]"
        )
    }

    #[test]
    fn both_pp_archs_root_labels_parse() {
        let flash = stage_label(2, 8, 10, 16);
        assert_eq!(
            parse_stage_label(&flash).unwrap(),
            Some(PipelineStage {
                index: 2,
                num_stages: 8,
                layers: (10, 16)
            })
        );
        let glm52 = "pp pipeline stage 3 of 4 (Glm52VllmNvfp4PpStageModel) [layers 60..78; \
                     EP1 on one GPU, no collectives; timing_context<=202752]";
        assert_eq!(
            parse_stage_label(glm52).unwrap().map(|stage| stage.layers),
            Some((60, 78))
        );
        assert_eq!(
            parse_stage_label("m [dense local, 32 layers]").unwrap(),
            None
        );
        assert!(parse_stage_label("pp pipeline stage 1 of 4 (X) no range").is_err());
        assert!(parse_stage_label(&stage_label(4, 4, 0, 1)).is_err());
    }

    #[test]
    fn stages_must_tile_their_pool_and_never_mix_with_plain_workers() {
        let pool = |ranges: &[(u32, u32)]| -> BTreeMap<(String, u16), ManifestDoc> {
            let num_stages = ranges.len() as u16;
            ranges
                .iter()
                .enumerate()
                .map(|(index, &(start, end))| {
                    (
                        ("stage".to_string(), index as u16),
                        manifest(&stage_label(index as u16, num_stages, start, end)),
                    )
                })
                .collect()
        };
        let stages = pipeline_stages(&pool(&[(0, 5), (5, 10), (10, 16)])).unwrap();
        assert_eq!(stages[&("stage".to_string(), 2)].layers, (10, 16));

        assert!(pipeline_stages(&pool(&[(0, 5), (6, 10)])).is_err());
        assert!(pipeline_stages(&pool(&[(1, 5), (5, 10)])).is_err());

        let mut mixed = pool(&[(0, 5), (5, 10)]);
        mixed.insert(("stage".to_string(), 2), manifest("plain worker"));
        assert!(pipeline_stages(&mixed).is_err());

        let mut other_pool = pool(&[(0, 5), (5, 10)]);
        other_pool.insert(("decode".to_string(), 0), manifest("plain worker"));
        assert_eq!(pipeline_stages(&other_pool).unwrap().len(), 2);
    }
}
