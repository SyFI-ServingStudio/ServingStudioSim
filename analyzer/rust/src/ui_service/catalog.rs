//! Metadata-fenced catalog projection and lifecycle fan-in cache.

use super::*;

const CATALOG_STAMP_NAMESPACE: &[u8] = b"vibesim-ui-catalog-stamp-v1\0";

#[derive(Debug)]
pub(super) struct CachedCatalog {
    stamp: String,
    etag: String,
    bytes: Vec<u8>,
}

#[derive(Debug, Serialize)]
struct RunCatalog {
    protocol_version: u32,
    generated_at: String,
    runs: Vec<RunCatalogEntry>,
}

#[derive(Debug, Serialize)]
struct RunCatalogEntry {
    run_id: String,
    kind: &'static str,
    display_name: String,
    descriptor_href: String,
    lifecycle: Lifecycle,
    updated_at: String,
}

#[derive(Clone, Copy)]
struct CatalogArtifact<'a> {
    relative: &'a Path,
    resource: &'static str,
    counts_toward_lifecycle_budget: bool,
}

pub(super) fn prepare_catalog(
    state: &ServiceState,
    headers: &HeaderMap,
    records: &[RunRecord],
) -> Result<ConditionalBody, ApiProblem> {
    prepare_catalog_inner(state, headers, records, |_| {})
}

fn prepare_catalog_inner<F>(
    state: &ServiceState,
    headers: &HeaderMap,
    records: &[RunRecord],
    mut after_build: F,
) -> Result<ConditionalBody, ApiProblem>
where
    F: FnMut(usize),
{
    let observed_stamp = catalog_stamp(records)?;
    if let Some(cached) = cached_catalog(state, &observed_stamp)? {
        return Ok(conditional_catalog_body(headers, cached));
    }

    // This mutex is taken only from the bounded blocking artifact pool. It
    // single-flights lifecycle JSON reads while allowing waiters to reuse the
    // first request's finished cache entry.
    let _build_guard = state
        .catalog_build
        .lock()
        .map_err(|_| catalog_cache_problem())?;

    for attempt in 0..MAX_GENERATION_READ_ATTEMPTS {
        let stamp_before = catalog_stamp(records)?;
        if let Some(cached) = cached_catalog(state, &stamp_before)? {
            return Ok(conditional_catalog_body(headers, cached));
        }

        let bytes = build_catalog(records)?;
        after_build(attempt);
        let stamp_after = catalog_stamp(records)?;
        if stamp_before != stamp_after {
            continue;
        }

        let cached = Arc::new(CachedCatalog {
            etag: format!("W/\"catalog-v1-{stamp_after}\""),
            stamp: stamp_after,
            bytes,
        });
        *state.catalog.write().map_err(|_| catalog_cache_problem())? = Some(Arc::clone(&cached));
        return Ok(conditional_catalog_body(headers, cached));
    }

    Err(ApiProblem::artifact_generation_changed(format!(
        "Analyzer catalog lifecycle metadata changed during all {MAX_GENERATION_READ_ATTEMPTS} bounded read attempts."
    )))
}

#[cfg(test)]
pub(super) fn prepare_catalog_with_build_hook<F>(
    state: &ServiceState,
    headers: &HeaderMap,
    records: &[RunRecord],
    after_build: F,
) -> Result<ConditionalBody, ApiProblem>
where
    F: FnMut(usize),
{
    prepare_catalog_inner(state, headers, records, after_build)
}

fn cached_catalog(
    state: &ServiceState,
    stamp: &str,
) -> Result<Option<Arc<CachedCatalog>>, ApiProblem> {
    Ok(state
        .catalog
        .read()
        .map_err(|_| catalog_cache_problem())?
        .as_ref()
        .filter(|cached| cached.stamp == stamp)
        .map(Arc::clone))
}

fn conditional_catalog_body(headers: &HeaderMap, cached: Arc<CachedCatalog>) -> ConditionalBody {
    if if_none_match(headers, &cached.etag) {
        ConditionalBody::NotModified {
            etag: cached.etag.clone(),
        }
    } else {
        ConditionalBody::Bytes {
            etag: cached.etag.clone(),
            bytes: cached.bytes.clone(),
        }
    }
}

fn build_catalog(records: &[RunRecord]) -> Result<Vec<u8>, ApiProblem> {
    let generated_at = records
        .iter()
        .map(|run| run.updated_at)
        .max()
        .unwrap_or(UNIX_EPOCH);
    let runs = records
        .iter()
        .map(|run| RunCatalogEntry {
            descriptor_href: format!("runs/{}/descriptor", run.run_id),
            run_id: run.run_id.clone(),
            kind: "simulation",
            display_name: run.display_name.clone(),
            lifecycle: lifecycle(run),
            updated_at: timestamp(run.updated_at),
        })
        .collect();
    encode_json(&RunCatalog {
        protocol_version: PROTOCOL_VERSION,
        generated_at: timestamp(generated_at),
        runs,
    })
}

/// Capture every input that can change the catalog bytes before reading any
/// lifecycle body. The opened-descriptor metadata both fences atomic replaces
/// and lets the service reject aggregate fan-in before parsing starts.
fn catalog_stamp(records: &[RunRecord]) -> Result<String, ApiProblem> {
    let mut hasher = Sha256::new();
    hasher.update(CATALOG_STAMP_NAMESPACE);
    hasher.update((records.len() as u64).to_be_bytes());
    let mut lifecycle_bytes = 0_u64;

    for run in records {
        hash_catalog_field(&mut hasher, run.run_id.as_bytes());
        hash_system_time(&mut hasher, run.updated_at);
        for artifact in catalog_artifacts() {
            hash_catalog_artifact(&mut hasher, run, artifact, &mut lifecycle_bytes)?;
        }
    }
    Ok(hex(&hasher.finalize()))
}

fn catalog_artifacts() -> [CatalogArtifact<'static>; 4] {
    [
        CatalogArtifact {
            relative: Path::new(".complete"),
            resource: "catalog completion marker",
            counts_toward_lifecycle_budget: false,
        },
        CatalogArtifact {
            relative: Path::new(".failed"),
            resource: "catalog failure marker",
            counts_toward_lifecycle_budget: false,
        },
        CatalogArtifact {
            relative: Path::new(PIPELINE_STATE_PATH),
            resource: "catalog analyzer pipeline",
            counts_toward_lifecycle_budget: true,
        },
        CatalogArtifact {
            relative: Path::new("reports/analyzer_timing.json"),
            resource: "catalog analyzer timing",
            counts_toward_lifecycle_budget: true,
        },
    ]
}

fn hash_catalog_artifact(
    hasher: &mut Sha256,
    run: &RunRecord,
    artifact: CatalogArtifact<'_>,
    lifecycle_bytes: &mut u64,
) -> Result<(), ApiProblem> {
    hash_catalog_field(hasher, artifact.relative.as_os_str().as_encoded_bytes());
    let (file, _) = match open_contained_artifact(run, artifact.relative, artifact.resource) {
        Ok(opened) => opened,
        Err(problem) if problem.code == "artifact_missing" => {
            hasher.update(b"\0missing\0");
            return Ok(());
        }
        Err(problem) => return Err(problem),
    };
    let metadata = file.metadata().map_err(|error| {
        ApiProblem::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "artifact_read_failed",
            "Artifact read failed",
            format!(
                "Cannot inspect opened {} metadata: {error}",
                artifact.resource
            ),
        )
    })?;
    hasher.update(b"\0present\0");
    hash_metadata(hasher, &metadata);

    if artifact.counts_toward_lifecycle_budget {
        *lifecycle_bytes = lifecycle_bytes
            .checked_add(metadata.len())
            .ok_or_else(catalog_too_large_problem)?;
        if *lifecycle_bytes > MAX_CATALOG_LIFECYCLE_BYTES {
            return Err(catalog_too_large_problem());
        }
    }
    Ok(())
}

fn hash_catalog_field(hasher: &mut Sha256, field: &[u8]) {
    hasher.update((field.len() as u64).to_be_bytes());
    hasher.update(field);
}

fn hash_system_time(hasher: &mut Sha256, time: SystemTime) {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            hasher.update([1]);
            hasher.update(duration.as_secs().to_be_bytes());
            hasher.update(duration.subsec_nanos().to_be_bytes());
        }
        Err(error) => {
            let duration = error.duration();
            hasher.update([0]);
            hasher.update(duration.as_secs().to_be_bytes());
            hasher.update(duration.subsec_nanos().to_be_bytes());
        }
    }
}

fn catalog_cache_problem() -> ApiProblem {
    ApiProblem::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "catalog_cache_failed",
        "Catalog cache failed",
        "The analyzer catalog cache lock is poisoned.",
    )
}

fn catalog_too_large_problem() -> ApiProblem {
    ApiProblem::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        "catalog_state_too_large",
        "Catalog lifecycle state is too large",
        "The aggregate analyzer pipeline and timing metadata exceeds the service-wide catalog safety limit.",
    )
}
