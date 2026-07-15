//! Metadata-fenced catalog projection and lifecycle fan-in cache.

use super::*;

const CATALOG_STAMP_NAMESPACE: &[u8] = b"vibesim-ui-catalog-stamp-v2\0";
// A cold attempt may retain two lifecycle descriptors per run. This cap makes
// that descriptor set, the metadata scan, and the serialized response finite
// even when every lifecycle file is missing or empty.
pub(super) const MAX_CATALOG_RUNS: usize = 1024;

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
    maximum_bytes: Option<u64>,
}

struct CapturedCatalog<'a> {
    stamp: String,
    runs: Vec<CapturedRun<'a>>,
}

struct CapturedRun<'a> {
    run: &'a RunRecord,
    simulation: StageStatus,
    pipeline: CapturedArtifact,
    timing: CapturedArtifact,
}

enum CapturedArtifact {
    Missing,
    Ready(BoundedArtifact),
    Invalid(ApiProblem),
}

impl CapturedArtifact {
    fn is_present(&self) -> bool {
        matches!(self, Self::Ready(_))
    }
}

pub(super) fn prepare_catalog(
    state: &ServiceState,
    headers: &HeaderMap,
    records: &[RunRecord],
) -> Result<ConditionalBody, ApiProblem> {
    prepare_catalog_inner(state, headers, records, || {}, |_| {}, |_, _| {})
}

fn prepare_catalog_inner<BeforeLock, AfterCapture, AfterBuild>(
    state: &ServiceState,
    headers: &HeaderMap,
    records: &[RunRecord],
    mut before_lock: BeforeLock,
    mut after_capture: AfterCapture,
    mut after_build: AfterBuild,
) -> Result<ConditionalBody, ApiProblem>
where
    BeforeLock: FnMut(),
    AfterCapture: FnMut(usize),
    AfterBuild: FnMut(usize, &[u8]),
{
    let observed_stamp = catalog_stamp(records)?;
    if let Some(cached) = cached_catalog(state, &observed_stamp)? {
        return Ok(conditional_catalog_body(headers, cached));
    }

    before_lock();
    // This mutex is taken only from the bounded blocking artifact pool. It
    // single-flights lifecycle JSON reads while allowing waiters to reuse the
    // first request's finished cache entry.
    let _build_guard = state
        .catalog_build
        .lock()
        .map_err(|_| catalog_cache_problem())?;

    for attempt in 0..MAX_GENERATION_READ_ATTEMPTS {
        // Retain the exact pipeline/timing descriptors whose metadata passed
        // both the per-file and aggregate limits. An atomic replacement after
        // this point cannot redirect body reads to a different, larger inode.
        let captured = capture_catalog(records)?;
        let stamp_before = captured.stamp.clone();
        if let Some(cached) = cached_catalog(state, &stamp_before)? {
            return Ok(conditional_catalog_body(headers, cached));
        }

        after_capture(attempt);
        let bytes = build_catalog(captured)?;
        after_build(attempt, &bytes);
        let stamp_after = catalog_stamp(records)?;
        if stamp_before != stamp_after {
            continue;
        }

        let cached = Arc::new(CachedCatalog {
            etag: catalog_etag(&bytes),
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
pub(super) fn prepare_catalog_with_hooks<BeforeLock, AfterCapture, AfterBuild>(
    state: &ServiceState,
    headers: &HeaderMap,
    records: &[RunRecord],
    before_lock: BeforeLock,
    after_capture: AfterCapture,
    after_build: AfterBuild,
) -> Result<ConditionalBody, ApiProblem>
where
    BeforeLock: FnMut(),
    AfterCapture: FnMut(usize),
    AfterBuild: FnMut(usize, &[u8]),
{
    prepare_catalog_inner(
        state,
        headers,
        records,
        before_lock,
        after_capture,
        after_build,
    )
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

fn build_catalog(captured: CapturedCatalog<'_>) -> Result<Vec<u8>, ApiProblem> {
    let generated_at = captured
        .runs
        .iter()
        .map(|captured| captured.run.updated_at)
        .max()
        .unwrap_or(UNIX_EPOCH);
    let runs = captured
        .runs
        .into_iter()
        .map(|captured| {
            let CapturedRun {
                run,
                simulation,
                pipeline,
                timing,
            } = captured;
            let pipeline = read_captured_pipeline(pipeline);
            let timing = read_captured_timing(timing);
            RunCatalogEntry {
                descriptor_href: format!("runs/{}/descriptor", run.run_id),
                run_id: run.run_id.clone(),
                kind: "simulation",
                display_name: run.display_name.clone(),
                lifecycle: lifecycle_from_simulation(simulation, &pipeline, &timing),
                updated_at: timestamp(run.updated_at),
            }
        })
        .collect();
    encode_json(&RunCatalog {
        protocol_version: PROTOCOL_VERSION,
        generated_at: timestamp(generated_at),
        runs,
    })
}

fn read_captured_pipeline(artifact: CapturedArtifact) -> PipelineStateRead {
    match artifact {
        CapturedArtifact::Missing => PipelineStateRead::Missing,
        CapturedArtifact::Invalid(problem) => PipelineStateRead::Invalid {
            code: problem.code,
            reason: problem.detail,
        },
        CapturedArtifact::Ready(artifact) => {
            let captured_length = artifact.metadata.len();
            match read_bounded_open_file(artifact, captured_length, "catalog analyzer pipeline") {
                Ok((bytes, _)) => parse_pipeline_state_bytes(&bytes),
                Err(problem) => PipelineStateRead::Invalid {
                    code: problem.code,
                    reason: problem.detail,
                },
            }
        }
    }
}

fn read_captured_timing(artifact: CapturedArtifact) -> TimingState {
    match artifact {
        CapturedArtifact::Missing => TimingState::Missing,
        CapturedArtifact::Invalid(problem) => TimingState::Invalid(problem.detail),
        CapturedArtifact::Ready(artifact) => {
            let captured_length = artifact.metadata.len();
            match read_bounded_open_file(artifact, captured_length, "catalog analyzer timing") {
                Ok((bytes, modified_at)) => parse_timing_bytes(bytes, modified_at),
                Err(problem) => TimingState::Invalid(problem.detail),
            }
        }
    }
}

/// Capture and retain every opened lifecycle descriptor used by one cold
/// attempt. Aggregate accounting and later parsing therefore refer to the same
/// file identities rather than two path resolutions separated by a race.
fn capture_catalog(records: &[RunRecord]) -> Result<CapturedCatalog<'_>, ApiProblem> {
    ensure_catalog_run_limit(records.len())?;
    let mut hasher = begin_catalog_stamp(records.len());
    let mut lifecycle_bytes = 0_u64;
    let mut captured_runs = Vec::with_capacity(records.len());

    for run in records {
        hash_catalog_run(&mut hasher, run);
        let complete =
            open_catalog_artifact(&mut hasher, run, completion_marker(), &mut lifecycle_bytes)?
                .is_present();
        let failed =
            open_catalog_artifact(&mut hasher, run, failure_marker(), &mut lifecycle_bytes)?
                .is_present();
        let pipeline =
            open_catalog_artifact(&mut hasher, run, pipeline_artifact(), &mut lifecycle_bytes)?;
        let timing =
            open_catalog_artifact(&mut hasher, run, timing_artifact(), &mut lifecycle_bytes)?;
        let simulation = if failed {
            StageStatus::Failed
        } else if complete {
            StageStatus::Complete
        } else {
            StageStatus::Pending
        };
        captured_runs.push(CapturedRun {
            run,
            simulation,
            pipeline,
            timing,
        });
    }

    Ok(CapturedCatalog {
        stamp: hex(&hasher.finalize()),
        runs: captured_runs,
    })
}

/// Metadata-only fast path used to validate an existing cache entry and as the
/// post-read fence. Cold attempts use [`capture_catalog`] so their body reads
/// remain bound to the exact descriptors that were accounted here.
fn catalog_stamp(records: &[RunRecord]) -> Result<String, ApiProblem> {
    ensure_catalog_run_limit(records.len())?;
    let mut hasher = begin_catalog_stamp(records.len());
    let mut lifecycle_bytes = 0_u64;

    for run in records {
        hash_catalog_run(&mut hasher, run);
        for artifact in catalog_artifacts() {
            drop(open_catalog_artifact(
                &mut hasher,
                run,
                artifact,
                &mut lifecycle_bytes,
            )?);
        }
    }
    Ok(hex(&hasher.finalize()))
}

fn begin_catalog_stamp(run_count: usize) -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(CATALOG_STAMP_NAMESPACE);
    hasher.update((run_count as u64).to_be_bytes());
    hasher
}

fn hash_catalog_run(hasher: &mut Sha256, run: &RunRecord) {
    hash_catalog_field(hasher, run.run_id.as_bytes());
    hash_system_time(hasher, run.updated_at);
}

fn catalog_artifacts() -> [CatalogArtifact<'static>; 4] {
    [
        completion_marker(),
        failure_marker(),
        pipeline_artifact(),
        timing_artifact(),
    ]
}

fn completion_marker() -> CatalogArtifact<'static> {
    CatalogArtifact {
        relative: Path::new(".complete"),
        resource: "catalog completion marker",
        maximum_bytes: None,
    }
}

fn failure_marker() -> CatalogArtifact<'static> {
    CatalogArtifact {
        relative: Path::new(".failed"),
        resource: "catalog failure marker",
        maximum_bytes: None,
    }
}

fn pipeline_artifact() -> CatalogArtifact<'static> {
    CatalogArtifact {
        relative: Path::new(PIPELINE_STATE_PATH),
        resource: "catalog analyzer pipeline",
        maximum_bytes: Some(MAX_PIPELINE_STATE_BYTES),
    }
}

fn timing_artifact() -> CatalogArtifact<'static> {
    CatalogArtifact {
        relative: Path::new("reports/analyzer_timing.json"),
        resource: "catalog analyzer timing",
        maximum_bytes: Some(MAX_JSON_BYTES),
    }
}

fn open_catalog_artifact(
    hasher: &mut Sha256,
    run: &RunRecord,
    artifact: CatalogArtifact<'_>,
    lifecycle_bytes: &mut u64,
) -> Result<CapturedArtifact, ApiProblem> {
    hash_catalog_field(hasher, artifact.relative.as_os_str().as_encoded_bytes());
    let opened = match open_artifact_with_metadata(run, artifact.relative, artifact.resource) {
        Ok(opened) => opened,
        Err(problem) if problem.code == "artifact_missing" => {
            hasher.update(b"\0missing\0");
            return Ok(CapturedArtifact::Missing);
        }
        Err(problem) => return Err(problem),
    };
    hasher.update(b"\0present\0");
    hash_metadata(hasher, &opened.metadata);

    if let Some(maximum_bytes) = artifact.maximum_bytes {
        // Aggregate accounting deliberately precedes the per-file policy. One
        // document larger than the total service budget is a catalog-level
        // error, while a smaller per-file violation remains row-local.
        *lifecycle_bytes = lifecycle_bytes
            .checked_add(opened.metadata.len())
            .ok_or_else(catalog_too_large_problem)?;
        if *lifecycle_bytes > MAX_CATALOG_LIFECYCLE_BYTES {
            return Err(catalog_too_large_problem());
        }
        if opened.metadata.len() > maximum_bytes {
            return Ok(CapturedArtifact::Invalid(artifact_too_large_problem(
                artifact.resource,
            )));
        }
    }
    Ok(CapturedArtifact::Ready(opened))
}

pub(super) fn catalog_etag(bytes: &[u8]) -> String {
    format!("\"sha256-{}\"", hex(&Sha256::digest(bytes)))
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

fn ensure_catalog_run_limit(run_count: usize) -> Result<(), ApiProblem> {
    if run_count > MAX_CATALOG_RUNS {
        return Err(ApiProblem::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "catalog_too_many_runs",
            "Catalog has too many runs",
            format!(
                "The analyzer catalog exceeds the service safety limit of {MAX_CATALOG_RUNS} runs."
            ),
        ));
    }
    Ok(())
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
