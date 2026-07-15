//! Containment-checked bounded artifact reads and conditional responses.

use super::*;

use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};

use tokio::io::{AsyncRead, ReadBuf};
use tokio::sync::OwnedSemaphorePermit;

fn acquire_artifact_read_permit(
    state: &Arc<ServiceState>,
) -> Result<OwnedSemaphorePermit, ApiProblem> {
    Arc::clone(&state.artifact_reads)
        .try_acquire_owned()
        .map_err(|_| {
            ApiProblem::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "artifact_read_busy",
                "Artifact readers are busy",
                "The bounded artifact read limit is already in use; retry shortly.",
            )
        })
}

/// Execute filesystem access and JSON parsing outside Tokio's async workers.
/// `try_acquire_owned` is intentionally fail-fast: a semaphore with an
/// unbounded waiter queue would merely move the request-amplification problem.
pub(super) async fn run_blocking_artifact_task<T, F>(
    state: &Arc<ServiceState>,
    operation: &'static str,
    task: F,
) -> Result<T, ApiProblem>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, ApiProblem> + Send + 'static,
{
    let permit = acquire_artifact_read_permit(state)?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        task()
    })
    .await
    .map_err(|error| {
        ApiProblem::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "artifact_read_failed",
            "Artifact read failed",
            format!("The blocking {operation} task failed: {error}"),
        )
    })?
}

/// Prepare a streamed artifact on a blocking worker, but return its permit to
/// the response body. Slow or abandoned clients therefore remain part of the
/// same finite admission budget until their stream is consumed or dropped.
pub(super) async fn run_blocking_stream_task<T, F>(
    state: &Arc<ServiceState>,
    operation: &'static str,
    task: F,
) -> Result<(T, OwnedSemaphorePermit), ApiProblem>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, ApiProblem> + Send + 'static,
{
    let permit = acquire_artifact_read_permit(state)?;
    let output = tokio::task::spawn_blocking(task).await.map_err(|error| {
        ApiProblem::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "artifact_read_failed",
            "Artifact read failed",
            format!("The blocking {operation} task failed: {error}"),
        )
    })??;
    Ok((output, permit))
}

pub(super) struct BoundedArtifact {
    pub(super) file: fs::File,
    pub(super) metadata: fs::Metadata,
    path: PathBuf,
}

#[cfg(test)]
static ARTIFACT_READ_OBSERVER: std::sync::OnceLock<
    std::sync::Mutex<Option<std::sync::mpsc::Sender<PathBuf>>>,
> = std::sync::OnceLock::new();

#[cfg(test)]
pub(super) fn set_artifact_read_observer(observer: Option<std::sync::mpsc::Sender<PathBuf>>) {
    *ARTIFACT_READ_OBSERVER
        .get_or_init(|| std::sync::Mutex::new(None))
        .lock()
        .expect("artifact read observer lock") = observer;
}

pub(super) enum ConditionalBody {
    NotModified { etag: String },
    Bytes { etag: String, bytes: Vec<u8> },
}

/// Open one run-relative artifact without permitting a symlink traversal.
/// Linux uses `openat2(RESOLVE_BENEATH|NO_SYMLINKS)` from the stable configured
/// root descriptor, so run containment and open are one kernel operation;
/// metadata, reads, and streaming reuse the resulting artifact descriptor.
/// The portable fallback retains canonical containment for non-Linux builds,
/// where configured logs roots remain a trusted-writer boundary.
pub(super) fn open_contained_artifact(
    run: &RunRecord,
    relative: &Path,
    resource: &str,
) -> Result<(fs::File, PathBuf), ApiProblem> {
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ApiProblem::new(
            StatusCode::BAD_REQUEST,
            "invalid_resource_path",
            "Invalid resource path",
            "The server-built artifact path is not a normalized relative path.",
        ));
    }

    #[cfg(target_os = "linux")]
    {
        use rustix::fs::{openat2, Mode, OFlags, ResolveFlags};
        use rustix::io::Errno;

        let run_relative = run.path.strip_prefix(&run.root).map_err(|_| {
            ApiProblem::new(
                StatusCode::FORBIDDEN,
                "artifact_outside_run",
                "Artifact escaped its run",
                "The resolved run is not relative to its configured logs root.",
            )
        })?;
        if run_relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(ApiProblem::new(
                StatusCode::FORBIDDEN,
                "artifact_outside_run",
                "Artifact escaped its run",
                "The resolved run path is not normalized beneath its logs root.",
            ));
        }
        let root_relative = run_relative.join(relative);
        let descriptor = openat2(
            &*run.root_directory,
            &root_relative,
            OFlags::RDONLY | OFlags::CLOEXEC,
            Mode::empty(),
            ResolveFlags::BENEATH | ResolveFlags::NO_MAGICLINKS | ResolveFlags::NO_SYMLINKS,
        )
        .map_err(|error| match error {
            Errno::NOENT | Errno::NOTDIR => ApiProblem::artifact_missing(resource),
            Errno::LOOP | Errno::XDEV => ApiProblem::new(
                StatusCode::FORBIDDEN,
                "artifact_outside_run",
                "Artifact escaped its run",
                format!("The bounded {resource} artifact traverses a forbidden symlink."),
            ),
            // ENOSYS/EINVAL includes kernels without usable openat2 support.
            // Linux deliberately fails closed instead of weakening containment.
            _ => ApiProblem::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "Artifact read failed",
                format!("Cannot securely open the bounded {resource}: {error}"),
            ),
        })?;
        let file = fs::File::from(descriptor);
        if !file
            .metadata()
            .map(|metadata| metadata.is_file())
            .unwrap_or(false)
        {
            return Err(ApiProblem::artifact_missing(resource));
        }
        Ok((file, run.path.join(relative)))
    }

    #[cfg(not(target_os = "linux"))]
    {
        let path = resolve_contained_file(&run.root, &run.path, relative, resource)?;
        let file = fs::File::open(&path).map_err(|error| {
            ApiProblem::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "Artifact read failed",
                format!("Cannot open the bounded {resource}: {error}"),
            )
        })?;
        Ok((file, path))
    }
}

pub(super) fn resolve_contained_file(
    root: &Path,
    run: &Path,
    relative: &Path,
    resource: &str,
) -> Result<PathBuf, ApiProblem> {
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ApiProblem::new(
            StatusCode::BAD_REQUEST,
            "invalid_resource_path",
            "Invalid resource path",
            "The server-built artifact path is not a normalized relative path.",
        ));
    }
    let candidate = run.join(relative);
    let canonical = match candidate.canonicalize() {
        Ok(path) => path,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            return Err(ApiProblem::artifact_missing(resource))
        }
        Err(error) => {
            return Err(ApiProblem::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "Artifact read failed",
                format!("Cannot resolve the bounded {resource} artifact: {error}"),
            ))
        }
    };
    if !canonical.starts_with(root) || !canonical.starts_with(run) {
        return Err(ApiProblem::new(
            StatusCode::FORBIDDEN,
            "artifact_outside_run",
            "Artifact escaped its run",
            format!("The bounded {resource} artifact resolves outside its canonical run."),
        ));
    }
    if !canonical.is_file() {
        return Err(ApiProblem::artifact_missing(resource));
    }
    Ok(canonical)
}

pub(super) fn read_json_value(
    run: &RunRecord,
    relative: &Path,
    resource: &str,
) -> Result<Value, ApiProblem> {
    let bytes = read_json_artifact(run, relative, resource)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| ApiProblem::artifact_incompatible(resource, error.to_string()))
}

pub(super) fn read_json_value_optional(run: &RunRecord, relative: &Path) -> Option<Value> {
    read_json_value(run, relative, "optional run metadata").ok()
}

pub(super) fn read_json_artifact(
    run: &RunRecord,
    relative: &Path,
    resource: &str,
) -> Result<Vec<u8>, ApiProblem> {
    let (bytes, _) = read_bounded_artifact(run, relative, MAX_JSON_BYTES, resource)?;
    serde_json::from_slice::<Value>(&bytes)
        .map_err(|error| ApiProblem::artifact_incompatible(resource, error.to_string()))?;
    Ok(bytes)
}

pub(super) fn read_bounded_artifact(
    run: &RunRecord,
    relative: &Path,
    maximum: u64,
    resource: &str,
) -> Result<(Vec<u8>, SystemTime), ApiProblem> {
    let artifact = open_bounded_artifact(run, relative, maximum, resource)?;
    read_bounded_open_file(artifact, maximum, resource)
}

pub(super) fn open_bounded_artifact(
    run: &RunRecord,
    relative: &Path,
    maximum: u64,
    resource: &str,
) -> Result<BoundedArtifact, ApiProblem> {
    let artifact = open_artifact_with_metadata(run, relative, resource)?;
    if artifact.metadata.len() > maximum {
        return Err(artifact_too_large_problem(resource));
    }
    Ok(artifact)
}

/// Open and stat one contained regular file without applying a size policy.
/// Callers that combine several artifacts can account the exact opened-file
/// metadata before deciding whether an individual body may be read.
pub(super) fn open_artifact_with_metadata(
    run: &RunRecord,
    relative: &Path,
    resource: &str,
) -> Result<BoundedArtifact, ApiProblem> {
    let (file, path) = open_contained_artifact(run, relative, resource)?;
    let metadata = file.metadata().map_err(|error| {
        ApiProblem::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "artifact_read_failed",
            "Artifact read failed",
            format!("Cannot inspect the opened bounded {resource}: {error}"),
        )
    })?;
    if !metadata.is_file() {
        return Err(ApiProblem::artifact_missing(resource));
    }
    Ok(BoundedArtifact {
        file,
        metadata,
        path,
    })
}

pub(super) fn artifact_too_large_problem(resource: &str) -> ApiProblem {
    ApiProblem::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        "artifact_too_large",
        "Artifact is too large",
        format!("The bounded {resource} exceeds the service safety limit."),
    )
}

pub(super) fn read_bounded_open_file(
    artifact: BoundedArtifact,
    maximum: u64,
    resource: &str,
) -> Result<(Vec<u8>, SystemTime), ApiProblem> {
    let BoundedArtifact {
        file,
        metadata,
        path,
    } = artifact;
    #[cfg(test)]
    if let Some(observer) = ARTIFACT_READ_OBSERVER
        .get_or_init(|| std::sync::Mutex::new(None))
        .lock()
        .expect("artifact read observer lock")
        .as_ref()
    {
        let _ = observer.send(path);
    }
    #[cfg(not(test))]
    let _ = path;
    let initial_capacity = usize::try_from(metadata.len().min(1024 * 1024)).unwrap_or(0);
    let mut bytes = Vec::with_capacity(initial_capacity);
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            ApiProblem::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "Artifact read failed",
                format!("Cannot read the bounded {resource}: {error}"),
            )
        })?;
    if bytes.len() as u64 > maximum {
        return Err(ApiProblem::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "artifact_too_large",
            "Artifact is too large",
            format!("The bounded {resource} exceeds the service safety limit."),
        ));
    }
    Ok((bytes, metadata.modified().unwrap_or(UNIX_EPOCH)))
}

/// Prepare a bounded JSON response from one stable file descriptor. The weak
/// ETag is derived from trusted-writer metadata, so an unchanged conditional
/// request returns before reading, hashing, or parsing the document.
pub(super) fn prepare_json_artifact<F>(
    headers: &HeaderMap,
    run: &RunRecord,
    relative: &Path,
    maximum: u64,
    resource: &str,
    validate: F,
) -> Result<ConditionalBody, ApiProblem>
where
    F: FnOnce(&Value) -> Result<(), ApiProblem>,
{
    let artifact = open_bounded_artifact(run, relative, maximum, resource)?;
    let etag = metadata_etag("json-artifact-v1", &[(relative, &artifact.metadata)]);
    if if_none_match_exact(headers, &etag) {
        return Ok(ConditionalBody::NotModified { etag });
    }
    let (bytes, _) = read_bounded_open_file(artifact, maximum, resource)?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| ApiProblem::artifact_incompatible(resource, error.to_string()))?;
    validate(&value)?;
    if if_none_match(headers, &etag) {
        return Ok(ConditionalBody::NotModified { etag });
    }
    Ok(ConditionalBody::Bytes { etag, bytes })
}

pub(super) fn metadata_etag(namespace: &str, artifacts: &[(&Path, &fs::Metadata)]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-ui-artifact-metadata-v1\0");
    hasher.update(namespace.as_bytes());
    for (relative, metadata) in artifacts {
        hasher.update(b"\0");
        hasher.update(relative.as_os_str().as_encoded_bytes());
        hash_metadata(&mut hasher, metadata);
    }
    format!("W/\"artifact-meta-v1-{}\"", hex(&hasher.finalize()))
}

pub(super) fn hash_metadata(hasher: &mut Sha256, metadata: &fs::Metadata) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        for value in [
            metadata.dev(),
            metadata.ino(),
            metadata.len(),
            metadata.mtime() as u64,
            metadata.mtime_nsec() as u64,
        ] {
            hasher.update(value.to_be_bytes());
        }
    }
    #[cfg(not(unix))]
    {
        hasher.update(metadata.len().to_be_bytes());
        let modified = metadata
            .modified()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        hasher.update(modified.to_be_bytes());
    }
}

pub(super) fn encode_json<T: Serialize>(value: &T) -> Result<Vec<u8>, ApiProblem> {
    serde_json::to_vec(value).map_err(|error| {
        ApiProblem::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "response_encoding_failed",
            "Response encoding failed",
            error.to_string(),
        )
    })
}

pub(super) fn conditional_response(
    content_type: &'static str,
    prepared: ConditionalBody,
) -> Result<Response, ApiProblem> {
    match prepared {
        ConditionalBody::NotModified { etag } => Response::builder()
            .status(StatusCode::NOT_MODIFIED)
            .header(header::ETAG, etag)
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::empty())
            .map_err(response_build_problem),
        ConditionalBody::Bytes { etag, bytes } => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, content_type)
            .header(header::ETAG, etag)
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from(bytes))
            .map_err(response_build_problem),
    }
}

pub(super) struct PreparedTrace {
    file: fs::File,
    byte_length: u64,
    etag: String,
}

struct PermitReader<R> {
    inner: R,
    _permit: OwnedSemaphorePermit,
}

impl<R: AsyncRead + Unpin> AsyncRead for PermitReader<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        context: &mut TaskContext<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(context, buffer)
    }
}

pub(super) fn conditional_prepared_trace(
    headers: &HeaderMap,
    content_type: &'static str,
    prepared: PreparedTrace,
    permit: OwnedSemaphorePermit,
) -> Result<Response, ApiProblem> {
    if if_none_match(headers, &prepared.etag) {
        drop(permit);
        return Response::builder()
            .status(StatusCode::NOT_MODIFIED)
            .header(header::ETAG, prepared.etag)
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::empty())
            .map_err(response_build_problem);
    }

    let reader = PermitReader {
        inner: tokio::fs::File::from_std(prepared.file).take(prepared.byte_length),
        _permit: permit,
    };
    let stream = ReaderStream::new(reader);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, prepared.byte_length)
        .header(header::ETAG, prepared.etag)
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .map_err(response_build_problem)
}

pub(super) fn prepare_trace(run: &RunRecord, path: PathBuf) -> Result<PreparedTrace, ApiProblem> {
    let relative = path.strip_prefix(&run.path).map_err(|_| {
        ApiProblem::new(
            StatusCode::FORBIDDEN,
            "artifact_outside_run",
            "Artifact escaped its run",
            "The selected Perfetto trace is not relative to its resolved run.",
        )
    })?;
    let (mut file, _) = open_contained_artifact(run, relative, "Perfetto trace")?;
    let metadata = file.metadata().map_err(|error| {
        ApiProblem::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "artifact_read_failed",
            "Artifact read failed",
            format!("Cannot inspect the opened bounded Perfetto trace: {error}"),
        )
    })?;
    if metadata.len() > MAX_TRACE_BYTES {
        return Err(ApiProblem::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "artifact_too_large",
            "Artifact is too large",
            "The bounded Perfetto trace exceeds the service safety limit.",
        ));
    }

    // fstat is the cheap early bound. A one-byte probe at max catches growth
    // between open and fstat without hashing/scanning the representation.
    file.seek(SeekFrom::Start(MAX_TRACE_BYTES))
        .and_then(|_| {
            let mut overflow = [0u8; 1];
            file.read(&mut overflow)
        })
        .map_err(|error| {
            ApiProblem::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "Artifact read failed",
                format!("Cannot enforce the Perfetto trace hard limit: {error}"),
            )
        })?
        .eq(&0)
        .then_some(())
        .ok_or_else(|| {
            ApiProblem::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "artifact_too_large",
                "Artifact is too large",
                "The bounded Perfetto trace exceeds the service safety limit.",
            )
        })?;
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        ApiProblem::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "artifact_read_failed",
            "Artifact read failed",
            format!("Cannot rewind the bounded Perfetto trace: {error}"),
        )
    })?;

    Ok(PreparedTrace {
        file,
        byte_length: metadata.len(),
        etag: trace_metadata_etag(&metadata),
    })
}

pub(super) fn trace_metadata_etag(metadata: &fs::Metadata) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        format!(
            "W/\"trace-v1-{:x}-{:x}-{:x}-{:x}-{:x}\"",
            metadata.dev(),
            metadata.ino(),
            metadata.len(),
            metadata.mtime(),
            metadata.mtime_nsec()
        )
    }
    #[cfg(not(unix))]
    {
        let modified = metadata
            .modified()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("W/\"trace-v1-{:x}-{modified:x}\"", metadata.len())
    }
}

pub(super) fn if_none_match(headers: &HeaderMap, etag: &str) -> bool {
    if_none_match_candidates(headers, etag, true)
}

/// Exact validators are safe before parsing a metadata-stable JSON artifact.
/// RFC wildcard semantics are preserved by a second check after validation.
pub(super) fn if_none_match_exact(headers: &HeaderMap, etag: &str) -> bool {
    if_none_match_candidates(headers, etag, false)
}

fn if_none_match_candidates(headers: &HeaderMap, etag: &str, accept_wildcard: bool) -> bool {
    headers
        .get_all(header::IF_NONE_MATCH)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .any(|candidate| {
            (accept_wildcard && candidate == "*")
                || candidate.trim_start_matches("W/") == etag.trim_start_matches("W/")
        })
}

pub(super) fn response_build_problem(error: axum::http::Error) -> ApiProblem {
    ApiProblem::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "response_build_failed",
        "Response build failed",
        error.to_string(),
    )
}

pub(super) fn timestamp(time: SystemTime) -> String {
    DateTime::<Utc>::from(time).to_rfc3339_opts(SecondsFormat::Nanos, true)
}

pub(super) fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}
