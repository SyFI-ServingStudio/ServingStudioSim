use thiserror::Error;

use crate::timing::bridge::{ArgsPayload, KernelKind};

#[derive(Debug, Error)]
pub enum PerfApiError {
    /// Raised when `get_times` finds at least one `MissingEntry` in the
    /// Python response. Only the **first** missing spec in the batch is
    /// surfaced — sim runtime treats any miss as a hard error, so a single
    /// representative spec is enough. Callers debugging large miss batches
    /// should re-run with `enable_jit_profiling` or read the profile DB to
    /// see the full miss set rather than depending on this variant's `spec`.
    #[error("missing profile entry for {kind}:{backend} spec {spec:?}")]
    MissingEntry {
        kind: KernelKind,
        backend: String,
        spec: ArgsPayload,
    },

    #[error("Python perf_api error: {0}")]
    Python(String),

    #[error("metric type mismatch: {0}")]
    TypeMismatch(String),

    #[error("invalid bridge request: {0}")]
    InvalidRequest(String),
}

/// Build-time errors raised while populating `*Kernel` caches.
///
/// Note: this enum intentionally does **not** implement `From<PerfApiError>`.
/// `?`-propagating a `PerfApiError` here would either lose the `&'static str`
/// backend invariant on `MissingEntry` (Python hands back a runtime `String`)
/// or silently bury a `MissingEntry` inside `BridgeError`, defeating the
/// typed-error path that `Kernel::init` relies on. All bridge callsites must
/// route through `BuildError::from_perf_api(kind, backend, err)` so the
/// source-literal backend is captured at the point of the call.
#[derive(Debug, Error)]
pub enum BuildError {
    #[error("missing profile entry for {kind}:{backend} spec {spec:?}")]
    MissingEntry {
        kind: KernelKind,
        backend: &'static str,
        spec: ArgsPayload,
    },

    #[error(transparent)]
    BridgeError(PerfApiError),

    #[error("cache fit failed for {kind}: {reason}")]
    FitFailed { kind: KernelKind, reason: String },
}

impl BuildError {
    pub fn from_perf_api(kind: KernelKind, backend: &'static str, error: PerfApiError) -> Self {
        match error {
            PerfApiError::MissingEntry { spec, .. } => Self::MissingEntry {
                kind,
                backend,
                spec,
            },
            other => Self::BridgeError(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::timing::bridge::{ArgsPayload, BuildError, PerfApiError};

    #[test]
    fn build_error_preserves_missing_entry_shape() {
        let mut spec = ArgsPayload::new();
        spec.insert("m", 128);

        let error = BuildError::from_perf_api(
            "single_gemm",
            "torch",
            PerfApiError::MissingEntry {
                kind: "single_gemm",
                backend: "torch".to_string(),
                spec: spec.clone(),
            },
        );

        assert!(matches!(
            error,
            BuildError::MissingEntry {
                kind: "single_gemm",
                backend: "torch",
                spec: returned_spec,
            } if returned_spec == spec
        ));
    }

    #[test]
    fn from_perf_api_wraps_non_missing_variants_in_bridge_error() {
        let error = BuildError::from_perf_api(
            "single_gemm",
            "torch",
            PerfApiError::InvalidRequest("boom".into()),
        );
        assert!(matches!(
            error,
            BuildError::BridgeError(PerfApiError::InvalidRequest(_)),
        ));
    }
}
