use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use serde_json::Value;

use crate::timing::bridge::{
    ArgsPayload, DbMetadata, KernelKind, KernelMetrics, PerfApiError, ProfilerVersion,
};

/// Convert `Result<T, pyo3::PyErr>` to `Result<T, PerfApiError>` by flattening
/// the PyO3 exception into `PerfApiError::Python(message)`. Defined as a
/// trait so the PyO3-heavy code below reads as `.py_err()?` instead of
/// `.py_err()?` at every step.
/// Loss of structural Python exception type is intentional at this boundary —
/// downstream callers reason in `PerfApiError` variants, not Python exception
/// classes.
trait PyErrExt<T> {
    fn py_err(self) -> Result<T, PerfApiError>;
}

impl<T> PyErrExt<T> for Result<T, pyo3::PyErr> {
    fn py_err(self) -> Result<T, PerfApiError> {
        self.map_err(|err| PerfApiError::Python(err.to_string()))
    }
}

/// Bridge to the Python `profiling.perf_api` facade.
///
/// Construct via `new()` or `with_gpu_name()`; both call `disable_jit_profiling`
/// as part of construction so a `PerfApiBridge` handle is always in the
/// "sim-runtime-safe" state by default. Build-cache-only paths that need JIT
/// re-enable it explicitly with `enable_jit_profiling`. There is intentionally
/// no `Default` impl — handing out an unconstructed bridge would skip the
/// `disable_jit_profiling` invariant (L1 design.md §5.1.4 / §1.2 invariant 4).
#[derive(Clone, Debug)]
pub struct PerfApiBridge {
    gpu_name: Option<String>,
}

impl PerfApiBridge {
    pub fn new() -> Result<Self, PerfApiError> {
        let bridge = Self { gpu_name: None };
        bridge.disable_jit_profiling()?;
        Ok(bridge)
    }

    pub fn with_gpu_name(gpu_name: impl Into<String>) -> Result<Self, PerfApiError> {
        let bridge = Self {
            gpu_name: Some(gpu_name.into()),
        };
        bridge.disable_jit_profiling()?;
        Ok(bridge)
    }

    /// Lock the perf_api into "sim-runtime-safe" mode: any spec that's not
    /// already cached in the profile DB will raise `MissingEntry` rather than
    /// kicking off a JIT profile. Called from `new()` / `with_gpu_name()`; the
    /// Python side is idempotent so repeated calls are safe.
    pub fn disable_jit_profiling(&self) -> Result<(), PerfApiError> {
        self.call_perf_api_void("disable_jit_profiling")
    }

    /// Re-enable JIT profiling. Used by the `--build-cache-only` entry point
    /// (L1 design.md §7.2): Rust main flips this back on *before* calling any
    /// `*Kernel::init`, so missing specs are profiled into the DB on demand
    /// instead of erroring out.
    pub fn enable_jit_profiling(&self) -> Result<(), PerfApiError> {
        self.call_perf_api_void("enable_jit_profiling")
    }

    fn call_perf_api_void(&self, attr: &str) -> Result<(), PerfApiError> {
        Python::with_gil(|py| {
            let perf_api = PyModule::import(py, "profiling.perf_api").py_err()?;
            perf_api
                .getattr(attr)
                .and_then(|func| func.call0())
                .py_err()?;
            Ok(())
        })
    }

    /// Look up profiled times for a batch of per-kernel specs.
    ///
    /// `kind` is both the error tag and the facade selector: the Python
    /// function name is derived as `get_{kind}_times`, matching the Python
    /// side's universal `_GENERATED_FACADES` convention (see
    /// `profiling/facade.py`). Per-kernel `enumerate` already produces wire
    /// payloads (`Vec<ArgsPayload>`), so the bridge takes them as-is and the
    /// Python `KernelArgs` dataclass owns the schema check on the other side.
    pub fn get_times(
        &self,
        payloads: Vec<ArgsPayload>,
        kind: KernelKind,
    ) -> Result<Vec<KernelMetrics>, PerfApiError> {
        if payloads.is_empty() {
            return Ok(Vec::new());
        }
        let backend = shared_backend(&payloads)?;
        Python::with_gil(|py| {
            let perf_api = PyModule::import(py, "profiling.perf_api").py_err()?;
            let py_specs = payloads_to_py_list(py, &payloads)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("backend", backend).py_err()?;
            if let Some(gpu_name) = &self.gpu_name {
                kwargs.set_item("gpu_name", gpu_name).py_err()?;
            }
            let fn_name = format!("get_{kind}_times");
            let results = perf_api
                .getattr(fn_name.as_str())
                .and_then(|func| func.call((py_specs,), Some(kwargs)))
                .py_err()?;
            py_results_to_metrics(kind, backend, &payloads, results)
        })
    }

    /// Count cache-miss specs for `kind` against the given backend's profile.
    /// Python function name derived as `count_missing_{kind}`.
    pub fn count_missing(
        &self,
        payloads: Vec<ArgsPayload>,
        kind: KernelKind,
        backend: &str,
    ) -> Result<usize, PerfApiError> {
        if payloads.is_empty() {
            return Ok(0);
        }
        ensure_payload_backends_match(&payloads, backend)?;
        Python::with_gil(|py| {
            let perf_api = PyModule::import(py, "profiling.perf_api").py_err()?;
            let py_specs = payloads_to_py_list(py, &payloads)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("backend", backend).py_err()?;
            if let Some(gpu_name) = &self.gpu_name {
                kwargs.set_item("gpu_name", gpu_name).py_err()?;
            }
            let fn_name = format!("count_missing_{kind}");
            perf_api
                .getattr(fn_name.as_str())
                .and_then(|func| func.call((py_specs,), Some(kwargs)))
                .and_then(|value| value.extract::<usize>())
                .py_err()
        })
    }

    pub fn get_db_metadata(&self) -> Result<DbMetadata, PerfApiError> {
        Python::with_gil(|py| {
            let perf_api = PyModule::import(py, "profiling.perf_api").py_err()?;
            let value = perf_api
                .getattr("get_db_metadata")
                .and_then(|func| func.call0())
                .py_err()?;
            Ok(DbMetadata {
                schema_version: value
                    .getattr("schema_version")
                    .and_then(|attr| attr.extract::<u32>())
                    .py_err()?,
                schema_hash: value
                    .getattr("schema_hash")
                    .and_then(|attr| attr.extract::<String>())
                    .py_err()?,
                created_at: optional_string(value, "created_at")?,
                last_migrated_at: optional_string(value, "last_migrated_at")?,
            })
        })
    }

    pub fn get_profiler_versions(
        &self,
        op_families: Vec<&str>,
    ) -> Result<Vec<ProfilerVersion>, PerfApiError> {
        Python::with_gil(|py| {
            let perf_api = PyModule::import(py, "profiling.perf_api").py_err()?;
            let families = PyList::new(py, op_families);
            let values = perf_api
                .getattr("get_profiler_versions")
                .and_then(|func| func.call1((families,)))
                .py_err()?;
            let mut versions = Vec::new();
            for item in values.iter().py_err()? {
                let item = item.py_err()?;
                versions.push(ProfilerVersion {
                    op_family: item
                        .getattr("op_family")
                        .and_then(|attr| attr.extract::<String>())
                        .py_err()?,
                    profiler_git_hash: item
                        .getattr("profiler_git_hash")
                        .and_then(|attr| attr.extract::<String>())
                        .py_err()?,
                });
            }
            Ok(versions)
        })
    }
}

fn payloads_to_py_list<'py>(
    py: Python<'py>,
    payloads: &[ArgsPayload],
) -> Result<&'py PyList, PerfApiError> {
    let list = PyList::empty(py);
    for payload in payloads {
        let dict = PyDict::new(py);
        for (key, value) in payload.fields() {
            dict.set_item(key, json_value_to_py(py, value)?).py_err()?;
        }
        list.append(dict).py_err()?;
    }
    Ok(list)
}

fn json_value_to_py(py: Python<'_>, value: &Value) -> Result<PyObject, PerfApiError> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(value) => Ok(value.into_py(py)),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                Ok(value.into_py(py))
            } else if let Some(value) = value.as_u64() {
                Ok(value.into_py(py))
            } else if let Some(value) = value.as_f64() {
                Ok(value.into_py(py))
            } else {
                Err(PerfApiError::InvalidRequest(
                    "unsupported JSON number".to_string(),
                ))
            }
        }
        Value::String(value) => Ok(value.into_py(py)),
        Value::Array(values) => {
            let list = PyList::empty(py);
            for item in values {
                list.append(json_value_to_py(py, item)?).py_err()?;
            }
            Ok(list.into_py(py))
        }
        Value::Object(_) => Err(PerfApiError::InvalidRequest(
            "nested object payloads are not supported yet".to_string(),
        )),
    }
}

fn py_results_to_metrics(
    kind: KernelKind,
    backend: &str,
    specs: &[ArgsPayload],
    results: &PyAny,
) -> Result<Vec<KernelMetrics>, PerfApiError> {
    let result_len = results.len().py_err()?;
    if result_len != specs.len() {
        return Err(PerfApiError::TypeMismatch(format!(
            "perf_api returned {result_len} result(s) for {} spec(s)",
            specs.len()
        )));
    }

    let mut metrics = Vec::new();
    for (idx, item) in results.iter().py_err()?.enumerate() {
        let item = item.py_err()?;
        let class_name = item
            .getattr("__class__")
            .and_then(|class| class.getattr("__name__"))
            .and_then(|name| name.extract::<String>())
            .py_err()?;
        if class_name == "MissingEntry" {
            return Err(PerfApiError::MissingEntry {
                kind,
                backend: backend.to_string(),
                spec: specs[idx].clone(),
            });
        }
        metrics.push(KernelMetrics {
            time_ms: extract_f64(item, "time_ms")?,
            tflops: optional_f64(item, "tflops")?,
            memory_bandwidth_gbps: optional_f64(item, "memory_bandwidth_gbps")?,
            algbw_gbps: optional_f64(item, "algbw_gbps")?,
            busbw_gbps: optional_f64(item, "busbw_gbps")?,
            message_size_bytes: optional_u64(item, "message_size_bytes")?,
            // `KernelMetrics::energy_j` is a required f64 (not `Option`);
            // profilers that don't measure energy report 0.0 via this default.
            // `is_finite` validates the resulting value, so a missing field is
            // semantically "zero energy" and does not bypass NaN/Inf checks.
            energy_j: optional_f64(item, "energy_j")?.unwrap_or(0.0),
        });
    }
    Ok(metrics)
}

fn shared_backend(payloads: &[ArgsPayload]) -> Result<&str, PerfApiError> {
    // perf_api keeps backend as an explicit kwarg (§3.2.3). The typed Rust
    // wrappers still embed it in ArgsPayload so this generic bridge can verify
    // one-backend batches before passing the kwarg through the PyO3 boundary.
    // `get_times_payload` early-returns on empty batches, but we still guard
    // here so any future caller cannot turn an empty slice into a panic.
    let backend = payloads
        .first()
        .and_then(ArgsPayload::backend)
        .ok_or_else(|| PerfApiError::InvalidRequest("spec is missing backend".to_string()))?;
    if payloads
        .iter()
        .any(|payload| payload.backend() != Some(backend))
    {
        return Err(PerfApiError::InvalidRequest(
            "bridge get_times requires one backend per batch".to_string(),
        ));
    }
    Ok(backend)
}

/// `count_missing` takes `backend` as an explicit argument (it doesn't share
/// `get_times`'s "one backend per batch" derivation because dry-run callers
/// need to pass a per-backend kwarg directly). The typed wrappers still embed
/// `backend` in each payload, so a wrapper bug could ship `payload.backend = A`
/// alongside `count_missing(..., backend = B)` and Python would silently count
/// against the wrong backend. Reject that mismatch at the bridge boundary.
fn ensure_payload_backends_match(
    payloads: &[ArgsPayload],
    backend: &str,
) -> Result<(), PerfApiError> {
    for (idx, payload) in payloads.iter().enumerate() {
        match payload.backend() {
            Some(payload_backend) if payload_backend == backend => continue,
            Some(payload_backend) => {
                return Err(PerfApiError::InvalidRequest(format!(
                    "count_missing backend kwarg '{backend}' disagrees with spec[{idx}].backend = '{payload_backend}'",
                )));
            }
            None => {
                return Err(PerfApiError::InvalidRequest(format!(
                    "count_missing spec[{idx}] is missing backend",
                )));
            }
        }
    }
    Ok(())
}

fn extract_f64(item: &PyAny, field: &str) -> Result<f64, PerfApiError> {
    item.getattr(field)
        .and_then(|attr| attr.extract::<f64>())
        .py_err()
}

// `optional_*` helpers below intentionally collapse `Err(AttributeError)` and
// `Ok(None)` into the same `Ok(None)` result. This is *not* leniency by
// accident: Python `ComputeMetrics` and `CommMetrics` are separate dataclasses
// with disjoint optional fields (`tflops` lives only on compute,
// `algbw_gbps`/`busbw_gbps`/`message_size_bytes` only on comm). When this
// bridge reads a row from either family, the cross-family fields legitimately
// don't exist on the Python object, and `AttributeError` is the expected
// signal. The cost is that a future field rename on the Python side will
// silently drop the value rather than fail loudly — guarded against by the
// `KernelMetrics::is_finite` invariant + integration tests on perf_api.

fn optional_f64(item: &PyAny, field: &str) -> Result<Option<f64>, PerfApiError> {
    match item.getattr(field) {
        Ok(value) if !value.is_none() => value.extract::<f64>().map(Some).py_err(),
        _ => Ok(None),
    }
}

fn optional_u64(item: &PyAny, field: &str) -> Result<Option<u64>, PerfApiError> {
    match item.getattr(field) {
        Ok(value) if !value.is_none() => value.extract::<u64>().map(Some).py_err(),
        _ => Ok(None),
    }
}

fn optional_string(item: &PyAny, field: &str) -> Result<Option<String>, PerfApiError> {
    match item.getattr(field) {
        Ok(value) if !value.is_none() => value.extract::<String>().map(Some).py_err(),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::{ensure_payload_backends_match, shared_backend};
    use crate::timing::bridge::{ArgsPayload, PerfApiError};
    use serde_json::Value;

    fn payload(backend: Option<&str>) -> ArgsPayload {
        let mut p = ArgsPayload::new();
        if let Some(backend) = backend {
            p.insert("backend", Value::from(backend));
        }
        p
    }

    #[test]
    fn ensure_payload_backends_match_passes_on_uniform_batch() {
        let batch = vec![payload(Some("torch")), payload(Some("torch"))];
        assert!(ensure_payload_backends_match(&batch, "torch").is_ok());
    }

    #[test]
    fn ensure_payload_backends_match_rejects_kwarg_payload_mismatch() {
        let batch = vec![payload(Some("torch")), payload(Some("triton"))];
        let err = ensure_payload_backends_match(&batch, "torch").unwrap_err();
        assert!(matches!(err, PerfApiError::InvalidRequest(_)));
    }

    #[test]
    fn ensure_payload_backends_match_rejects_missing_backend_field() {
        let batch = vec![payload(None)];
        let err = ensure_payload_backends_match(&batch, "torch").unwrap_err();
        assert!(matches!(err, PerfApiError::InvalidRequest(_)));
    }

    #[test]
    fn shared_backend_rejects_empty_batch_instead_of_panic() {
        let err = shared_backend(&[]).unwrap_err();
        assert!(matches!(err, PerfApiError::InvalidRequest(_)));
    }
}
