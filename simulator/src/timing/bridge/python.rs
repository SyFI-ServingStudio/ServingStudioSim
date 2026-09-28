//! The PyO3 calls into Python `profiling.perf_api`, behind the `python` cargo
//! feature. [`PerfApiBridge`](super::PerfApiBridge) routes every perf_api call
//! through these functions (or answers it from replay samples first), so the
//! rest of the crate builds without PyO3 — the wasm32 build turns the feature
//! off and gets the stub module below, whose calls all fail.

#[cfg(not(feature = "python"))]
pub(super) use disabled::*;
#[cfg(feature = "python")]
pub(super) use enabled::*;

#[cfg(feature = "python")]
mod enabled {
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList, PyModule};
    use serde_json::Value;

    use crate::timing::bridge::{
        ArgsPayload, DbMetadata, KernelKind, KernelMetrics, PerfApiError, ProfilerVersion,
    };

    pub(crate) fn call_perf_api_void(attr: &str) -> Result<(), PerfApiError> {
        Python::with_gil(|py| {
            let perf_api = PyModule::import(py, "profiling.perf_api").py_err()?;
            perf_api
                .getattr(attr)
                .and_then(|func| func.call0())
                .py_err()?;
            Ok(())
        })
    }

    pub(crate) fn issue_collected(expected_specs: usize) -> Result<(), PerfApiError> {
        Python::with_gil(|py| {
            let perf_api = PyModule::import(py, "profiling.perf_api").py_err()?;
            perf_api
                .getattr("issue_collected")
                .and_then(|func| func.call1((expected_specs,)))
                .py_err()?;
            Ok(())
        })
    }

    pub(crate) fn get_times(
        payloads: &[ArgsPayload],
        kind: KernelKind,
        backend: &str,
        gpu_name: &str,
    ) -> Result<Vec<KernelMetrics>, PerfApiError> {
        Python::with_gil(|py| {
            let perf_api = PyModule::import(py, "profiling.perf_api").py_err()?;
            let py_specs = payloads_to_py_list(py, payloads)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("backend", backend).py_err()?;
            kwargs.set_item("gpu_name", gpu_name).py_err()?;
            let fn_name = format!("get_{kind}_times");
            let results = perf_api
                .getattr(fn_name.as_str())
                .and_then(|func| func.call((py_specs,), Some(kwargs)))
                .py_err()?;
            py_results_to_metrics(kind, backend, payloads, results)
        })
    }

    pub(crate) fn count_missing(
        payloads: &[ArgsPayload],
        kind: KernelKind,
        backend: &str,
        gpu_name: &str,
    ) -> Result<usize, PerfApiError> {
        Python::with_gil(|py| {
            let perf_api = PyModule::import(py, "profiling.perf_api").py_err()?;
            let py_specs = payloads_to_py_list(py, payloads)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("backend", backend).py_err()?;
            kwargs.set_item("gpu_name", gpu_name).py_err()?;
            let fn_name = format!("count_missing_{kind}");
            perf_api
                .getattr(fn_name.as_str())
                .and_then(|func| func.call((py_specs,), Some(kwargs)))
                .and_then(|value| value.extract::<usize>())
                .py_err()
        })
    }

    pub(crate) fn get_current_gpu_name() -> Result<String, PerfApiError> {
        Python::with_gil(|py| {
            let perf_api = PyModule::import(py, "profiling.perf_api").py_err()?;
            perf_api
                .getattr("get_current_gpu_name")
                .and_then(|func| func.call0())
                .and_then(|value| value.extract::<String>())
                .py_err()
        })
    }

    pub(crate) fn get_db_metadata() -> Result<DbMetadata, PerfApiError> {
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

    pub(crate) fn get_profiler_versions(
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
                // `KernelMetrics::energy_j` is a required f64 (not `Option`);
                // profilers that don't measure energy report 0.0 via this default.
                // `is_finite` validates the resulting value, so a missing field is
                // semantically "zero energy" and does not bypass NaN/Inf checks.
                energy_j: optional_f64(item, "energy_j")?.unwrap_or(0.0),
            });
        }
        Ok(metrics)
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
    // `algbw_gbps`/`busbw_gbps` only on comm). When this
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

    fn optional_string(item: &PyAny, field: &str) -> Result<Option<String>, PerfApiError> {
        match item.getattr(field) {
            Ok(value) if !value.is_none() => value.extract::<String>().map(Some).py_err(),
            _ => Ok(None),
        }
    }
}

#[cfg(not(feature = "python"))]
mod disabled {
    use crate::timing::bridge::{
        ArgsPayload, DbMetadata, KernelKind, KernelMetrics, PerfApiError, ProfilerVersion,
    };

    fn unavailable() -> PerfApiError {
        PerfApiError::Python(
            "built without the `python` feature: no perf_api; use PerfApiBridge::replay"
                .to_string(),
        )
    }

    pub(crate) fn call_perf_api_void(_attr: &str) -> Result<(), PerfApiError> {
        Err(unavailable())
    }
    pub(crate) fn issue_collected(_expected_specs: usize) -> Result<(), PerfApiError> {
        Err(unavailable())
    }
    pub(crate) fn get_times(
        _payloads: &[ArgsPayload],
        _kind: KernelKind,
        _backend: &str,
        _gpu_name: &str,
    ) -> Result<Vec<KernelMetrics>, PerfApiError> {
        Err(unavailable())
    }
    pub(crate) fn count_missing(
        _payloads: &[ArgsPayload],
        _kind: KernelKind,
        _backend: &str,
        _gpu_name: &str,
    ) -> Result<usize, PerfApiError> {
        Err(unavailable())
    }
    pub(crate) fn get_current_gpu_name() -> Result<String, PerfApiError> {
        Err(unavailable())
    }
    pub(crate) fn get_db_metadata() -> Result<DbMetadata, PerfApiError> {
        Err(unavailable())
    }
    pub(crate) fn get_profiler_versions(
        _op_families: Vec<&str>,
    ) -> Result<Vec<ProfilerVersion>, PerfApiError> {
        Err(unavailable())
    }
}
