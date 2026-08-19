use std::cell::RefCell;
use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use serde_json::Value;

use crate::timing::bridge::{
    intern_backend, ArgsPayload, DType, DbMetadata, KernelKind, KernelMetrics, PerfApiError,
    ProfilerVersion,
};

/// One kernel's profile-coverage line for the `dry-run` report: how many of its
/// enumerated specs are missing from `profile.db` (i.e. would be JIT-profiled on a
/// real build). `name` is the kernel's dotted path; counts are summed over its
/// backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelMissing {
    pub name: String,
    pub kind: KernelKind,
    pub missing: usize,
    pub total: usize,
}

/// One distinct kernel's structural facts for the `emit-backends` enumerator: its
/// pool, dotted role `name`, `kind`, structured `describe_config`, and the current
/// const-default candidate `backends`. Collected with NO profiling / GPU — the
/// enumerate bridge mode records this and returns an empty kernel before any
/// `profile.db` lookup. Reused sites (folded layers, unrolled experts) emit one
/// record each; the launcher dedups by `(pool, name)` and counts occurrences.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct KernelEnum {
    pub pool: String,
    pub name: String,
    pub kind: KernelKind,
    /// The run GPU (nvidia-smi name) — a typed field for the launcher's capability
    /// GPU axis (e.g. trt = Blackwell-only), so it need not scrape `config`.
    pub gpu: String,
    /// Compute/kv dtypes as typed fields (serialized to the wire literal `"bf16"`
    /// / `"fp8_e4m3"` by `DType`, matching the launcher's `DType` enum), `None`
    /// for dtype-agnostic kernels. These drive capability filtering; `config`
    /// below is kept ONLY for the launcher's informational shape annotation.
    pub compute_dtype: Option<DType>,
    pub kv_dtype: Option<DType>,
    pub config: Value,
    pub backends: Vec<String>,
}

/// Convert `Result<T, pyo3::PyErr>` to `Result<T, PerfApiError>` by flattening
/// the `PyO3` exception into `PerfApiError::Python(message)`. Defined as a
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
/// Construct via `new()`, which calls `disable_jit_profiling` so a
/// `PerfApiBridge` handle is always in the "sim-runtime-safe" state by default.
/// Build-cache-only paths that need JIT re-enable it explicitly with
/// `enable_jit_profiling`. There is intentionally no `Default` impl — handing
/// out an unconstructed bridge would skip the `disable_jit_profiling` invariant
/// (L1 design.md §5.1.4 / §1.2 invariant 4).
///
/// The bridge is GPU-agnostic: the DB `gpu_name` key travels per call from the
/// kernel's `*KernelConfig` (`KernelConfig::gpu_name`), not from bridge state,
/// so one bridge serves kernels modeling different GPUs.
#[derive(Clone, Debug)]
pub struct PerfApiBridge {
    /// `Some(..)` puts the bridge in dry-run mode: `Kernel::build` counts missing
    /// specs into this report instead of fitting caches (see `enable_dry_run`).
    /// Interior-mutable because `build` borrows the bridge by `&` only.
    dry_run: RefCell<Option<Vec<KernelMissing>>>,
    /// Active per-role backend override map (dotted role `name` → interned
    /// candidate backends). `Kernel::build` consults it by role `name` on the RUN
    /// path to replace a kernel's const-default candidate set. `None` (the default,
    /// and always the case under `emit-backends`) = every kernel keeps its
    /// arch-declared backends. Set + restored, together with `active_pool`, by the
    /// deployment's single [`with_backend_overrides`] call per pool.
    ///
    /// [`with_backend_overrides`]: PerfApiBridge::with_backend_overrides
    backend_overrides: RefCell<Option<HashMap<String, Vec<&'static str>>>>,
    /// The pool whose model is currently building. Read ONLY by [`record_enum`] on
    /// the EMIT path to pool-prefix enumerate role names (worker-local names alone
    /// don't distinguish AFD's two pools — both are `afd.…`); inert on the run
    /// path. Set + restored alongside `backend_overrides` by the same
    /// [`with_backend_overrides`] call — the deployment names the pool once, and
    /// that name serves both override-scoping (run) and record-tagging (emit).
    ///
    /// [`with_backend_overrides`]: PerfApiBridge::with_backend_overrides
    /// [`record_enum`]: PerfApiBridge::record_enum
    active_pool: RefCell<Option<String>>,
    /// `Some(..)` puts the bridge in enumerate mode: `Kernel::build` records one
    /// [`KernelEnum`] and returns an empty kernel WITHOUT any `profile.db` lookup
    /// (no GPU, no profiling) — the `emit-backends` structural walk. Distinct from
    /// `dry_run`, which still calls `count_missing`.
    enumerate: RefCell<Option<Vec<KernelEnum>>>,
}

impl PerfApiBridge {
    pub fn new() -> Result<Self, PerfApiError> {
        let bridge = Self {
            dry_run: RefCell::new(None),
            backend_overrides: RefCell::new(None),
            active_pool: RefCell::new(None),
            enumerate: RefCell::new(None),
        };
        bridge.disable_jit_profiling()?;
        Ok(bridge)
    }

    /// Switch the bridge into dry-run mode: subsequent `Kernel::build` calls only
    /// `count_missing` (no cache fit) and accumulate one [`KernelMissing`] per
    /// kernel. Drain the result with [`take_dry_run_report`](Self::take_dry_run_report).
    pub fn enable_dry_run(&self) {
        *self.dry_run.borrow_mut() = Some(Vec::new());
    }

    /// Whether the bridge is in dry-run mode (set by [`enable_dry_run`](Self::enable_dry_run)).
    pub fn is_dry_run(&self) -> bool {
        self.dry_run.borrow().is_some()
    }

    /// Record one kernel's coverage line. No-op when not in dry-run mode.
    pub fn record_missing(&self, name: String, kind: KernelKind, missing: usize, total: usize) {
        if let Some(report) = self.dry_run.borrow_mut().as_mut() {
            report.push(KernelMissing {
                name,
                kind,
                missing,
                total,
            });
        }
    }

    /// Take the accumulated dry-run report, leaving the bridge in dry-run mode
    /// with an empty report. Empty `Vec` if dry-run was never enabled.
    pub fn take_dry_run_report(&self) -> Vec<KernelMissing> {
        match self.dry_run.borrow_mut().as_mut() {
            Some(report) => std::mem::take(report),
            None => Vec::new(),
        }
    }

    // ── per-pool build scope: overrides (run) + pool tag (emit) ─────────────

    /// Scope one pool's model build. The deployment calls this ONCE per pool,
    /// naming the pool and passing its override submap; nothing else on the run
    /// path touches backend/pool state, and the emit path adds no call of its own
    /// (it is driven solely by [`enable_enumerate`](Self::enable_enumerate)). The
    /// one pool name serves both concerns:
    ///
    /// - RUN — activate the pool's `role → backends` override submap so
    ///   `Kernel::build` replaces a kernel's const-default candidate set by role
    ///   `name`. `submap` is the pool's slice of the run config's
    ///   `pool → role → backends` (`None` = keep arch defaults). Names are interned
    ///   to `&'static str` here (once per distinct name per process).
    /// - EMIT — record the pool so [`record_enum`](Self::record_enum) can prefix
    ///   enumerate role names (inert unless in enumerate mode).
    ///
    /// The returned guard save/restores BOTH the previous override map and pool tag
    /// on drop (even on an early `?` return), so pools built in sequence never leak
    /// into one another.
    pub fn with_backend_overrides(
        &self,
        pool: &str,
        submap: Option<&HashMap<String, Vec<String>>>,
    ) -> BackendOverrideGuard<'_> {
        let interned = submap.map(|submap| {
            submap
                .iter()
                .map(|(role, backends)| {
                    (
                        role.clone(),
                        backends.iter().map(|b| intern_backend(b)).collect(),
                    )
                })
                .collect()
        });
        let prev_overrides = std::mem::replace(&mut *self.backend_overrides.borrow_mut(), interned);
        let prev_pool = self.active_pool.borrow_mut().replace(pool.to_string());
        BackendOverrideGuard {
            bridge: self,
            prev_pool,
            prev_overrides,
        }
    }

    /// The override candidate set for a kernel's dotted role `name`, if the active
    /// override submap names it. `Kernel::build` calls this before fitting caches;
    /// `None` means "keep the arch-declared const-default backends".
    pub fn backend_override_for(&self, name: &str) -> Option<Vec<&'static str>> {
        self.backend_overrides
            .borrow()
            .as_ref()
            .and_then(|map| map.get(name).cloned())
    }

    // ── enumerate mode (emit-backends structural walk) ──────────────────────

    /// Switch the bridge into enumerate mode: subsequent `Kernel::build` calls
    /// record one [`KernelEnum`] each (no cache fit, no `profile.db` lookup) and
    /// return an empty kernel. Drain with [`take_enum_report`](Self::take_enum_report).
    pub fn enable_enumerate(&self) {
        *self.enumerate.borrow_mut() = Some(Vec::new());
    }

    /// Whether the bridge is in enumerate mode.
    pub fn is_enumerate(&self) -> bool {
        self.enumerate.borrow().is_some()
    }

    /// Record one kernel's structural facts, tagged with the active pool. No-op
    /// when not in enumerate mode.
    #[allow(clippy::too_many_arguments)]
    pub fn record_enum(
        &self,
        name: String,
        kind: KernelKind,
        gpu: &str,
        compute_dtype: Option<DType>,
        kv_dtype: Option<DType>,
        config: Value,
        backends: &[&'static str],
    ) {
        if let Some(report) = self.enumerate.borrow_mut().as_mut() {
            report.push(KernelEnum {
                pool: self.active_pool.borrow().clone().unwrap_or_default(),
                name,
                kind,
                gpu: gpu.to_string(),
                compute_dtype,
                kv_dtype,
                config,
                backends: backends
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect(),
            });
        }
    }

    /// Take the accumulated enumerate report, leaving the bridge in enumerate mode
    /// with an empty report. Empty `Vec` if enumerate was never enabled.
    pub fn take_enum_report(&self) -> Vec<KernelEnum> {
        match self.enumerate.borrow_mut().as_mut() {
            Some(report) => std::mem::take(report),
            None => Vec::new(),
        }
    }

    /// Lock the `perf_api` into "sim-runtime-safe" mode: any spec that's not
    /// already cached in the profile DB will raise `MissingEntry` rather than
    /// kicking off a JIT profile. Called from `new()`; the Python side is
    /// idempotent so repeated calls are safe.
    pub fn disable_jit_profiling(&self) -> Result<(), PerfApiError> {
        self.call_perf_api_void("disable_jit_profiling")
    }

    /// Re-enable JIT profiling. Used by the `--build-cache-only` entry point
    /// (L1 design.md §7.2): Rust main flips this back on *before* calling any
    /// `*Kernel::build`, so missing specs are profiled into the DB on demand
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
        gpu_name: &str,
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
            kwargs.set_item("gpu_name", gpu_name).py_err()?;
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
        gpu_name: &str,
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
            kwargs.set_item("gpu_name", gpu_name).py_err()?;
            let fn_name = format!("count_missing_{kind}");
            perf_api
                .getattr(fn_name.as_str())
                .and_then(|func| func.call((py_specs,), Some(kwargs)))
                .and_then(pyo3::PyAny::extract::<usize>)
                .py_err()
        })
    }

    /// Resolve the current CUDA device's DB `gpu_name` key via the Python
    /// facade. Source of truth for the `gpu_name` threaded through worklet / op
    /// / kernel inputs (L3 INV-13); callers targeting a non-current or remote
    /// GPU should supply the name explicitly rather than calling this.
    pub fn get_current_gpu_name(&self) -> Result<String, PerfApiError> {
        Python::with_gil(|py| {
            let perf_api = PyModule::import(py, "profiling.perf_api").py_err()?;
            perf_api
                .getattr("get_current_gpu_name")
                .and_then(|func| func.call0())
                .and_then(pyo3::PyAny::extract::<String>)
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
                    .and_then(pyo3::PyAny::extract::<u32>)
                    .py_err()?,
                schema_hash: value
                    .getattr("schema_hash")
                    .and_then(pyo3::PyAny::extract::<String>)
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
                        .and_then(pyo3::PyAny::extract::<String>)
                        .py_err()?,
                    profiler_git_hash: item
                        .getattr("profiler_git_hash")
                        .and_then(pyo3::PyAny::extract::<String>)
                        .py_err()?,
                });
            }
            Ok(versions)
        })
    }
}

/// RAII guard from [`PerfApiBridge::with_backend_overrides`]: restores BOTH the
/// previous override map and the previous pool tag on drop (even on an early `?`
/// return), so a pool's build scope never leaks into the next. Holds the bridge by
/// shared ref — its state is `RefCell`, so no `&mut` is needed.
#[must_use = "the pool scope is active only while this guard is alive"]
pub struct BackendOverrideGuard<'a> {
    bridge: &'a PerfApiBridge,
    prev_pool: Option<String>,
    prev_overrides: Option<HashMap<String, Vec<&'static str>>>,
}

impl Drop for BackendOverrideGuard<'_> {
    fn drop(&mut self) {
        *self.bridge.backend_overrides.borrow_mut() = self.prev_overrides.take();
        *self.bridge.active_pool.borrow_mut() = self.prev_pool.take();
    }
}

#[cfg(test)]
impl PerfApiBridge {
    /// Test-only constructor that skips the Python `disable_jit_profiling` init
    /// (which imports `profiling.perf_api`). Lets the backend-override unit tests
    /// exercise pure bridge state with no live perf_api / GIL.
    pub(crate) fn new_uninit_for_test() -> Self {
        Self {
            dry_run: RefCell::new(None),
            backend_overrides: RefCell::new(None),
            active_pool: RefCell::new(None),
            enumerate: RefCell::new(None),
        }
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
            .and_then(pyo3::PyAny::extract::<String>)
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
        .and_then(pyo3::PyAny::extract::<f64>)
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

#[cfg(test)]
mod tests {
    use super::{ensure_payload_backends_match, shared_backend, PerfApiBridge};
    use crate::timing::bridge::payload::intern_backend;
    use crate::timing::bridge::{ArgsPayload, PerfApiError};
    use serde_json::Value;
    use std::collections::HashMap;

    fn submap(pairs: &[(&str, &[&str])]) -> HashMap<String, Vec<String>> {
        pairs
            .iter()
            .map(|(role, backends)| {
                (
                    role.to_string(),
                    backends.iter().map(|b| b.to_string()).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn override_scope_sets_by_role_and_clears_on_drop() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        // No overrides active by default.
        assert_eq!(
            bridge.backend_override_for("afd.moe_expert_compute.gate_up"),
            None
        );

        let map = submap(&[
            ("afd.moe_expert_compute.gate_up", &["fa3"]),
            ("afd.attn_block.qkv", &["fa2", "fa3"]),
        ]);
        {
            let _scope = bridge.with_backend_overrides("ffn", Some(&map));
            // Named roles resolve to their (interned) candidate lists...
            assert_eq!(
                bridge.backend_override_for("afd.moe_expert_compute.gate_up"),
                Some(vec!["fa3"])
            );
            assert_eq!(
                bridge.backend_override_for("afd.attn_block.qkv"),
                Some(vec!["fa2", "fa3"])
            );
            // ...an unnamed role keeps its const default (no override).
            assert_eq!(bridge.backend_override_for("afd.some.other"), None);
        }
        // Guard dropped: overrides restored to none, so a later pool can't inherit them.
        assert_eq!(
            bridge.backend_override_for("afd.moe_expert_compute.gate_up"),
            None
        );
    }

    #[test]
    fn override_scope_none_is_noop() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        let _scope = bridge.with_backend_overrides("main", None);
        assert_eq!(bridge.backend_override_for("anything"), None);
    }

    #[test]
    fn sequential_scopes_do_not_leak_across_pools() {
        // Mirrors afd/pd: pool A's submap is scoped, dropped, then pool B's — B
        // must not see A's overrides (and vice versa).
        let bridge = PerfApiBridge::new_uninit_for_test();
        let attn = submap(&[("afd.attn_block.qkv", &["fa3"])]);
        let ffn = submap(&[("afd.moe_expert_compute.gate_up", &["deepgemm"])]);
        {
            let _a = bridge.with_backend_overrides("attn", Some(&attn));
            assert_eq!(
                bridge.backend_override_for("afd.attn_block.qkv"),
                Some(vec!["fa3"])
            );
        }
        {
            let _f = bridge.with_backend_overrides("ffn", Some(&ffn));
            // ffn scope: attn's role is gone, ffn's role is present.
            assert_eq!(bridge.backend_override_for("afd.attn_block.qkv"), None);
            assert_eq!(
                bridge.backend_override_for("afd.moe_expert_compute.gate_up"),
                Some(vec!["deepgemm"])
            );
        }
    }

    #[test]
    fn nested_override_scopes_restore_the_outer_map() {
        // save/restore (not clear-to-none): an inner scope shadows the outer one,
        // and dropping it restores the OUTER overrides rather than wiping them.
        let bridge = PerfApiBridge::new_uninit_for_test();
        let outer = submap(&[("r", &["fa2"])]);
        let inner = submap(&[("r", &["fa3"])]);
        let _o = bridge.with_backend_overrides("p", Some(&outer));
        assert_eq!(bridge.backend_override_for("r"), Some(vec!["fa2"]));
        {
            let _i = bridge.with_backend_overrides("p", Some(&inner));
            assert_eq!(bridge.backend_override_for("r"), Some(vec!["fa3"]));
        }
        assert_eq!(bridge.backend_override_for("r"), Some(vec!["fa2"]));
    }

    #[test]
    fn enumerate_records_are_pool_tagged_and_drained() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        assert!(!bridge.is_enumerate());
        bridge.enable_enumerate();
        assert!(bridge.is_enumerate());
        {
            // The deployment's single per-pool call sets the pool tag (its override
            // submap is None here — the emit path strips backends).
            let _s = bridge.with_backend_overrides("ffn", None);
            bridge.record_enum(
                "afd.moe.gate_up".into(),
                "grouped_gemm",
                "NVIDIA H200",
                Some(super::DType::Fp8E4m3),
                None,
                "dtype=Fp8E4m3".into(),
                &["deepgemm"],
            );
        }
        // outside any pool scope → empty pool tag (deployment-level kernel).
        bridge.record_enum(
            "afd_qkv_transfer".into(),
            "p2p_inter",
            "NVIDIA H200",
            None,
            None,
            "fabric=Infiniband".into(),
            &["nccl", "nvshmem"],
        );
        let report = bridge.take_enum_report();
        assert_eq!(report.len(), 2);
        assert_eq!(report[0].pool, "ffn");
        assert_eq!(report[0].name, "afd.moe.gate_up");
        assert_eq!(report[0].backends, vec!["deepgemm".to_string()]);
        assert_eq!(report[0].gpu, "NVIDIA H200");
        assert_eq!(report[0].compute_dtype, Some(super::DType::Fp8E4m3));
        assert_eq!(report[1].compute_dtype, None); // comm is dtype-agnostic
        assert_eq!(report[1].pool, ""); // deployment-level, no active pool
                                        // draining leaves an empty report while still in enumerate mode.
        assert!(bridge.take_enum_report().is_empty());
        assert!(bridge.is_enumerate());
    }

    #[test]
    fn intern_backend_dedups_to_one_pointer() {
        // Same name → same &'static (leaked once); distinct names differ.
        let a = intern_backend("fa3");
        let b = intern_backend(&String::from("fa3"));
        assert_eq!(a, b);
        assert!(
            std::ptr::eq(a, b),
            "same backend name must intern to one pointer"
        );
        assert_ne!(intern_backend("fa2"), intern_backend("fa3"));
    }

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
