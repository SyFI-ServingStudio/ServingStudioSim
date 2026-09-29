use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};

use std::sync::Arc;

use serde_json::Value;

use super::kernel_data::KernelData;
use super::python;

use crate::timing::bridge::{
    intern_backend, ArgsPayload, BuildError, DType, DbMetadata, KernelKind, KernelMetrics,
    PerfApiError, ProfilerVersion,
};

/// One kernel's profile-coverage line for the `dry-run` report: how many of its
/// enumerated specs are missing from `profile.db` (i.e. would be JIT-profiled on a
/// real build). `name` is the kernel's dotted path; counts are summed over its
/// backends. Shared profile identities count only at their first role in a
/// dry-run report, matching the number of rows JIT would need to measure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelMissing {
    pub name: String,
    pub kind: KernelKind,
    pub missing: usize,
    pub total: usize,
}

#[derive(Clone, Debug, Default)]
struct DryRunReport {
    rows: Vec<KernelMissing>,
    // Backend is inside the canonical ArgsPayload JSON; role is not a DB key.
    seen: HashSet<(KernelKind, String, String)>,
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

/// One kernel config a build asked profile.db for, as profile.db's kernel-config
/// registry stores it: the config's identity, the GPU, the cache grid and the
/// DB args behind each grid cell. Every role that builds the same
/// `(kind, gpu_name, identity)` shares one record and is listed in `uses`.
///
/// The grid is recorded rather than recomputed later from `identity` because
/// recomputing is not always possible: a corpus-routed MoE config names a
/// payload file that the reader may not have.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct KernelConfigRecord {
    /// The kernel's `KernelSpec::KIND`: what `kernel-query` deserializes the
    /// identity as.
    pub kind: KernelKind,
    /// The profile.db table the grid's args are rows of (`profile_kind`).
    pub profile_kind: KernelKind,
    pub gpu_name: String,
    /// [`KernelConfig::identity`](crate::timing::KernelConfig::identity).
    pub identity: Value,
    pub grid: ConfigGrid,
    pub uses: Vec<ConfigUse>,
}

/// A config's sweep grid and the profile.db args of each cell.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ConfigGrid {
    /// One name per axis: the kernel input's `SweepCoords::coord_field_names`.
    pub cache_coords: &'static [&'static str],
    pub axes: Vec<Vec<f64>>,
    /// Row-major over `axes`, one entry per cell: the profile.db args columns of
    /// that cell's row. `backend` is left out because it is the only column that
    /// differs between the config's backends.
    pub cells: Vec<BTreeMap<String, Value>>,
    /// Cells the kernel marks physically infeasible; they are never profiled.
    pub infeasible: Vec<usize>,
}

/// Where a config was built: the pool and the kernel's dotted role name.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ConfigUse {
    pub pool: String,
    pub role: String,
}

/// Version of the document [`config_records_document`] builds.
pub const CONFIG_RECORDS_SCHEMA_VERSION: u32 = 1;

/// `records` as the JSON document the launcher registers into profile.db's
/// kernel-config registry (`profiling/db/kernel_config.py`).
pub fn config_records_document(records: &[KernelConfigRecord]) -> serde_json::Value {
    serde_json::json!({
        "schema_version": CONFIG_RECORDS_SCHEMA_VERSION,
        "configs": records,
    })
}

/// Write [`config_records_document`] of `records` to `path`.
pub fn write_config_records(
    path: &std::path::Path,
    records: &[KernelConfigRecord],
) -> anyhow::Result<()> {
    use anyhow::Context;
    let text =
        serde_json::to_string(&config_records_document(records)).expect("config records serialize");
    std::fs::write(path, text)
        .with_context(|| format!("writing kernel config records {}", path.display()))
}

#[derive(Clone, Debug, Default)]
struct ConfigRecords {
    records: Vec<KernelConfigRecord>,
    /// `(kind, gpu_name, identity JSON)` -> index into `records`.
    index: HashMap<(KernelKind, String, String), usize>,
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
    dry_run: RefCell<Option<DryRunReport>>,
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
    /// `Some(..)` makes `Kernel::build` record each config it builds, in any
    /// mode, for profile.db's kernel-config registry (see
    /// [`enable_config_records`](Self::enable_config_records)).
    config_records: RefCell<Option<ConfigRecords>>,
    /// `Some(..)` makes `Kernel::build` fit every cache from these config
    /// documents instead of asking Python (see [`kernel_data`](Self::kernel_data)):
    /// no profile.db, no JIT, no GPU.
    kernel_data: Option<Arc<KernelData>>,
}

impl PerfApiBridge {
    pub fn new() -> Result<Self, PerfApiError> {
        let bridge = Self::inert();
        bridge.disable_jit_profiling()?;
        Ok(bridge)
    }

    /// A bridge whose kernels read their measured grids from `data`, the
    /// kernel API's config documents, and nothing else: a config without a
    /// complete document fails to build. Needs no Python, so it is the bridge
    /// of a build without the `python` feature (wasm32).
    pub fn kernel_data(data: Arc<KernelData>) -> Self {
        Self {
            kernel_data: Some(data),
            ..Self::inert()
        }
    }

    /// The config documents a [`kernel_data`](Self::kernel_data) bridge reads.
    pub fn documents(&self) -> Option<&KernelData> {
        self.kernel_data.as_deref()
    }

    /// A bridge that can only build structure: enumerate mode, and no Python
    /// perf_api at all, so nothing can reach `profile.db` or a GPU. For
    /// commands that need a model's cost tree and nothing measured.
    pub fn structure_only() -> Self {
        Self {
            enumerate: RefCell::new(Some(Vec::new())),
            ..Self::inert()
        }
    }

    /// Every mode off and no perf_api attached; each constructor starts here.
    fn inert() -> Self {
        Self {
            dry_run: RefCell::new(None),
            backend_overrides: RefCell::new(None),
            active_pool: RefCell::new(None),
            enumerate: RefCell::new(None),
            config_records: RefCell::new(None),
            kernel_data: None,
        }
    }

    /// Switch the bridge into dry-run mode: subsequent `Kernel::build` calls only
    /// `count_missing` (no cache fit) and accumulate one [`KernelMissing`] per
    /// kernel. Drain the result with [`take_dry_run_report`](Self::take_dry_run_report).
    pub fn enable_dry_run(&self) {
        *self.dry_run.borrow_mut() = Some(DryRunReport::default());
    }

    /// Whether the bridge is in dry-run mode (set by [`enable_dry_run`](Self::enable_dry_run)).
    pub fn is_dry_run(&self) -> bool {
        self.dry_run.borrow().is_some()
    }

    /// Record one kernel's coverage line. No-op when not in dry-run mode.
    pub fn record_missing(&self, name: String, kind: KernelKind, missing: usize, total: usize) {
        if let Some(report) = self.dry_run.borrow_mut().as_mut() {
            report.rows.push(KernelMissing {
                name,
                kind,
                missing,
                total,
            });
        }
    }

    /// Assign each complete profile identity to its first role in this report.
    /// Repeated roles share DB rows and therefore do not add JIT work.
    pub fn unique_dry_run_specs(
        &self,
        kind: KernelKind,
        gpu_name: &str,
        specs: Vec<ArgsPayload>,
    ) -> Vec<ArgsPayload> {
        let mut state = self.dry_run.borrow_mut();
        let report = state
            .as_mut()
            .expect("unique_dry_run_specs requires dry-run mode");
        specs
            .into_iter()
            .filter(|spec| {
                let fields = serde_json::to_string(spec.fields())
                    .expect("ArgsPayload fields are JSON values");
                report.seen.insert((kind, gpu_name.to_owned(), fields))
            })
            .collect()
    }

    /// Take the accumulated dry-run report, leaving the bridge in dry-run mode
    /// with an empty report. Empty `Vec` if dry-run was never enabled.
    pub fn take_dry_run_report(&self) -> Vec<KernelMissing> {
        match self.dry_run.borrow_mut().as_mut() {
            Some(report) => std::mem::take(report).rows,
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
                backends: backends.iter().map(|b| b.to_string()).collect(),
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

    // ── kernel-config records (profile.db registry) ──────────────────────────

    /// Record every kernel config subsequent `Kernel::build` calls build, in any
    /// bridge mode. Drain with [`take_config_records`](Self::take_config_records).
    pub fn enable_config_records(&self) {
        *self.config_records.borrow_mut() = Some(ConfigRecords::default());
    }

    /// Whether `Kernel::build` should record its config.
    pub fn records_configs(&self) -> bool {
        self.config_records.borrow().is_some()
    }

    /// Record that role `role` of the active pool built the config with this
    /// `identity` on `gpu_name`. `grid` runs only for the first role that builds
    /// a given `(kind, gpu_name, identity)`; later roles are added to its uses.
    /// No-op when records are not enabled.
    pub fn record_config(
        &self,
        role: &str,
        kind: KernelKind,
        profile_kind: KernelKind,
        gpu_name: &str,
        identity: Value,
        grid: impl FnOnce() -> Result<ConfigGrid, BuildError>,
    ) -> Result<(), BuildError> {
        let mut state = self.config_records.borrow_mut();
        let Some(records) = state.as_mut() else {
            return Ok(());
        };
        let config_use = ConfigUse {
            pool: self.active_pool.borrow().clone().unwrap_or_default(),
            role: role.to_string(),
        };
        let key = (
            kind,
            gpu_name.to_string(),
            serde_json::to_string(&identity).expect("a config identity is JSON"),
        );
        if let Some(&i) = records.index.get(&key) {
            let uses = &mut records.records[i].uses;
            if !uses.contains(&config_use) {
                uses.push(config_use);
            }
            return Ok(());
        }
        records.records.push(KernelConfigRecord {
            kind,
            profile_kind,
            gpu_name: gpu_name.to_string(),
            identity,
            grid: grid()?,
            uses: vec![config_use],
        });
        records.index.insert(key, records.records.len() - 1);
        Ok(())
    }

    /// Take the recorded configs in first-built order, leaving recording on
    /// with nothing recorded. Empty if records were never enabled.
    pub fn take_config_records(&self) -> Vec<KernelConfigRecord> {
        match self.config_records.borrow_mut().as_mut() {
            Some(records) => std::mem::take(records).records,
            None => Vec::new(),
        }
    }

    /// Lock the perf_api into "sim-runtime-safe" mode: any spec that's not
    /// already cached in the profile DB will raise `MissingEntry` rather than
    /// kicking off a JIT profile. Called from `new()`; the Python side is
    /// idempotent so repeated calls are safe. A kernel-data bridge is always strict.
    pub fn disable_jit_profiling(&self) -> Result<(), PerfApiError> {
        if self.kernel_data.is_some() {
            return Ok(());
        }
        python::call_perf_api_void("disable_jit_profiling")
    }

    /// Re-enable JIT profiling. Used by the `--build-cache-only` entry point
    /// (L1 design.md §7.2): Rust main flips this back on *before* calling any
    /// `*Kernel::build`, so missing specs are profiled into the DB on demand
    /// instead of erroring out. A no-op on a kernel-data bridge, which cannot
    /// profile: a missing sample stays an error.
    pub fn enable_jit_profiling(&self) -> Result<(), PerfApiError> {
        if self.kernel_data.is_some() {
            return Ok(());
        }
        python::call_perf_api_void("enable_jit_profiling")
    }

    /// Start the collect pass: `count_missing` records what is absent instead of
    /// only counting it. Pair with [`issue_collected`](Self::issue_collected),
    /// and run the build cascade in dry-run mode in between.
    pub fn begin_collect(&self) -> Result<(), PerfApiError> {
        python::call_perf_api_void("begin_collect")
    }

    /// Measure everything the collect pass recorded, one whole `(kind, backend)`
    /// unit per GPU. Errors if any unit could not run.
    ///
    /// `expected_specs` is this side's own missing count, from the dry-run
    /// report. The counts and the work list travel separately — the report holds
    /// counts for display, the Python collector holds the specs — so passing it
    /// across lets a disagreement fail here rather than resurface as a
    /// kernel-at-a-time JIT during the next real run.
    pub fn issue_collected(&self, expected_specs: usize) -> Result<(), PerfApiError> {
        python::issue_collected(expected_specs)
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
        python::get_times(&payloads, kind, backend, gpu_name)
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
        python::count_missing(&payloads, kind, backend, gpu_name)
    }

    /// Resolve the current CUDA device's DB `gpu_name` key via the Python
    /// facade. Source of truth for the `gpu_name` threaded through worklet / op
    /// / kernel inputs (L3 INV-13); callers targeting a non-current or remote
    /// GPU should supply the name explicitly rather than calling this.
    pub fn get_current_gpu_name(&self) -> Result<String, PerfApiError> {
        python::get_current_gpu_name()
    }

    pub fn get_db_metadata(&self) -> Result<DbMetadata, PerfApiError> {
        python::get_db_metadata()
    }

    pub fn get_profiler_versions(
        &self,
        op_families: Vec<&str>,
    ) -> Result<Vec<ProfilerVersion>, PerfApiError> {
        python::get_profiler_versions(op_families)
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
        Self::inert()
    }
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

#[cfg(test)]
mod tests {
    use super::{ensure_payload_backends_match, shared_backend, ConfigGrid, PerfApiBridge};
    use crate::timing::bridge::payload::intern_backend;
    use crate::timing::bridge::{ArgsPayload, BuildError, PerfApiError};
    use serde_json::Value;
    use std::collections::{BTreeMap, HashMap};

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
    fn dry_run_counts_complete_cache_keys_once_and_resets_with_report() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_dry_run();
        let spec = payload(Some("torch"));
        let count = |kind, gpu, specs| bridge.unique_dry_run_specs(kind, gpu, specs).len();
        assert_eq!(
            count("single_gemm", "B200", vec![spec.clone(), spec.clone()]),
            1
        );
        assert_eq!(count("single_gemm", "B200", vec![spec.clone()]), 0);
        assert_eq!(count("batched_gemm", "B200", vec![spec.clone()]), 1);
        assert_eq!(count("single_gemm", "H200", vec![spec.clone()]), 1);
        assert_eq!(
            count("single_gemm", "B200", vec![payload(Some("triton"))]),
            1
        );
        let mut other_shape = spec.clone();
        other_shape.insert("m", Value::from(4096));
        assert_eq!(count("single_gemm", "B200", vec![other_shape]), 1);

        bridge.record_missing("target.gemm".to_string(), "single_gemm", 1, 1);
        assert_eq!(bridge.take_dry_run_report().len(), 1);
        assert!(bridge.is_dry_run());
        assert_eq!(count("single_gemm", "B200", vec![spec]), 1);
        assert!(bridge.take_dry_run_report().is_empty());
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

    fn grid() -> ConfigGrid {
        ConfigGrid {
            cache_coords: &["m"],
            axes: vec![vec![1.0]],
            cells: vec![BTreeMap::from([("m".to_string(), Value::from(1))])],
            infeasible: Vec::new(),
        }
    }

    #[test]
    fn record_config_shares_one_record_per_identity_and_gpu() {
        let bridge = PerfApiBridge::structure_only();
        let identity = serde_json::json!({"n": 1});
        // Off until enabled: nothing is recorded and the grid is never built.
        bridge
            .record_config(
                "a",
                "single_gemm",
                "single_gemm",
                "B200",
                identity.clone(),
                || panic!("grid built while records are off"),
            )
            .unwrap();
        bridge.enable_config_records();
        assert!(bridge.records_configs());

        let _scope = bridge.with_backend_overrides("main", None);
        let mut built = 0;
        for (role, gpu) in [("a", "B200"), ("b", "B200"), ("a", "B200"), ("a", "H200")] {
            bridge
                .record_config(
                    role,
                    "single_gemm",
                    "single_gemm",
                    gpu,
                    identity.clone(),
                    || {
                        built += 1;
                        Ok(grid())
                    },
                )
                .unwrap();
        }

        assert_eq!(built, 2);
        let records = bridge.take_config_records();
        assert_eq!(records.len(), 2);
        let roles: Vec<_> = records[0]
            .uses
            .iter()
            .map(|u| (u.pool.as_str(), u.role.as_str()))
            .collect();
        assert_eq!(roles, vec![("main", "a"), ("main", "b")]);
        assert_eq!(records[1].gpu_name, "H200");
        assert!(bridge.take_config_records().is_empty());
    }

    #[test]
    fn record_config_propagates_a_grid_error() {
        let bridge = PerfApiBridge::structure_only();
        bridge.enable_config_records();
        let err = bridge
            .record_config("a", "k", "k", "B200", serde_json::json!({}), || {
                Err(BuildError::BackendDependentArgs {
                    kind: "k",
                    first: "x",
                    other: "y",
                })
            })
            .unwrap_err();
        assert!(matches!(err, BuildError::BackendDependentArgs { .. }));
        assert!(bridge.take_config_records().is_empty());
    }

    #[test]
    fn shared_backend_rejects_empty_batch_instead_of_panic() {
        let err = shared_backend(&[]).unwrap_err();
        assert!(matches!(err, PerfApiError::InvalidRequest(_)));
    }
}
