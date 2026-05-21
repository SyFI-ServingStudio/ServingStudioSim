//! PyO3 bridge smoke test — the only end-to-end exercise of the Rust↔Python
//! `perf_api` boundary (unit tests cover pure-Rust helpers only).
//!
//! `#[ignore]` by default: it needs the project venv interpreter, not the
//! system Python (3.9, which can't import `profiling` — `StrEnum` is 3.11+).
//! Run it explicitly:
//!
//!     LD_LIBRARY_PATH="$(uv run python -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))'):$LD_LIBRARY_PATH" \
//!         uv run cargo test --test bridge_smoke -- --ignored
//!
//! `uv run` makes pyo3 link the `.venv` 3.12 interpreter; `LD_LIBRARY_PATH`
//! points the runtime loader at that interpreter's `libpython3.12.so`. The test
//! also inserts the repo root into `sys.path` so `import profiling...` resolves
//! even when the embedded interpreter's path doesn't include it.
//!
//! Coverage: facade-name resolution (`get_{kind}_times` / `count_missing_{kind}`),
//! payload marshalling, `DbMetadata` + `usize` unmarshalling, a real
//! `ComputeMetrics` hit, and the `MissingEntry` recognition path.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use simulator::timing::bridge::{ArgsPayload, DType, PerfApiError};
use simulator::timing::PerfApiBridge;

const KIND: &str = "single_gemm";
const GPU: &str = "TestGPU";

/// A single_gemm wire payload matching the Python `SingleGemmArgs` schema.
fn payload(m: u32) -> ArgsPayload {
    ArgsPayload::new()
        .with("backend", "torch")
        .with("m", m)
        .with("n", 32u32)
        .with("k", 64u32)
        .with("dtype", DType::Fp16.as_str())
}

/// Point `perf_api.DB_PATH` at `db_path` and seed one single_gemm/torch row, so
/// the `m == 16` lookups below hit. The Table/ProfileRow schema lives in Python,
/// so seeding is done there.
fn seed_db(db_path: &str) {
    Python::with_gil(|py| {
        let ns = PyDict::new(py);
        ns.set_item("db_path", db_path).unwrap();
        ns.set_item("gpu", GPU).unwrap();
        ns.set_item("repo_root", env!("CARGO_MANIFEST_DIR"))
            .unwrap();
        py.run(
            r#"
import sys
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pathlib import Path
import profiling.perf_api as perf_api
perf_api.DB_PATH = Path(db_path)

from profiling.db.registry import find_kernel_profiler_spec
from profiling.db.table import Table, ProfileRow
from profiling.db.args import DType
from profiling.kernels.single_gemm import SingleGemmArgs
from profiling.runners.metrics import ComputeMetrics

spec = find_kernel_profiler_spec("single_gemm", "torch")
Table(spec, db_path).insert([ProfileRow(
    args=SingleGemmArgs(m=16, n=32, k=64, dtype=DType.FP16),
    metrics=ComputeMetrics(time_ms=2.5, tflops=1.0, memory_bandwidth_gbps=2.0, energy_j=0.25),
    gpu_name=gpu,
    backend="torch",
)])
"#,
            Some(ns),
            None,
        )
        .inspect_err(|err| err.print(py))
        .expect("seeding the temp profile DB via Python failed");
    });
}

#[test]
#[ignore = "needs the venv 3.12 interpreter; see module docs for the run command"]
fn bridge_round_trips_metadata_count_hit_and_missing() {
    let db_path =
        std::env::temp_dir().join(format!("mlsim_bridge_smoke_{}.db", std::process::id()));
    let db_path_str = db_path.to_str().unwrap().to_string();
    let _ = std::fs::remove_file(&db_path);
    seed_db(&db_path_str);

    let bridge = PerfApiBridge::new().expect("bridge construction (disable_jit)");

    // 1. DbMetadata unmarshalling: migrate() ran during seeding, so a real
    //    schema_version comes back across the boundary.
    let metadata = bridge.get_db_metadata().expect("get_db_metadata");
    assert!(
        metadata.schema_version >= 1,
        "expected a migrated schema_version, got {}",
        metadata.schema_version
    );
    assert!(!metadata.schema_hash.is_empty());

    // 2. count_missing: facade name `count_missing_single_gemm` + usize extract.
    //    m=16 is seeded (hit), m=999 is absent (miss) → exactly one missing.
    let missing = bridge
        .count_missing(vec![payload(16), payload(999)], KIND, "torch", GPU)
        .expect("count_missing");
    assert_eq!(missing, 1, "only m=999 should be missing");

    // 3. get_times hit: facade name `get_single_gemm_times` + full ComputeMetrics
    //    unmarshalling (including the optional_f64 path for tflops/mem_bw, and
    //    the AttributeError-swallow for the absent comm-only fields).
    let hit = bridge
        .get_times(vec![payload(16)], KIND, GPU)
        .expect("get_times hit");
    assert_eq!(hit.len(), 1);
    assert_eq!(hit[0].time_ms, 2.5);
    assert_eq!(hit[0].energy_j, 0.25);
    assert_eq!(hit[0].tflops, Some(1.0));
    assert_eq!(hit[0].memory_bandwidth_gbps, Some(2.0));
    // Comm-only fields are absent on a ComputeMetrics row → None, not an error.
    assert_eq!(hit[0].algbw_gbps, None);

    // 4. get_times miss: an absent spec must surface as a typed MissingEntry
    //    (JIT is disabled by construction), not a zeroed metric or a panic.
    let miss = bridge.get_times(vec![payload(999)], KIND, GPU);
    match miss {
        Err(PerfApiError::MissingEntry { kind, backend, .. }) => {
            assert_eq!(kind, KIND);
            assert_eq!(backend, "torch");
        }
        other => panic!("expected MissingEntry for m=999, got {other:?}"),
    }

    let _ = std::fs::remove_file(&db_path);
}
