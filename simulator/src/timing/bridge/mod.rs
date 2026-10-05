//! Thin Rust/Python bridge for L1 perf_api.

pub mod core;
pub mod error;
pub mod payload;

pub use core::{
    config_records_document, dry_run_document, print_dry_run, write_config_records,
    write_dry_run_report, BackendOverrideGuard, ConfigGrid, ConfigUse, KernelConfigRecord,
    KernelEnum, KernelMissing, PerfApiBridge, CONFIG_RECORDS_SCHEMA_VERSION,
    DRY_RUN_REPORT_SCHEMA_VERSION,
};
pub use error::{BuildError, PerfApiError};
pub(crate) use payload::{de_backends, intern_backend};
pub use payload::{ArgsPayload, DType, DbMetadata, KernelKind, KernelMetrics, ProfilerVersion};
