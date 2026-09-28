//! Thin Rust/Python bridge for L1 perf_api.

pub mod core;
pub mod error;
pub mod payload;
mod python;
pub mod replay;

pub use core::{
    config_records_document, write_config_records, BackendOverrideGuard, ConfigGrid, ConfigUse,
    KernelConfigRecord, KernelEnum, KernelMissing, PerfApiBridge, CONFIG_RECORDS_SCHEMA_VERSION,
};
pub use error::{BuildError, PerfApiError};
pub(crate) use payload::{de_backends, intern_backend};
pub use payload::{ArgsPayload, DType, DbMetadata, KernelKind, KernelMetrics, ProfilerVersion};
pub use replay::{write_samples, ReplaySamples, SampleMetrics, SampleRow, SAMPLE_FORMAT};
