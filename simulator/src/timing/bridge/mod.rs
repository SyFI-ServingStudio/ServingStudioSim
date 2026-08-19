//! Thin Rust/Python bridge for L1 `perf_api`.

pub mod core;
pub mod error;
pub mod payload;

pub use core::{BackendOverrideGuard, KernelEnum, KernelMissing, PerfApiBridge};
pub use error::{BuildError, PerfApiError};
pub(crate) use payload::{de_backends, intern_backend};
pub use payload::{ArgsPayload, DType, DbMetadata, KernelKind, KernelMetrics, ProfilerVersion};
