//! Thin Rust/Python bridge for L1 perf_api.

pub mod core;
pub mod error;
pub mod payload;

pub use core::{KernelMissing, PerfApiBridge};
pub use error::{BuildError, PerfApiError};
pub use payload::{ArgsPayload, DType, DbMetadata, KernelKind, KernelMetrics, ProfilerVersion};
