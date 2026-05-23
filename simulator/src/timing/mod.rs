//! L1 Rust timing layer: Python bridge, cache fitting, and per-kind kernels.

pub mod bridge;
pub mod cache;
pub mod cost_tree;
pub mod jit;
pub mod kernels;
pub mod result;
pub mod routing;
pub mod sweep;

pub use bridge::{BuildError, PerfApiBridge};
pub use cache::interp::{CoverageFlags, LeafMetrics, Metrics4};
pub use cost_tree::{CostNode, CostTree, CostTreeBuilder, FlatCostNode, LeafDesc};
pub use jit::{BackendJitPlan, DryRun, JitPlan};
pub use kernels::engine::KernelConfig;
pub use result::{CoverageKind, CoverageWarning, Describe, LookupResult, Probe};
pub use sweep::{Axis, Coords, SweepCoords, SweepGrid};
// Re-export derive macros under the same names as their traits so users only
// import `crate::timing::{SweepCoords, KernelConfig}` once for both `impl`
// and `#[derive(...)]`.
pub use timing_kernel_derive::{KernelConfig, SweepCoords};
