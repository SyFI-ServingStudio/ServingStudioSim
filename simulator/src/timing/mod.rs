//! L1 Rust timing layer: Python bridge, cache fitting, and per-kind kernels.

pub mod bridge;
pub mod cache;
pub mod cost_tree;
pub mod dims;
pub mod kernels;
pub mod result;
pub mod routing;
pub mod slot_input;
pub mod sweep;

pub use bridge::{
    BackendOverrideGuard, BuildError, DType, KernelEnum, KernelMissing, PerfApiBridge,
};
pub use cache::interp::{CoverageFlags, LeafMetrics, Metrics4};
pub use cache::PeakRates;
pub use dims::Dim;
pub use cost_tree::{
    CostManifest, CostManifestDoc, CostManifestSection, CostNode, CostTree, CostTreeBuilder,
    Evaluator, FlatCostNode, LeafDesc,
};
pub use slot_input::{AttnPrefillLog, SlotInput};
pub use kernels::engine::KernelConfig;
pub use result::{CacheProbe, Probe};
pub use sweep::{Axis, Coords, SweepCoords, SweepGrid};
// Re-export derive macros under the same names as their traits so users only
// import `crate::timing::{SweepCoords, KernelConfig}` once for both `impl`
// and `#[derive(...)]`.
pub use timing_kernel_derive::{KernelConfig, SweepCoords};
