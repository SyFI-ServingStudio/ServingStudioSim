// Make `::simulator::...` resolve to this crate so in-tree proc-macros (e.g.
// `#[derive(SweepCoords)]` in `timing-kernel-derive`) can emit absolute paths
// that work both inside this crate and from downstream callers.
extern crate self as simulator;

pub mod arch;
pub mod common;
pub mod deployment;
pub mod introspect;
pub mod log;
pub mod op;
pub mod orchestrator;
pub mod schema;
pub mod sim;
pub mod timing;
pub mod timing_predict;
pub mod worker;
pub mod worklet;

/// Shared helpers for unit-test modules across the crate (FakeModel,
/// `test_cluster`, request-store builders). Test-only; not exported.
#[cfg(test)]
mod test_helpers;
