// Make `::simulator::...` resolve to this crate so in-tree proc-macros (e.g.
// `#[derive(SweepCoords)]` in `timing-kernel-derive`) can emit absolute paths
// that work both inside this crate and from downstream callers.
extern crate self as simulator;

pub mod arch;
pub mod common;
pub mod deployment;
pub mod log;
pub mod op;
pub mod orchestrator;
pub mod schema;
pub mod sim;
pub mod timing;
pub mod worker;
pub mod worklet;
