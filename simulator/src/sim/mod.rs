//! `sim` (L7-β/γ) — the deployment-independent sim core: the trace frontend
//! (L7-γ arrival queue) and the single tick driver (L7-β `run_sim`). See
//! doc/detailed_design/L7.md.

pub mod frontend;
pub mod repro;
pub mod run;

pub use frontend::{TraceEntry, TraceFrontend};
pub use run::{run_sim, RunSummary, TerminationCause, TickCfg};
