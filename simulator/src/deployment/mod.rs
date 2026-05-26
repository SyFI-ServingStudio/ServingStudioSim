//! Deployment registry — each deployment deserializes a structured run config
//! and assembles the L6b `Flow` it represents.
//!
//! Per new-interface-design §2/§10: the `deployment` tag (top-level serde tag on
//! `config::RunConfig`) fixes the orchestrator (L6, 1:1), the topology, and each
//! pool's contract class. A `Deployment` impl turns its concrete config into a
//! `Box<dyn Flow>`; the per-arch/worker providers inside the config carry their
//! own params (no global param union). The launcher-facing param schema lives in
//! `schema::dump` (a structural registry), not on this trait.

pub mod config;
pub mod unified;

pub use config::{AfdConfig, IoSpec, LogLevel, PdConfig, RunConfig, UnifiedConfig, WorkloadSpec};

use crate::common::SharedRequests;
use crate::orchestrator::Flow;
use crate::timing::PerfApiBridge;

/// Dispatch a deserialized `RunConfig` to its deployment's `build`. The single
/// place the `deployment` tag routes to a concrete topology. `pd` / `afd` parse
/// (so `list-params` advertises them) but error cleanly until wired.
pub fn build_flow(
    cfg: &RunConfig,
    bridge: &PerfApiBridge,
    store: SharedRequests,
) -> anyhow::Result<Box<dyn Flow>> {
    match cfg {
        RunConfig::Unified(c) => unified::UnifiedDeployment::build(c, bridge, store),
        RunConfig::Pd(_) => anyhow::bail!("pd deployment not wired yet"),
        RunConfig::Afd(_) => anyhow::bail!("afd deployment not wired yet"),
    }
}

/// One simulation topology. `Config` is the deserialized per-deployment config
/// (a variant of `config::RunConfig`); `build` runs the L4 cascade (via the
/// `bridge`) and assembles the L6b `Flow` the tick driver runs. `store` is the
/// shared `RequestStore` injected into every worker.
pub trait Deployment {
    /// serde `deployment` tag value + `list-params` grammar key (e.g. `"unified"`).
    const NAME: &'static str;

    /// The deserialized config this deployment consumes.
    type Config;

    fn build(
        cfg: &Self::Config,
        bridge: &PerfApiBridge,
        store: SharedRequests,
    ) -> anyhow::Result<Box<dyn Flow>>;
}
