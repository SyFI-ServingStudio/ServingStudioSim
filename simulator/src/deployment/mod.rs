//! Deployment registry — each deployment is a clap-parsed param surface plus
//! the schema slices the launcher reads via `simulator list-params`.
//!
//! Per L7 design.md §1.8: a deployment is the **active selector** of which pool
//! fragments (`schema::common_pool`) it accepts, plus its own fields. It pairs:
//!   - a clap `Args` struct (`type Args`) flattening the chosen fragments, and
//!   - `PARAM_GROUPS`: the matching `ParamDef` slices, in CLI order.
//!
//! ## Why `PARAM_GROUPS: &[&[ParamDef]]` and not a single concatenated `PARAMS`
//!
//! design.md §1.8.3 sketches `const PARAMS = const_concat::concat_slices!(...)`.
//! Stable Rust cannot concatenate slices in `const` context, and we avoid
//! pulling in the unmaintained `const_concat` crate. A slice-of-slices is the
//! stable, zero-dependency equivalent: `schema::dump` and the per-deployment
//! sync test flatten it with [`flatten_params`]. The launcher-facing JSON is
//! identical — this is purely how the Rust side stores the composition.
//!
//! ## No `build()` yet
//!
//! design.md §2.1 gives `Deployment::build(...) -> Box<dyn Flow>`, but the L6
//! `Flow` trait / L7-β tick driver are not implemented. Until they land this
//! trait carries only the schema surface; `main.rs` parses a deployment's args
//! (proving CLI routing) and exits with a "pending L7-β" message for `run`.

pub mod unified;

use crate::common::SharedRequests;
use crate::orchestrator::Flow;
use crate::schema::{ParamDef, ParamSchema};
use crate::timing::PerfApiBridge;

/// One simulation topology's CLI + schema surface plus the L6b `Flow` it builds.
/// See module docs for the `PARAM_GROUPS` rationale.
pub trait Deployment {
    /// CLI subcommand name + `list-params` JSON key (e.g. `"unified"`).
    const NAME: &'static str;

    /// Pool-fragment slices + this deployment's own slice, in CLI order.
    /// Flatten with [`flatten_params`] for dumping / validation. Defaults to
    /// the schema `#[derive(DeploymentParams)]` composed on `Args`, so a
    /// deployment normally only declares `NAME` + `Args`.
    const PARAM_GROUPS: &'static [&'static [ParamDef]] =
        <Self::Args as ParamSchema>::PARAM_GROUPS;

    /// clap-derived struct that parses this deployment's flags. Its arg ids
    /// must equal `flatten_params(PARAM_GROUPS)` names (pinned by a unit test).
    type Args: clap::Args + ParamSchema;

    /// Run the L4 model cascade (via the `bridge`) and assemble the L6b `Flow`
    /// the L7-β tick driver runs. `store` is the shared `RequestStore` injected
    /// into every worker; the same handle is held by the driver for logging.
    /// This is the heavy step (issues `profile.db` queries) — design §2.1 Phase B.
    fn build(
        args: &Self::Args,
        bridge: &PerfApiBridge,
        store: SharedRequests,
    ) -> anyhow::Result<Box<dyn Flow>>;
}

/// Flatten a deployment's `PARAM_GROUPS` into one ordered list. Used by the
/// schema dump and the clap/ParamDef sync tests; allocates, so off the hot path.
pub fn flatten_params(groups: &[&[ParamDef]]) -> Vec<ParamDef> {
    groups.iter().flat_map(|g| g.iter().copied()).collect()
}
