//! CLI parameter schema — the single source of truth shared between the
//! Rust binary (clap structs) and the Python launcher (via `simulator
//! list-params` JSON). See `docs/detailed_design/L7/design.md` §1.8.1.

pub mod dump;
pub mod param_def;

pub use dump::list_params;
pub use param_def::{DefaultValue, ParamDef, ParamType};
