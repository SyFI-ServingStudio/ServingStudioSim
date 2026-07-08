//! CLI parameter schema — the single source of truth shared between the
//! Rust binary (clap structs) and the Python launcher (via `simulator
//! list-params` JSON). See `doc/detailed_design/L7.md`.

pub mod dump;
pub mod param_def;

pub use dump::list_params;
pub use param_def::{DefaultValue, ParamDef, ParamType};
