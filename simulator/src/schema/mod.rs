//! CLI parameter schema — the single source of truth shared between the
//! Rust binary (clap structs) and the Python launcher (via `simulator
//! list-params` JSON). See `docs/detailed_design/L7/design.md` §1.8.1.

pub mod common;
pub mod param_def;

pub use common::COMMON_PARAMS;
pub use param_def::{DefaultValue, ParamDef, ParamSection, ParamType};
