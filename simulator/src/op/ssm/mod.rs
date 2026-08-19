//! `op/ssm` — compound state-space-model operations.

pub mod gdn_decode;
pub mod gdn_prefill;

pub use gdn_decode::{GdnDecodeOp, GdnDecodeOpConfig, GdnDecodeOpInput};
pub use gdn_prefill::{GdnPrefillOp, GdnPrefillOpConfig, GdnPrefillOpInput};
