//! `op/ssm` — compound state-space-model operations.

pub mod gdn_decode;
pub mod gdn_prefill;
pub mod tensorfold_gdn_chain;

pub use gdn_decode::{GdnDecodeOp, GdnDecodeOpConfig, GdnDecodeOpInput};
pub use gdn_prefill::{GdnPrefillOp, GdnPrefillOpConfig, GdnPrefillOpInput};
pub use tensorfold_gdn_chain::{
    TensorfoldGdnChainOp, TensorfoldGdnChainOpConfig, TensorfoldGdnChainOpInput,
};
