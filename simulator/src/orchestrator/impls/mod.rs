//! Concrete deployment implementations. See L6 design.md.

pub mod afd;
pub mod afd_attn_pool;
pub mod afd_ffn_pool;
pub mod pd;
pub mod simple_dp;

pub use afd::{AfdFlow, AFD_ATTN_POOL, AFD_FFN_POOL};
pub use afd_attn_pool::AfdAttnPoolController;
pub use afd_ffn_pool::AfdFfnPoolController;
pub use pd::{PdFlow, PD_DECODE_POOL, PD_PREFILL_POOL};
pub use simple_dp::{
    DpPlacementPolicy, GroupBy, SimpleDpConfig, SimpleDpFlow, SimpleDpPoolConfig,
    SimpleDpPoolController,
};
