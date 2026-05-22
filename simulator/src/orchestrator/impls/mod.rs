//! Concrete deployment implementations. See L6 design.md.

pub mod simple_dp;

pub use simple_dp::{
    DpPlacementPolicy, SimpleDpConfig, SimpleDpFlow, SimpleDpPoolConfig, SimpleDpPoolController,
};
