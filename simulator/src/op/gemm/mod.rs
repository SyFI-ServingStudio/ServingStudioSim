//! Compound GEMM operations whose preparation launches are inseparable from
//! the GEMM they feed.

mod single_fp8;

pub use single_fp8::{
    SingleFp8GemmWithQuantConfig, SingleFp8GemmWithQuantInput, SingleFp8GemmWithQuantOp,
};
