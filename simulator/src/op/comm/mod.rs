//! `op/comm` — collective boundaries composed from several L1 kernels.

mod tp_all_reduce_rms_norm;

pub use tp_all_reduce_rms_norm::{TpAllReduceRmsNormConfig, TpAllReduceRmsNormOp};
