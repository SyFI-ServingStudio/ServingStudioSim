//! AFD-FFN family: double-buffered transfer/compute shell.

mod buffered_ffn_worker;
mod build_afd_ffn_worker;

pub use buffered_ffn_worker::DisaggFfnWorker;
pub(crate) use build_afd_ffn_worker::build_afd_ffn_worker;
