//! AFD-FFN cadence family: a double-buffered task worker with no KV/admission axes.

mod buffered_ffn_worker;
mod build_afd_ffn_worker;

pub use buffered_ffn_worker::BufferedFfnWorker;
pub use build_afd_ffn_worker::build_afd_ffn_worker;
