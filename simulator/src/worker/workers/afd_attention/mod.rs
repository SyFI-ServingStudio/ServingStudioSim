//! AFD-attention family: fresh-request ingress around a shared slot pipeline.

mod attention_build_essentials;
mod attention_slot_pipeline;
mod build_afd_attention_worker;
mod slot_attention_worker;

pub(crate) use build_afd_attention_worker::build_afd_attention_worker;
pub use slot_attention_worker::DisaggAttnWorker;
