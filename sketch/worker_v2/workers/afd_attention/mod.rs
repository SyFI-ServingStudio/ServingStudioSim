//! AFD-attention cadence family.
//!
//! Both ingress variants share `AttentionSlotPipeline`; each concrete essentials
//! recipe remains isolated in its own `build_*` file.

mod attention_build_essentials;
mod attention_slot_pipeline;
mod build_afd_attention_worker;
mod build_pd_afd_decode_attention_worker;
mod pull_slot_attention_worker;
mod slot_attention_worker;

pub use build_afd_attention_worker::build_afd_attention_worker;
pub use build_pd_afd_decode_attention_worker::build_pd_afd_decode_attention_worker;
pub use pull_slot_attention_worker::{PullAttentionWorkerMsg, PullSlotAttentionWorker};
pub use slot_attention_worker::SlotAttentionWorker;
