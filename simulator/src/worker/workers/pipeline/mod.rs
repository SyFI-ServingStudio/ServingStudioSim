//! Pipeline-parallel family: a head stage that owns admission and KV for the
//! whole pipeline, and follower stages that only pull and compute.

mod build_pipeline_head_worker;
mod build_pipeline_stage_worker;
mod pipeline_head_worker;
mod pipeline_stage_worker;

pub(crate) use build_pipeline_head_worker::build_pipeline_head_worker;
pub(crate) use build_pipeline_stage_worker::build_pipeline_stage_worker;
pub use pipeline_head_worker::{PipelineHeadWorker, PipelineLayout};
pub use pipeline_stage_worker::PipelineStageWorker;

use crate::worker::admission::{PendingOrder, PipelinedPrefillAdmission};
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::kv::FullAttnKv;

/// Production pipeline head: full-attention KV sized for the stage with the most
/// KV per token, prefill-only pipelined chunk admission, and stage-0 execution.
pub type PipelineHead<M> = PipelineHeadWorker<
    FullAttnKv,
    PipelinedPrefillAdmission<PendingOrder>,
    UnifiedIterExecution<M>,
>;
/// Production follower stage.
pub type PipelineStage<M> = PipelineStageWorker<UnifiedIterExecution<M>>;
