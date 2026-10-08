//! Pipeline-parallel family: a head stage that owns admission and KV for the
//! whole pipeline, and follower stages that only pull and compute.

mod build_pipeline_head_worker;
mod build_pipeline_stage_worker;
mod head_prefix_tiers;
mod pipeline_head_worker;
mod pipeline_stage_worker;

pub use build_pipeline_head_worker::PipelineHybridState;
pub(crate) use build_pipeline_head_worker::{
    build_hybrid_pipeline_head_worker, build_pipeline_head_worker,
};
pub(crate) use build_pipeline_stage_worker::build_pipeline_stage_worker;
pub use pipeline_head_worker::{PipelineHeadWorker, PipelineLayout};
pub use pipeline_stage_worker::PipelineStageWorker;

use crate::worker::admission::{PendingOrder, PipelinedChunkedPrefillAdmission};
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::kv::{FullAttnKv, HybridGdnKv};

/// Production pipeline head: full-attention KV sized for the stage with the most
/// KV per token, pipelined chunked prefill and decode admission, and stage-0
/// execution.
pub type PipelineHead<M> = PipelineHeadWorker<
    FullAttnKv,
    PipelinedChunkedPrefillAdmission<PendingOrder>,
    UnifiedIterExecution<M>,
>;
/// Pipeline head of a hybrid recurrent + full-attention model: one block pool
/// for per-token attention KV and per-request recurrent state.
pub type HybridPipelineHead<M> = PipelineHeadWorker<
    HybridGdnKv,
    PipelinedChunkedPrefillAdmission<PendingOrder>,
    UnifiedIterExecution<M>,
>;
/// Production follower stage.
pub type PipelineStage<M> = PipelineStageWorker<UnifiedIterExecution<M>>;
