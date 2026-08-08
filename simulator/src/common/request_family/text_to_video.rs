use crate::common::request::Request;

use super::{GeneratedMediaProgress, RequestDefinition, VideoExtent};

/// Text-in, step-generated-video-out request definition.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoGenerationDefinition {
    pub text_prompt_tokens: u32,
    pub target_generation_steps: u32,
    pub extent: VideoExtent,
}

impl RequestDefinition for VideoGenerationDefinition {
    type Progress = GeneratedMediaProgress;

    fn initial_progress(&self) -> Self::Progress {
        GeneratedMediaProgress::default()
    }

    fn is_complete(&self, progress: &Self::Progress) -> bool {
        progress.generation_steps_completed >= self.target_generation_steps
    }
}

pub type VideoGenerationRequest = Request<VideoGenerationDefinition>;
pub type TextToVideoDefinition = VideoGenerationDefinition;
pub type TextToVideoRequest = VideoGenerationRequest;
