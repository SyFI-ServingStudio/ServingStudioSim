use crate::common::request::Request;

use super::{GeneratedMediaProgress, ImageExtent, RequestDefinition, VideoExtent};

/// Image-and-text in, step-generated-video-out request definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageToVideoDefinition {
    pub text_prompt_tokens: u32,
    pub encoded_input_tokens: u32,
    pub input_extent: ImageExtent,
    pub output_extent: VideoExtent,
    pub target_generation_steps: u32,
}

impl RequestDefinition for ImageToVideoDefinition {
    type Progress = GeneratedMediaProgress;

    fn initial_progress(&self) -> Self::Progress {
        GeneratedMediaProgress::default()
    }

    fn is_complete(&self, progress: &Self::Progress) -> bool {
        progress.generation_steps_completed >= self.target_generation_steps
    }
}

pub type ImageToVideoRequest = Request<ImageToVideoDefinition>;
