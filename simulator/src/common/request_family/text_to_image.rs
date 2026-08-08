use crate::common::request::Request;

use super::{GeneratedMediaProgress, ImageExtent, RequestDefinition};

/// Text-in, step-generated-image-out request definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageGenerationDefinition {
    pub text_prompt_tokens: u32,
    pub target_generation_steps: u32,
    pub extent: ImageExtent,
}

impl RequestDefinition for ImageGenerationDefinition {
    type Progress = GeneratedMediaProgress;

    fn initial_progress(&self) -> Self::Progress {
        GeneratedMediaProgress::default()
    }

    fn is_complete(&self, progress: &Self::Progress) -> bool {
        progress.generation_steps_completed >= self.target_generation_steps
    }
}

pub type ImageGenerationRequest = Request<ImageGenerationDefinition>;
pub type TextToImageDefinition = ImageGenerationDefinition;
pub type TextToImageRequest = ImageGenerationRequest;
