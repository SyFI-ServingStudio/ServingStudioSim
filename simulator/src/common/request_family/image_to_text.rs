use crate::common::request::Request;

use super::{
    DecodingStrategy, ImageExtent, RequestDefinition, SessionInput, TextGenerationProgress,
};

/// Image-and-text in, autoregressive-text-out request definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageTextGenerationDefinition {
    pub text_prompt_tokens: u32,
    pub encoded_input_tokens: u32,
    pub target_output_tokens: u32,
    pub extent: ImageExtent,
    pub session: SessionInput,
    pub decoding: DecodingStrategy,
}

impl RequestDefinition for ImageTextGenerationDefinition {
    type Progress = TextGenerationProgress;

    fn initial_progress(&self) -> Self::Progress {
        TextGenerationProgress::default()
    }

    fn is_complete(&self, progress: &Self::Progress) -> bool {
        progress.output_tokens_emitted >= self.target_output_tokens
    }
}

pub type ImageTextGenerationRequest = Request<ImageTextGenerationDefinition>;
pub type ImageToTextDefinition = ImageTextGenerationDefinition;
pub type ImageToTextRequest = ImageTextGenerationRequest;
