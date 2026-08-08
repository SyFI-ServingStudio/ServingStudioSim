use crate::common::request::Request;

use super::{
    DecodingStrategy, PrefixInput, RequestDefinition, TextGenerationProgress, VideoExtent,
};

/// Video-and-text in, autoregressive-text-out request definition.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoTextGenerationDefinition {
    pub text_prompt_tokens: u32,
    pub encoded_input_tokens: u32,
    pub target_output_tokens: u32,
    pub extent: VideoExtent,
    pub prefix: PrefixInput,
    pub decoding: DecodingStrategy,
}

impl RequestDefinition for VideoTextGenerationDefinition {
    type Progress = TextGenerationProgress;

    fn initial_progress(&self) -> Self::Progress {
        TextGenerationProgress::default()
    }

    fn is_complete(&self, progress: &Self::Progress) -> bool {
        progress.output_tokens_emitted >= self.target_output_tokens
    }
}

pub type VideoTextGenerationRequest = Request<VideoTextGenerationDefinition>;
pub type VideoToTextDefinition = VideoTextGenerationDefinition;
pub type VideoToTextRequest = VideoTextGenerationRequest;
