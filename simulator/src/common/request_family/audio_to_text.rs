use crate::common::request::Request;

use super::{
    AudioExtent, DecodingStrategy, RequestDefinition, SessionInput, TextGenerationProgress,
};

/// Audio-and-text in, autoregressive-text-out request definition.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioTextGenerationDefinition {
    pub text_prompt_tokens: u32,
    pub encoded_input_tokens: u32,
    pub target_output_tokens: u32,
    pub extent: AudioExtent,
    pub session: SessionInput,
    pub decoding: DecodingStrategy,
}

impl RequestDefinition for AudioTextGenerationDefinition {
    type Progress = TextGenerationProgress;

    fn initial_progress(&self) -> Self::Progress {
        TextGenerationProgress::default()
    }

    fn is_complete(&self, progress: &Self::Progress) -> bool {
        progress.output_tokens_emitted >= self.target_output_tokens
    }
}

pub type AudioTextGenerationRequest = Request<AudioTextGenerationDefinition>;
pub type AudioToTextDefinition = AudioTextGenerationDefinition;
pub type AudioToTextRequest = AudioTextGenerationRequest;
