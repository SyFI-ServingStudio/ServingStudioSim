use crate::common::request::Request;

use super::{AudioExtent, GeneratedMediaProgress, RequestDefinition};

/// Text-in, step-generated-speech-out request definition.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeechGenerationDefinition {
    pub text_prompt_tokens: u32,
    pub target_generation_steps: u32,
    pub extent: AudioExtent,
}

impl RequestDefinition for SpeechGenerationDefinition {
    type Progress = GeneratedMediaProgress;

    fn initial_progress(&self) -> Self::Progress {
        GeneratedMediaProgress::default()
    }

    fn is_complete(&self, progress: &Self::Progress) -> bool {
        progress.generation_steps_completed >= self.target_generation_steps
    }
}

pub type SpeechGenerationRequest = Request<SpeechGenerationDefinition>;
pub type TextToSpeechDefinition = SpeechGenerationDefinition;
pub type TextToSpeechRequest = SpeechGenerationRequest;
