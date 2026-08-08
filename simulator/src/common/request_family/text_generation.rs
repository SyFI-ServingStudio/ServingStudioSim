use crate::common::request::{ActiveRequest, Request};
use crate::common::Time;

use super::{DecodingStrategy, PrefixInput, RequestDefinition, TextGenerationProgress};

/// Text-in, autoregressive-token-out request definition.
#[derive(Clone, Debug, PartialEq)]
pub struct TextGenerationDefinition {
    pub prompt_tokens: u32,
    pub target_output_tokens: u32,
    pub prefix: PrefixInput,
    pub decoding: DecodingStrategy,
}

impl RequestDefinition for TextGenerationDefinition {
    type Progress = TextGenerationProgress;

    fn initial_progress(&self) -> Self::Progress {
        TextGenerationProgress::default()
    }

    fn is_complete(&self, progress: &Self::Progress) -> bool {
        progress.output_tokens_emitted >= self.target_output_tokens
    }
}

pub type TextGenerationRequest = Request<TextGenerationDefinition>;

impl ActiveRequest<TextGenerationDefinition> {
    pub fn is_prefill(&self) -> bool {
        self.progress.prefill_tokens_processed < self.request.definition.prompt_tokens
    }

    pub fn record_first_token(&mut self, now: Time, log_tokens: bool) {
        self.progress.output_tokens_emitted = 1;
        self.telemetry.first_output_time = Some(now);
        self.telemetry.last_output_time = Some(now);
        self.lifecycle.completed = self.is_complete();
        if log_tokens {
            self.telemetry
                .output_times
                .reserve_exact(self.request.definition.target_output_tokens as usize);
            self.telemetry.output_times.push(now);
        }
    }

    pub fn record_token(&mut self, now: Time, log_tokens: bool) {
        self.progress.output_tokens_emitted += 1;
        self.telemetry.last_output_time = Some(now);
        if log_tokens {
            self.telemetry.output_times.push(now);
        }
        if self.is_complete() {
            self.lifecycle.completed = true;
        }
    }
}
