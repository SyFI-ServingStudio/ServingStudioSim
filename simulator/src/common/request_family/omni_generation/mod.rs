//! One request family for a model that natively consumes and emits arbitrary
//! mixtures of text, image, audio, and video.

mod input;
mod output;
mod progress;

pub use input::OmniInputSegment;
pub use output::OmniOutputSpec;
pub use progress::OmniGenerationProgress;

use crate::common::request::Request;

use super::{DecodingStrategy, RequestDefinition, SessionInput};

/// Heterogeneous input and output plans accepted by one omni worker family.
#[derive(Clone, Debug, PartialEq)]
pub struct OmniGenerationDefinition {
    pub input: Vec<OmniInputSegment>,
    pub output: Vec<OmniOutputSpec>,
    pub session: SessionInput,
    pub decoding: DecodingStrategy,
}

impl RequestDefinition for OmniGenerationDefinition {
    type Progress = OmniGenerationProgress;

    fn initial_progress(&self) -> Self::Progress {
        OmniGenerationProgress {
            output_tokens_emitted: vec![0; self.output.len()],
        }
    }

    fn is_complete(&self, progress: &Self::Progress) -> bool {
        progress.output_tokens_emitted.len() == self.output.len()
            && self
                .output
                .iter()
                .zip(&progress.output_tokens_emitted)
                .all(|(spec, emitted)| *emitted >= spec.target_tokens())
    }
}

pub type OmniGenerationRequest = Request<OmniGenerationDefinition>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{ImageExtent, RequestDefinition};

    #[test]
    fn all_output_segments_must_complete() {
        let definition = OmniGenerationDefinition {
            input: vec![OmniInputSegment::Text { tokens: 8 }],
            output: vec![
                OmniOutputSpec::Text { target_tokens: 2 },
                OmniOutputSpec::Image {
                    extent: ImageExtent {
                        width: 512,
                        height: 512,
                    },
                    target_tokens: 4,
                },
            ],
            session: SessionInput::Standalone,
            decoding: DecodingStrategy::Standard,
        };
        let mut progress = definition.initial_progress();
        progress.output_tokens_emitted = vec![2, 3];
        assert!(!definition.is_complete(&progress));
        progress.output_tokens_emitted[1] = 4;
        assert!(definition.is_complete(&progress));
    }
}
