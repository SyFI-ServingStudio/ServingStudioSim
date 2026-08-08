use serde::{Deserialize, Serialize};

use super::super::{AudioExtent, ImageExtent, VideoExtent};

/// One requested output segment from a token-generating omni model.
///
/// Image/audio/video targets count model or codec tokens here. Step-based
/// diffusion pipelines remain separate request families such as text-to-video
/// and image-to-video.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OmniOutputSpec {
    Text {
        target_tokens: u32,
    },
    Image {
        extent: ImageExtent,
        target_tokens: u32,
    },
    Audio {
        extent: AudioExtent,
        target_tokens: u32,
    },
    Video {
        extent: VideoExtent,
        target_tokens: u32,
    },
}

impl OmniOutputSpec {
    pub fn target_tokens(&self) -> u32 {
        match self {
            Self::Text { target_tokens }
            | Self::Image { target_tokens, .. }
            | Self::Audio { target_tokens, .. }
            | Self::Video { target_tokens, .. } => *target_tokens,
        }
    }
}
