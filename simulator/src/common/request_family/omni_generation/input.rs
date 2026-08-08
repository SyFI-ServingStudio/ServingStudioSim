use serde::{Deserialize, Serialize};

use super::super::{AudioExtent, ImageExtent, VideoExtent};

/// One item in an omni model's heterogeneous input sequence.
///
/// The vector containing these segments may mix modalities and repeat a
/// modality. Encoded-token counts stay explicit because the trace layer must
/// not guess a model-specific tokenizer expansion.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OmniInputSegment {
    Text {
        tokens: u32,
    },
    Image {
        extent: ImageExtent,
        encoded_tokens: u32,
    },
    Audio {
        extent: AudioExtent,
        encoded_tokens: u32,
    },
    Video {
        extent: VideoExtent,
        encoded_tokens: u32,
    },
}
