//! Concrete request families and their family-owned state.
//!
//! Every request family lives in its own file. Shared family vocabulary is
//! limited to explicit building blocks such as autoregressive decoding state
//! and physical media extents; the family-agnostic request container, lifecycle,
//! telemetry, and store remain in `common::request`.

use std::fmt::Debug;

mod audio_to_text;
mod autoregressive;
mod generated_media;
mod image_to_text;
mod image_to_video;
mod media;
mod omni_generation;
mod text_generation;
mod text_to_image;
mod text_to_speech;
mod text_to_video;
mod video_to_text;

pub use audio_to_text::{
    AudioTextGenerationDefinition, AudioTextGenerationRequest, AudioToTextDefinition,
    AudioToTextRequest,
};
pub use autoregressive::{
    AcceptanceProfile, DecodingStrategy, SessionInput, TextGenerationProgress,
};
pub use generated_media::GeneratedMediaProgress;
pub use image_to_text::{
    ImageTextGenerationDefinition, ImageTextGenerationRequest, ImageToTextDefinition,
    ImageToTextRequest,
};
pub use image_to_video::{ImageToVideoDefinition, ImageToVideoRequest};
pub use media::{AudioExtent, ImageExtent, VideoExtent};
pub use omni_generation::{
    OmniGenerationDefinition, OmniGenerationProgress, OmniGenerationRequest, OmniInputSegment,
    OmniOutputSpec,
};
pub use text_generation::{TextGenerationDefinition, TextGenerationRequest};
pub use text_to_image::{
    ImageGenerationDefinition, ImageGenerationRequest, TextToImageDefinition, TextToImageRequest,
};
pub use text_to_speech::{
    SpeechGenerationDefinition, SpeechGenerationRequest, TextToSpeechDefinition,
    TextToSpeechRequest,
};
pub use text_to_video::{
    TextToVideoDefinition, TextToVideoRequest, VideoGenerationDefinition, VideoGenerationRequest,
};
pub use video_to_text::{
    VideoTextGenerationDefinition, VideoTextGenerationRequest, VideoToTextDefinition,
    VideoToTextRequest,
};

/// Immutable request definitions provide their own mutable progress and
/// completion rule. They do not own worker cadence or resource accounting.
pub trait RequestDefinition: Clone + Debug {
    type Progress: Clone + Debug;

    fn initial_progress(&self) -> Self::Progress;
    fn is_complete(&self, progress: &Self::Progress) -> bool;
}
