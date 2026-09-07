//! Cross-cutting types referenced by every simulator layer.
//!
//! See `doc/architecture.md` (the `common/` cross-cutting module).

pub mod fabric;
pub mod id;
pub mod request;
pub mod request_family;
pub mod request_stage;
pub mod time;

pub use fabric::Fabric;
pub use id::{BatchId, ExpertId, GroupId, IdMap, PoolId, RequestId, WorkerId};
pub use request::{
    ActiveRequest, Request, RequestCore, RequestLifecycle, RequestRecord, RequestStore,
    RequestTelemetry, SchedulingContract, SharedRequests, SloContract,
};
pub use request_family::{
    AcceptanceProfile, AudioExtent, AudioTextGenerationDefinition, AudioTextGenerationRequest,
    AudioToTextDefinition, AudioToTextRequest, DecodingStrategy, GeneratedMediaProgress,
    ImageExtent, ImageGenerationDefinition, ImageGenerationRequest, ImageTextGenerationDefinition,
    ImageTextGenerationRequest, ImageToTextDefinition, ImageToTextRequest, ImageToVideoDefinition,
    ImageToVideoRequest, OmniGenerationDefinition, OmniGenerationProgress, OmniGenerationRequest,
    OmniInputSegment, OmniOutputSpec, RequestDefinition, SessionInput, SpeechGenerationDefinition,
    SpeechGenerationRequest, TextGenerationDefinition, TextGenerationProgress,
    TextGenerationRequest, TextToImageDefinition, TextToImageRequest, TextToSpeechDefinition,
    TextToSpeechRequest, TextToVideoDefinition, TextToVideoRequest, VideoExtent,
    VideoGenerationDefinition, VideoGenerationRequest, VideoTextGenerationDefinition,
    VideoTextGenerationRequest, VideoToTextDefinition, VideoToTextRequest,
};
pub use request_stage::{AfdStage, PdStage, StageEvent, StageVocab, UnifiedStage};
pub use time::Time;
