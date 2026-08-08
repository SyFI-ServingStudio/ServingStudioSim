/// Session/prefix declaration attached to an autoregressive request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefixInput {
    #[default]
    None,
    Session {
        session_id: u32,
        /// Trace-declared reusable KV. This is a requirement, not an observed
        /// cache hit; a prefix-capable KV implementation must resolve it.
        declared_prefix_tokens: u32,
    },
}

impl PrefixInput {
    pub const fn session_id(self) -> Option<u32> {
        match self {
            Self::None => None,
            Self::Session { session_id, .. } => Some(session_id),
        }
    }

    pub const fn declared_prefix_tokens(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Session {
                declared_prefix_tokens,
                ..
            } => declared_prefix_tokens,
        }
    }
}

/// Decode behavior requested by an autoregressive request.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DecodingStrategy {
    #[default]
    Standard,
    Speculative {
        accept_rate: f32,
    },
}

/// Mutable progress shared by autoregressive token-output families.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextGenerationProgress {
    pub prefill_tokens_processed: u32,
    pub output_tokens_emitted: u32,
}
