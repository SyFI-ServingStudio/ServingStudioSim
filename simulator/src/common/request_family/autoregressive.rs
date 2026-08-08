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
