use crate::common::Time;

/// Conversation context declared for one autoregressive request.
///
/// A session's identity, trace-declared start, and reusable-prefix requirement
/// stay in one variant so downstream code cannot observe only a subset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SessionInput {
    #[default]
    Standalone,
    /// A standalone request whose first `prefix_tokens` prompt tokens the trace
    /// declares already resident in KV when it arrives. Unlike a session prefix
    /// this is a fact, not a requirement: every prefix-capable KV resolves it as
    /// a full hit on whichever partition admission picks, whatever that
    /// partition's cache holds. The prefix is reserved and released with the
    /// request and is never a shared, retained, or evictable cache entry.
    PinnedPrefix { prefix_tokens: u32 },
    Session {
        session_id: u32,
        /// The first `arrival_time` declared for this session in the trace.
        session_start_time: Time,
        /// Trace-declared reusable KV. This is a requirement, not an observed
        /// cache hit; a prefix-capable KV implementation must resolve it.
        declared_prefix_tokens: u32,
    },
}

impl SessionInput {
    pub const fn session_id(self) -> Option<u32> {
        match self {
            Self::Standalone | Self::PinnedPrefix { .. } => None,
            Self::Session { session_id, .. } => Some(session_id),
        }
    }

    pub const fn session_start_time(self) -> Option<Time> {
        match self {
            Self::Standalone | Self::PinnedPrefix { .. } => None,
            Self::Session {
                session_start_time, ..
            } => Some(session_start_time),
        }
    }

    /// Oldest-session-first key, with a standalone request treated as its own
    /// one-request conversation.
    pub const fn session_start_or(self, standalone_arrival_time: Time) -> Time {
        match self {
            Self::Standalone | Self::PinnedPrefix { .. } => standalone_arrival_time,
            Self::Session {
                session_start_time, ..
            } => session_start_time,
        }
    }

    /// Prefix tokens the request's context includes before its fresh prompt:
    /// a session's reusable-prefix requirement, or a pinned prefix.
    pub const fn declared_prefix_tokens(self) -> u32 {
        match self {
            Self::Standalone => 0,
            Self::PinnedPrefix { prefix_tokens } => prefix_tokens,
            Self::Session {
                declared_prefix_tokens,
                ..
            } => declared_prefix_tokens,
        }
    }
}

/// Decode behaviour requested by an autoregressive request.
///
/// Defined in the shared trace crate: it is what the `accept_rate` column means,
/// and the file says it rather than this simulator deciding it. The same column
/// accepts either one legacy geometric probability or a JSON list of
/// per-position conditional probabilities.
pub use req_frontend::schema::{AcceptanceProfile, DecodingStrategy};

/// Mutable progress shared by autoregressive token-output families.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextGenerationProgress {
    pub prefill_tokens_processed: u32,
    pub output_tokens_emitted: u32,
}
