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
        /// How many rounds the trace declares for this conversation, this one
        /// included. Trace-given like `target_output_tokens`, not a prediction.
        ///
        /// It is what lets a consumer tell "this round finished" from "this
        /// conversation finished". Under `SessionDependency::Chained` only one
        /// round of a session is ever in flight, so a counter of what is still
        /// outstanding hits zero after *every* round; only the declared total
        /// says which of those zeros is the last one. A count rather than an
        /// is-final flag because under independent release the rounds may
        /// complete out of order.
        rounds_in_session: u32,
        /// Trace-declared tokens, counted from the start of this round's full
        /// context (`declared_prefix_tokens` + fresh prompt), that are resident
        /// on the request's first admission whatever the chosen partition's
        /// cache holds; 0 when the row declares none. It may exceed
        /// `declared_prefix_tokens` (a cross-session system prompt on a first
        /// turn) and never extends the context. A re-prefill drops it.
        pinned_prefix_tokens: u32,
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

    /// Rounds this conversation declares. A standalone request is its own
    /// one-round conversation, which is the same reading `session_start_or`
    /// takes.
    pub const fn rounds_in_session(self) -> u32 {
        match self {
            Self::Standalone | Self::PinnedPrefix { .. } => 1,
            Self::Session {
                rounds_in_session, ..
            } => rounds_in_session,
        }
    }

    /// Leading context tokens the trace forces resident on first admission,
    /// whatever any cache holds: a standalone pinned prefix, or a session
    /// row's `prefix_len`.
    pub const fn pinned_prefix_tokens(self) -> u32 {
        match self {
            Self::Standalone => 0,
            Self::PinnedPrefix { prefix_tokens } => prefix_tokens,
            Self::Session {
                pinned_prefix_tokens,
                ..
            } => pinned_prefix_tokens,
        }
    }

    /// The declaration a re-prefill of this request resolves, and the
    /// formerly pinned tokens it must now compute as prompt.
    ///
    /// The trace's pinned hit describes the request's first admission only. A
    /// retracted request recomputes its whole context: a standalone pinned
    /// prefix becomes leading prompt tokens of a plain standalone request
    /// (returned so the caller adds them to the recomputed prompt), and a
    /// session keeps its `declared_prefix_tokens` requirement, which only the
    /// partition's retained session KV can satisfy, but loses its pin.
    pub const fn without_pinned_prefix(self) -> (Self, u32) {
        match self {
            Self::Standalone => (Self::Standalone, 0),
            Self::PinnedPrefix { prefix_tokens } => (Self::Standalone, prefix_tokens),
            Self::Session {
                session_id,
                session_start_time,
                declared_prefix_tokens,
                rounds_in_session,
                ..
            } => (
                Self::Session {
                    session_id,
                    session_start_time,
                    declared_prefix_tokens,
                    rounds_in_session,
                    pinned_prefix_tokens: 0,
                },
                0,
            ),
        }
    }

    /// The declaration a re-prefill on another worker resolves, and the
    /// leading tokens it must now compute as prompt.
    ///
    /// A migrated request reaches a worker that holds none of its KV, so
    /// nothing of its context counts as resident: the pin goes the way
    /// [`Self::without_pinned_prefix`] takes it, and a session's declared
    /// prefix also becomes prompt, which leaves no requirement for a cache to
    /// satisfy. The session id stays, so the context it builds is retained for
    /// the conversation's later rounds as usual.
    pub const fn without_resident_prefix(self) -> (Self, u32) {
        match self.without_pinned_prefix() {
            (
                Self::Session {
                    session_id,
                    session_start_time,
                    declared_prefix_tokens,
                    rounds_in_session,
                    pinned_prefix_tokens,
                },
                unpinned_prefix_tokens,
            ) => (
                Self::Session {
                    session_id,
                    session_start_time,
                    declared_prefix_tokens: 0,
                    rounds_in_session,
                    pinned_prefix_tokens,
                },
                unpinned_prefix_tokens + declared_prefix_tokens,
            ),
            unpinned => unpinned,
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
