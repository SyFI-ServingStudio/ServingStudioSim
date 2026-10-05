//! A request longer than the context its arch or pool serves.

/// The refusal of a request longer than `max_model_len`, by
/// `deployment::check_trace` (a run's trace) and `timing-predict` (a case).
/// Its `Display` is the whole message; `--error-json` also writes its fields
/// as `too_long`, so a caller reads the limit and the counts instead of
/// parsing the message.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TooLong {
    pub max_model_len: u32,
    /// How many of a trace's requests are too long; None for one case.
    pub requests: Option<usize>,
    /// How many requests the trace has; None for one case.
    pub total: Option<usize>,
    #[serde(skip)]
    pub message: String,
}

impl std::fmt::Display for TooLong {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TooLong {}
