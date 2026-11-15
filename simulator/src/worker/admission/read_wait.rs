//! How long admission lets a host-tier read wait for its tier's channel
//! before it starts it (not vLLM).
//!
//! A read holds its request's whole HBM footprint from its start until its
//! context is put back, so a long queue of reads on a slow channel holds HBM
//! for its whole wait, and a blocked queue head keeps the HBM and DRAM hits
//! behind it out. A bound starts a read only while its tier would begin it
//! within `max`; past it, the policy decides.

use serde::Deserialize;

use crate::common::Time;

/// What a request whose read would wait past the bound gets.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ReadWaitPolicy {
    /// Passed over for this admission pass, holding nothing: the requests
    /// behind it go ahead, and it returns to the front of its queue.
    #[default]
    PassOver,
    /// Admission stops at it, as at a read that does not fit.
    Block,
}

/// A bound on a read's wait for its tier's channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadWait {
    pub max: Time,
    pub policy: ReadWaitPolicy,
}

impl ReadWait {
    /// Whether a read that would wait `queue_wait` is past the bound.
    pub(crate) fn exceeded(self, queue_wait: Time) -> bool {
        queue_wait > self.max
    }
}
