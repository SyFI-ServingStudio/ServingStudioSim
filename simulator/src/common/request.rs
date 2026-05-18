//! Single-round inference request representation.
//!
//! Phase 0 carries the minimum fields the first-milestone (Llama3-8B dense /
//! local / single-round trace) needs. Multi-round fields (`round_idx`,
//! `preserved_prefix_kv`, ...) land alongside L7 lifecycle work.

use serde::{Deserialize, Serialize};

use super::id::RequestId;
use super::time::Time;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Request {
    pub id: RequestId,
    pub prompt_len: u32,
    pub decode_len: u32,
    pub arrival_time: Time,
}

impl Request {
    pub const fn new(id: RequestId, prompt_len: u32, decode_len: u32, arrival_time: Time) -> Self {
        Self {
            id,
            prompt_len,
            decode_len,
            arrival_time,
        }
    }
}
