//! Completion rule for a prefill head whose decode runs elsewhere.
//!
//! A request arrives with its real output length; the head runs it to its
//! first token only, so it completes there and frees its slot. The decode
//! elsewhere then hands the context back: every output but the last extends
//! the prompt's context HBM retained, and host tiers are written the extended
//! round. Hand-backs run after every release of the microbatch, as the
//! context returns once its decode finishes, not inside the release order.

use std::collections::HashMap;

use crate::common::{RequestId, TextGenerationDefinition, Time};
use crate::worker::kv::{HandBack, PrefixKv};
use crate::worker::shared::advance_scope::PartitionId;

#[derive(Debug, Default)]
pub(crate) struct ExternalDecode {
    /// Each running request's real output length.
    output_tokens: HashMap<RequestId, u32>,
    /// Contexts released in the microbatch being completed, waiting to be
    /// handed back with their output lengths.
    handing_back: Vec<(HandBack, u32)>,
}

impl ExternalDecode {
    /// Keep the real output length and run the request to its first token.
    pub(crate) fn accept(&mut self, request: RequestId, definition: &mut TextGenerationDefinition) {
        self.output_tokens
            .insert(request, definition.target_output_tokens);
        definition.target_output_tokens = 1;
    }

    /// Release a request that reached its first token, keeping its prompt's
    /// context in HBM until the decode elsewhere hands it back.
    pub(crate) fn release<K: PrefixKv>(
        &mut self,
        kv_store: &mut K,
        request: RequestId,
        partition: PartitionId,
        at: Time,
    ) {
        let output_tokens = self
            .output_tokens
            .remove(&request)
            .expect("an external decode completes a request it accepted");
        if let Some(hand_back) = kv_store.release_handing_back(request, partition, at) {
            self.handing_back.push((hand_back, output_tokens));
        }
    }

    /// Hand back every context released since the last call, in release order.
    pub(crate) fn hand_back<K: PrefixKv>(
        &mut self,
        kv_store: &mut K,
        partition: PartitionId,
        at: Time,
    ) {
        for (hand_back, output_tokens) in self.handing_back.drain(..) {
            kv_store.hand_back_context(partition, hand_back, output_tokens, at);
        }
    }
}
