//! A typed trace row before replay decides when it enters L6.
//!
//! The request family is the `Definition` parameter. There is deliberately no
//! runtime `ArrivalKind`: once a text trace has been parsed, every later layer
//! sees `ScheduledRequest<TextGenerationDefinition>` and the Rust type system
//! prevents it from becoming a generated-media request.

use crate::common::{
    Request, RequestCore, RequestDefinition, RequestId, SchedulingContract, SloContract, Time,
};

/// Session facts used only to order releases. Prefix requirements live in the
/// concrete request definition, not in this pacing metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SessionReleaseMetadata {
    pub session_id: u32,
    pub tool_wait_after: Time,
}

/// The only facts the replay scheduler may inspect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReleaseMetadata {
    pub request_id: RequestId,
    /// Rate-1-normalized trace time in milliseconds.
    pub trace_arrival_time_ms: f64,
    pub session: Option<SessionReleaseMetadata>,
}

/// Scheduling policy declared by the trace.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SchedulingDeclaration {
    pub priority: i32,
}

/// One parsed row with a statically known request definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ScheduledRequest<Definition: RequestDefinition> {
    pub release: ReleaseMetadata,
    pub slo: SloContract,
    pub scheduling: SchedulingDeclaration,
    pub definition: Definition,
}

impl<Definition: RequestDefinition> ScheduledRequest<Definition> {
    /// Stamp release-dependent core facts while preserving the typed definition.
    pub fn realize_at(&self, arrival_time: Time) -> Request<Definition> {
        Request::new(
            RequestCore {
                id: self.release.request_id,
                arrival_time,
                slo: self.slo,
                scheduling: SchedulingContract {
                    priority: self.scheduling.priority,
                },
            },
            self.definition.clone(),
        )
    }
}
