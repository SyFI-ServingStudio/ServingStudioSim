/// Mutable completion state for an omni output plan.
///
/// Entry N always tracks output spec N. Input preparation and worker cadence
/// remain owned by the future omni execution family.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OmniGenerationProgress {
    pub output_tokens_emitted: Vec<u32>,
}
