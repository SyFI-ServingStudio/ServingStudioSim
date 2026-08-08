/// Mutable progress shared by non-autoregressive generated-media families.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GeneratedMediaProgress {
    pub generation_steps_completed: u32,
}
