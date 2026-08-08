use serde::{Deserialize, Serialize};

/// Physical input/output shapes remain concrete so a family consumer never
/// handles unrelated media variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageExtent {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoExtent {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioExtent {
    pub samples: u64,
}
