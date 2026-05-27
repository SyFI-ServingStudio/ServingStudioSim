//! `SlotInput` — one leaf's typed kernel input, captured for the `cost_log`
//! `slot_input` column (the per-kernel Perfetto trace). A **closed** enum over
//! every leaf input type, stored inline in the per-iter capture buffer (no
//! per-slot box) and serialized to JSON on the writer thread.
//!
//! This is a deliberate exception to the codebase's "no central enum, kernels
//! self-register via `inventory`" rule (`timing/kernels/engine.rs`): a closed
//! enum is the price of inline capture — it removes the per-slot heap box the
//! `erased_serde` variant pays (measured ~28% sim-thread wall on a saturating
//! run; see memory `slot-input-capture-centralized-enum`). Adding a kernel whose
//! input reaches a leaf means one new `log_inputs!` line below; the
//! `K::Input: Into<SlotInput>` bound on `Op::eval` turns a missing entry into a
//! compile error, so the registry can't silently drift.

use serde::Serialize;

use crate::timing::kernels::{
    AllReduceKernelInput, ElementwiseKernelInput, FlashinferAttnDecodeKernelInput,
    FlashinferAttnRectKernelInput, GroupedGemmKernelInput, P2pInterKernelInput,
    P2pIntraKernelInput, RmsNormKernelInput, SingleGemmKernelInput,
};

/// The prefill aggregating leaf's input: the full `(prefix_len, append_len)`
/// fan-out the attention op summed into one slot (many prefill requests fold to a
/// single leaf).
#[derive(Clone, Serialize)]
pub struct AttnPrefillLog {
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
}

/// Declare the `SlotInput` enum + a `From<Input>` per variant from one central
/// list. `#[serde(untagged)]` so each variant serializes as just its inner input
/// object (e.g. `{"m":512}`) — the slot's kernel kind is recovered from the
/// `cost_manifest.json` sidecar, so no tag is needed in the row.
macro_rules! log_inputs {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        #[derive(Clone, Serialize)]
        #[serde(untagged)]
        pub enum SlotInput {
            $( $variant($ty), )+
        }
        $(
            impl From<$ty> for SlotInput {
                fn from(v: $ty) -> Self { SlotInput::$variant(v) }
            }
        )+
    };
}

log_inputs! {
    Gemm        => SingleGemmKernelInput,
    GroupedGemm => GroupedGemmKernelInput,
    RmsNorm     => RmsNormKernelInput,
    Elementwise => ElementwiseKernelInput,
    AttnPrefill => AttnPrefillLog,
    AttnDecode  => FlashinferAttnDecodeKernelInput,
    AttnRect    => FlashinferAttnRectKernelInput,
    AllReduce   => AllReduceKernelInput,
    P2pIntra    => P2pIntraKernelInput,
    P2pInter    => P2pInterKernelInput,
}
