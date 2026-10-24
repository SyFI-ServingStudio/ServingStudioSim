//! Shared pieces of the TensorFold Qwen3.8-27B worklets
//! (`qwen35_tensorfold_nvfp4_gdn_dense_dflash2`): the backend names TensorFold's
//! kernels are profiled under, a few construction / fan-in helpers, and the two
//! per-case shapes every TensorFold worklet reads — one prefill step
//! ([`PrefillForward`]) and one verify round ([`VerifyRound`]).
//!
//! TensorFold sources cited below are under
//! `tensorfold-0.6.6/src/tensorfold/families/qwen3_5/cuda/`.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    GdnStateReplayKernelInput, GdnTreeSlotClass, GdnTreeVerifyRecurrenceKernelInput,
    SingleGemmKernelConfig, TreeVerifyAttentionKernelInput,
};
use crate::timing::{Evaluator, FanInLog, LeafMetrics, Probe, SlotInput};

// The GLM construction helpers, re-exported for the TensorFold worklets.
#[allow(unused_imports)]
pub(crate) use super::glm53_common::{atomic, elementwise, push_or_zero, repeated};

// ─── backend names ───────────────────────────────────────────────────────────

/// `single_gemm`: `lane_kernel`, every verify linear (FP8 and NVFP4) and the
/// drafter head slice.
pub const TENSORFOLD_LANE: &str = "tensorfold_lane";
/// `single_gemm` (prefill linears) and `gemm_swiglu_quant` (the fused prefill
/// MLP): `gemm_kernel`.
pub const TENSORFOLD_CK: &str = "tensorfold_ck";
/// `single_gemm` at `w4a16_affine_g64`: the drafter's and the taps' linears.
pub const TENSORFOLD_QMM: &str = "tensorfold_qmm";
/// `single_gemm` at bf16, n = 96: the GDN b/a pair in verify.
pub const TENSORFOLD_B16: &str = "tensorfold_b16";
/// `single_gemm` at bf16, n = 96: the GDN b/a pair in prefill.
pub const TENSORFOLD_B16_PROMPT: &str = "tensorfold_b16_prompt";
/// `flashinfer_attn_prefill`: `pattn_kernel`, one stream's prompt chunk.
pub use crate::timing::kernels::flashinfer_attn_prefill::TENSORFOLD_PATTN;
/// `gdn_chunk_delta_rule`: `chain_kernel`, one stream's prompt chunk.
pub use crate::timing::kernels::gdn_chunk_delta_rule::TENSORFOLD_CHAIN;
/// CUDA kernels: `gdn_tree_verify_recurrence`, `gdn_state_replay`,
/// `nvfp4_quant`, `static_fp8_quant`.
pub const TENSORFOLD_CUDA: &str = "tensorfold_cuda";
/// Triton glue: norms, `_gdn_pre`, `_attn_prep`, `_gate_mul`, `_swiglu`,
/// `tree_verify_attention`, the drafter's KV append and block attention.
pub const TENSORFOLD_TRITON: &str = "tensorfold_triton";
/// Plain torch calls: `logits_topk` (drafter head, sampling) and the drafter's
/// `F.rms_norm`.
pub const TENSORFOLD_TORCH: &str = "tensorfold_torch";

/// A TensorFold `single_gemm` config: one backend, `n × k` at `dtype`
/// (`fp8_e4m3` / `nvfp4_e2m1` for lane and ck, `w4a16_affine_g64` for qmm, bf16
/// for the b16 pair).
pub fn gemm(
    backend: &'static str,
    gpu_name: &str,
    n: u32,
    k: u32,
    dtype: DType,
) -> SingleGemmKernelConfig {
    SingleGemmKernelConfig {
        backends: vec![backend],
        gpu_name: gpu_name.to_string(),
        n: n.into(),
        k: k.into(),
        dtype,
    }
}

/// Several launches of one atomic op summed into its one slot (INV-1): the
/// per-chunk forwards of a single-piece prefill, or the m = 1 lm_head of each
/// finishing prompt. No launches pushes zero without a lookup. The slot logs
/// every launch's input ([`FanInLog`]).
pub(crate) fn push_sum<K>(op: &Op<K>, inputs: &[K::Input], ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let mut metrics = LeafMetrics::ZERO;
    for input in inputs {
        metrics.add_fanin(op.kernel.eval(input));
    }
    ev.push(metrics, || {
        FanInLog {
            launches: inputs.iter().cloned().map(Into::into).collect(),
        }
        .into()
    });
}

// ─── one prefill step ────────────────────────────────────────────────────────

/// Rows of one `prefill_state` chunk (`prefill.py` `CHUNK`; GB10 runs the
/// real weights, whose `prompt_rows` is unset).
pub const PREFILL_CHUNK_ROWS: u32 = 4096;

/// One stream's share of a prefill step: rows `[prefix_len, prefix_len + rows)`
/// of its prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillPiece {
    /// Prompt rows already committed (the cached prefix the attention reads).
    pub prefix_len: u32,
    /// Rows this step commits; positive.
    pub rows: u32,
    /// The step reaches the prompt's end, so its last row runs the final norm
    /// and an m = 1 lm_head for the first token.
    pub finishes_prompt: bool,
}

/// How TensorFold runs a prefill step (`multi.py` `_fill`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillPath {
    /// One piece: `prefill_state`, a forward per even chunk of at most
    /// [`PREFILL_CHUNK_ROWS`] rows (`prefill.py` `chunks`).
    State,
    /// Two or more pieces: one `prefill_rows` forward. Linears, norms and the
    /// MLP run over every row at once; the GDN chain and the attention launch
    /// once per piece.
    Rows,
}

/// One per-sequence launch of the prompt attention or the GDN chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillLaunch {
    pub prefix_len: u32,
    pub rows: u32,
}

/// One prefill step: the pieces TensorFold prefills before a verify round.
/// Empty means no prefill this iteration.
///
/// Not modeled: a piece that keeps a state inside its rows (`keep_at`, a
/// cached prompt prefix) splits its GDN chain into two launches.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefillForward {
    pub pieces: Vec<PrefillPiece>,
}

impl PrefillForward {
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// `None` when the step is empty; otherwise fixed by the piece count.
    pub fn path(&self) -> Option<PrefillPath> {
        match self.pieces.len() {
            0 => None,
            1 => Some(PrefillPath::State),
            _ => Some(PrefillPath::Rows),
        }
    }

    /// Rows over all pieces.
    pub fn total_rows(&self) -> u32 {
        self.pieces.iter().map(|piece| piece.rows).sum()
    }

    /// Rows of each forward the step runs, in order: every chunk of the one
    /// piece on [`PrefillPath::State`], else one forward over all rows. The
    /// row-wise leaves (linears, norms, quants, MLP) cost one launch per entry.
    pub fn forward_rows(&self) -> Vec<u32> {
        match self.path() {
            None => Vec::new(),
            Some(PrefillPath::State) => self.launches().iter().map(|l| l.rows).collect(),
            Some(PrefillPath::Rows) => vec![self.total_rows()],
        }
    }

    /// Per-sequence launches of the prompt attention and the GDN chain, in
    /// order: each chunk of the one piece, or each piece of a batched step.
    pub fn launches(&self) -> Vec<PrefillLaunch> {
        match self.path() {
            None => Vec::new(),
            Some(PrefillPath::State) => {
                let piece = self.pieces[0];
                even_chunks(
                    piece.prefix_len,
                    piece.prefix_len + piece.rows,
                    PREFILL_CHUNK_ROWS,
                )
                .into_iter()
                .map(|(start, end)| PrefillLaunch {
                    prefix_len: start,
                    rows: end - start,
                })
                .collect()
            }
            Some(PrefillPath::Rows) => self
                .pieces
                .iter()
                .map(|piece| PrefillLaunch {
                    prefix_len: piece.prefix_len,
                    rows: piece.rows,
                })
                .collect(),
        }
    }

    /// Prompts the step finishes: each runs one m = 1 lm_head.
    pub fn finishing_prompts(&self) -> u32 {
        self.pieces
            .iter()
            .filter(|piece| piece.finishes_prompt)
            .count() as u32
    }

    /// Every piece commits at least one row.
    pub fn check(&self) -> Result<(), String> {
        for (index, piece) in self.pieces.iter().enumerate() {
            if piece.rows == 0 {
                return Err(format!("prefill piece {index} has no rows"));
            }
        }
        Ok(())
    }
}

/// `prefill.py` `chunks`: `[start, end)` cut into the fewest pieces of at most
/// `size` rows, as even as integer bounds allow.
pub fn even_chunks(start: u32, end: u32, size: u32) -> Vec<(u32, u32)> {
    assert!(
        size > 0 && start <= end,
        "invalid chunking [{start}, {end}) / {size}"
    );
    let span = u64::from(end - start);
    let n = span.div_ceil(u64::from(size));
    let bound = |j: u64| start + (span * j / n) as u32;
    (0..n).map(|j| (bound(j), bound(j + 1))).collect()
}

// ─── one verify round ────────────────────────────────────────────────────────

/// A stream's window this round (`multi.py` `COPY, TREE, ONE = 0, 1, 2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WindowMode {
    /// A chain proposed from the stream's own context (`CopyIndex`).
    Copy,
    /// A DFlash2 tree; only these streams run the drafter.
    Tree,
    /// The pending token alone.
    One,
}

/// One stream's verify window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifyStream {
    /// Committed KV length before the round; excludes the window.
    pub committed_len: u32,
    /// Window rows, the pending root included; positive.
    pub rows: u32,
    /// Draft rows accepted (the root not counted); below `rows`.
    pub accepted: u32,
    /// Shared-memory states the stream's tree needs in `tree_kernel`; 0 for a
    /// chain.
    pub tree_slots: u32,
    pub mode: WindowMode,
}

/// One batched verify forward over every decoding stream's window. Empty means
/// no stream decodes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerifyRound {
    pub streams: Vec<VerifyStream>,
}

/// The `tree_kernel` class for a stream needing `tree_slots` states; `None`
/// above 32, which no kernel variant holds.
pub fn slot_class_of(tree_slots: u32) -> Option<GdnTreeSlotClass> {
    Some(match tree_slots {
        0 => GdnTreeSlotClass::Chain,
        1..=2 => GdnTreeSlotClass::Slots1To2,
        3..=4 => GdnTreeSlotClass::Slots3To4,
        5..=8 => GdnTreeSlotClass::Slots5To8,
        9..=16 => GdnTreeSlotClass::Slots9To16,
        17..=32 => GdnTreeSlotClass::Slots17To32,
        _ => return None,
    })
}

impl VerifyRound {
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    pub fn num_streams(&self) -> u32 {
        self.streams.len() as u32
    }

    /// Verify rows W over all windows: the m of every row-wise verify leaf.
    pub fn total_rows(&self) -> u32 {
        self.streams.iter().map(|s| s.rows).sum()
    }

    /// The launch's `tree_kernel` class: the class of the largest `tree_slots`
    /// over streams (`chain` when empty).
    ///
    /// Panics above 32 slots; [`Self::check`] rejects that first.
    pub fn slot_class(&self) -> GdnTreeSlotClass {
        let slots = self.streams.iter().map(|s| s.tree_slots).max().unwrap_or(0);
        slot_class_of(slots).unwrap_or_else(|| panic!("{slots} tree slots exceed slots_17_32"))
    }

    /// Streams in [`WindowMode::Tree`]: the drafter runs 16 rows for each.
    pub fn tree_streams(&self) -> u32 {
        self.streams
            .iter()
            .filter(|s| s.mode == WindowMode::Tree)
            .count() as u32
    }

    /// The one `gdn_tree_verify_recurrence` launch of a GDN layer.
    pub fn gdn_tree_input(&self) -> GdnTreeVerifyRecurrenceKernelInput {
        GdnTreeVerifyRecurrenceKernelInput {
            batch_size: self.num_streams(),
            num_tokens: self.total_rows(),
            slot_class: self.slot_class(),
        }
    }

    /// The one `tree_verify_attention` launch of a GQA layer:
    /// `(rows, committed_len)` per stream.
    pub fn attention_input(&self) -> TreeVerifyAttentionKernelInput {
        TreeVerifyAttentionKernelInput {
            query_context_pairs: self
                .streams
                .iter()
                .map(|s| (s.rows, s.committed_len))
                .collect(),
        }
    }

    /// The commit's one `gdn_state_replay` launch: every stream replays its
    /// accepted rows plus the root.
    pub fn gdn_replay_input(&self) -> GdnStateReplayKernelInput {
        GdnStateReplayKernelInput {
            batch_size: self.num_streams(),
            num_tokens: self.streams.iter().map(|s| s.accepted + 1).sum(),
        }
    }

    /// Every window has a row, accepts fewer drafts than it has, and needs at
    /// most 32 tree slots.
    pub fn check(&self) -> Result<(), String> {
        for (index, s) in self.streams.iter().enumerate() {
            if s.rows == 0 {
                return Err(format!("verify stream {index} has no rows"));
            }
            if s.accepted >= s.rows {
                return Err(format!(
                    "verify stream {index} accepts {} of {} rows (the root is not a draft)",
                    s.accepted, s.rows
                ));
            }
            if slot_class_of(s.tree_slots).is_none() {
                return Err(format!(
                    "verify stream {index} needs {} tree slots; at most 32",
                    s.tree_slots
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};
    use crate::timing::kernels::SingleGemmKernelInput;
    use std::sync::Arc;

    fn piece(prefix_len: u32, rows: u32, finishes_prompt: bool) -> PrefillPiece {
        PrefillPiece {
            prefix_len,
            rows,
            finishes_prompt,
        }
    }

    fn stream(rows: u32, tree_slots: u32, mode: WindowMode) -> VerifyStream {
        VerifyStream {
            committed_len: 100,
            rows,
            accepted: rows - 1,
            tree_slots,
            mode,
        }
    }

    #[test]
    fn even_chunks_match_prefill_py() {
        // chunks(0, 10000) = [(0, 3333), (3333, 6666), (6666, 10000)]
        assert_eq!(
            even_chunks(0, 10000, 4096),
            [(0, 3333), (3333, 6666), (6666, 10000)]
        );
        assert_eq!(even_chunks(500, 4596, 4096), [(500, 4596)]);
        assert_eq!(even_chunks(7, 7, 4096), []);
        assert_eq!(even_chunks(0, 8193, 4096).len(), 3);
    }

    #[test]
    fn single_piece_runs_a_forward_per_chunk() {
        let step = PrefillForward {
            pieces: vec![piece(1000, 9000, true)],
        };
        assert_eq!(step.path(), Some(PrefillPath::State));
        assert_eq!(step.forward_rows(), [3000, 3000, 3000]);
        assert_eq!(
            step.launches(),
            [
                PrefillLaunch {
                    prefix_len: 1000,
                    rows: 3000
                },
                PrefillLaunch {
                    prefix_len: 4000,
                    rows: 3000
                },
                PrefillLaunch {
                    prefix_len: 7000,
                    rows: 3000
                },
            ]
        );
        assert_eq!(step.finishing_prompts(), 1);
    }

    #[test]
    fn several_pieces_run_one_forward_and_a_launch_each() {
        let step = PrefillForward {
            pieces: vec![piece(0, 600, false), piece(2048, 424, true)],
        };
        assert_eq!(step.path(), Some(PrefillPath::Rows));
        assert_eq!(step.forward_rows(), [1024]);
        assert_eq!(
            step.launches(),
            [
                PrefillLaunch {
                    prefix_len: 0,
                    rows: 600
                },
                PrefillLaunch {
                    prefix_len: 2048,
                    rows: 424
                },
            ]
        );
        assert_eq!(step.finishing_prompts(), 1);
        assert!(step.check().is_ok());
    }

    #[test]
    fn empty_step_has_no_forwards_and_zero_rows_fail_check() {
        let empty = PrefillForward::default();
        assert_eq!(empty.path(), None);
        assert!(empty.forward_rows().is_empty() && empty.launches().is_empty());
        let bad = PrefillForward {
            pieces: vec![piece(0, 0, false)],
        };
        assert!(bad.check().unwrap_err().contains("no rows"));
    }

    #[test]
    fn slot_class_follows_the_arch_notes_table() {
        use GdnTreeSlotClass::*;
        for (slots, class) in [
            (0, Chain),
            (1, Slots1To2),
            (2, Slots1To2),
            (3, Slots3To4),
            (4, Slots3To4),
            (5, Slots5To8),
            (8, Slots5To8),
            (9, Slots9To16),
            (16, Slots9To16),
            (17, Slots17To32),
            (32, Slots17To32),
        ] {
            assert_eq!(slot_class_of(slots), Some(class), "{slots} slots");
        }
        assert_eq!(slot_class_of(33), None);
    }

    #[test]
    fn round_takes_the_max_class_and_derives_launch_inputs() {
        let round = VerifyRound {
            streams: vec![
                stream(16, 4, WindowMode::Tree),
                stream(5, 0, WindowMode::Copy),
                stream(1, 0, WindowMode::One),
                stream(9, 2, WindowMode::Tree),
            ],
        };
        assert!(round.check().is_ok());
        assert_eq!(round.slot_class(), GdnTreeSlotClass::Slots3To4);
        assert_eq!(round.tree_streams(), 2);
        let tree = round.gdn_tree_input();
        assert_eq!((tree.batch_size, tree.num_tokens), (4, 31));
        assert_eq!(round.attention_input().query_context_pairs[1], (5, 100));
        let replay = round.gdn_replay_input();
        // Every row of these windows is accepted: Σ(accepted + 1) = Σ rows.
        assert_eq!((replay.batch_size, replay.num_tokens), (4, 31));
        assert_eq!(VerifyRound::default().slot_class(), GdnTreeSlotClass::Chain);
    }

    #[test]
    fn round_check_rejects_bad_windows() {
        let mut s = stream(4, 0, WindowMode::Copy);
        s.accepted = 4;
        assert!(VerifyRound { streams: vec![s] }
            .check()
            .unwrap_err()
            .contains("accepts"));
        let s = stream(16, 33, WindowMode::Tree);
        assert!(VerifyRound { streams: vec![s] }
            .check()
            .unwrap_err()
            .contains("33"));
    }

    /// The GDN verify launch is atomic: [`VerifyRound::gdn_tree_input`] is
    /// exactly the kernel's input. Against the GB10 `tensorfold_cuda` rows in
    /// `profiling/profile.db`. Run from the repo root with libpython on
    /// `LD_LIBRARY_PATH`:
    /// `PYTHONPATH=$PWD uv run cargo test -p simulator --lib tensorfold_common -- --ignored`.
    #[test]
    #[ignore = "needs profile.db GB10 rows and the venv interpreter"]
    fn gb10_rows_price_the_gdn_tree_launch() {
        use crate::timing::kernels::{
            GdnTreeVerifyRecurrenceKernel, GdnTreeVerifyRecurrenceKernelConfig,
        };
        use crate::timing::PerfApiBridge;

        let bridge = PerfApiBridge::new().expect("perf_api bridge");
        let config = GdnTreeVerifyRecurrenceKernelConfig {
            backends: vec![TENSORFOLD_CUDA],
            gpu_name: "NVIDIA GB10".to_string(),
            num_qk_heads: 16.into(),
            num_value_heads: 48.into(),
            key_head_dim: 128.into(),
            value_head_dim: 128.into(),
            dtype: DType::Bf16,
            state_dtype: DType::Fp32,
        };
        let op = atomic(
            "m.gdn",
            "tree",
            config,
            GdnTreeVerifyRecurrenceKernel::build,
            &bridge,
        )
        .expect("gdn_tree_verify_recurrence rows on GB10");
        let price = |round: &VerifyRound| op.kernel.eval(&round.gdn_tree_input());
        let chain = VerifyRound {
            streams: vec![stream(16, 0, WindowMode::Copy); 4],
        };
        let tree = VerifyRound {
            streams: vec![stream(16, 4, WindowMode::Tree); 4],
        };
        let mixed = VerifyRound {
            streams: vec![
                stream(16, 4, WindowMode::Tree),
                stream(5, 0, WindowMode::Copy),
                stream(1, 0, WindowMode::One),
            ],
        };
        let (c, t, m) = (price(&chain), price(&tree), price(&mixed));
        eprintln!(
            "gdn tree GB10 ms: 4x16 chain {:.4}, 4x16 slots_3_4 {:.4}, 16+5+1 slots_3_4 {:.4}",
            c.m.time_ms, t.m.time_ms, m.m.time_ms
        );
        for leaf in [c, t, m] {
            assert!(leaf.m.time_ms > 0.0 && leaf.m.time_ms.is_finite());
        }
        // A tree class costs more than a chain of the same rows.
        assert!(t.m.time_ms > c.m.time_ms);
    }

    struct RowsKernel;

    impl Probe for RowsKernel {
        type Input = SingleGemmKernelInput;
        fn eval(&self, input: &SingleGemmKernelInput) -> LeafMetrics {
            LeafMetrics {
                m: Metrics4 {
                    time_ms: input.m as f32,
                    flops: 1.0,
                    bytes: 2.0,
                    energy_j: 0.0,
                },
                coverage: CoverageFlags::EMPTY,
                backend_index: 1,
            }
        }
        fn kind(&self) -> &'static str {
            "single_gemm"
        }
        fn describe_config(&self) -> serde_json::Value {
            serde_json::Value::Null
        }
    }

    #[test]
    fn push_sum_folds_launches_into_one_slot_and_logs_each() {
        let op = Op::new("m.qkv".to_string(), Arc::new(RowsKernel));
        let mut buf = vec![LeafMetrics::MISS; 2];
        let mut inputs = Vec::new();
        let mut ev = Evaluator::with_inputs(&mut buf, &mut inputs);
        push_sum(
            &op,
            &[
                SingleGemmKernelInput { m: 3000 },
                SingleGemmKernelInput { m: 3000 },
            ],
            &mut ev,
        );
        push_sum(&op, &[], &mut ev);
        assert_eq!(ev.filled(), 2);
        assert_eq!((buf[0].m.time_ms, buf[0].backend_index), (6000.0, 1));
        assert_eq!(buf[1].m.time_ms, 0.0);
        assert_eq!(
            serde_json::to_value(&inputs).unwrap(),
            serde_json::json!([
                {"launches": [{"m": 3000}, {"m": 3000}]},
                {"launches": []},
            ])
        );
    }
}
